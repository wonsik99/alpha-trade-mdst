"""
Alpaca Paper Trading

Workflow: Fetches live 15-min bars market data (via IEX WebSocket),
-> builds an observation
-> gets an action from the policy
-> places orders via the Alpaca REST API

TWO CONNECTIONS:
  - REST API (alpaca_trade_api): used to read account/positions and to
    submit orders (get_account, list_positions, get_latest_trade,
    submit_order). These are "request–response" API calls.
    We send a request, we get one response.
    API Documentation: https://github.com/alpacahq/alpaca-trade-api-python?tab=readme-ov-file

  - WebSocket (alpaca_websocket.py): used only for market data. We open
    one long-lived connection; Alpaca pushes minute bars to us. We
    aggregate them into 15-min bars and read them in the main loop.
    WebSocket Documentation: https://alpaca.markets/sdks/python/api_reference/data/stock/live.html


Every 15 minutes the engine runs: 
(1) Read latest 15-min bars from the WebSocket fetcher.
(2) Add technical indicators.
(3) Build observation.
(4) Policy predicts action.
(5) Place orders via REST.
The WebSocket runs in a background thread the whole time (see alpaca_websocket.py).

Run: python alpaca_live.py
Config: MODE and TRADE_FREQUENCY_MINUTES below (default paper, every 15 min).
"""

import certifi
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import torch

# FIX: Force Python to use certifi's CA bundle to avoid SSL errors on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()

# Add parent directory so we can import agent/ and src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

from agent.actor_critic import ActorCritic
from alpaca_utils import (
    TRAINING_TICKERS,
    prepare_features,
    build_observation,
    place_orders_from_actions,
    get_current_positions,
)

# Make sure to install alpaca-py package: pip install alpaca-py
from alpaca_websocket import IEXStream15MinFetcher

# Load .env file
load_dotenv(Path(__file__).parent / '.env')

# Risk limits: stop trading if we hit these limits
RISK_PARAMS = {
    'max_position_size': 0.15,   # Max 15 % per stock
    'daily_loss_limit': 0.05,    # Stop if down 2 % in a day
    'max_drawdown': 0.30,        # Stop if down 15 % from peak
    'min_trade_value': 100,      # Minimum $100 per trade
}

# Trained PPO policy model 
DEFAULT_MODEL_PATH = str(Path(__file__).parent.parent / 'models' / 'ppo_trading.pt')

class AlpacaPPOTrader:
    """Live trading manager"""
    def __init__(self, api, model_path: str, api_key: str = None, secret_key: str = None):
        self.api = api
        # Include your API keys from Alpaca to .env
        self._iex_fetcher = IEXStream15MinFetcher(api_key, secret_key)
        self.running = False

        # Performance tracking
        self.start_time = datetime.now()
        self.initial_value = None
        self.peak_value = None
        self.daily_start_value = None
        self.portfolio_history = []
        self.trade_step = 0

        # Load model
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load trained PPO policy from .pt checkpoint."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        obs_dim = checkpoint['obs_shape']
        act_dim = checkpoint['action_shape']

        self.policy = ActorCritic(obs_dim, act_dim)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()

        print(f"Policy loaded (obs={obs_dim}, act={act_dim})\n")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Get action from the policy"""
        state = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.policy.forward(state)
            action = torch.tanh(dist.mean)
        return action.cpu().numpy()[0]

    def get_portfolio_value(self):
        """Get total portfolio value """
        try:
            return float(self.api.get_account().portfolio_value)
        except Exception as e:
            print(f"Error getting portfolio value: {e}")
            return None

    def is_market_open(self):
        """Check if the US equity market is currently open"""
        try:
            return self.api.get_clock().is_open
        except Exception as e:
            print(f"  Error checking market status: {e}")
            return False

    def get_market_hours(self):
        """Get next market open and close times (for display when closed)"""
        try:
            clock = self.api.get_clock()
            return clock.next_open, clock.next_close
        except Exception:
            return None, None

    # validate risk limits
    def check_risk_limits(self):
        value = self.get_portfolio_value()
        if value is None:
            return True

        if self.initial_value is None:
            self.initial_value = value
            self.peak_value = value
            self.daily_start_value = value

        if value > self.peak_value:
            self.peak_value = value

        daily_loss = (self.daily_start_value - value) / self.daily_start_value
        if daily_loss > RISK_PARAMS['daily_loss_limit']:
            print(f"\n RISK LIMIT BREACHED: Daily loss {daily_loss*100:.2f}% > limit {RISK_PARAMS['daily_loss_limit']*100:.1f}%")
            return False

        drawdown = (self.peak_value - value) / self.peak_value
        if drawdown > RISK_PARAMS['max_drawdown']:
            print(f"\n RISK LIMIT BREACHED: Drawdown {drawdown*100:.2f}% > limit {RISK_PARAMS['max_drawdown']*100:.1f}%")
            return False

        return True

    # Trading cycle: data -> observation -> action -> orders
    def fetch_latest_data(self):
        """
        Read latest 15-min bars from the WebSocket fetcher
        Check for any missing tickers
        """
        raw = self._iex_fetcher.get_latest_15min_data()
        if raw is None:
            print("No 15-min data from IEX stream yet. Wait for next 15-min bar")
            return None
        if len(raw) < len(TRAINING_TICKERS):
            print(f"Only got {len(raw)}/{len(TRAINING_TICKERS)} tickers")
            return None
        return prepare_features(raw)

    def execute_trade(self):
        """One cycle: get 15-min data, add features, build obs, get action, place orders via REST."""
        print(f"\n{'='*70}")
        print(f"Trade {self.trade_step}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        if not self.check_risk_limits():
            return False

        portfolio_value = self.get_portfolio_value()
        if portfolio_value is None:
            print("Can't read portfolio value. Skipping trade")
            return True

        print(f"Portfolio value: ${portfolio_value:,.2f}")

        # Fetch data
        stock_data = self.fetch_latest_data()
        if stock_data is None:
            print("Data fetch failed. Skipping trade")
            return True

        # Get current positions and cash for observation and order sizing
        positions = get_current_positions(self.api)
        cash = float(self.api.get_account().cash)
        initial = self.initial_value or portfolio_value
        max_nw = self.peak_value or portfolio_value

        obs = build_observation(
            stock_data, balance=cash, shares_held=positions,
            net_worth=portfolio_value, max_net_worth=max_nw,
            current_step=self.trade_step, max_steps=252,
            initial_balance=initial,
        )

        # Policy outputs one action per ticker in [-1, 1]
        action = self.predict(obs)

        orders = place_orders_from_actions(
            self.api, action, TRAINING_TICKERS,
            portfolio_value, positions,
            min_trade_value=RISK_PARAMS['min_trade_value'],
        )
        print(f"Placed {len(orders)} orders")

        self.portfolio_history.append({'time': datetime.now(), 'value': portfolio_value})
        self.trade_step += 1
        return True

    def run(self, trade_frequency_minutes: int):
        """
        Start the WebSocket in a background thread, then every
        trade_frequency_minutes (when market is open) run execute_trade()
        """
        print(f"{'='*70}")
        print(f"Trade interval: every {trade_frequency_minutes} min")
        print(f"Data source: IEX WebSocket (15-min bars)")
        # Start WebSocket in a background thread so we receive bars while the loop runs.
        self._iex_fetcher.start() 
        pv = self.get_portfolio_value()
        print(f"Starting portfolio: ${pv:,.2f}\n")

        self.running = True
        last_trade = None
        last_date = None

        try:
            while self.running:
                now = datetime.now()

                # Reset daily-start value at the beginning of each calendar day.
                if last_date is None or now.date() != last_date:
                    self.daily_start_value = self.get_portfolio_value()
                    print(f"\n New day - Current portfolio value: ${self.daily_start_value:,.2f}")
                    last_date = now.date()

                # Don't trade when market is closed; sleep 2.5 hours then recheck 
                # (4pm - 9:30am is 17.5 hours so sleeping for 2.5 hours guarantees we're rerun when market opens)
                if not self.is_market_open():
                    nxt, _ = self.get_market_hours()
                    if nxt:
                        print(f"Market closed. Next open: {nxt}")
                    time.sleep(9000) # 2.5 hours = 150 minutes = 9000 seconds
                    continue

                # Only run a trade cycle if enough time has passed since the last one.
                elapsed = (now - last_trade).total_seconds() if last_trade else float('inf')
                if elapsed >= trade_frequency_minutes * 60:
                    if not self.execute_trade():
                        print("\n Risk limit breached. Stopping trade cycle")
                        break
                    last_trade = now

                    cv = self.get_portfolio_value()
                    if self.initial_value and cv:
                        tot = (cv - self.initial_value) / self.initial_value * 100
                        day = (cv - self.daily_start_value) / self.daily_start_value * 100
                        print(f"Total return: {tot:+.2f}%  |  Today's return: {day:+.2f}%")
                else:
                    time.sleep(60)
        
        # Handle keyboard interrupt and exceptions
        except KeyboardInterrupt:
            print("\n\n  Interrupted by user")
        except Exception as e:
            print(f"\n\n  Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.running = False
            self._iex_fetcher.stop()
            self._print_summary()

    def _print_summary(self):
        print(f"\nFINAL SUMMARY")
        fv = self.get_portfolio_value()
        if self.initial_value and fv:
            ret = (fv - self.initial_value) / self.initial_value * 100
            print(f"  Initial : ${self.initial_value:,.2f}")
            print(f"  Final   : ${fv:,.2f}")
            print(f"  Return  : {ret:+.2f}%")
            if self.peak_value:
                dd = (self.peak_value - fv) / self.peak_value * 100
                print(f"  Max DD  : {dd:.2f}%")
        print(f"  Runtime : {datetime.now() - self.start_time}")
        print(f"{'='*70}\n")


TRADE_FREQUENCY_MINUTES = 15

def main():
    """Load env, connect to Alpaca REST, then run the trader."""
    api_key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_SECRET_KEY')
    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    if not api_key or not secret:
        print("  Error: Alpaca API keys not found!")
        print("  Create backtest/.env with ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return
    
    # Initialize Alpaca REST API
    api = tradeapi.REST(api_key, secret, base, api_version='v2')
    
    # Check connection 
    try:
        acct = api.get_account()
        print(f"Connected — status: {acct.status}")
        print(f"Portfolio : ${float(acct.portfolio_value):,.2f}")
        print(f"Buying pwr: ${float(acct.buying_power):,.2f}")
    except Exception as e:
        print(f"  Connection failed: {e}")
        return

    trader = AlpacaPPOTrader(
        api, DEFAULT_MODEL_PATH,
        api_key=api_key,
        secret_key=secret,
    )
    trader.run(trade_frequency_minutes=TRADE_FREQUENCY_MINUTES)


if __name__ == '__main__':
    main()