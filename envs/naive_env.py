import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AlphaTradeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    def __init__(self, stock_data, transaction_cost_percent=0.005):
        super().__init__()

        self.stock_data = {ticker: df for ticker, df in stock_data.items() if not df.empty}
        self.tickers = list(self.stock_data.keys())

        if not self.tickers:
            raise ValueError("No stock data provided")

        sample_df = next(iter(self.stock_data.values()))
        self.n_features = len(sample_df.columns)

        # Action space: one continuous value per stock in [-1, 1] (sell ... hold ... buy)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.tickers),), dtype=np.float32)

        # Observation space: price data for each stock + balance + shares held + net worth + max net worth + current step 
        self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 1
        # Use bounded observation space to prevent extreme values that cause NaN
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.obs_shape,), dtype=np.float32)

        # Initialize account balance and portfolio state
        self.initial_balance = 100000
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        self.current_step = 0
        self.max_steps = max(0, min(len(df) for df in self.stock_data.values()) - 1)
        self.transaction_cost_percent = transaction_cost_percent

        # Pre-compute feature normalization stats (z-score) so raw prices/volumes
        # don't all clip to 10.0 and destroy information for the network
        all_feat = np.concatenate([df.values for df in self.stock_data.values()], axis=0)
        self._feat_mean = np.mean(all_feat, axis=0).astype(np.float32)
        self._feat_std  = np.std(all_feat, axis=0).astype(np.float32)
        self._feat_std[self._feat_std < 1e-8] = 1.0  # avoid /0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset all state variables for a new episode
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        self.current_step = 0

        return self._next_observation(), {} 

    def _next_observation(self):
        # Initialize the frame 
        frame = np.zeros(self.obs_shape, dtype=np.float32)

        # Add stock data for each ticker 
        idx = 0 
        for ticker in self.tickers: 
            df = self.stock_data[ticker]  
            # If the current step is less than the length of the dataframe, add the data to the frame  
            if self.current_step < len(df): 
                values = df.iloc[self.current_step].values.astype(np.float32)
            elif len(df) > 0: 
                values = df.iloc[-1].values.astype(np.float32)
            else:
                values = np.zeros(self.n_features, dtype=np.float32)
            # Z-score normalize so prices/volumes are on a reasonable scale
            values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
            values = (values - self._feat_mean) / self._feat_std
            frame[idx:idx+self.n_features] = values
            # If the current step is greater than or equal to the length of the dataframe, add the last value to the frame  
            idx += self.n_features  

        # Add balance, shares held, net worth, max net worth, and current step 
        # Normalize balance and net worth by initial balance to prevent large values
        # Clip all values to prevent inf/nan
        balance_norm = np.clip(self.balance / self.initial_balance, -10.0, 10.0)
        frame[-4-len(self.tickers)] = balance_norm

        shares_norm = [np.clip(self.shares_held[ticker] / 1000.0, -10.0, 10.0) for ticker in self.tickers]
        frame[-3-len(self.tickers):-3] = shares_norm

        net_worth_norm = np.clip(self.net_worth / self.initial_balance, -10.0, 10.0)
        frame[-3] = net_worth_norm

        max_net_worth_norm = np.clip(self.max_net_worth / self.initial_balance, -10.0, 10.0)
        frame[-2] = max_net_worth_norm

        step_norm = np.clip(self.current_step / max(self.max_steps, 1), 0.0, 1.0)
        frame[-1] = step_norm

        # Final check: replace any remaining NaN or Inf values
        frame = np.nan_to_num(frame, nan=0.0, posinf=10.0, neginf=-10.0)
        frame = np.clip(frame, -10.0, 10.0)  # Final safety clip
        
        return frame 

    def step(self, action):
        self.current_step += 1  
        
        # Check if the episode is done (reached max steps) 
        if self.current_step >= self.max_steps: 
            return self._next_observation(), 0.0, True, False, {} 
        
        # Flatten action if it comes from DummyVecEnv (2D array)
        if isinstance(action, np.ndarray):
            action = action.flatten()
        
        current_prices = {}
        # Loop through each ticker and perform action 
        for i, ticker in enumerate(self.tickers): 
            current_prices[ticker] = self.stock_data[ticker].iloc[self.current_step]['Close']
            ticker_action = action[i] 

            # Calculate cost and update balance based on selected action
            if ticker_action > 0:  # Buy 
                # Spend a % of balance based on action magnitude
                shares_to_buy = int(self.balance * ticker_action / current_prices[ticker])
                
                # Gross cost
                cost = shares_to_buy * current_prices[ticker]
                
                # Transaction fee
                transaction_cost = cost * self.transaction_cost_percent
                
                # Update balance
                if shares_to_buy > 0 and cost + transaction_cost <= self.balance:
                    self.balance -= cost + transaction_cost
                    self.shares_held[ticker] += shares_to_buy

            elif ticker_action < 0:  # Sell 
                # Sell a % of held shares (clamp to what we actually own)
                shares_to_sell = min(int(self.shares_held[ticker] * abs(ticker_action)),
                                     self.shares_held[ticker])
                
                # Gross revenue
                sale = shares_to_sell * current_prices[ticker]
                
                # Transaction fee
                transaction_cost = sale * self.transaction_cost_percent
                
                # Update balance and inventory
                if shares_to_sell > 0:
                    self.balance += sale - transaction_cost
                    self.shares_held[ticker] -= shares_to_sell    
                    self.total_shares_sold[ticker] += shares_to_sell     
                    self.total_sales_value[ticker] += sale  

        # Calculate the net worth 
        self.net_worth = self.balance + sum(self.shares_held[ticker] * current_prices[ticker] for ticker in self.tickers) 
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Reward: normalized change from initial balance
        reward = (self.net_worth - self.initial_balance) / self.initial_balance  
        # Clip reward to prevent extreme values that can cause NaN
        reward = np.clip(reward, -10.0, 10.0)
        # Done if negative net worth or reached max steps
        done = self.net_worth <= 0 or self.current_step >= self.max_steps 

        obs = self._next_observation()
        return obs, reward, done, False, {}

    def render(self, mode="human"):
        # Print the current step, balance, shares held, net worth, and profit 
        profit = self.net_worth - self.initial_balance
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:,.2f}")
        for ticker in self.tickers:
            print(f"{ticker} Shares held: {self.shares_held[ticker]}")
        print(f"Net worth: ${self.net_worth:,.2f}")
        print(f"Profit: ${profit:,.2f}")

    def close(self):
        pass
