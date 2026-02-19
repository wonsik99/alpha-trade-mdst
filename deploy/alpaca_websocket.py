"""
IEX WebSocket stream: live 15-minute bar data for the trading bot.

WHAT IS STREAMING?
  Instead of repeatedly asking the server "give me the latest price" (polling),
  we open a long-lived connection (WebSocket). The server pushes new data to us
  as it arrives (e.g. every minute a new "bar" = OHLCV for that minute). This
  is more efficient and real-time than REST API calls.

WHY A BACKGROUND THREAD?
  The WebSocket library (alpaca-py) runs an "event loop" that waits for
  incoming messages. That loop blocks: it never returns until we stop it.
  Our main program also needs to run (e.g. every 15 minutes: get data,
  run the policy, place orders). So we run the WebSocket in a separate
  "background thread". The main thread and the stream thread run at the
  same time: the stream keeps receiving bars and storing them; the main
  thread periodically reads the stored bars via get_latest_15min_data().

THREAD SAFETY:
  The stream thread writes to _minute_buffers and _fifteen_min_bars; the
  main thread reads them in get_latest_15min_data(). To avoid one thread
  reading while the other is writing (which can cause corrupt data or
  crashes), we use a Lock. Only one thread can hold the lock at a time:
  the stream thread acquires it when updating buffers; the main thread
  acquires it when copying data out. So reads and writes never overlap.

Docs: https://alpaca.markets/sdks/python/api_reference/data/stock/live.html
"""

import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from alpaca.data.enums import DataFeed
from alpaca.data.live.stock import StockDataStream

from alpaca_utils import TRAINING_TICKERS

# How many 15-min bars to keep per symbol for technical indicators lookback
MAX_15MIN_BARS = 200


def _bar_to_tuple(bar) -> tuple:
    """
    Convert one bar from Alpaca's format to (timestamp, open, high, low, close, volume).
    The stream can send either a Bar objects or a raw dict (keys like 't', 'o', 'h').
    """
    if hasattr(bar, "timestamp"):
        ts = bar.timestamp
        o, h, l, c = bar.open, bar.high, bar.low, bar.close
        v = getattr(bar, "volume", 0) or 0
    else:
        ts = pd.Timestamp(bar["t"]).to_pydatetime()
        o, h, l, c = float(bar["o"]), float(bar["h"]), float(bar["l"]), float(bar["c"])
        v = int(bar.get("v", 0))
    return (ts, o, h, l, c, v)


def _bucket_15min(ts: datetime) -> datetime:
    """
    Round a timestamp down to the start of its 15-minute window (e.g. 10:07 -> 10:00).
    Used to group minute bars into 15-min buckets before aggregating.
    """
    return ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)


class IEXStream15MinFetcher:
    """
    Connects to Alpaca's IEX WebSocket, receives minute bars, aggregates them
    into 15-minute bars, and exposes the latest data for the trading bot.

    - The stream runs in a background thread (start() / stop()).
    - The main thread calls get_latest_15min_data() to read the current
      15-min bar data without blocking the stream.
    """

    def __init__(self, api_key: str, secret_key: str):
        self._api_key = api_key
        self._secret_key = secret_key
        # Buffers used by the stream thread 
        self._minute_buffers: Dict[str, List[tuple]] = defaultdict(list)   # current 15-min window of minute bars per symbol
        self._current_bucket_ts: Dict[str, Optional[datetime]] = defaultdict(lambda: None)  # which 15-min window we're filling
        self._fifteen_min_bars: Dict[str, List[tuple]] = defaultdict(list)  # completed 15-min bars (ts, o, h, l, c, v)
        self._lock = threading.Lock()
        self._stream: Optional[StockDataStream] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()  # set once we have at least one 15-min bar (optional for callers)

    def _aggregate_bucket(self, symbol: str, bars: List[tuple]) -> tuple:
        """Turn a list of minute bars in one 15-min window into one 15-min bar: open=first, high=max, low=min, close=last, volume=sum."""
        if not bars:
            return None
        ts = _bucket_15min(bars[0][0])
        o = bars[0][1]
        h = max(b[2] for b in bars)
        l = min(b[3] for b in bars)
        c = bars[-1][4]
        v = sum(b[5] for b in bars)
        return (ts, o, h, l, c, v)

    async def _on_bar(self, bar):
        """
        Called by the WebSocket library every time a new minute bar arrives.
        We add it to the current 15-min bucket for that symbol. When the
        next minute belongs to a new 15-min window, we "flush" the previous
        bucket (aggregate to one 15-min bar) and store it.
        """
        if hasattr(bar, "symbol"):
            symbol = bar.symbol
        else:
            symbol = bar.get("S") or bar.get("symbol")
        row = _bar_to_tuple(bar)
        ts = row[0]
        bucket_ts = _bucket_15min(ts)

        with self._lock:
            prev_bucket = self._current_bucket_ts[symbol]
            if prev_bucket is not None and bucket_ts != prev_bucket:
                # New 15-min window started; save the previous window as one 15-min bar.
                agg = self._aggregate_bucket(symbol, self._minute_buffers[symbol])
                if agg:
                    lst = self._fifteen_min_bars[symbol]
                    lst.append(agg)
                    if len(lst) > MAX_15MIN_BARS:
                        lst.pop(0)
                    self._ready.set()
                self._minute_buffers[symbol] = []
            self._current_bucket_ts[symbol] = bucket_ts
            self._minute_buffers[symbol].append(row)

    def _run_stream(self):
        """
        Entry point for the background thread. Creates the WebSocket client,
        subscribes to minute bars for our tickers, and runs the event loop.
        This method blocks until the stream is stopped.
        """
        self._stream = StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            feed=DataFeed.IEX,
            raw_data=False,
        )
        self._stream.subscribe_bars(self._on_bar, *TRAINING_TICKERS)
        self._stream.run()

    def start(self):
        """
        Start the WebSocket in a background thread so the main program can
        keep running. The thread is "daemon" so it won't prevent the process
        from exiting when the main thread exits.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_stream, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=30)

    def stop(self):
        """Stop the WebSocket and the background thread."""
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            self._stream = None

    def get_latest_15min_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Called from the main thread. Returns the latest 15-min bar data for
        all symbols as Dict[symbol -> DataFrame]. Each DataFrame has columns
        Open, High, Low, Close, Volume and a DatetimeIndex; we keep up to
        MAX_15MIN_BARS rows for indicator lookback. Returns None if we don't
        yet have at least one 15-min bar for every symbol.
        """
        with self._lock:
            if not self._fifteen_min_bars:
                return None
            out = {}
            for symbol in TRAINING_TICKERS:
                rows = self._fifteen_min_bars.get(symbol)
                if not rows:
                    continue
                df = pd.DataFrame(
                    rows,
                    columns=["Date", "Open", "High", "Low", "Close", "Volume"],
                )
                df.set_index("Date", inplace=True)
                df.index = pd.to_datetime(df.index)
                out[symbol] = df
            if len(out) < len(TRAINING_TICKERS):
                return None
            return out