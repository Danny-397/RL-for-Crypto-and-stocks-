"""Download real daily OHLCV data for the RL-Trader basket via Yahoo Finance.

Writes one clean CSV per ticker to ``data/raw/<market>/<TICKER>.csv`` with the
columns the framework's loader expects: ``date,open,high,low,close,volume``.

Yahoo aggressively rate-limits bursts, so this fetches **one ticker at a time**
with exponential backoff and a polite delay, and **caches** — re-running skips
tickers already downloaded. Run from the repo root:

    python tools/fetch_data.py                 # fetch everything (cached)
    python tools/fetch_data.py --force         # re-download even if cached
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

# Modest baskets — enough diversity for the agent to generalize, small enough to
# stay under Yahoo's rate limit. Edit freely.
BASKETS = {
    "stock": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "JNJ", "XOM", "SPY", "QQQ"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD", "ADA-USD"],
}
# yfinance 1.x is far more reliable with a relative ``period`` than an absolute
# start date for long histories.
PERIOD = {"stock": "10y", "crypto": "max"}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a (field, ticker) MultiIndex even for one ticker."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _download(ticker: str, period: str, retries: int = 5):
    """Download one ticker with exponential backoff; return a clean DataFrame."""
    import yfinance as yf

    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker, period=period, auto_adjust=True, progress=False, threads=False
            )
            raw = _flatten_columns(raw)
            if raw is not None and len(raw) > 100:
                out = pd.DataFrame({
                    "date": pd.to_datetime(raw.index).strftime("%Y-%m-%d"),
                    "open": raw["Open"].to_numpy(),
                    "high": raw["High"].to_numpy(),
                    "low": raw["Low"].to_numpy(),
                    "close": raw["Close"].to_numpy(),
                    "volume": raw["Volume"].to_numpy(),
                })
                return out.dropna().reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001 - report and retry
            print(f"    attempt {attempt + 1} failed: {exc}")
        wait = 8 * (2 ** attempt)
        print(f"    rate-limited/empty; waiting {wait}s before retry")
        time.sleep(wait)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real OHLCV for the basket.")
    parser.add_argument("--out", default="data/raw", help="Output directory root.")
    parser.add_argument("--force", action="store_true", help="Ignore the cache.")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between tickers.")
    args = parser.parse_args()

    got, missed = 0, []
    for market, tickers in BASKETS.items():
        out_dir = os.path.join(args.out, market)
        os.makedirs(out_dir, exist_ok=True)
        for ticker in tickers:
            path = os.path.join(out_dir, f"{ticker}.csv")
            if os.path.exists(path) and os.path.getsize(path) > 0 and not args.force:
                print(f"[cache] {market}/{ticker}")
                got += 1
                continue
            print(f"[fetch] {market}/{ticker} ...")
            df = _download(ticker, PERIOD[market])
            if df is None or df.empty:
                print(f"    !! gave up on {ticker}")
                missed.append(ticker)
                continue
            df.to_csv(path, index=False)
            print(f"    saved {len(df)} rows -> {path}")
            got += 1
            time.sleep(args.delay)

    print(f"\nDone: {got} cached/saved, {len(missed)} missing {missed}")


if __name__ == "__main__":
    main()
