"""Download real daily OHLCV data for the RL-Trader basket via Yahoo Finance.

Writes one clean CSV per ticker to ``data/raw/<market>/<TICKER>.csv`` with the
columns the framework's loader expects: ``date,open,high,low,close,volume``.

Yahoo aggressively rate-limits bursts, so this fetches **one ticker at a time**
with exponential backoff and a polite delay, and **caches** — re-running skips
tickers already downloaded. Run from the repo root:

    python tools/fetch_data.py                 # fetch everything (cached)
    python tools/fetch_data.py --force         # re-download even if cached

Pinning a snapshot (reproducibility)
------------------------------------
``PERIOD`` requests a *relative* window ("10y", "max"), so the data you get
depends on **when you run this**. Re-fetching months later slides the whole
train/val/test split and the resulting figures will not match an earlier run —
the same recipe on a different slice of history is a different experiment.

Pass ``--end`` (optionally with ``--start``) to truncate every series to a fixed
calendar range, and the run records ``data/SNAPSHOT.json``: a per-file SHA-256,
row count, and date range. Commit that file and any later fetch can be checked
against it::

    python tools/fetch_data.py --end 2026-06-17          # pin and record
    python tools/fetch_data.py --verify                  # check against the pin

``--verify`` exits non-zero on any drift, so it can gate a rebuild in CI.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
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


DEFAULT_SNAPSHOT = os.path.join("data", "SNAPSHOT.json")


def _clip(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Truncate to a fixed calendar range so the series stops being time-dependent."""
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    return df.reset_index(drop=True)


def _digest(path: str) -> dict:
    """Content hash plus the shape facts worth eyeballing in a diff."""
    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    df = pd.read_csv(path)
    return {
        "sha256": sha,
        "rows": int(len(df)),
        "first_date": str(df["date"].iloc[0]) if len(df) else None,
        "last_date": str(df["date"].iloc[-1]) if len(df) else None,
    }


def _scan(root: str) -> dict:
    """Digest every CSV under ``root``, keyed by ``market/TICKER``."""
    files = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "*.csv"))):
        market = os.path.basename(os.path.dirname(path))
        ticker = os.path.splitext(os.path.basename(path))[0]
        files[f"{market}/{ticker}"] = _digest(path)
    return files


def write_snapshot(root: str, out_path: str, start: str | None, end: str | None) -> dict:
    """Record the current dataset so a later fetch can be verified against it."""
    snapshot = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pinned_start": start,
        "pinned_end": end,
        "root": root.replace("\\", "/"),
        "files": _scan(root),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return snapshot


def verify_snapshot(root: str, snapshot_path: str) -> list[str]:
    """Return a list of human-readable drifts against a recorded snapshot."""
    with open(snapshot_path, encoding="utf-8") as fh:
        snapshot = json.load(fh)
    expected = snapshot.get("files", {})
    actual = _scan(root)

    problems = []
    for key, want in sorted(expected.items()):
        have = actual.get(key)
        if have is None:
            problems.append(f"{key}: missing (expected {want['rows']} rows)")
        elif have["sha256"] != want["sha256"]:
            problems.append(
                f"{key}: content differs — expected {want['rows']} rows "
                f"({want['first_date']}..{want['last_date']}), found {have['rows']} rows "
                f"({have['first_date']}..{have['last_date']})"
            )
    for key in sorted(set(actual) - set(expected)):
        problems.append(f"{key}: present but not in the snapshot")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real OHLCV for the basket.")
    parser.add_argument("--out", default="data/raw", help="Output directory root.")
    parser.add_argument("--force", action="store_true", help="Ignore the cache.")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between tickers.")
    parser.add_argument("--start", default=None, help="Pin the first date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Pin the last date (YYYY-MM-DD).")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT,
                        help="Where to write/read the dataset snapshot.")
    parser.add_argument("--verify", action="store_true",
                        help="Check existing CSVs against the snapshot and exit.")
    args = parser.parse_args()

    if args.verify:
        if not os.path.exists(args.snapshot):
            raise SystemExit(f"no snapshot at {args.snapshot} — run with --end to create one")
        problems = verify_snapshot(args.out, args.snapshot)
        if problems:
            print(f"Dataset drift vs {args.snapshot}:")
            for p in problems:
                print(f"  !! {p}")
            raise SystemExit(1)
        print(f"Dataset matches {args.snapshot}.")
        return

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
            df = _clip(df, args.start, args.end)
            if df.empty:
                print(f"    !! {ticker} has no rows inside {args.start}..{args.end}")
                missed.append(ticker)
                continue
            df.to_csv(path, index=False)
            print(f"    saved {len(df)} rows -> {path}")
            got += 1
            time.sleep(args.delay)

    print(f"\nDone: {got} cached/saved, {len(missed)} missing {missed}")

    if args.start or args.end:
        snap = write_snapshot(args.out, args.snapshot, args.start, args.end)
        print(
            f"Pinned {len(snap['files'])} files -> {args.snapshot} "
            f"(range {args.start or 'earliest'}..{args.end or 'latest'}). "
            "Commit it, then gate rebuilds with --verify."
        )
    else:
        print(
            "NOTE: unpinned fetch — PERIOD is relative, so this window moves over "
            "time and results will not reproduce. Pass --end YYYY-MM-DD to pin it."
        )


if __name__ == "__main__":
    main()
