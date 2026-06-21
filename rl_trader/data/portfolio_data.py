"""Multi-asset (cross-sectional) data pipeline.

Where :mod:`rl_trader.data.data_loader` prepares *one* asset, this module aligns
a whole **basket** onto a shared calendar so a portfolio agent can reason across
assets at the same instant — the prerequisite for cross-sectional allocation
(long the strong names, short the weak ones) rather than timing a single ticker.

The agent consumes two aligned arrays:

    features  ndarray[T, N, F]   scaled model inputs, per (time, asset, feature)
    prices    ndarray[T, N]      raw close prices per (time, asset)

Assets are **inner-joined on date**, so every row is a day on which *every* asset
traded — no forward-filling, no look-ahead. Feature scaling statistics are fit on
the **training rows only** (pooled across assets, per feature) and applied to all
splits, so validation/test never leak into training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data_loader import FEATURE_COLUMNS, add_technical_indicators, load_ohlcv_csv


@dataclass
class PortfolioData:
    """A scaled feature tensor aligned with its raw price matrix.

    ``features`` is ``[T, N, F]`` and ``prices`` is ``[T, N]`` for ``N`` assets
    over ``T`` shared dates.
    """

    features: np.ndarray   # [T, N, F], float32, scaled
    prices: np.ndarray     # [T, N],    float32, raw close prices
    feature_names: List[str]
    tickers: List[str]

    def __len__(self) -> int:
        return len(self.prices)

    @property
    def n_assets(self) -> int:
        return self.prices.shape[1]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _featured_frames(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Engineer indicators per asset; keep a date index for alignment."""
    out: Dict[str, pd.DataFrame] = {}
    for ticker, df in dfs.items():
        feat = add_technical_indicators(df)
        if "date" not in feat.columns:
            raise ValueError(f"{ticker}: a 'date' column is required to align assets.")
        out[ticker] = feat.set_index("date")
    return out


def build_portfolio_data(
    raw: Dict[str, pd.DataFrame],
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.0,
) -> Dict[str, PortfolioData]:
    """Align a basket of raw OHLCV frames and split it chronologically.

    Parameters
    ----------
    raw:
        ``{ticker: ohlcv_dataframe}`` (each with a ``date`` column).
    train_frac / val_frac:
        Chronological split fractions; the remainder is the test split. The
        scaler is fit on the training rows only.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"`` → :class:`PortfolioData`.
    """
    if len(raw) < 2:
        raise ValueError("a portfolio needs at least two assets")

    featured = _featured_frames(raw)
    tickers = sorted(featured)

    # Inner-join on date so every retained row is a day all assets traded.
    common = None
    for df in featured.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()
    if len(common) < 100:
        raise ValueError(f"only {len(common)} shared dates across the basket — too few")

    # Stack into [T, N, F] (features) and [T, N] (prices), asset order = `tickers`.
    feats = np.stack(
        [featured[t].loc[common, FEATURE_COLUMNS].to_numpy(np.float32) for t in tickers],
        axis=1,
    )
    prices = np.stack(
        [featured[t].loc[common, "close"].to_numpy(np.float32) for t in tickers], axis=1
    )

    n = len(common)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    # Fit per-feature mean/std on training rows, pooled across all assets.
    train_block = feats[:train_end].reshape(-1, feats.shape[2])
    mean = train_block.mean(axis=0)
    std = train_block.std(axis=0)
    std[std < 1e-8] = 1.0

    def scale(a: np.ndarray) -> np.ndarray:
        return ((a - mean) / std).astype(np.float32)

    def make(lo: int, hi: int) -> PortfolioData:
        return PortfolioData(scale(feats[lo:hi]), prices[lo:hi], FEATURE_COLUMNS, tickers)

    return {
        "train": make(0, train_end),
        "val": make(train_end, val_end),
        "test": make(val_end, n),
    }


def load_portfolio(
    data_dir: str,
    market: str = "stock",
    *,
    train_frac: float = 0.6,
    tickers: Optional[List[str]] = None,
) -> Dict[str, PortfolioData]:
    """Load every ticker CSV under ``data_dir/market`` into a portfolio split."""
    import glob
    import os

    raw: Dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(os.path.join(data_dir, market, "*.csv"))):
        ticker = os.path.splitext(os.path.basename(path))[0]
        if tickers and ticker not in tickers:
            continue
        raw[ticker] = load_ohlcv_csv(path)
    if not raw:
        raise SystemExit(f"No CSVs found in {os.path.join(data_dir, market)}.")
    return build_portfolio_data(raw, train_frac=train_frac, val_frac=0.0)


def synthetic_portfolio(
    n_assets: int = 5,
    market: str = "stock",
    seed: Optional[int] = None,
    n_steps: int = 1_400,
) -> Dict[str, PortfolioData]:
    """Build a self-contained synthetic basket (no external data needed).

    Each asset is an independent trending series sharing the market regime, so the
    portfolio machinery runs end-to-end in tests and demos with zero downloads.
    """
    from .data_loader import generate_synthetic_ohlcv, market_regime

    rng = np.random.default_rng(seed)
    vol, drift, mom = market_regime(market)
    raw: Dict[str, pd.DataFrame] = {}
    base = pd.Timestamp("2010-01-01")
    for i in range(n_assets):
        df = generate_synthetic_ohlcv(
            n_steps=n_steps, annual_vol=vol, annual_drift=drift, momentum=mom,
            seed=int(rng.integers(1e9)),
        )
        # A shared business-day calendar so the assets inner-join cleanly.
        df.insert(0, "date", pd.bdate_range(base, periods=len(df)))
        raw[f"SYN{i}"] = df
    return build_portfolio_data(raw, train_frac=0.6, val_frac=0.0)
