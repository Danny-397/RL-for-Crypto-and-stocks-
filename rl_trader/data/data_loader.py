"""Data pipeline: load OHLCV, engineer features, scale, and split.

The environments consume two aligned NumPy arrays:

    features  ndarray[T, F]   scaled model inputs (indicators + returns)
    prices    ndarray[T]      raw close prices used for execution / PnL

Keeping *raw prices* separate from *scaled features* is deliberate: the agent
should reason over normalised, stationary features, but trades must execute at
true prices so portfolio accounting stays correct.

A synthetic geometric-Brownian-motion generator is included so the whole
framework trains end-to-end with zero external data. Swap in real CSVs (or
yfinance) by passing a DataFrame to :func:`prepare_market_data`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Columns every loader is expected to provide (case-insensitive on read).
OHLCV_COLUMNS: List[str] = ["open", "high", "low", "close", "volume"]


@dataclass
class MarketData:
    """A scaled feature matrix aligned with its raw close-price series."""

    features: np.ndarray  # shape [T, F], float32, already scaled
    prices: np.ndarray    # shape [T],    float32, raw close prices
    feature_names: List[str]

    def __len__(self) -> int:
        return len(self.prices)


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load_ohlcv_csv(path: str, date_column: Optional[str] = "date") -> pd.DataFrame:
    """Load an OHLCV CSV into a clean, lower-cased DataFrame sorted by time.

    The CSV is expected to contain (at least) open/high/low/close/volume
    columns. Column names are matched case-insensitively. Extra columns are
    preserved so custom features can be added downstream.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path!r} is missing required columns: {missing}")
    return df


def generate_synthetic_ohlcv(
    n_steps: int = 4_000,
    start_price: float = 100.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.30,
    bars_per_year: int = 252,
    momentum: float = 0.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a realistic-looking OHLCV series.

    With ``momentum == 0`` this is pure geometric Brownian motion (memoryless).
    Real markets, however, **trend** — returns are positively autocorrelated over
    short horizons. Setting ``momentum`` in (0, 1) adds an AR(1) component to the
    returns so that recent-return / MACD features carry genuine predictive signal
    for the agent to exploit. Total volatility is preserved regardless of the
    momentum level, so ``annual_vol`` keeps its meaning.

    ``annual_vol`` remains the main regime knob (higher ≈ crypto-like).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / bars_per_year
    mu = (annual_drift - 0.5 * annual_vol**2) * dt
    sigma = annual_vol * np.sqrt(dt)

    if momentum > 0.0:
        # AR(1) signal carries a fixed fraction of the variance; the remainder is
        # unpredictable noise. This yields autocorrelated (trending) returns.
        # Kept modest so the resulting return autocorrelation stays in a
        # plausible "trending regime" band rather than making the market trivial.
        signal_frac = 0.40
        ar = np.zeros(n_steps)
        innovations = rng.normal(0.0, 1.0, size=n_steps)
        for t in range(1, n_steps):
            ar[t] = momentum * ar[t - 1] + innovations[t]
        ar /= ar.std() + 1e-8  # normalise to unit variance
        noise = rng.normal(0.0, 1.0, size=n_steps)
        log_returns = mu + sigma * (
            signal_frac * ar + np.sqrt(1.0 - signal_frac**2) * noise
        )
    else:
        log_returns = rng.normal(loc=mu, scale=sigma, size=n_steps)

    close = start_price * np.exp(np.cumsum(log_returns))

    # Build plausible OHLC around each close using small intrabar noise.
    intrabar = np.abs(rng.normal(0.0, annual_vol * np.sqrt(dt) * 0.5, size=n_steps))
    open_ = np.empty(n_steps)
    open_[0] = start_price
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1.0 + intrabar)
    low = np.minimum(open_, close) * (1.0 - intrabar)
    volume = rng.lognormal(mean=12.0, sigma=0.5, size=n_steps)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


# --------------------------------------------------------------------------- #
# Feature engineering                                                          #
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing approximation)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line and its signal line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder), the standard volatility-of-range measure.

    True range captures gaps that a simple high-low miss, so ATR is a cleaner
    volatility proxy than close-to-close std for assets that gap overnight.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: how many std-devs the latest value sits from its mean.

    A compact, scale-free way to express "is this unusually high/low *for this
    asset, right now*" — the regime context a single raw level can't convey.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append a standard set of technical indicators to an OHLCV frame.

    Features are intentionally *stationary-ish* (returns, oscillators, ratios)
    rather than raw price levels, which helps the policy generalise across the
    train/val/test split and across markets.
    """
    out = df.copy()
    close = out["close"]

    # --- Multi-horizon momentum: short, medium, and long look-backs let the
    # policy distinguish a one-day blip from a durable trend.
    out["return_1"] = close.pct_change()
    out["return_5"] = close.pct_change(5)
    out["return_20"] = close.pct_change(20)
    out["log_return"] = np.log(close).diff()

    # --- Trend / mean-reversion context relative to moving averages.
    out["sma_10_ratio"] = close / close.rolling(10).mean() - 1.0
    out["sma_30_ratio"] = close / close.rolling(30).mean() - 1.0
    out["ema_12_ratio"] = close / close.ewm(span=12, adjust=False).mean() - 1.0
    # Long-horizon trend: a regime proxy (above/below the ~quarterly average).
    out["sma_50_ratio"] = close / close.rolling(50).mean() - 1.0

    # --- Oscillators.
    out["rsi_14"] = _rsi(close, 14) / 100.0  # scale to ~[0, 1]
    macd, macd_signal = _macd(close)
    # Normalise MACD by price so it is comparable across assets / regimes.
    out["macd"] = macd / close
    out["macd_signal"] = macd_signal / close

    # --- Bollinger %B: position of price within its 20-day ±2σ band.
    # ~0.5 = at the mean, >1 = above the upper band, <0 = below the lower band.
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    out["bollinger_pct_b"] = (close - (sma_20 - 2 * std_20)) / (4 * std_20)

    # --- Donchian position: where price sits in its 20-day high/low range,
    # centred to [-0.5, 0.5]. Captures breakouts the band measures miss.
    roll_high = out["high"].rolling(20).max()
    roll_low = out["low"].rolling(20).min()
    out["donchian_pos"] = (close - roll_low) / (roll_high - roll_low) - 0.5

    # --- Volatility level + regime: ATR (gap-aware) and a z-score of realised
    # vol so the agent knows whether *this* moment is calm or turbulent.
    out["volatility_10"] = out["return_1"].rolling(10).std()
    out["atr_norm"] = _atr(out, 14) / close
    out["vol_regime"] = _zscore(out["volatility_10"], 60)

    # --- Range + volume microstructure.
    out["high_low_range"] = (out["high"] - out["low"]) / close
    out["volume_change"] = out["volume"].pct_change()
    out["volume_zscore"] = _zscore(out["volume"], 20)

    # --- Longer-horizon momentum, trend, and risk regime. These look-backs give
    # the policy the *market context* the short windows miss: a durable multi-month
    # trend, how deep the current drawdown is, and whether volatility is expanding.
    out["return_60"] = close.pct_change(60)
    out["return_120"] = close.pct_change(120)
    out["sma_100_ratio"] = close / close.rolling(100).mean() - 1.0
    # Distance below the trailing ~6-month high (<= 0): a drawdown / risk-off signal.
    out["high_120_dist"] = close / out["high"].rolling(120).max() - 1.0
    # Volatility regime: short vs longer realised vol (>0 expanding, <0 contracting).
    vol_60 = out["return_1"].rolling(60).std()
    out["vol_ratio"] = out["volatility_10"] / (vol_60 + 1e-8) - 1.0

    # --- Market context (cross-asset). When a reference-index close is supplied via
    # a `_mkt_close` column (e.g. SPY for stocks, BTC-USD for crypto), encode the
    # asset's strength *relative to the market* and the market's own trend/momentum —
    # genuinely exogenous information a single ticker's OHLCV can't provide. Absent a
    # reference, these default to 0.0 so the feature vector stays a fixed width.
    if "_mkt_close" in out.columns:
        mkt = out["_mkt_close"]
        out["rel_return_5"] = close.pct_change(5) - mkt.pct_change(5)
        out["rel_return_20"] = close.pct_change(20) - mkt.pct_change(20)
        out["market_trend"] = mkt / mkt.rolling(50).mean() - 1.0
        out["market_ret_20"] = mkt.pct_change(20)
        out = out.drop(columns=["_mkt_close"])
    else:
        out["rel_return_5"] = 0.0
        out["rel_return_20"] = 0.0
        out["market_trend"] = 0.0
        out["market_ret_20"] = 0.0

    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


# Feature columns fed to the agent (raw OHLCV levels are deliberately excluded).
# Grouped by what they encode: momentum, trend context, oscillators, band/range
# position, volatility regime, and volume microstructure.
FEATURE_COLUMNS: List[str] = [
    # momentum (multi-horizon)
    "return_1",
    "return_5",
    "return_20",
    "log_return",
    # trend / mean-reversion context
    "sma_10_ratio",
    "sma_30_ratio",
    "ema_12_ratio",
    "sma_50_ratio",
    # oscillators
    "rsi_14",
    "macd",
    "macd_signal",
    # band / range position
    "bollinger_pct_b",
    "donchian_pos",
    # volatility level + regime
    "volatility_10",
    "atr_norm",
    "vol_regime",
    # range + volume microstructure
    "high_low_range",
    "volume_change",
    "volume_zscore",
    # longer-horizon momentum / trend / risk regime
    "return_60",
    "return_120",
    "sma_100_ratio",
    "high_120_dist",
    "vol_ratio",
    # market context (cross-asset; relative strength + market regime)
    "rel_return_5",
    "rel_return_20",
    "market_trend",
    "market_ret_20",
]


# --------------------------------------------------------------------------- #
# Scaling + splitting                                                          #
# --------------------------------------------------------------------------- #
def _fit_scaler(train_features: np.ndarray):
    """Return (mean, std) computed on the *training* split only (no leakage)."""
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-8] = 1.0  # guard against constant columns
    return mean.astype(np.float32), std.astype(np.float32)


def market_regime(market: str) -> tuple[float, float, float]:
    """Return (annual_vol, annual_drift, momentum) for a synthetic regime.

    Crypto is more volatile and slightly noisier (lower momentum) than equities.
    The momentum term gives returns exploitable autocorrelation so the agent has
    real signal to learn — see :func:`generate_synthetic_ohlcv`.
    """
    # Low net drift + meaningful trend signal: passive buy-&-hold is dragged by
    # volatility, leaving room for an active momentum-timing agent to add value.
    if market == "crypto":
        return 0.55, 0.06, 0.65
    return 0.22, 0.04, 0.70


def synthetic_market_data(
    market: str = "stock", seed: Optional[int] = None, n_steps: int = 1_400
) -> MarketData:
    """Build one self-contained, self-scaled synthetic series.

    Designed to be called repeatedly (with ``seed=None``) as a *factory* for
    domain-randomized training: each call returns an independent price path that
    shares the market's drift/volatility regime but has fresh noise. Features are
    scaled by this series' own statistics — appropriate here because every such
    series is training data.
    """
    vol, drift, mom = market_regime(market)
    df = generate_synthetic_ohlcv(
        n_steps=n_steps, annual_vol=vol, annual_drift=drift, momentum=mom, seed=seed
    )
    return market_data_from_df(df)


def market_data_from_df(df: pd.DataFrame) -> MarketData:
    """Build a single self-scaled :class:`MarketData` from a raw OHLCV frame.

    Used both by the synthetic generator and by the live-serving backend, which
    feeds it recent real prices. Features are scaled by this series' own
    statistics — appropriate for a single self-contained inference window.
    """
    featured = add_technical_indicators(df)
    feat = featured[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    prices = featured["close"].to_numpy(dtype=np.float32)
    mean, std = _fit_scaler(feat)
    return MarketData(((feat - mean) / std).astype(np.float32), prices, FEATURE_COLUMNS)


def prepare_market_data(
    df: Optional[pd.DataFrame],
    *,
    market: str = "stock",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    synthetic_steps: int = 5_000,
    seed: Optional[int] = None,
) -> Dict[str, MarketData]:
    """End-to-end: -> indicators -> chronological split -> train-fit scaling.

    Parameters
    ----------
    df:
        Raw OHLCV DataFrame. If ``None``, synthetic data is generated. Crypto
        synthetic data uses a higher volatility to emulate its regime.
    market:
        ``"stock"`` or ``"crypto"`` — only affects the synthetic generator.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to
    :class:`MarketData`. The scaler is fit on the training split and applied to
    all three, so validation/test statistics never leak into training.
    """
    if df is None:
        # Realistic-but-learnable regimes (see :func:`market_regime`). Swap in
        # real CSVs for production backtests.
        annual_vol, annual_drift, momentum = market_regime(market)
        df = generate_synthetic_ohlcv(
            n_steps=synthetic_steps,
            annual_vol=annual_vol,
            annual_drift=annual_drift,
            momentum=momentum,
            seed=seed,
        )

    featured = add_technical_indicators(df)
    feat = featured[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    prices = featured["close"].to_numpy(dtype=np.float32)

    n = len(prices)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    mean, std = _fit_scaler(feat[:train_end])

    def _scale(a: np.ndarray) -> np.ndarray:
        return ((a - mean) / std).astype(np.float32)

    return {
        "train": MarketData(_scale(feat[:train_end]), prices[:train_end], FEATURE_COLUMNS),
        "val": MarketData(_scale(feat[train_end:val_end]), prices[train_end:val_end], FEATURE_COLUMNS),
        "test": MarketData(_scale(feat[val_end:]), prices[val_end:], FEATURE_COLUMNS),
    }
