"""Synthetic market regimes for controlled distribution-shift tests.

The agents were trained on one distribution. The sharpest way to ask whether
they learned a *strategy* rather than a *dataset* is to move the distribution
deliberately and watch what survives. These regimes are built on the project's
existing, tested generator
(:func:`rl_trader.data.data_loader.generate_synthetic_ohlcv`) so the price
process is the same one the research code uses — only its parameters change.

The generator's ``momentum`` term is an AR(1) coefficient on returns, which
makes three of these regimes a pure parameter choice:

* ``phi = 0``  -> memoryless geometric Brownian motion (random walk)
* ``phi > 0``  -> positively autocorrelated returns (trending / momentum)
* ``phi < 0``  -> negatively autocorrelated returns (mean reverting)

``regime_switch`` is the one genuinely new construction: it splices segments of
differing volatility and drift in *log-return* space, so the resulting price path
is continuous across each switch rather than gapping.

These are **not** claimed to be realistic market simulators. They are controlled
distributions for measuring generalization, and every response that carries one
is labelled ``synthetic: true``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rl_trader.data.data_loader import generate_synthetic_ohlcv, market_data_from_df


@dataclass
class Regime:
    """A named synthetic distribution with the parameters that define it."""

    key: str
    label: str
    description: str
    params: Dict[str, float]


REGIMES: Dict[str, Regime] = {
    "random_walk": Regime(
        key="random_walk",
        label="Random Walk",
        description=(
            "Memoryless geometric Brownian motion. Returns carry no autocorrelation, "
            "so there is no timing signal to extract — the control condition."
        ),
        params={"annual_vol": 0.30, "annual_drift": 0.05, "momentum": 0.0},
    ),
    "momentum": Regime(
        key="momentum",
        label="Momentum",
        description=(
            "Positively autocorrelated returns (AR(1), phi = +0.70): trends persist, "
            "so a trend-following policy has genuine signal to exploit."
        ),
        params={"annual_vol": 0.25, "annual_drift": 0.05, "momentum": 0.70},
    ),
    "mean_reversion": Regime(
        key="mean_reversion",
        label="Mean Reversion",
        description=(
            "Negatively autocorrelated returns (AR(1), phi = -0.50): moves reverse. "
            "A momentum policy should actively lose here — the sharpest shift test."
        ),
        params={"annual_vol": 0.28, "annual_drift": 0.04, "momentum": -0.50},
    ),
    "high_volatility": Regime(
        key="high_volatility",
        label="High Volatility",
        description=(
            "Same weak trend structure, roughly triple the volatility. Tests whether "
            "position sizing degrades when the risk scale leaves the training range."
        ),
        params={"annual_vol": 0.90, "annual_drift": 0.05, "momentum": 0.30},
    ),
}

SWITCHING_KEY = "regime_switch"


def list_regimes() -> List[dict]:
    """Describe every regime, for the frontend's selector."""
    out = [
        {"key": r.key, "label": r.label, "description": r.description, "params": r.params}
        for r in REGIMES.values()
    ]
    out.append(
        {
            "key": SWITCHING_KEY,
            "label": "Regime Switching",
            "description": (
                "Alternating calm-trending and turbulent-choppy segments spliced in "
                "log-return space (continuous price). Tests adaptation to a "
                "distribution that changes mid-episode."
            ),
            "params": {"n_segments": 4},
        }
    )
    return out


def _switching_ohlcv(n_steps: int, seed: Optional[int], n_segments: int = 4) -> pd.DataFrame:
    """Splice alternating calm/turbulent segments into one continuous path.

    Each segment is generated independently, converted to log returns, and the
    returns are concatenated — so the price level carries across boundaries
    without an artificial jump, while the *statistics* switch abruptly.
    """
    rng = np.random.default_rng(seed)
    seg_len = max(60, n_steps // n_segments)
    calm = {"annual_vol": 0.20, "annual_drift": 0.08, "momentum": 0.60}
    wild = {"annual_vol": 0.85, "annual_drift": -0.04, "momentum": 0.05}

    log_returns: List[np.ndarray] = []
    boundaries: List[int] = []
    cursor = 0
    for i in range(n_segments):
        spec = calm if i % 2 == 0 else wild
        length = seg_len if i < n_segments - 1 else max(60, n_steps - cursor)
        seg = generate_synthetic_ohlcv(
            n_steps=length + 1,
            seed=int(rng.integers(0, 2**31 - 1)),
            **spec,
        )
        close = seg["close"].to_numpy(dtype=float)
        log_returns.append(np.diff(np.log(close)))
        cursor += length
        boundaries.append(cursor)

    all_lr = np.concatenate(log_returns)
    close = 100.0 * np.exp(np.cumsum(np.insert(all_lr, 0, 0.0)))

    # Rebuild plausible OHLC around the spliced close series.
    n = len(close)
    intrabar = np.abs(rng.normal(0.0, 0.006, size=n))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1.0 + intrabar),
            "low": np.minimum(open_, close) * (1.0 - intrabar),
            "close": close,
            "volume": rng.lognormal(mean=12.0, sigma=0.5, size=n),
        }
    )


def build_regime_frame(
    regime: str, seed: Optional[int] = None, n_steps: int = 650
) -> pd.DataFrame:
    """The raw OHLCV frame for ``regime``, before any feature engineering.

    Callers that need to control the *scaling* themselves — walk-forward fits a
    separate scaler per fold — need the frame rather than a pre-scaled
    :class:`MarketData`, so the two construction paths share this one generator
    and cannot diverge.
    """
    if regime == SWITCHING_KEY:
        return _switching_ohlcv(n_steps, seed)
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    return generate_synthetic_ohlcv(n_steps=n_steps, seed=seed, **REGIMES[regime].params)


def build_regime_data(regime: str, seed: Optional[int] = None, n_steps: int = 650):
    """Generate one synthetic series for ``regime`` as scaled :class:`MarketData`.

    Returns ``(market_data, meta)``. ``meta`` carries the realised statistics of
    the path actually generated — measured, not assumed — so the UI can show
    that a "mean reversion" path really does have negative return autocorrelation.
    """
    df = build_regime_frame(regime, seed=seed, n_steps=n_steps)
    if regime == SWITCHING_KEY:
        params: Dict[str, float] = {"n_segments": 4}
        label = "Regime Switching"
    else:
        params = dict(REGIMES[regime].params)
        label = REGIMES[regime].label

    data = market_data_from_df(df)

    # Realised (not nominal) statistics of this specific path.
    close = df["close"].to_numpy(dtype=float)
    lr = np.diff(np.log(close))
    lag1 = float(np.corrcoef(lr[:-1], lr[1:])[0, 1]) if len(lr) > 2 else float("nan")
    meta = {
        "regime": regime,
        "label": label,
        "synthetic": True,
        "seed": seed,
        "requested_steps": n_steps,
        "params": params,
        "realised": {
            "return_autocorr_lag1": round(lag1, 4),
            "annualised_vol": round(float(lr.std() * np.sqrt(252)), 4),
            "total_return": round(float(close[-1] / close[0] - 1.0), 4),
            "bars": int(len(close)),
        },
    }
    return data, meta
