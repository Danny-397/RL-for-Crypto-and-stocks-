"""Backtesting: run a trained agent over a held-out series and score it.

The agent acts *deterministically* (policy mean) here — we want to measure the
learned strategy, not exploration noise. Metrics are intentionally the ones a
quant or recruiter expects to see: total return, annualised Sharpe, and maximum
drawdown, plus the equity and action trajectories for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# Per-bar periods used to annualise the Sharpe ratio. Crypto trades every day of
# the year; equities trade ~252 sessions. Adjust if you use intraday bars.
ANNUALISATION = {"stock": 252, "crypto": 365}


@dataclass
class BacktestResult:
    """Container for a single backtest's trajectories and headline metrics."""

    equity_curve: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)


def compute_metrics(equity_curve: np.ndarray, periods_per_year: int) -> Dict[str, float]:
    """Compute total return, annualised Sharpe, and max drawdown.

    Parameters
    ----------
    equity_curve:
        Portfolio value at each step (first element is the starting equity).
    periods_per_year:
        Number of bars per year, for Sharpe annualisation.
    """
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    step_returns = np.diff(equity_curve) / equity_curve[:-1]

    total_return = equity_curve[-1] / equity_curve[0] - 1.0

    if step_returns.std() > 1e-12:
        sharpe = float(np.sqrt(periods_per_year) * step_returns.mean() / step_returns.std())
    else:
        sharpe = 0.0

    running_peak = np.maximum.accumulate(equity_curve)
    drawdowns = (running_peak - equity_curve) / running_peak
    max_drawdown = float(drawdowns.max())

    return {
        "total_return": float(total_return),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_equity": float(equity_curve[-1]),
    }


def backtest(agent, env, market: str = "stock") -> BacktestResult:
    """Run ``agent`` deterministically through ``env`` once and score it.

    The environment should be constructed with ``random_start=False`` so the
    backtest covers the full held-out series from the beginning.
    """
    obs, info = env.reset()
    equity_curve: List[float] = [info["equity"]]
    actions: List[float] = []

    done = False
    while not done:
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        equity_curve.append(info["equity"])
        actions.append(float(action[0]))
        done = terminated or truncated

    equity_arr = np.asarray(equity_curve, dtype=np.float64)
    periods = ANNUALISATION.get(market, 252)
    metrics = compute_metrics(equity_arr, periods)

    return BacktestResult(
        equity_curve=equity_arr,
        actions=np.asarray(actions, dtype=np.float64),
        returns=np.diff(equity_arr) / equity_arr[:-1],
        metrics=metrics,
    )
