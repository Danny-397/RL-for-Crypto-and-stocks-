"""Backtesting + baselines for the cross-sectional portfolio agent.

Every strategy here — the learned agent and each baseline — is driven through the
*same* :class:`~rl_trader.envs.portfolio_env.PortfolioTradingEnv`, so they pay
identical transaction costs and slippage and are scored on the identical price
path. The baselines are the ones a quant reaches for first, so any RL edge has to
be earned against real competition:

* ``equal_weight``   — 1/N long, rebalanced (the passive portfolio benchmark).
* ``cross_sectional_momentum`` — long the top-k recent winners, short the bottom-k
  losers (market-neutral): the canonical cross-sectional factor strategy.
* ``random``         — random weights (is the agent better than noise?).
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from ..config.training_config import EnvConfig, RewardConfig
from ..data.portfolio_data import PortfolioData
from ..envs.portfolio_env import PortfolioTradingEnv
from .evaluate_agent import compute_metrics

WeightFn = Callable[[PortfolioTradingEnv], np.ndarray]


def _run(env: PortfolioTradingEnv, weight_fn: WeightFn, periods: int) -> Dict:
    """Step a weight policy through the env once; score the equity curve."""
    _obs, info = env.reset()
    equity: List[float] = [info["equity"]]
    done = False
    while not done:
        w = np.asarray(weight_fn(env), dtype=np.float32)
        _obs, _r, term, trunc, info = env.step(w)
        equity.append(info["equity"])
        done = term or trunc
    arr = np.asarray(equity, dtype=np.float64)
    metrics = compute_metrics(arr, periods)
    return {"equity": arr, "metrics": metrics}


def portfolio_backtest(agent, env: PortfolioTradingEnv, periods: int = 252) -> Dict:
    """Run the trained agent deterministically (policy mean) through the env."""
    obs, info = env.reset()
    equity = [info["equity"]]
    done = False
    while not done:
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        equity.append(info["equity"])
        done = term or trunc
    arr = np.asarray(equity, dtype=np.float64)
    return {"equity": arr, "metrics": compute_metrics(arr, periods)}


def _equal_weight(env: PortfolioTradingEnv) -> np.ndarray:
    n = env.n_assets
    return np.ones(n, dtype=np.float64) / n


def _cross_sectional_momentum(lookback: int = 60, k: int | None = None) -> WeightFn:
    """Long the top-k trailing-return names, short the bottom-k (market-neutral)."""
    def weight(env: PortfolioTradingEnv) -> np.ndarray:
        n = env.n_assets
        kk = k if k is not None else max(1, n // 3)
        prices = env.data.prices[: env.t + 1]
        if len(prices) <= lookback:
            return np.zeros(n)
        mom = prices[-1] / prices[-1 - lookback] - 1.0
        order = np.argsort(mom)
        w = np.zeros(n)
        w[order[-kk:]] = 1.0 / kk       # long winners
        w[order[:kk]] = -1.0 / kk       # short losers
        return w  # gross ~2*kk*(1/kk)=2 -> env scales to its gross cap
    return weight


def _random_weights(seed: int) -> WeightFn:
    rng = np.random.default_rng(seed)

    def weight(env: PortfolioTradingEnv) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=env.n_assets)
    return weight


def evaluate_portfolio_baselines(
    data: PortfolioData,
    env_config: EnvConfig,
    reward_config: RewardConfig,
    periods: int = 252,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Run every baseline on ``data`` and return ``{name: {equity, metrics}}``."""
    policies: Dict[str, WeightFn] = {
        "equal_weight": _equal_weight,
        "cross_sectional_momentum": _cross_sectional_momentum(),
        "random": _random_weights(seed),
    }
    results: Dict[str, Dict] = {}
    for name, fn in policies.items():
        env = PortfolioTradingEnv(data, env_config, reward_config, random_start=False)
        results[name] = _run(env, fn, periods)
    return results
