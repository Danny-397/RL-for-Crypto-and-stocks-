"""Non-learned baseline strategies for honest benchmarking.

An RL trading agent only looks impressive if it is measured against the
alternatives a skeptic would reach for first. These baselines run through the
*same* environment as the agent — identical transaction costs, slippage, and
accounting — so every comparison is apples-to-apples.

Strategies
----------
* ``buy_and_hold``   — always fully long (the passive benchmark).
* ``flat``           — never trade (pure cash; sanity floor).
* ``random``         — uniform random target position (is the agent better
                       than luck?).
* ``ma_crossover``   — go long when a fast moving average is above a slow one,
                       else flat: a classic momentum rule the agent must beat
                       to justify its complexity.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from ..config.training_config import EnvConfig, RewardConfig
from ..data.data_loader import MarketData
from ..envs import make_env
from .evaluate_agent import ANNUALISATION, compute_metrics

# An action function maps the live environment to a target position in [-1, 1].
ActionFn = Callable[[object], float]


def _run_policy(env, action_fn: ActionFn, periods: int) -> Dict[str, float]:
    """Step ``action_fn`` through ``env`` once and score the equity curve."""
    obs, info = env.reset()
    equity = [info["equity"]]
    done = False
    while not done:
        action = np.array([action_fn(env)], dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        equity.append(info["equity"])
        done = terminated or truncated
    return compute_metrics(np.asarray(equity, dtype=np.float64), periods)


def _ma_crossover_action(fast: int = 10, slow: int = 30) -> ActionFn:
    """Long when the fast SMA exceeds the slow SMA, else flat."""
    def action(env) -> float:
        prices = env.data.prices[: env.t + 1]
        if len(prices) < slow:
            return 0.0
        fast_ma = prices[-fast:].mean()
        slow_ma = prices[-slow:].mean()
        return 1.0 if fast_ma > slow_ma else 0.0
    return action


def evaluate_baselines(
    data: MarketData,
    env_config: EnvConfig,
    reward_config: RewardConfig,
    market: str = "stock",
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Run every baseline on ``data`` and return ``{name: metrics}``."""
    periods = ANNUALISATION.get(market, 252)
    rng = np.random.default_rng(seed)

    policies: Dict[str, ActionFn] = {
        "buy_and_hold": lambda env: 1.0,
        "flat": lambda env: 0.0,
        "random": lambda env: float(rng.uniform(-1.0, 1.0)),
        "ma_crossover": _ma_crossover_action(),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, fn in policies.items():
        env = make_env(market, data, env_config, reward_config, random_start=False)
        results[name] = _run_policy(env, fn, periods)
    return results
