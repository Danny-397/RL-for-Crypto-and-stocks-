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

Two further baselines are *learned*, and appear only when a training split is
supplied (see :mod:`rl_trader.evaluation.supervised`):

* ``ridge_forecast``      — ridge regression of the next bar's return on the
                            same features the agent sees.
* ``logistic_direction``  — logistic regression on the *direction* of that move.

They exist to answer the objection the rule-based baselines cannot: whether an
apparent absence of signal is a property of the market or merely of PPO. Both
are fit on the training split alone and traded through this same environment.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ..config.training_config import EnvConfig, RewardConfig
from ..data.data_loader import MarketData
from ..envs import make_env
from .evaluate_agent import ANNUALISATION, compute_metrics
from .supervised import supervised_policies

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
    train_data: Optional[MarketData] = None,
) -> Dict[str, Dict[str, float]]:
    """Run every baseline on ``data`` and return ``{name: metrics}``.

    ``train_data`` is optional. Supplying it adds the supervised baselines, fit
    on that split alone; omitting it leaves them out entirely rather than
    fitting them on the evaluation data, which would be the leakage this
    project spends most of its effort avoiding.
    """
    periods = ANNUALISATION.get(market, 252)
    rng = np.random.default_rng(seed)

    policies: Dict[str, ActionFn] = {
        "buy_and_hold": lambda env: 1.0,
        "flat": lambda env: 0.0,
        "random": lambda env: float(rng.uniform(-1.0, 1.0)),
        "ma_crossover": _ma_crossover_action(),
    }
    if train_data is not None:
        policies.update(
            supervised_policies(train_data.features, train_data.prices))

    results: Dict[str, Dict[str, float]] = {}
    for name, fn in policies.items():
        env = make_env(market, data, env_config, reward_config, random_start=False)
        results[name] = _run_policy(env, fn, periods)
    return results
