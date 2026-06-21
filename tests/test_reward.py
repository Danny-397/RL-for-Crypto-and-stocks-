"""Tests for the reward formulations, including the Differential Sharpe Ratio."""

import numpy as np

from rl_trader.config.training_config import stock_config
from rl_trader.data.data_loader import synthetic_market_data
from rl_trader.envs import make_env


def _run(reward_kind: str, seed: int = 0):
    cfg = stock_config()
    cfg.reward.kind = reward_kind
    data = synthetic_market_data("stock", seed=3)
    env = make_env("stock", data, cfg.env, cfg.reward, random_start=False)
    obs, _ = env.reset()
    rng = np.random.default_rng(seed)
    rewards = []
    done = False
    while not done:
        action = np.array([rng.uniform(-1.0, 1.0)], dtype=np.float32)
        obs, r, term, trunc, _ = env.step(action)
        rewards.append(r)
        done = term or trunc
    return np.array(rewards)


def test_both_reward_kinds_are_finite():
    for kind in ("return", "dsr"):
        rewards = _run(kind)
        assert np.isfinite(rewards).all()
        assert len(rewards) > 0


def test_dsr_differs_from_return_reward():
    # The two formulations should produce genuinely different signals.
    assert not np.allclose(_run("return"), _run("dsr"))


def test_dsr_rewards_steady_gains_over_volatile_ones():
    # The DSR is a risk-adjusted measure: a smooth up-drift should accumulate
    # more reward than a jagged path with the *same* total return.
    cfg = stock_config()
    cfg.reward.kind = "dsr"
    data = synthetic_market_data("stock", seed=1)
    env = make_env("stock", data, cfg.env, cfg.reward, random_start=False)
    env.reset()

    smooth = sum(env._differential_sharpe(0.001) for _ in range(200))
    env.reset()
    jagged = sum(env._differential_sharpe(0.001 if i % 2 else -0.0005)
                 for i in range(200))
    # Same-ish mean drift, but the smooth stream is far less volatile -> higher
    # cumulative Differential Sharpe.
    assert smooth > jagged
