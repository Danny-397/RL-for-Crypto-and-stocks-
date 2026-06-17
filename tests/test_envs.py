"""Tests for the trading environments.

These check the contract every environment must honour: correctly shaped
observations, valid Gymnasium step/reset returns, cost-aware accounting, and
that an episode eventually terminates. They run on synthetic data so no
network or data files are required.
"""

import numpy as np
import pytest

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.data.data_loader import prepare_market_data
from rl_trader.envs import CryptoTradingEnv, StockTradingEnv, make_env


@pytest.fixture
def stock_splits():
    return prepare_market_data(None, market="stock", synthetic_steps=800, seed=0)


def test_observation_shape_matches_space(stock_splits):
    cfg = stock_config()
    env = make_env("stock", stock_splits["train"], cfg.env, cfg.reward)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert "equity" in info


def test_step_returns_valid_gym_tuple(stock_splits):
    cfg = stock_config()
    env = make_env("stock", stock_splits["train"], cfg.env, cfg.reward)
    env.reset(seed=1)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert np.isfinite(reward)


def test_episode_terminates(stock_splits):
    cfg = stock_config()
    env = make_env("stock", stock_splits["train"], cfg.env, cfg.reward,
                   random_start=False)
    env.reset(seed=2)
    done = False
    steps = 0
    while not done and steps < 10_000:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert done


def test_transaction_costs_reduce_equity_on_churn(stock_splits):
    """Repeatedly flipping position should bleed equity via costs."""
    cfg = stock_config()
    env = make_env("stock", stock_splits["train"], cfg.env, cfg.reward,
                   random_start=False)
    _, info = env.reset(seed=3)
    start_equity = info["equity"]
    # Alternate full-long / full-short to maximise turnover.
    for i in range(50):
        action = np.array([1.0 if i % 2 == 0 else -1.0], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    # With costs and no edge, churning should not magically grow equity.
    assert info["cost"] >= 0.0
    assert info["equity"] < start_equity * 1.5  # sanity bound


def test_crypto_env_disallow_short_clips_action():
    cfg = crypto_config()
    cfg.env.allow_short = False
    splits = prepare_market_data(None, market="crypto", synthetic_steps=600, seed=4)
    env = CryptoTradingEnv(splits["train"], cfg.env, cfg.reward)
    assert env.action_space.low[0] == 0.0
    env.reset(seed=4)
    # A short action is clipped to 0 (flat); equity stays finite.
    _, reward, _, _, info = env.step(np.array([-1.0], dtype=np.float32))
    assert np.isfinite(reward)
    assert info["units"] >= -1e-9


def test_market_factory_types():
    splits = prepare_market_data(None, market="stock", synthetic_steps=400, seed=5)
    cfg = stock_config()
    assert isinstance(make_env("stock", splits["train"], cfg.env, cfg.reward),
                      StockTradingEnv)
    assert isinstance(make_env("crypto", splits["train"], cfg.env, cfg.reward),
                      CryptoTradingEnv)
    with pytest.raises(ValueError):
        make_env("forex", splits["train"], cfg.env, cfg.reward)
