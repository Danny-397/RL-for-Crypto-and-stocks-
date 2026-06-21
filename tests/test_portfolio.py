"""Tests for the cross-sectional portfolio data, environment, and baselines."""

import numpy as np
import pytest

from rl_trader.config.training_config import stock_config
from rl_trader.data.portfolio_data import build_portfolio_data, synthetic_portfolio
from rl_trader.data.data_loader import generate_synthetic_ohlcv
from rl_trader.envs.portfolio_env import PortfolioTradingEnv
from rl_trader.evaluation.portfolio_eval import evaluate_portfolio_baselines, portfolio_backtest


def _splits(n_assets=4, seed=0):
    return synthetic_portfolio(n_assets=n_assets, seed=seed)


def test_portfolio_data_shapes_and_alignment():
    splits = _splits(n_assets=4, seed=1)
    tr = splits["train"]
    assert tr.features.ndim == 3 and tr.features.shape[1] == 4
    assert tr.prices.shape[1] == 4
    assert tr.features.shape[0] == tr.prices.shape[0]
    assert np.isfinite(tr.features).all()
    # train and test must not overlap in length-0 weirdness
    assert len(splits["test"]) > 0


def test_build_requires_multiple_assets():
    df = generate_synthetic_ohlcv(n_steps=300, seed=0)
    import pandas as pd
    df.insert(0, "date", pd.bdate_range("2010-01-01", periods=len(df)))
    with pytest.raises(ValueError):
        build_portfolio_data({"ONLY": df})


def test_env_obs_and_action_dimensions():
    splits = _splits(n_assets=5, seed=2)
    cfg = stock_config()
    env = PortfolioTradingEnv(splits["train"], cfg.env, cfg.reward, random_start=False)
    obs, _ = env.reset()
    w, f = cfg.env.window_size, splits["train"].n_features
    assert env.action_space.shape == (5,)
    assert obs.shape[0] == w * 5 * f + 5 + 2
    assert np.isfinite(obs).all()


def test_gross_exposure_is_capped():
    splits = _splits(n_assets=5, seed=3)
    cfg = stock_config()
    cfg.env.max_position = 1.0
    env = PortfolioTradingEnv(splits["test"], cfg.env, cfg.reward, random_start=False)
    env.reset()
    # An all-ones target (gross 5) must be scaled so gross exposure stays <= ~1.
    done = False
    grosses = []
    while not done:
        _o, _r, term, trunc, info = env.step(np.ones(5, dtype=np.float32))
        grosses.append(info["gross_exposure"])
        done = term or trunc
    assert max(grosses) <= 1.05  # small slack for intra-step price drift


def test_single_asset_equivalence_of_weight_scaling():
    # With one effective direction, the gross cap leaves a sub-budget action alone.
    splits = _splits(n_assets=3, seed=4)
    cfg = stock_config()
    env = PortfolioTradingEnv(splits["test"], cfg.env, cfg.reward, random_start=False)
    small = np.array([0.2, -0.1, 0.0], dtype=np.float64)  # gross 0.3 < cap
    assert np.allclose(env._target_weights(small), small)


def test_baselines_run_and_are_finite():
    splits = _splits(n_assets=5, seed=5)
    cfg = stock_config()
    out = evaluate_portfolio_baselines(splits["test"], cfg.env, cfg.reward, periods=252)
    assert set(out) == {"equal_weight", "cross_sectional_momentum", "random"}
    for r in out.values():
        assert np.isfinite(r["metrics"]["total_return"])
        assert len(r["equity"]) > 1


def test_portfolio_backtest_with_untrained_agent_runs():
    from rl_trader.models.ppo_agent import PPOAgent

    splits = _splits(n_assets=4, seed=6)
    cfg = stock_config()
    env = PortfolioTradingEnv(splits["test"], cfg.env, cfg.reward, random_start=False)
    obs, _ = env.reset()
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], cfg.ppo)
    res = portfolio_backtest(agent, env, periods=252)
    assert np.isfinite(res["metrics"]["sharpe"])
    assert res["equity"][0] == cfg.env.initial_balance
