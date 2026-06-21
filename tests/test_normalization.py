"""Tests for running normalization and training reproducibility."""

import numpy as np

from rl_trader.config.training_config import stock_config
from rl_trader.data.data_loader import synthetic_market_data
from rl_trader.envs import make_env
from rl_trader.models.ppo_agent import PPOAgent
from rl_trader.training.normalization import RunningNormalizer
from rl_trader.training.utils import run_ppo_training


def test_running_normalizer_matches_batch_statistics():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=[3.0, -1.0], scale=[2.0, 0.5], size=(5000, 2))
    norm = RunningNormalizer(2)
    # Stream the data in uneven chunks; the streaming stats must match the
    # one-shot batch statistics.
    for chunk in np.array_split(data, 37):
        norm.update(chunk)
    assert np.allclose(norm.mean, data.mean(axis=0), atol=1e-6)
    assert np.allclose(norm.var, data.var(axis=0), atol=1e-4)


def test_normalizer_output_is_standardized_and_clipped():
    rng = np.random.default_rng(1)
    data = rng.normal(5.0, 3.0, size=(4000, 3))
    norm = RunningNormalizer(3, clip=4.0)
    norm.update(data)
    out = norm.normalize(data)
    assert abs(out.mean()) < 0.05          # ~zero mean
    assert abs(out.std() - 1.0) < 0.05     # ~unit std
    assert out.max() <= 4.0 and out.min() >= -4.0  # clipped


def test_normalizer_state_roundtrip():
    norm = RunningNormalizer(4)
    norm.update(np.random.default_rng(2).normal(size=(100, 4)))
    restored = RunningNormalizer.from_state_dict(norm.state_dict())
    assert np.allclose(restored.mean, norm.mean)
    assert np.allclose(restored.var, norm.var)
    assert restored.count == norm.count


def _short_run(seed: int):
    cfg = stock_config()
    cfg.train.total_timesteps = 1600
    cfg.train.rollout_length = 800
    cfg.train.eval_interval = 0
    cfg.train.log_interval = 999
    cfg.train.seed = seed
    cfg.train.checkpoint_dir = "checkpoints/_test"
    agent, _ = run_ppo_training(cfg, df=None)
    return agent


def test_training_is_reproducible_with_seed():
    a1 = _short_run(seed=11)
    a2 = _short_run(seed=11)
    data = synthetic_market_data("stock", seed=99)
    env = make_env("stock", data, stock_config().env, stock_config().reward, random_start=False)
    obs, _ = env.reset()
    act1 = a1.select_action(obs, deterministic=True)[0]
    act2 = a2.select_action(obs, deterministic=True)[0]
    assert np.allclose(act1, act2), "same seed must produce identical policies"


def test_agent_has_normalizer_by_default_and_can_disable():
    cfg = stock_config()
    data = synthetic_market_data("stock", seed=0)
    env = make_env("stock", data, cfg.env, cfg.reward)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    on = PPOAgent(obs_dim, act_dim, cfg.ppo)
    assert on.obs_rms is not None

    cfg.ppo.normalize_obs = False
    off = PPOAgent(obs_dim, act_dim, cfg.ppo)
    assert off.obs_rms is None
