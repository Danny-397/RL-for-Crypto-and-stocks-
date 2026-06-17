"""Tests for the PPO agent and rollout buffer.

These verify the agent's action/value interfaces, that a PPO update runs and
returns finite losses, that save/load round-trips weights, and that GAE
produces correctly shaped advantages. A tiny synthetic run keeps them fast.
"""

import numpy as np
import torch

from rl_trader.config.training_config import stock_config
from rl_trader.data.data_loader import prepare_market_data
from rl_trader.envs import make_env
from rl_trader.models.ppo_agent import PPOAgent
from rl_trader.training.utils import RolloutBuffer


def _make_env():
    cfg = stock_config()
    splits = prepare_market_data(None, market="stock", synthetic_steps=600, seed=0)
    return make_env("stock", splits["train"], cfg.env, cfg.reward), cfg


def test_select_action_shapes():
    env, cfg = _make_env()
    obs, _ = env.reset(seed=0)
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], cfg.ppo)
    action, log_prob, value = agent.select_action(obs)
    assert action.shape == (env.action_space.shape[0],)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_deterministic_action_is_repeatable():
    env, cfg = _make_env()
    obs, _ = env.reset(seed=0)
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], cfg.ppo)
    a1, _, _ = agent.select_action(obs, deterministic=True)
    a2, _, _ = agent.select_action(obs, deterministic=True)
    assert np.allclose(a1, a2)


def test_ppo_update_runs_and_is_finite():
    env, cfg = _make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, cfg.ppo)

    buffer = RolloutBuffer(size=128, obs_dim=obs_dim, act_dim=act_dim)
    obs, _ = env.reset(seed=1)
    for _ in range(128):
        action, log_prob, value = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, log_prob, reward, value, done)
        obs = next_obs if not done else env.reset()[0]

    buffer.compute_gae(last_value=0.0, gamma=cfg.ppo.gamma, gae_lambda=cfg.ppo.gae_lambda)
    stats = agent.update(buffer)
    for key in ("policy_loss", "value_loss", "entropy", "clip_fraction"):
        assert np.isfinite(stats[key])


def test_gae_shapes_and_finiteness():
    buffer = RolloutBuffer(size=32, obs_dim=4, act_dim=1)
    rng = np.random.default_rng(0)
    for _ in range(32):
        buffer.add(rng.normal(size=4), rng.normal(size=1), -0.5,
                   rng.normal(), rng.normal(), False)
    buffer.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    assert buffer.advantages.shape == (32,)
    assert buffer.returns.shape == (32,)
    assert np.all(np.isfinite(buffer.advantages))


def test_save_and_load_roundtrip(tmp_path):
    env, cfg = _make_env()
    obs, _ = env.reset(seed=0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, cfg.ppo)

    path = tmp_path / "agent.pt"
    agent.save(str(path))

    reloaded = PPOAgent.from_checkpoint(str(path))
    a1, _, _ = agent.select_action(obs, deterministic=True)
    a2, _, _ = reloaded.select_action(obs, deterministic=True)
    assert np.allclose(a1, a2, atol=1e-5)
