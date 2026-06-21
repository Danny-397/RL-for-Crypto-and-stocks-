"""Tests for the recurrent (LSTM) PPO agent and its rollout buffer."""

import numpy as np

from rl_trader.config.training_config import stock_config
from rl_trader.data.data_loader import synthetic_market_data
from rl_trader.envs import make_env
from rl_trader.training.recurrent import (
    RecurrentPPOAgent,
    RecurrentRolloutBuffer,
    run_recurrent_ppo_training,
)


def _make_env():
    cfg = stock_config()
    data = synthetic_market_data("stock", seed=0)
    return make_env("stock", data, cfg.env, cfg.reward, random_start=False), cfg


def test_recurrent_action_threads_hidden_state():
    env, cfg = _make_env()
    obs, _ = env.reset(seed=0)
    agent = RecurrentPPOAgent(
        env.observation_space.shape[0], env.action_space.shape[0], cfg.ppo
    )
    hidden = agent.initial_state()
    action, log_prob, value, hidden = agent.select_action(obs, hidden)
    assert action.shape == (env.action_space.shape[0],)
    assert isinstance(log_prob, float) and isinstance(value, float)
    # Hidden state advanced (no longer all-zero after a step).
    assert float(hidden[0].abs().sum()) > 0.0


def test_recurrent_buffer_segments_split_on_done_and_cap():
    buf = RecurrentRolloutBuffer(size=10, obs_dim=3, act_dim=1, hidden_size=4)
    import torch

    h = (torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))
    for t in range(10):
        done = t == 4  # one mid-buffer episode boundary
        buf.add(np.zeros(3), np.zeros(1), 0.0, 0.0, 0.0, done, h)
    # seq_len cap of 3 + a done at index 4 should yield several segments that
    # exactly tile [0, 10) with no gaps or overlaps.
    bounds = buf._segment_bounds(seq_len=3)
    assert bounds[0][0] == 0 and bounds[-1][1] == 10
    for (s0, e0), (s1, _e1) in zip(bounds, bounds[1:]):
        assert e0 == s1  # contiguous
    # The episode boundary at t=4 must end a segment (next starts at 5).
    assert any(e == 5 for _s, e in bounds)


def test_recurrent_update_runs_and_is_finite():
    env, cfg = _make_env()
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    cfg.ppo.recurrent_seq_len = 16
    agent = RecurrentPPOAgent(obs_dim, act_dim, cfg.ppo)
    buf = RecurrentRolloutBuffer(64, obs_dim, act_dim, agent.hidden_size)

    obs, _ = env.reset(seed=1)
    hidden = agent.initial_state()
    for _ in range(64):
        action, lp, val, next_hidden = agent.select_action(obs, hidden)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        buf.add(obs, action, lp, reward, val, done, hidden)
        obs, hidden = next_obs, next_hidden
        if done:
            obs, _ = env.reset()
            hidden = agent.initial_state()
    buf.compute_gae(0.0, cfg.ppo.gamma, cfg.ppo.gae_lambda)
    stats = agent.update(buf)
    for key in ("policy_loss", "value_loss", "entropy", "clip_fraction"):
        assert np.isfinite(stats[key])


def test_recurrent_checkpoint_roundtrip(tmp_path):
    env, cfg = _make_env()
    obs, _ = env.reset(seed=0)
    agent = RecurrentPPOAgent(
        env.observation_space.shape[0], env.action_space.shape[0], cfg.ppo
    )
    path = tmp_path / "lstm.pt"
    agent.save(str(path))
    reloaded = RecurrentPPOAgent.from_checkpoint(str(path))

    h1, h2 = agent.initial_state(), reloaded.initial_state()
    a1, _, _, _ = agent.select_action(obs, h1, deterministic=True)
    a2, _, _, _ = reloaded.select_action(obs, h2, deterministic=True)
    assert np.allclose(a1, a2, atol=1e-5)


def test_recurrent_training_smoke():
    cfg = stock_config()
    cfg.ppo.use_lstm = True
    cfg.ppo.recurrent_seq_len = 16
    cfg.train.total_timesteps = 600
    cfg.train.rollout_length = 300
    cfg.train.eval_interval = 0
    cfg.train.log_interval = 999
    agent, history = run_recurrent_ppo_training(
        cfg, train_series_factory=lambda: synthetic_market_data("stock")
    )
    assert isinstance(agent, RecurrentPPOAgent)
    assert len(history["update"]) == 2
    assert np.isfinite(history["policy_loss"]).all()
