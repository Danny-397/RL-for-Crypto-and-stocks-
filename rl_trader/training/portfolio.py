"""Training loop for the cross-sectional portfolio agent.

This reuses the exact PPO machinery the single-asset path uses — :class:`PPOAgent`
(with its observation normaliser), :class:`RolloutBuffer` (GAE), the clipped
update — and only swaps the environment for :class:`PortfolioTradingEnv`. The
agent is unchanged because it is domain-agnostic: it just sees a bigger
observation and emits an ``N``-dimensional action. That the same PPO code trains
both a one-asset timer and an ``N``-asset allocator is the payoff of the
"unified agent, swappable environment" design.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from ..config.training_config import Config
from ..data.portfolio_data import PortfolioData
from ..envs.portfolio_env import PortfolioTradingEnv
from ..models.ppo_agent import PPOAgent, resolve_device
from .utils import RolloutBuffer, get_logger, set_seed


def evaluate_portfolio_policy(agent: PPOAgent, env: PortfolioTradingEnv) -> float:
    """Deterministic full-pass total return (lightweight in-loop validation)."""
    obs, info = env.reset()
    start = info["equity"]
    done = False
    while not done:
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return info["equity"] / start - 1.0


def train_portfolio(
    config: Config,
    splits: Dict[str, PortfolioData],
    logger=None,
    train_series_factory=None,
):
    """Train a PPO portfolio agent on a basket; returns ``(agent, history)``.

    ``train_series_factory`` (optional) supplies a fresh :class:`PortfolioData`
    per episode for domain randomization, mirroring the single-asset trainer.
    """
    log = logger or get_logger("rl_trader", config.train.log_dir)
    set_seed(config.train.seed, getattr(config.train, "deterministic", False))
    device = resolve_device(config.train.device)

    train_data = train_series_factory() if train_series_factory else splits["train"]
    env = PortfolioTradingEnv(train_data, config.env, config.reward)
    eval_env = PortfolioTradingEnv(
        splits.get("val") if len(splits.get("val", [])) else splits["test"],
        config.env, config.reward, random_start=False,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, config.ppo, device=device)
    buffer = RolloutBuffer(config.train.rollout_length, obs_dim, act_dim)

    log.info(
        "Training PPO portfolio | assets=%d obs_dim=%d act_dim=%d device=%s",
        env.n_assets, obs_dim, act_dim, device,
    )

    history: Dict[str, list] = {
        "update": [], "mean_episode_return": [], "mean_episode_equity": [],
        "policy_loss": [], "value_loss": [], "entropy": [], "val_return": [],
    }

    n_updates = max(1, config.train.total_timesteps // config.train.rollout_length)
    obs, _ = env.reset(seed=config.train.seed)
    ep_return, ep_returns, ep_equities = 0.0, [], []

    for update in range(1, n_updates + 1):
        buffer.reset()
        for _ in range(config.train.rollout_length):
            agent.observe(obs)
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, log_prob, reward, value, done)
            obs = next_obs
            ep_return += reward
            if done:
                ep_returns.append(ep_return)
                ep_equities.append(info["equity"])
                ep_return = 0.0
                if train_series_factory:
                    env.reload(train_series_factory())
                obs, _ = env.reset()

        last_value = 0.0 if done else agent.value(obs)
        buffer.compute_gae(last_value, config.ppo.gamma, config.ppo.gae_lambda)
        stats = agent.update(buffer)

        mean_ret = float(np.mean(ep_returns[-20:])) if ep_returns else 0.0
        mean_eq = float(np.mean(ep_equities[-20:])) if ep_equities else config.env.initial_balance
        val_return = float("nan")
        if config.train.eval_interval and update % config.train.eval_interval == 0:
            val_return = evaluate_portfolio_policy(agent, eval_env)

        history["update"].append(update)
        history["mean_episode_return"].append(mean_ret)
        history["mean_episode_equity"].append(mean_eq)
        history["policy_loss"].append(stats["policy_loss"])
        history["value_loss"].append(stats["value_loss"])
        history["entropy"].append(stats["entropy"])
        history["val_return"].append(val_return)

        if update % config.train.log_interval == 0:
            val_str = "" if np.isnan(val_return) else f" | val_return {val_return:+.2%}"
            log.info(
                f"update {update}/{n_updates} | ep_return {mean_ret:.4f} | "
                f"ep_equity {mean_eq:,.0f} | pi_loss {stats['policy_loss']:.4f} | "
                f"entropy {stats['entropy']:.3f}{val_str}"
            )

    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(config.train.checkpoint_dir, "ppo_portfolio.pt")
    agent.save(ckpt_path)
    log.info("Saved portfolio checkpoint -> %s", ckpt_path)
    return agent, history
