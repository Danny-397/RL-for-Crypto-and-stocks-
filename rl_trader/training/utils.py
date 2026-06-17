"""Training utilities: rollout buffer (with GAE), logging, and seeding."""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "rl_trader", log_dir: str | None = None) -> logging.Logger:
    """Return a console logger, optionally also writing to ``log_dir/train.log``."""
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on repeated calls
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


class RolloutBuffer:
    """Fixed-size on-policy buffer that computes GAE advantages and returns.

    PPO is on-policy: we collect a batch of transitions with the *current*
    policy, compute advantages with Generalized Advantage Estimation, then do a
    few epochs of mini-batch updates before discarding everything and repeating.
    """

    def __init__(self, size: int, obs_dim: int, act_dim: int) -> None:
        self.size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.reset()

    def reset(self) -> None:
        """Clear the buffer for a fresh rollout."""
        self.observations = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.size, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros(self.size, dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.values = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.float32)
        self.advantages = np.zeros(self.size, dtype=np.float32)
        self.returns = np.zeros(self.size, dtype=np.float32)
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Store one transition. ``done`` flags an episode boundary at this step."""
        i = self.ptr
        self.observations[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = float(done)
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr >= self.size

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """Fill ``advantages`` and ``returns`` via GAE-lambda.

        ``last_value`` bootstraps the value of the state following the final
        stored transition (zero if that state was terminal).
        """
        adv = 0.0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            adv = delta + gamma * gae_lambda * next_non_terminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def iter_minibatches(
        self, minibatch_size: int, device: torch.device
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield shuffled mini-batches of tensors for the PPO update.

        Advantages are normalised across the whole rollout, which materially
        stabilises PPO updates.
        """
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = np.random.permutation(self.size)
        for start in range(0, self.size, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield {
                "observations": torch.as_tensor(self.observations[idx], device=device),
                "actions": torch.as_tensor(self.actions[idx], device=device),
                "old_log_probs": torch.as_tensor(self.log_probs[idx], device=device),
                "old_values": torch.as_tensor(self.values[idx], device=device),
                "advantages": torch.as_tensor(adv[idx], device=device),
                "returns": torch.as_tensor(self.returns[idx], device=device),
            }


def run_ppo_training(
    config,
    df: Optional["pd.DataFrame"] = None,
    logger: Optional[logging.Logger] = None,
):
    """Train a PPO agent on a single market end-to-end.

    This is the shared engine behind :func:`train_stock` and :func:`train_crypto`
    — neither wrapper duplicates loop logic; they only supply a market-specific
    :class:`~rl_trader.config.Config`. The loop is the textbook PPO cycle:

        collect a fixed-length rollout  ->  compute GAE  ->  PPO update  ->  repeat

    Parameters
    ----------
    config:
        A fully-populated :class:`~rl_trader.config.Config`.
    df:
        Optional raw OHLCV DataFrame. If ``None``, synthetic data is generated
        so training runs with no external dependencies.

    Returns
    -------
    (agent, history) where ``history`` is a dict of per-update metric lists.
    """
    # Imported lazily to avoid a heavy import chain when only the buffer/logger
    # utilities are needed (e.g. in unit tests).
    from ..data.data_loader import prepare_market_data
    from ..envs import make_env
    from ..models.ppo_agent import PPOAgent, resolve_device

    log = logger or get_logger("rl_trader", config.train.log_dir)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)

    splits = prepare_market_data(df, market=config.market, seed=config.train.seed)
    env = make_env(config.market, splits["train"], config.env, config.reward)
    eval_env = make_env(
        config.market, splits["val"], config.env, config.reward, random_start=False
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, config.ppo, device=device)
    buffer = RolloutBuffer(config.train.rollout_length, obs_dim, act_dim)

    log.info(
        "Training PPO on %s | obs_dim=%d act_dim=%d device=%s",
        config.market, obs_dim, act_dim, device,
    )

    history: Dict[str, list] = {
        "update": [], "mean_episode_return": [], "mean_episode_equity": [],
        "policy_loss": [], "value_loss": [], "entropy": [], "val_return": [],
    }

    n_updates = max(1, config.train.total_timesteps // config.train.rollout_length)
    obs, _ = env.reset()
    ep_return, ep_returns, ep_equities = 0.0, [], []

    for update in range(1, n_updates + 1):
        buffer.reset()
        for _ in range(config.train.rollout_length):
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
                obs, _ = env.reset()

        last_value = 0.0 if done else agent.value(obs)
        buffer.compute_gae(last_value, config.ppo.gamma, config.ppo.gae_lambda)
        stats = agent.update(buffer)

        mean_ret = float(np.mean(ep_returns[-20:])) if ep_returns else 0.0
        mean_eq = float(np.mean(ep_equities[-20:])) if ep_equities else config.env.initial_balance

        val_return = float("nan")
        if config.train.eval_interval and update % config.train.eval_interval == 0:
            val_return = evaluate_policy(agent, eval_env)

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
                f"v_loss {stats['value_loss']:.3f} | entropy {stats['entropy']:.3f}{val_str}"
            )

    # Persist the trained agent.
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(config.train.checkpoint_dir, f"ppo_{config.market}.pt")
    agent.save(ckpt_path)
    log.info("Saved checkpoint -> %s", ckpt_path)

    return agent, history


def evaluate_policy(agent, env) -> float:
    """Roll the deterministic policy through one full pass; return total return.

    Used as a lightweight in-loop validation signal. The richer, metric-heavy
    backtest lives in :mod:`rl_trader.evaluation.evaluate_agent`.
    """
    obs, info = env.reset()
    start_equity = info["equity"]
    done = False
    while not done:
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info["equity"] / start_equity - 1.0
