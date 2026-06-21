"""Recurrent (LSTM) PPO: buffer, agent, and training loop.

This is the sequence-modelling counterpart to the feed-forward PPO in
:mod:`rl_trader.models.ppo_agent`. The feed-forward agent sees one flattened
window per step; the recurrent agent instead carries an LSTM hidden state across
time, so it can model dependencies longer than the observation window.

Making PPO recurrent requires two things the feed-forward loop doesn't:

1. **Hidden-state continuity during collection** — the rollout is gathered one
   step at a time while threading ``(h, c)`` forward, resetting to zero at each
   episode boundary.
2. **Sequence replay during the update** — instead of shuffling individual
   transitions, we replay contiguous *sequences* from their stored initial
   hidden state (truncated back-propagation through time). Sequences are split
   at episode boundaries and capped at ``seq_len`` so gradients stay stable.

The two agents share the trading domain unchanged — only the network and the
buffer differ — which keeps the "unified PPO, swappable policy" design intact.
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config.training_config import PPOConfig
from ..models.networks import RecurrentActorCritic
from ..models.ppo_agent import resolve_device
from .normalization import RunningNormalizer

Hidden = Tuple[torch.Tensor, torch.Tensor]


class RecurrentRolloutBuffer:
    """On-policy buffer that also records the hidden state entering each step.

    Storing the per-step input hidden state lets the update replay any sequence
    *exactly* from where the policy actually was, rather than approximating with
    a zero state — the difference between correct and merely-plausible recurrent
    PPO.
    """

    def __init__(self, size: int, obs_dim: int, act_dim: int, hidden_size: int) -> None:
        self.size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.size, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros(self.size, dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.values = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.float32)
        self.h_in = np.zeros((self.size, self.hidden_size), dtype=np.float32)
        self.c_in = np.zeros((self.size, self.hidden_size), dtype=np.float32)
        self.advantages = np.zeros(self.size, dtype=np.float32)
        self.returns = np.zeros(self.size, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done, hidden: Hidden) -> None:
        i = self.ptr
        self.observations[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = float(done)
        self.h_in[i] = hidden[0].detach().cpu().numpy().reshape(-1)
        self.c_in[i] = hidden[1].detach().cpu().numpy().reshape(-1)
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr >= self.size

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        adv = 0.0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            adv = delta + gamma * gae_lambda * next_non_terminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def _segment_bounds(self, seq_len: int) -> List[Tuple[int, int]]:
        """Index ranges to replay as sequences: split at episode ends, cap length."""
        bounds: List[Tuple[int, int]] = []
        start = 0
        for t in range(self.size):
            ended = self.dones[t] > 0.5
            capped = (t - start + 1) >= seq_len
            if ended or capped or t == self.size - 1:
                bounds.append((start, t + 1))
                start = t + 1
        return bounds

    def iter_sequences(
        self, seq_len: int, device: torch.device, shuffle: bool = True
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield padded sequence batches (here, one sequence at a time).

        Advantages are normalised across the whole rollout before slicing — the
        same variance-reduction trick the feed-forward buffer uses.
        """
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        bounds = self._segment_bounds(seq_len)
        if shuffle:
            np.random.shuffle(bounds)

        for s, e in bounds:
            h0 = torch.as_tensor(self.h_in[s], device=device).view(1, 1, -1)
            c0 = torch.as_tensor(self.c_in[s], device=device).view(1, 1, -1)
            yield {
                "observations": torch.as_tensor(self.observations[s:e], device=device).unsqueeze(0),
                "actions": torch.as_tensor(self.actions[s:e], device=device).unsqueeze(0),
                "old_log_probs": torch.as_tensor(self.log_probs[s:e], device=device).unsqueeze(0),
                "old_values": torch.as_tensor(self.values[s:e], device=device).unsqueeze(0),
                "advantages": torch.as_tensor(adv[s:e], device=device).unsqueeze(0),
                "returns": torch.as_tensor(self.returns[s:e], device=device).unsqueeze(0),
                "hidden": (h0, c0),
            }


class RecurrentPPOAgent:
    """PPO over an :class:`RecurrentActorCritic`, mirroring :class:`PPOAgent`'s API.

    Acting threads a hidden state through time; the update replays stored
    sequences (truncated BPTT). Everything else — clipped surrogate, clipped
    value loss, entropy bonus, grad clipping — is identical to the feed-forward
    agent, so the comparison between the two is clean.
    """

    def __init__(self, obs_dim, act_dim, config: PPOConfig, device="cpu") -> None:
        self.cfg = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = config.hidden_sizes[-1]
        self.ac = RecurrentActorCritic(
            obs_dim, act_dim, hidden_size=self.hidden_size, init_log_std=config.init_log_std
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=config.learning_rate)
        self.obs_rms = (
            RunningNormalizer(obs_dim, clip=config.obs_clip)
            if getattr(config, "normalize_obs", False)
            else None
        )

    def initial_state(self) -> Hidden:
        return self.ac.initial_state(batch_size=1, device=self.device)

    def observe(self, observation) -> None:
        if self.obs_rms is not None:
            self.obs_rms.update(np.asarray(observation, dtype=np.float64))

    def _norm(self, observation):
        return self.obs_rms.normalize(observation) if self.obs_rms is not None else observation

    @torch.no_grad()
    def select_action(self, observation, hidden: Hidden, deterministic: bool = False):
        """Return ``(action, log_prob, value, next_hidden)`` for one step."""
        obs = torch.as_tensor(self._norm(observation), dtype=torch.float32, device=self.device)
        action, log_prob, value, hidden = self.ac.act(obs, hidden, deterministic)
        return action.cpu().numpy(), float(log_prob.item()), float(value.item()), hidden

    @torch.no_grad()
    def value(self, observation, hidden: Hidden) -> float:
        obs = torch.as_tensor(self._norm(observation), dtype=torch.float32, device=self.device)
        _, _, value, _ = self.ac.act(obs, hidden, deterministic=True)
        return float(value.item())

    def update(self, buffer: RecurrentRolloutBuffer) -> Dict[str, float]:
        policy_losses, value_losses, entropies, clip_fracs = [], [], [], []
        seq_len = self.cfg.recurrent_seq_len

        norm_mean = norm_std = None
        if self.obs_rms is not None:
            norm_mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=self.device)
            norm_std = torch.as_tensor(
                np.sqrt(self.obs_rms.var + self.obs_rms.epsilon), dtype=torch.float32, device=self.device
            )

        for _ in range(self.cfg.update_epochs):
            for batch in buffer.iter_sequences(seq_len, self.device):
                obs_seq = batch["observations"]
                if norm_mean is not None:
                    obs_seq = torch.clamp((obs_seq - norm_mean) / norm_std, -self.obs_rms.clip, self.obs_rms.clip)
                log_probs, entropy, values = self.ac.evaluate_sequence(
                    obs_seq, batch["actions"], batch["hidden"]
                )
                log_probs = log_probs.squeeze(0)
                entropy = entropy.squeeze(0)
                values = values.squeeze(0)
                old_log_probs = batch["old_log_probs"].squeeze(0)
                old_values = batch["old_values"].squeeze(0)
                adv = batch["advantages"].squeeze(0)
                returns = batch["returns"].squeeze(0)

                ratio = torch.exp(log_probs - old_log_probs)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                v_clipped = old_values + torch.clamp(
                    values - old_values, -self.cfg.clip_ratio, self.cfg.clip_ratio
                )
                v_loss = 0.5 * torch.max(
                    (values - returns) ** 2, (v_clipped - returns) ** 2
                ).mean()

                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    + self.cfg.value_coef * v_loss
                    - self.cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > self.cfg.clip_ratio).float().mean().item()
                    )
                policy_losses.append(policy_loss.item())
                value_losses.append(v_loss.item())
                entropies.append(entropy_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clip_fracs)),
        }

    # ------------------------------------------------------------------ #
    # Persistence (compatible signature with PPOAgent)                   #
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.ac.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "config": self.cfg,
                "recurrent": True,
                "obs_rms": self.obs_rms.state_dict() if self.obs_rms is not None else None,
            },
            path,
        )

    def load(self, path: str, map_location: str | None = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.ac.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("obs_rms") is not None:
            self.obs_rms = RunningNormalizer.from_state_dict(ckpt["obs_rms"])

    @classmethod
    def from_checkpoint(cls, path: str, device="cpu") -> "RecurrentPPOAgent":
        device = torch.device(device) if isinstance(device, str) else device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], ckpt["config"], device=device)
        agent.ac.load_state_dict(ckpt["model_state"])
        if ckpt.get("obs_rms") is not None:
            agent.obs_rms = RunningNormalizer.from_state_dict(ckpt["obs_rms"])
        return agent


def evaluate_recurrent_policy(agent: RecurrentPPOAgent, env) -> float:
    """Deterministic full-pass total return for the recurrent policy."""
    obs, info = env.reset()
    start_equity = info["equity"]
    hidden = agent.initial_state()
    done = False
    while not done:
        action, _, _, hidden = agent.select_action(obs, hidden, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info["equity"] / start_equity - 1.0


def run_recurrent_ppo_training(config, df=None, logger=None, train_series_factory=None):
    """Train a recurrent PPO agent — the LSTM analogue of ``run_ppo_training``.

    Same end-to-end contract (data → randomized/fixed training series → PPO loop
    → checkpoint), but the rollout threads ``(h, c)`` across steps and resets it
    at episode boundaries. Returns ``(agent, history)``.
    """
    from ..data.data_loader import prepare_market_data
    from ..envs import make_env
    from .utils import get_logger, set_seed

    log = logger or get_logger("rl_trader", config.train.log_dir)
    set_seed(config.train.seed, getattr(config.train, "deterministic", False))
    device = resolve_device(config.train.device)

    splits = prepare_market_data(df, market=config.market, seed=config.train.seed)
    train_data = train_series_factory() if train_series_factory else splits["train"]
    env = make_env(config.market, train_data, config.env, config.reward)
    eval_env = make_env(
        config.market, splits["val"], config.env, config.reward, random_start=False
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = RecurrentPPOAgent(obs_dim, act_dim, config.ppo, device=device)
    buffer = RecurrentRolloutBuffer(
        config.train.rollout_length, obs_dim, act_dim, agent.hidden_size
    )

    log.info(
        "Training recurrent PPO on %s | obs_dim=%d act_dim=%d device=%s",
        config.market, obs_dim, act_dim, device,
    )

    history: Dict[str, list] = {
        "update": [], "mean_episode_return": [], "mean_episode_equity": [],
        "policy_loss": [], "value_loss": [], "entropy": [], "val_return": [],
    }

    n_updates = max(1, config.train.total_timesteps // config.train.rollout_length)
    obs, _ = env.reset(seed=config.train.seed)
    hidden = agent.initial_state()
    ep_return, ep_returns, ep_equities = 0.0, [], []

    for update in range(1, n_updates + 1):
        buffer.reset()
        for _ in range(config.train.rollout_length):
            agent.observe(obs)
            action, log_prob, value, next_hidden = agent.select_action(obs, hidden)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, log_prob, reward, value, done, hidden)
            obs = next_obs
            hidden = next_hidden
            ep_return += reward
            if done:
                ep_returns.append(ep_return)
                ep_equities.append(info["equity"])
                ep_return = 0.0
                if train_series_factory:
                    env.reload(train_series_factory())
                obs, _ = env.reset()
                hidden = agent.initial_state()  # fresh memory for a new episode

        last_value = 0.0 if done else agent.value(obs, hidden)
        buffer.compute_gae(last_value, config.ppo.gamma, config.ppo.gae_lambda)
        stats = agent.update(buffer)

        mean_ret = float(np.mean(ep_returns[-20:])) if ep_returns else 0.0
        mean_eq = float(np.mean(ep_equities[-20:])) if ep_equities else config.env.initial_balance

        val_return = float("nan")
        if config.train.eval_interval and update % config.train.eval_interval == 0:
            val_return = evaluate_recurrent_policy(agent, eval_env)

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

    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(config.train.checkpoint_dir, f"ppo_lstm_{config.market}.pt")
    agent.save(ckpt_path)
    log.info("Saved recurrent checkpoint -> %s", ckpt_path)
    return agent, history
