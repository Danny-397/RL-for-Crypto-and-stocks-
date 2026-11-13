"""Proximal Policy Optimization agent.

A compact, readable PPO implementation built on the :class:`ActorCritic`
network. The agent is deliberately decoupled from the trading domain: it speaks
only in observation/action tensors, so the same class trains on the stock and
crypto environments unchanged — exactly the "unified agent, separate
environments" design the project is built around.

Key PPO ingredients implemented here:
    * clipped surrogate policy objective
    * clipped value-function loss
    * entropy bonus for exploration
    * mini-batch updates over multiple epochs per rollout
    * gradient clipping for stability
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from ..config.training_config import PPOConfig
from ..training.normalization import RunningNormalizer
from .networks import ActorCritic


def resolve_device(preference: str = "auto") -> torch.device:
    """Map a device preference string to a concrete ``torch.device``."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


class PPOAgent:
    """A clip-objective PPO agent with a shared-trunk actor-critic network."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: PPOConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.ac = ActorCritic(
            obs_dim,
            act_dim,
            hidden_sizes=config.hidden_sizes,
            init_log_std=config.init_log_std,
            shared_trunk=getattr(config, "shared_trunk", True),
            squash_mean=getattr(config, "squash_mean", True),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(), lr=config.learning_rate, eps=getattr(config, "adam_eps", 1e-8))

        # Running observation normaliser (fitted during training, frozen at eval).
        self.obs_rms = (
            RunningNormalizer(obs_dim, clip=config.obs_clip)
            if getattr(config, "normalize_obs", False)
            else None
        )

    def observe(self, observation: np.ndarray) -> None:
        """Fold a raw observation into the normaliser (called during rollout only)."""
        if self.obs_rms is not None:
            self.obs_rms.update(np.asarray(observation, dtype=np.float64))

    def _norm(self, observation: np.ndarray) -> np.ndarray:
        return self.obs_rms.normalize(observation) if self.obs_rms is not None else observation

    @property
    def _frozen_norm(self) -> bool:
        return self.obs_rms is not None and getattr(self.cfg, "normalize_at_collection", False)

    def buffer_obs(self, observation: np.ndarray) -> np.ndarray:
        """What the rollout buffer should store for ``observation``.

        Published behaviour: the raw observation, normalised at update time with
        the statistics as they stand *after* the rollout, which differ from the
        ones the action was sampled under, so the first PPO epoch starts with a
        probability ratio that is not 1. With ``normalize_at_collection`` the
        observation is stored already normalised, exactly as the policy saw it
        (what Stable-Baselines3's VecNormalize does).
        """
        if self._frozen_norm:
            return np.asarray(self._norm(observation), dtype=np.float32)
        return observation

    # ------------------------------------------------------------------ #
    # Acting                                                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def select_action(self, observation: np.ndarray, deterministic: bool = False):
        """Choose an action for a single observation.

        Returns ``(action, log_prob, value)`` as NumPy/floats. When
        ``deterministic`` is True the policy mean is used (for evaluation).
        """
        obs = torch.as_tensor(self._norm(observation), dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, value = self.ac(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def value(self, observation: np.ndarray) -> float:
        """Estimate the state value (used to bootstrap the final GAE step)."""
        obs = torch.as_tensor(self._norm(observation), dtype=torch.float32, device=self.device).unsqueeze(0)
        _, value = self.ac(obs)
        return float(value.item())

    # ------------------------------------------------------------------ #
    # Learning                                                           #
    # ------------------------------------------------------------------ #
    def update(self, buffer) -> Dict[str, float]:
        """Run several PPO epochs over a completed rollout buffer.

        ``buffer`` must expose ``iter_minibatches(minibatch_size)`` yielding
        dicts of tensors with keys: observations, actions, old_log_probs,
        advantages, returns, old_values.
        """
        policy_losses, value_losses, entropies, clip_fracs = [], [], [], []

        # The buffer stores *raw* observations; normalise them here with the stats
        # accumulated over the rollout (frozen for the duration of the update).
        norm_mean = norm_std = None
        if self.obs_rms is not None and not self._frozen_norm:
            norm_mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=self.device)
            norm_std = torch.as_tensor(
                np.sqrt(self.obs_rms.var + self.obs_rms.epsilon), dtype=torch.float32, device=self.device
            )

        per_minibatch = getattr(self.cfg, "advantage_norm", "rollout") == "minibatch"
        for _ in range(self.cfg.update_epochs):
            extra = {"normalize_advantages": False} if per_minibatch else {}
            for batch in buffer.iter_minibatches(self.cfg.minibatch_size, self.device, **extra):
                obs = batch["observations"]
                if norm_mean is not None:
                    obs = torch.clamp((obs - norm_mean) / norm_std, -self.obs_rms.clip, self.obs_rms.clip)
                log_probs, entropy, values = self.ac.evaluate(obs, batch["actions"])
                ratio = torch.exp(log_probs - batch["old_log_probs"])
                adv = batch["advantages"]
                if per_minibatch:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Clipped surrogate policy objective.
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Clipped value loss (mirrors the policy clip for stability).
                v_clipped = batch["old_values"] + torch.clamp(
                    values - batch["old_values"], -self.cfg.clip_ratio, self.cfg.clip_ratio
                )
                v_loss_unclipped = (values - batch["returns"]) ** 2
                v_loss_clipped = (v_clipped - batch["returns"]) ** 2
                if getattr(self.cfg, "value_loss", "max") == "clipped":
                    # Stable-Baselines3's form: squared error of the clipped value.
                    value_loss = v_loss_clipped.mean()
                else:
                    # Published form (OpenAI Baselines): half the larger of the two.
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
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
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clip_fracs)),
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Serialise network + optimiser + dimensions for exact reloading."""
        torch.save(
            {
                "model_state": self.ac.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "config": self.cfg,
                "obs_rms": self.obs_rms.state_dict() if self.obs_rms is not None else None,
            },
            path,
        )

    def load(self, path: str, map_location: str | None = None) -> None:
        """Load weights saved by :meth:`save`."""
        ckpt = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.ac.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("obs_rms") is not None:
            self.obs_rms = RunningNormalizer.from_state_dict(ckpt["obs_rms"])

    @staticmethod
    def _upgrade_config(cfg):
        """Fill settings an older checkpoint's config predates with what it used.

        A pickled PPOConfig from before a field existed has no value for it, and
        attribute lookup would fall through to the *current* class default. For
        the PPO repairs (RESULTS.md §10a) that default changed in v2.3.0, so a
        v2.2.1 checkpoint would rebuild the repaired architecture and fail to
        load. Anything missing was, by construction, the published setting.
        """
        from ..config.training_config import PUBLISHED_PPO

        for k, v in PUBLISHED_PPO.items():
            if k not in vars(cfg):
                setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_checkpoint(
        cls, path: str, device: torch.device | str = "cpu"
    ) -> "PPOAgent":
        """Rebuild an agent directly from a checkpoint file."""
        device = torch.device(device) if isinstance(device, str) else device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        agent = cls(ckpt["obs_dim"], ckpt["act_dim"], cls._upgrade_config(ckpt["config"]), device=device)
        agent.ac.load_state_dict(ckpt["model_state"])
        if ckpt.get("obs_rms") is not None:
            agent.obs_rms = RunningNormalizer.from_state_dict(ckpt["obs_rms"])
        return agent
