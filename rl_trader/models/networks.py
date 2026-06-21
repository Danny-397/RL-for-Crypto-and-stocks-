"""Neural-network architectures for the PPO agent.

The default :class:`ActorCritic` is a shared-trunk MLP with two heads: a
Gaussian policy (continuous position sizing) and a scalar value estimate.
Sharing the trunk is parameter-efficient and lets both objectives benefit from
the same learned feature representation.

A :class:`RecurrentActorCritic` (LSTM) is also provided and **fully trainable**
via :mod:`rl_trader.training.recurrent`, which threads hidden states through the
rollout and replays whole sequences (truncated BPTT) during the PPO update.
Select it with ``PPOConfig.use_lstm = True``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(
    sizes: Sequence[int],
    activation: type[nn.Module] = nn.Tanh,
    output_activation: type[nn.Module] = nn.Identity,
) -> nn.Sequential:
    """Build a simple multi-layer perceptron from a list of layer sizes."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = i == len(sizes) - 2
        layers.append(output_activation() if is_last else activation())
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Orthogonal weight init — a small but reliable PPO stability win."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)
    return module


class ActorCritic(nn.Module):
    """Shared-trunk MLP with a Gaussian policy head and a value head.

    The policy mean is squashed through ``tanh`` so it stays within the action
    bounds; a state-independent ``log_std`` parameter controls exploration. The
    action itself is *not* tanh-squashed (the environment clips it), which keeps
    the log-probability exact and avoids the change-of-variables correction —
    a common, well-understood simplification for bounded-action PPO.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.trunk = mlp([obs_dim, *hidden_sizes], activation=nn.Tanh)
        self.trunk.apply(lambda m: _orthogonal_init(m, gain=2.0**0.5))

        last = hidden_sizes[-1]
        # Small policy-head gain keeps the initial policy near-deterministic and
        # well-behaved; the value head uses unit gain.
        self.policy_mean = _orthogonal_init(nn.Linear(last, act_dim), gain=0.01)
        self.value_head = _orthogonal_init(nn.Linear(last, 1), gain=1.0)
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)

    def forward(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """Return the action distribution and the state-value estimate."""
        features = self.trunk(obs)
        mean = torch.tanh(self.policy_mean(features))
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.value_head(features).squeeze(-1)
        return Normal(mean, std), value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action; return (action, summed log-prob, value)."""
        dist, value = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate stored actions: return (log_probs, entropy, values)."""
        dist, values = self.forward(obs)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return log_probs, entropy, values


class RecurrentActorCritic(nn.Module):
    """LSTM-based actor-critic for sequence-aware trading policies.

    Where :class:`ActorCritic` sees a fixed window flattened into one vector,
    this network carries an LSTM hidden state across time, so the policy can in
    principle model longer-range temporal dependencies than the window exposes.
    It is trained by :class:`~rl_trader.training.recurrent.RecurrentPPOAgent`,
    which preserves hidden states across the rollout and replays whole sequences
    (truncated BPTT) during the PPO update.

    A small pre-LSTM encoder + orthogonal init mirror the MLP's stability tricks.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 128,
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = mlp([obs_dim, hidden_size], activation=nn.Tanh)
        self.encoder.apply(lambda m: _orthogonal_init(m, gain=2.0**0.5))
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        self.policy_mean = _orthogonal_init(nn.Linear(hidden_size, act_dim), gain=0.01)
        self.value_head = _orthogonal_init(nn.Linear(hidden_size, 1), gain=1.0)
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)

    def initial_state(self, batch_size: int = 1, device=None):
        """Return a zeroed ``(h, c)`` hidden state for a fresh sequence."""
        shape = (1, batch_size, self.hidden_size)
        h = torch.zeros(shape, device=device)
        c = torch.zeros(shape, device=device)
        return h, c

    def forward(self, obs_seq: torch.Tensor, hidden=None):
        """Run a sequence through the network.

        ``obs_seq`` has shape ``[batch, seq_len, obs_dim]``. Returns the action
        distribution, the value estimate ``[batch, seq_len]``, and the final
        hidden state (for continuing the rollout).
        """
        encoded = torch.tanh(self.encoder(obs_seq))
        out, hidden = self.lstm(encoded, hidden)
        mean = torch.tanh(self.policy_mean(out))
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.value_head(out).squeeze(-1)
        return Normal(mean, std), value, hidden

    @torch.no_grad()
    def act(self, obs: torch.Tensor, hidden, deterministic: bool = False):
        """Advance one step. ``obs`` is ``[obs_dim]``; returns action, log-prob,
        value, and the updated hidden state."""
        dist, value, hidden = self.forward(obs.view(1, 1, -1), hidden)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.view(-1), log_prob.view(()), value.view(()), hidden

    def evaluate_sequence(
        self, obs_seq: torch.Tensor, actions: torch.Tensor, hidden
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate stored actions over a sequence for the PPO update.

        ``obs_seq`` is ``[batch, seq_len, obs_dim]`` and ``actions`` is
        ``[batch, seq_len, act_dim]``; returns per-step ``(log_probs, entropy,
        values)``, each ``[batch, seq_len]``.
        """
        dist, values, _ = self.forward(obs_seq, hidden)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy, values
