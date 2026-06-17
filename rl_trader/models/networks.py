"""Neural-network architectures for the PPO agent.

The default :class:`ActorCritic` is a shared-trunk MLP with two heads: a
Gaussian policy (continuous position sizing) and a scalar value estimate.
Sharing the trunk is parameter-efficient and lets both objectives benefit from
the same learned feature representation.

A :class:`RecurrentActorCritic` (LSTM) is provided as a documented extension
point for sequence modelling. The bundled PPO loop is feed-forward, so wiring
the recurrent variant into training requires sequential mini-batching with
preserved hidden states — intentionally left as an exercise rather than a
half-working default.
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
    """LSTM-based actor-critic (experimental extension point).

    Provided so sequence modelling is a natural next step. Training it requires
    a recurrent rollout buffer that preserves hidden states across time and
    samples whole sequences — not supported by the feed-forward PPO loop here.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 128,
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_size, batch_first=True)
        self.policy_mean = nn.Linear(hidden_size, act_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)

    def forward(self, obs_seq: torch.Tensor, hidden=None):
        """``obs_seq`` has shape [batch, seq_len, obs_dim]."""
        out, hidden = self.lstm(obs_seq, hidden)
        mean = torch.tanh(self.policy_mean(out))
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.value_head(out).squeeze(-1)
        return Normal(mean, std), value, hidden
