"""Train a hedging policy with the repo's PPO agent on batches of simulated paths.

Hedging episodes are short (21 steps), so instead of one environment stepping alone
this steps ``n_envs`` episodes together and hands the update the same mini-batches
:class:`~rl_trader.models.ppo_agent.PPOAgent` already trains on. Every rollout
draws fresh paths with a fresh volatility per path (domain randomization): the
agent never sees the same path twice, so it cannot memorise one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np
import torch

from ..config.training_config import PPOConfig
from ..models.ppo_agent import PPOAgent
from ..training.utils import set_seed
from .core import OBS_DIM, HedgeSpec, VecHedgeEnv, action_to_hedge, simulate_paths


@dataclass
class HedgeTrainConfig:
    n_envs: int = 256                 # episodes stepped together per rollout
    updates: int = 150                # PPO updates (each on n_envs * n_steps transitions)
    gamma: float = 1.0                # a finite 21-step episode: no discounting
    gae_lambda: float = 0.95
    seed: int = 0
    lr_anneal: bool = True            # linear decay to zero: settles the policy at the end
    ppo: PPOConfig = field(default_factory=lambda: PPOConfig(
        learning_rate=3e-4, entropy_coef=0.0, init_log_std=-1.0, hidden_sizes=(64, 64),
        update_epochs=8, minibatch_size=512, normalize_obs=True))


class _Batch:
    """The rollout, flattened, in the shape PPOAgent.update expects."""

    def __init__(self, obs, act, logp, val, adv, ret):
        self.obs, self.act, self.logp, self.val, self.adv, self.ret = obs, act, logp, val, adv, ret

    def iter_minibatches(self, minibatch_size, device):
        adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)
        idx = np.random.permutation(len(self.obs))
        for s in range(0, len(idx), minibatch_size):
            j = idx[s:s + minibatch_size]
            yield {'observations': torch.as_tensor(self.obs[j], device=device),
                   'actions': torch.as_tensor(self.act[j], device=device),
                   'old_log_probs': torch.as_tensor(self.logp[j], device=device),
                   'old_values': torch.as_tensor(self.val[j], device=device),
                   'advantages': torch.as_tensor(adv[j], device=device),
                   'returns': torch.as_tensor(self.ret[j], device=device)}


@torch.no_grad()
def _act(agent: PPOAgent, obs: np.ndarray, deterministic: bool = False):
    x = agent.obs_rms.normalize(obs) if agent.obs_rms is not None else obs
    dist, value = agent.ac(torch.as_tensor(x, dtype=torch.float32, device=agent.device))
    a = dist.mean if deterministic else dist.sample()
    return a.cpu().numpy(), dist.log_prob(a).sum(-1).cpu().numpy(), value.cpu().numpy()


def agent_policy(agent: PPOAgent):
    """The trained policy (its mean action) as a ``policy(env) -> hedges`` function."""
    def policy(env: VecHedgeEnv) -> np.ndarray:
        a, _, _ = _act(agent, env.obs(), deterministic=True)
        return action_to_hedge(a[:, 0])
    return policy


def train_hedger(spec: HedgeSpec, cfg: Optional[HedgeTrainConfig] = None,
                 log_every: int = 0) -> (PPOAgent, Dict[str, List[float]]):
    """Train one policy for ``spec``; return the agent and its learning curve."""
    cfg = cfg or HedgeTrainConfig()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    agent = PPOAgent(OBS_DIM, 1, cfg.ppo, device='cpu')
    T, N = spec.n_steps, cfg.n_envs
    hist: Dict[str, List[float]] = {'update': [], 'objective': [], 'pnl_std': [], 'cost': []}
    t0 = time.time()
    for u in range(1, cfg.updates + 1):
        S, sig = simulate_paths(spec, N, rng)
        env = VecHedgeEnv(S, sig, spec)
        obs = env.reset()
        OB = np.zeros((T, N, OBS_DIM), np.float32)
        A = np.zeros((T, N, 1), np.float32)
        LP = np.zeros((T, N), np.float32)
        V = np.zeros((T, N), np.float32)
        R = np.zeros((T, N), np.float32)
        for t in range(T):
            if agent.obs_rms is not None:
                agent.obs_rms.update(obs)
            a, lp, v = _act(agent, obs)
            OB[t], A[t], LP[t], V[t] = obs, a, lp, v
            obs, r, done, _ = env.step(action_to_hedge(a[:, 0]))
            R[t] = r
        # GAE over each episode; the episode ends at T, so nothing is bootstrapped.
        adv = np.zeros((T, N), np.float32)
        last = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nxt = V[t + 1] if t + 1 < T else np.zeros(N, np.float32)
            delta = R[t] + cfg.gamma * nxt - V[t]
            last = delta + cfg.gamma * cfg.gae_lambda * last * (t + 1 < T)
            adv[t] = last
        ret = adv + V
        if cfg.lr_anneal:
            for g in agent.optimizer.param_groups:
                g['lr'] = cfg.ppo.learning_rate * (1.0 - (u - 1) / cfg.updates)
        batch = _Batch(OB.reshape(-1, OBS_DIM), A.reshape(-1, 1), LP.ravel(), V.ravel(), adv.ravel(), ret.ravel())
        agent.update(batch)
        hist['update'].append(u)
        hist['objective'].append(float(R.sum(axis=0).mean()))
        hist['pnl_std'].append(float(100 * env.pnl.std()))
        hist['cost'].append(float(100 * env.costs.mean()))
        if log_every and u % log_every == 0:
            print(f'  update {u}/{cfg.updates}  objective {hist["objective"][-1]:+.3f}  '
                  f'P&L std {hist["pnl_std"][-1]:.3f}%  cost {hist["cost"][-1]:.3f}%  ({time.time() - t0:.0f}s)')
    return agent, hist


def with_seed(cfg: HedgeTrainConfig, seed: int) -> HedgeTrainConfig:
    return replace(cfg, seed=seed)
