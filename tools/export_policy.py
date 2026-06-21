"""Export the trained PPO actor to a tiny NumPy archive for serving.

The deployed backend runs the deterministic policy with a few NumPy matmuls —
no PyTorch, no ONNX runtime — so the Render container stays featherweight and
starts instantly. This script extracts the actor's weights from the PyTorch
checkpoint into ``server/models/ppo_<market>.npz``.

The policy is a small MLP::

    h = obs
    for each trunk Linear (Tanh after all but the last):  h = act(h @ W.T + b)
    action = tanh(h @ W_mean.T + b_mean)      # target position in [-1, 1]

Run from the repo root (PyTorch needed here, not on the server):

    python tools/export_policy.py
"""

from __future__ import annotations

import os

import numpy as np
import torch.nn as nn

from rl_trader.models.ppo_agent import PPOAgent


def export(market: str, ckpt_dir: str = "checkpoints", out_dir: str = "server/models") -> None:
    agent = PPOAgent.from_checkpoint(os.path.join(ckpt_dir, f"ppo_{market}.pt"))
    trunk_linears = [m for m in agent.ac.trunk if isinstance(m, nn.Linear)]

    arrays = {"n_trunk": np.array(len(trunk_linears)),
              "obs_dim": np.array(agent.obs_dim)}
    for i, lin in enumerate(trunk_linears):
        arrays[f"w{i}"] = lin.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float32)
    arrays["wm"] = agent.ac.policy_mean.weight.detach().cpu().numpy().astype(np.float32)
    arrays["bm"] = agent.ac.policy_mean.bias.detach().cpu().numpy().astype(np.float32)

    # Carry the observation normaliser so the server standardises inputs exactly
    # as training did (otherwise the served policy sees out-of-distribution obs).
    if getattr(agent, "obs_rms", None) is not None:
        arrays["obs_mean"] = agent.obs_rms.mean.astype(np.float32)
        arrays["obs_std"] = np.sqrt(agent.obs_rms.var + agent.obs_rms.epsilon).astype(np.float32)
        arrays["obs_clip"] = np.array(agent.obs_rms.clip, dtype=np.float32)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"ppo_{market}.npz")
    np.savez(out, **arrays)
    print(f"exported {market}: obs_dim={agent.obs_dim}, trunk_layers={len(trunk_linears)} -> {out}")


def main() -> None:
    for market in ("stock", "crypto"):
        export(market)


if __name__ == "__main__":
    main()
