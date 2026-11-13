"""The NumPy server must compute exactly what the PyTorch agent computes.

The deployed backend runs exported weights without PyTorch. Any architecture the
trainer can produce has to round-trip through tools/export_policy.py and
server/policy.py, or the site serves a different policy from the one measured.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_trader.config.training_config import PPOConfig, published_ppo
from rl_trader.models.ppo_agent import PPOAgent
from server.policy import Policy


@pytest.mark.parametrize("published", [False, True], ids=["repaired", "published"])
def test_exported_policy_matches_torch(tmp_path, monkeypatch, published):
    from tools import export_policy

    torch.manual_seed(0)
    cfg = PPOConfig(hidden_sizes=(16, 16))
    if published:
        published_ppo(cfg)
    agent = PPOAgent(7, 1, cfg)
    rng = np.random.default_rng(0)
    for _ in range(50):
        agent.observe(rng.normal(size=7))
    # Push the mean outside [-1, 1] on some inputs so clipping is exercised.
    with torch.no_grad():
        agent.ac.policy_mean.bias.fill_(0.8)
    agent.save(str(tmp_path / "ppo_stock.pt"))
    export_policy.export("stock", ckpt_dir=str(tmp_path), out_dir=str(tmp_path))
    pol = Policy.load(str(tmp_path / "ppo_stock.npz"), "stock")

    assert pol.squash_mean is published
    assert (pol.n_vtrunk == 0) is published
    for _ in range(20):
        obs = rng.normal(size=7) * 3
        a, _, v = agent.select_action(obs, deterministic=True)
        out = pol.evaluate(obs)
        assert out.action == pytest.approx(float(np.clip(a[0], -1, 1)), abs=1e-5)
        assert out.value == pytest.approx(v, abs=1e-4)
