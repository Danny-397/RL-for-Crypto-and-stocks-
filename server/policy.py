"""Serving-side policy: the trained actor-critic as a few NumPy matmuls.

The deployed backend deliberately runs **without PyTorch** — the exported
archives in ``server/models/*.npz`` hold the actor's weights and are evaluated
directly with NumPy, which keeps the Render container tiny and cold-starts fast.
This module owns that forward pass so the Flask layer stays a thin router.

The network mirrors :class:`rl_trader.models.networks.ActorCritic` exactly::

    h = clip((obs - obs_mean) / obs_std, -c, +c)     # if a normaliser was exported
    for each trunk Linear (Tanh after all but the LAST):
        h = act(h @ W.T + b)
    action = tanh(h @ W_mean.T + b_mean)             # target position in [-1, 1]
    value  = h @ W_value.T + b_value                 # critic, when exported

**The critic is optional.** Older archives exported only the actor, so
:attr:`Policy.has_value` reports whether a value estimate is available and
:meth:`Policy.evaluate` returns ``value=None`` when it is not. Callers must
surface that absence rather than substituting a placeholder — an invented
critic reading would be indistinguishable from a real one to a visitor.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


@dataclass
class PolicyOutput:
    """One forward pass: the deterministic action and (optionally) the critic."""

    action: float
    value: Optional[float]


class Policy:
    """A deterministic actor(-critic) restored from an exported ``.npz``."""

    def __init__(self, arrays: Dict[str, np.ndarray], name: str, sha256: str) -> None:
        self._p = arrays
        self.name = name
        self.sha256 = sha256
        self.n_trunk = int(arrays["n_trunk"])
        self.obs_dim = int(arrays["obs_dim"]) if "obs_dim" in arrays else -1
        # The critic head is exported only by newer runs of tools/export_policy.py.
        self.has_value = "wv" in arrays and "bv" in arrays
        self.has_normalizer = "obs_mean" in arrays

    # -- construction ---------------------------------------------------- #
    @classmethod
    def load(cls, path: str, name: str) -> "Policy":
        """Load an exported archive, recording its hash for provenance."""
        with open(path, "rb") as fh:
            sha256 = hashlib.sha256(fh.read()).hexdigest()
        with np.load(path) as d:
            arrays = {k: d[k] for k in d.files}
        return cls(arrays, name=name, sha256=sha256)

    # -- inference ------------------------------------------------------- #
    def _trunk(self, obs: np.ndarray) -> np.ndarray:
        """Run the shared trunk, returning its (un-activated) final features."""
        x = obs.reshape(1, -1).astype(np.float32)
        if self.has_normalizer:
            p = self._p
            clip = float(p["obs_clip"]) if "obs_clip" in p else 10.0
            x = np.clip((x - p["obs_mean"]) / p["obs_std"], -clip, clip).astype(np.float32)
        for i in range(self.n_trunk):
            x = x @ self._p[f"w{i}"].T + self._p[f"b{i}"]
            if i < self.n_trunk - 1:
                x = np.tanh(x)
        return x

    def evaluate(self, obs: np.ndarray) -> PolicyOutput:
        """Return the deterministic action and the critic's value (if exported)."""
        h = self._trunk(obs)
        action = float(np.tanh(h @ self._p["wm"].T + self._p["bm"]).reshape(-1)[0])
        value = None
        if self.has_value:
            value = float((h @ self._p["wv"].T + self._p["bv"]).reshape(-1)[0])
        return PolicyOutput(action=action, value=value)

    def act(self, obs: np.ndarray) -> float:
        """Convenience: just the deterministic target position in ``[-1, 1]``."""
        return self.evaluate(obs).action

    def describe(self) -> dict:
        """Provenance for the reproducibility receipt."""
        return {
            "name": self.name,
            "sha256": self.sha256[:16],
            "obs_dim": self.obs_dim,
            "trunk_layers": self.n_trunk,
            "has_value_head": self.has_value,
            "has_obs_normalizer": self.has_normalizer,
        }


def load_policies(models_dir: str = MODELS_DIR) -> Dict[str, Policy]:
    """Load every ``ppo_<name>.npz`` in ``models_dir``.

    A missing or unreadable archive is skipped rather than fatal: the API should
    still serve the markets whose policies *did* load, and ``/health`` reports
    exactly which ones those are.
    """
    policies: Dict[str, Policy] = {}
    if not os.path.isdir(models_dir):
        return policies
    for fname in sorted(os.listdir(models_dir)):
        if not (fname.startswith("ppo_") and fname.endswith(".npz")):
            continue
        name = fname[len("ppo_"):-len(".npz")]
        try:
            policies[name] = Policy.load(os.path.join(models_dir, fname), name=name)
        except Exception:  # pragma: no cover - a corrupt archive must not 500 the app
            continue
    return policies
