"""Running input/reward normalization — a standard, high-impact PPO stabiliser.

Neural policies train best on inputs that are roughly zero-mean, unit-variance.
The data pipeline already standardises the *market features*, but the observation
also carries **account state** (position fraction, cash fraction, normalised
equity) that drifts as equity compounds, and domain-randomized synthetic paths
each arrive on their own scale. A *running* normaliser fixes both: it tracks the
mean/variance of whatever the agent actually sees and standardises on the fly.

:class:`RunningNormalizer` uses Welford's parallel (Chan et al.) update so the
statistics are computed in a single streaming pass with no stored history and no
numerical blow-up — the same algorithm Stable-Baselines3's ``VecNormalize`` uses.
The fitted ``mean``/``var`` travel with the checkpoint and are exported for
inference, so the deployed policy normalises observations exactly as training did.
"""

from __future__ import annotations

import numpy as np


class RunningNormalizer:
    """Streaming mean/variance estimator with clip-and-standardise."""

    def __init__(self, dim: int, clip: float = 10.0, epsilon: float = 1e-8) -> None:
        self.dim = dim
        self.clip = clip
        self.epsilon = epsilon
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = epsilon  # tiny non-zero prior so the first update is stable

    def update(self, x: np.ndarray) -> None:
        """Fold a batch of observations ``x`` (shape ``[batch, dim]``) into the stats."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        # Parallel variance combination (Chan's algorithm).
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Standardise (and clip) using the current statistics; does not update."""
        std = np.sqrt(self.var + self.epsilon)
        out = (np.asarray(x, dtype=np.float64) - self.mean) / std
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    # -- (de)serialisation: plain arrays so it rides along in a checkpoint/npz -- #
    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
            "clip": float(self.clip),
            "epsilon": float(self.epsilon),
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
        self.clip = float(state["clip"])
        self.epsilon = float(state["epsilon"])

    @classmethod
    def from_state_dict(cls, state: dict) -> "RunningNormalizer":
        norm = cls(len(state["mean"]), clip=state.get("clip", 10.0))
        norm.load_state_dict(state)
        return norm
