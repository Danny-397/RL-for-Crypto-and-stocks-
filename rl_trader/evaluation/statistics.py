"""Statistical tools for honest, defensible performance claims.

A single backtest number is an anecdote. To say something *true* about a trading
strategy you need to quantify uncertainty (how much would this change under a
different seed or a different slice of history?) and test significance (is the
agent's edge over a benchmark distinguishable from noise?).

This module provides the two primitives the evaluation scripts build on:

* :func:`bootstrap_ci` — a distribution-free confidence interval for any summary
  statistic, via resampling. No normality assumption, which matters for fat-
  tailed return data.
* :func:`paired_permutation_test` — a paired test of whether strategy A beats
  strategy B across a set of assets/seeds, using the exact sign-flip
  permutation distribution. Returns a two-sided p-value.

Both are deliberately dependency-light (NumPy only) so they run anywhere the
rest of the framework does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np


@dataclass
class Estimate:
    """A point estimate with a bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.mean:+.4f} (95% CI [{self.low:+.4f}, {self.high:+.4f}], n={self.n})"


def bootstrap_ci(
    samples: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = 10_000,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for ``statistic`` over ``samples``.

    Resamples ``samples`` with replacement ``n_boot`` times, recomputes the
    statistic each time, and takes the empirical percentiles as the interval.
    """
    data = np.asarray(samples, dtype=np.float64)
    n = len(data)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    point = float(statistic(data))
    if n == 1:
        return Estimate(point, point, point, 1)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(data[i]) for i in idx])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return Estimate(point, float(low), float(high), n)


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = 20_000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Two-sided paired permutation test on the mean difference ``a - b``.

    Pairs are matched (e.g. the same asset evaluated under strategy A and B).
    Under the null "A and B are exchangeable within each pair", we randomly flip
    the sign of each pair's difference and rebuild the null distribution of the
    mean difference. Returns ``(observed_mean_difference, p_value)``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired test requires equal-length samples")
    diff = a - b
    observed = float(diff.mean())
    n = len(diff)
    if n == 0:
        return observed, float("nan")

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null = (signs * diff).mean(axis=1)
    # +1 in numerator/denominator: the observed assignment is itself one valid
    # permutation, which keeps the p-value from ever being exactly zero.
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, float(p)


def summarize(samples: Sequence[float], **kwargs) -> dict:
    """Convenience: mean / std / min / max plus a bootstrap CI as a flat dict."""
    data = np.asarray(samples, dtype=np.float64)
    est = bootstrap_ci(data, **kwargs)
    return {
        "mean": est.mean,
        "ci_low": est.low,
        "ci_high": est.high,
        "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "min": float(data.min()) if len(data) else float("nan"),
        "max": float(data.max()) if len(data) else float("nan"),
        "n": est.n,
    }
