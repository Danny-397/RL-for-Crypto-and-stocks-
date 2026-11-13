"""Reporting statistics recommended for deep-RL evaluation.

Agarwal et al. (2021), *Deep Reinforcement Learning at the Edge of the Statistical
Precipice*, argue that a mean over a handful of seeds is a poor summary of an RL
result and recommend instead:

* the **interquartile mean** (IQM): the mean of the middle 50% of runs, robust
  to one outlier seed but less noisy than the median;
* **bootstrap confidence intervals** for that statistic, resampling runs;
* **performance profiles**: the fraction of runs scoring above each threshold,
  which shows the whole distribution instead of one number;
* the **probability of improvement** P(X > Y): how often a run of one method
  beats a run of another.

This module implements those with NumPy only (no ``rliable`` dependency), plus
two tests used across the study:

* :func:`two_sample_diff` — difference of means between two independent sets of
  runs, with a bootstrap CI and an exact-or-Monte-Carlo permutation p-value;
* :func:`holm` — the Holm-Bonferroni step-down correction for a family of tests.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Sequence

import numpy as np


def iqm(x: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle 50% (trimmed 25% each side).

    Uses fractional weighting at the quartile boundaries, as ``rliable`` does
    (scipy.stats.trim_mean semantics for n divisible by 4, and a weighted
    generalisation otherwise), so it is defined for any n >= 1.
    """
    a = np.sort(np.asarray(x, dtype=float))
    n = len(a)
    if n == 0:
        return float("nan")
    lo, hi = 0.25 * n, 0.75 * n
    w = np.clip(np.minimum(np.arange(1, n + 1), hi) - np.maximum(np.arange(n), lo), 0, 1)
    return float((w * a).sum() / w.sum())


def bootstrap_interval(x: Sequence[float], stat=iqm, n_boot: int = 10_000,
                       confidence: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for ``stat`` over runs."""
    a = np.asarray(x, dtype=float)
    if len(a) < 2:
        v = stat(a) if len(a) else float("nan")
        return float(v), float(v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boots = np.array([stat(a[i]) for i in idx])
    q = (1 - confidence) / 2
    return float(np.quantile(boots, q)), float(np.quantile(boots, 1 - q))


def aggregate_scores(x: Sequence[float]) -> Dict[str, float]:
    """Mean, median and IQM with a 95% bootstrap CI on the IQM."""
    a = np.asarray(x, dtype=float)
    lo, hi = bootstrap_interval(a, iqm)
    return {
        "n": int(len(a)),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "iqm": round(iqm(a), 4),
        "iqm_low": round(lo, 4),
        "iqm_high": round(hi, 4),
    }


def performance_profile(x: Sequence[float], thresholds: Sequence[float]) -> list[float]:
    """Fraction of runs scoring strictly above each threshold."""
    a = np.asarray(x, dtype=float)
    return [float((a > t).mean()) for t in thresholds]


def probability_of_improvement(x: Sequence[float], y: Sequence[float]) -> float:
    """P(X > Y) over all pairs of runs, ties counted as half (Mann-Whitney U / nm)."""
    a, b = np.asarray(x, dtype=float)[:, None], np.asarray(y, dtype=float)[None, :]
    return float(((a > b) + 0.5 * (a == b)).mean())


def two_sample_diff(x: Sequence[float], y: Sequence[float], n_boot: int = 10_000,
                    n_perm: int = 100_000, seed: int = 0) -> Dict[str, float]:
    """mean(x) - mean(y) for two independent sets of runs.

    Returns the difference, a 95% bootstrap CI (each group resampled on its own),
    a two-sided permutation p-value on the difference of means (exact when the
    number of relabellings is at most ``n_perm``, Monte Carlo otherwise), and
    P(X > Y).
    """
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    diff = float(a.mean() - b.mean())
    ba = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
    bb = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
    lo, hi = np.quantile(ba - bb, [0.025, 0.975])

    pooled = np.concatenate([a, b])
    n, total = len(a), pooled.sum()
    from math import comb
    if comb(len(pooled), n) <= n_perm:
        sums = np.array([pooled[list(c)].sum() for c in combinations(range(len(pooled)), n)])
    else:
        sums = np.array([rng.permutation(pooled)[:n].sum() for _ in range(n_perm)])
    d = sums / n - (total - sums) / len(b)
    p = float((np.abs(d) >= abs(diff) - 1e-12).mean())
    return {
        "diff": round(diff, 4), "low": round(float(lo), 4), "high": round(float(hi), 4),
        "p": round(p, 4), "p_improve": round(probability_of_improvement(a, b), 3),
    }


def holm(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Holm-Bonferroni step-down adjustment for a family of tests.

    Returns, per test, the adjusted p-value (monotone, capped at 1) and whether it
    is rejected at ``alpha`` after adjustment. Controls the family-wise error rate
    under any dependence between the tests.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = {"p": p, "p_holm": round(running, 4), "reject": running <= alpha}
    return out
