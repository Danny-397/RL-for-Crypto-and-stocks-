"""Live statistical inference over the project's real seed-level results.

This is the computational core of the "is the result real — or luck?" experiment.
The data is precomputed and real (see :mod:`server.precomputed`); the *inference*
runs on demand using the exact estimators the research code and the paper use —
:func:`rl_trader.evaluation.statistics.bootstrap_ci` and
:func:`~rl_trader.evaluation.statistics.paired_permutation_test`. Nothing is
reimplemented here, so the site cannot drift from the paper.

Two axes, two questions — do not conflate them
----------------------------------------------
The project's real-market study reports two numbers that live on *different*
axes, and pairing on the wrong one silently changes the claim:

* **Across training seeds** (n = 5): "how repeatable is this?" Answered with a
  bootstrap CI over per-seed basket-mean returns.
* **Across held-out tickers** (n = 10 stocks / 6 crypto): "is the cross-sectional
  edge distinguishable from noise?" Answered with a paired permutation test of
  agent vs. buy-&-hold, per asset. This is where the published p-value comes
  from (``tools/real_significance.py``).

The sign-flip resolution floor
------------------------------
A two-sided paired sign-flip test over ``n`` pairs draws from only ``2**n``
equally-likely sign assignments. When every difference shares a sign — the most
extreme case possible — exactly 2 of them match or exceed the observed statistic,
so the smallest attainable p-value is ``2 / 2**n``:

    n = 5  -> p >= 0.0625   (can never reach significance at 0.05)
    n = 6  -> p >= 0.031
    n = 10 -> p >= 0.002

:func:`permutation_floor` reports this alongside every test, because a p-value of
0.0625 at n = 5 means "this design cannot resolve it", not "there is no effect".
Surfacing that distinction is the honest thing to do and is itself the kind of
methodological point the project exists to make.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from rl_trader.evaluation.statistics import bootstrap_ci, paired_permutation_test

# Bounds keep a public endpoint from becoming a CPU sink; both sit far above the
# values these estimators need to be stable.
MAX_BOOT = 50_000
MAX_PERM = 100_000


def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def permutation_floor(n_pairs: int) -> dict:
    """Smallest p-value a two-sided sign-flip test over ``n_pairs`` can return."""
    n = max(0, int(n_pairs))
    floor = (2.0 / (2**n)) if 0 < n <= 30 else 0.0
    return {
        "n_pairs": n,
        # 2/2**30 needs ~10 significant places; rounding shorter would misreport it.
        "min_attainable_p": round(floor, 12),
        "can_reach_05": bool(floor < 0.05),
        "explanation": (
            f"A two-sided sign-flip test over {n} pairs draws from 2^{n} = {2**n} "
            f"sign assignments; at most 2 are as extreme as the observed statistic, "
            f"so p can never fall below {floor:.4f}."
        )
        if 0 < n <= 30
        else "n is large enough that the resolution floor is negligible.",
    }


def analyze(
    values: Sequence[float],
    benchmark: Optional[float] = None,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    n_perm: int = 20_000,
    seed: int = 0,
) -> dict:
    """Single-seed vs multi-seed analysis of one real per-training-seed series.

    The bootstrap CI is the estimator the paper uses on this axis, and it is the
    headline. When ``benchmark`` is supplied a one-sample sign-flip test against
    that constant is also reported — but it is explicitly labelled as the
    *seed axis*, distinct from the paper's cross-sectional test, and it always
    carries its resolution floor so an unreachable p-value cannot be misread.
    """
    data = np.asarray(list(values), dtype=np.float64)
    n = len(data)
    if n == 0:
        raise ValueError("no values to analyse")

    confidence = float(min(0.999, max(0.5, confidence)))
    n_boot = _clamp(n_boot, 100, MAX_BOOT, 10_000)
    n_perm = _clamp(n_perm, 100, MAX_PERM, 20_000)

    est = bootstrap_ci(data, confidence=confidence, n_boot=n_boot, seed=seed)
    best_idx, worst_idx = int(np.argmax(data)), int(np.argmin(data))

    out: dict = {
        "axis": "training_seed",
        "n_seeds": n,
        "confidence": confidence,
        "n_boot": n_boot,
        "values": [round(float(v), 6) for v in data],
        # What a single lucky run would have let you claim...
        "single_seed": {
            "best": round(float(data[best_idx]), 6),
            "best_index": best_idx,
            "worst": round(float(data[worst_idx]), 6),
            "worst_index": worst_idx,
            "spread": round(float(data[best_idx] - data[worst_idx]), 6),
        },
        # ...versus what the whole set actually supports.
        "multi_seed": {
            "mean": round(float(est.mean), 6),
            "median": round(float(np.median(data)), 6),
            "std": round(float(data.std(ddof=1)), 6) if n > 1 else 0.0,
            "ci_low": round(float(est.low), 6),
            "ci_high": round(float(est.high), 6),
            "ci_excludes_zero": bool(est.low > 0.0 or est.high < 0.0),
        },
    }

    if benchmark is not None:
        bench = float(benchmark)
        observed, p = paired_permutation_test(
            data, np.full(n, bench), n_perm=n_perm, seed=seed
        )
        floor = permutation_floor(n)
        beats = int(np.sum(data > bench))
        out["benchmark"] = {
            "value": round(bench, 6),
            "mean_difference": round(float(observed), 6),
            "p_value": round(float(p), 6),
            "n_perm": n_perm,
            "significant_at_05": bool(p < 0.05),
            "seeds_beating_benchmark": beats,
            "resolution": floor,
            "axis_note": (
                "Sign-flip test on the TRAINING-SEED axis against a constant "
                "benchmark. The published p-value in RESULTS.md is a different "
                "test — paired across held-out tickers — so these will not match."
            ),
            "verdict": _verdict(observed, p, beats, n, floor),
        }
    return out


def _verdict(observed: float, p: float, beats: int, n: int, floor: dict) -> str:
    """A plain-language reading, stated carefully and never oversold."""
    if not floor["can_reach_05"]:
        return (
            f"Underpowered by construction: with {n} pairs this test cannot return "
            f"p below {floor['min_attainable_p']:.4f}, so it can never reach "
            f"significance at 0.05. Observed p = {p:.4f}, mean difference "
            f"{observed:+.3f}, {beats}/{n} seeds beat the benchmark. Read the "
            "confidence interval instead."
        )
    if p >= 0.05:
        return (
            f"No significant difference from the benchmark (p = {p:.3f}). "
            f"{beats}/{n} beat it — not distinguishable from luck."
        )
    direction = "better than" if observed > 0 else "worse than"
    return (
        f"Significantly {direction} the benchmark (p = {p:.3f}, mean difference "
        f"{observed:+.3f}). {beats}/{n} beat it."
    )


def compare(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_perm: int = 20_000,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
    axis: str = "paired",
) -> dict:
    """Paired permutation test between two genuinely paired arms.

    Valid pairings in this project are *by training seed* (both ablation arms
    used the same seed set) and *by ticker* (agent vs. buy-&-hold on the same
    held-out asset). ``axis`` is recorded on the response so a reader always
    knows which question was asked.
    """
    a = np.asarray(list(values_a), dtype=np.float64)
    b = np.asarray(list(values_b), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"paired comparison needs equal-length arms (got {len(a)} and {len(b)})"
        )
    n_perm = _clamp(n_perm, 100, MAX_PERM, 20_000)
    n_boot = _clamp(n_boot, 100, MAX_BOOT, 10_000)

    observed, p = paired_permutation_test(a, b, n_perm=n_perm, seed=seed)
    diff_est = bootstrap_ci(a - b, confidence=confidence, n_boot=n_boot, seed=seed)
    floor = permutation_floor(len(a))
    return {
        "axis": axis,
        "n_pairs": len(a),
        "mean_a": round(float(a.mean()), 6),
        "mean_b": round(float(b.mean()), 6),
        "mean_difference": round(float(observed), 6),
        "difference_ci": [round(float(diff_est.low), 6), round(float(diff_est.high), 6)],
        "p_value": round(float(p), 6),
        "n_perm": n_perm,
        "n_boot": n_boot,
        "confidence": confidence,
        "significant_at_05": bool(p < 0.05),
        "resolution": floor,
        "a_wins": int(np.sum(a > b)),
        "pairs": [
            {"index": i, "a": round(float(x), 6), "b": round(float(y), 6)}
            for i, (x, y) in enumerate(zip(a, b))
        ],
    }


def bootstrap_distribution(
    values: Sequence[float], n_boot: int = 2_000, seed: int = 0, bins: int = 40
) -> Dict[str, List[float]]:
    """Histogram of the bootstrap distribution of the mean, for plotting.

    Returned as counts + bin edges so the payload stays small; the resampling
    itself is genuine.
    """
    data = np.asarray(list(values), dtype=np.float64)
    n_boot = _clamp(n_boot, 100, MAX_BOOT, 2_000)
    if len(data) < 2:
        return {"counts": [], "edges": [], "n_boot": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(data), size=(n_boot, len(data)))
    means = data[idx].mean(axis=1)
    counts, edges = np.histogram(means, bins=max(5, min(80, int(bins))))
    return {
        "counts": [int(c) for c in counts],
        "edges": [round(float(e), 6) for e in edges],
        "n_boot": n_boot,
    }
