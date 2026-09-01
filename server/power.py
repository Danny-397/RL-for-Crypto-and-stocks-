"""How many runs would you actually need?

Every panel in this lab reports the resolution floor ``2 / 2**n`` — the smallest
p-value a paired sign-flip test over ``n`` pairs can produce at all. That tells a
reader when a design *cannot* reach significance. It does not answer the question
that immediately follows, and that nothing else here answered:

    **How many runs would it take to detect an effect this size?**

This computes it, for the test the project actually uses.

Why not a textbook formula
--------------------------
The usual power formula assumes a t-test. This project reports a two-sided
**sign-flip permutation test**: under the null, each paired difference's sign is
equally likely to flip, and the p-value is the fraction of the ``2**n`` sign
assignments whose mean is at least as extreme as the observed one. That test is
discrete, and at the sample sizes here the discreteness is the whole story — its
attainable p-values are a coarse lattice, not a continuum. A t-test formula would
give a smooth, confident, and wrong answer.

So power is measured the same way the p-value is: simulate draws under the
alternative, run the real test on each, and count rejections. The permutation
distribution is enumerated exactly whenever ``2**n`` is small enough to enumerate
(which covers every sample size this project actually has), and sampled above
that — with the response saying which was used.

Effect size is expressed as a mean difference and a standard deviation, both in
the units the caller is already looking at (fractional returns), so a visitor can
put in the project's own numbers and read off what it would have taken.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# 2**n sign assignments are enumerated below this; above it they are sampled.
# 2**16 = 65,536 rows is still cheap, and every sample size in this project
# (5 seeds, 6 and 10 held-out tickers, 4 folds) sits far below it.
EXACT_LIMIT = 16
DEFAULT_SIMS = 2000
DEFAULT_PERMS = 4096
MAX_N = 200


def _sign_matrix(n: int) -> np.ndarray:
    """All ``2**n`` combinations of +1/-1, as a (2**n, n) array."""
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    return (1 - 2 * bits).astype(np.float32)


def _sampled_sign_matrix(n: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_perm, n))


def sign_flip_p(diffs, n_perm: int = DEFAULT_PERMS, seed: int = 0) -> float:
    """Two-sided sign-flip p-value for one set of paired differences.

    Exact when the sign space is small enough to enumerate. This mirrors the
    estimator in :mod:`rl_trader.evaluation.statistics`; it is reimplemented here
    only because power measurement needs to run it a few million times in a
    vectorised inner loop.
    """
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    if n == 0:
        return 1.0
    observed = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    signs = _sign_matrix(n) if n <= EXACT_LIMIT else _sampled_sign_matrix(n, n_perm, rng)
    means = np.abs(signs @ d) / n
    # ``>=`` with a tolerance: the observed assignment is itself one of the
    # permutations, so a strict comparison would report p = 0 for a perfect
    # separation, which is not attainable.
    return float((means >= observed - 1e-12).mean())


def power_at_n(
    n: int,
    effect: float,
    sd: float,
    alpha: float = 0.05,
    n_sims: int = DEFAULT_SIMS,
    n_perm: int = DEFAULT_PERMS,
    seed: int = 0,
) -> dict:
    """Probability the sign-flip test rejects, given a true effect of ``effect``.

    Paired differences are drawn as ``Normal(effect, sd)``. Every simulated study
    is put through the real test, so the reported power inherits the test's
    discreteness rather than smoothing it away.
    """
    n = int(n)
    if n < 2:
        return {"n": n, "power": 0.0, "floor": 1.0, "attainable": False,
                "exact": True, "n_sims": 0}

    floor = 2.0 / (2 ** n) if n <= 60 else 0.0
    rng = np.random.default_rng(seed)
    exact = n <= EXACT_LIMIT
    signs = _sign_matrix(n) if exact else _sampled_sign_matrix(n, n_perm, rng)

    # A design whose smallest attainable p-value already exceeds alpha has zero
    # power by construction. Saying so costs nothing and is more informative than
    # a simulation that returns 0.000 for a reason the reader has to guess.
    if floor > alpha:
        return {"n": n, "power": 0.0, "floor": round(floor, 12), "attainable": False,
                "exact": exact, "n_sims": 0,
                "note": f"{n} pairs cannot produce p <= {alpha}; the floor is {floor:.4f}."}

    if sd <= 0:
        # No variance: every draw is identical, so the test either always or
        # never rejects. Handled explicitly rather than dividing by zero.
        rejects = 1.0 if abs(effect) > 0 and floor <= alpha else 0.0
        return {"n": n, "power": rejects, "floor": round(floor, 12), "attainable": True,
                "exact": exact, "n_sims": 0,
                "note": "zero variance: the outcome is deterministic."}

    rejected = 0
    done = 0
    chunk = max(1, min(int(n_sims), int(4_000_000 / max(1, signs.shape[0]))))
    while done < n_sims:
        size = min(chunk, n_sims - done)
        draws = rng.normal(effect, sd, size=(size, n))
        observed = np.abs(draws.mean(axis=1))
        # (perms x n) @ (n x sims) -> every permuted mean for every simulated study
        perm_means = np.abs(signs @ draws.T) / n
        p = (perm_means >= observed[None, :] - 1e-12).mean(axis=0)
        rejected += int((p <= alpha).sum())
        done += size

    return {
        "n": n,
        "power": round(rejected / n_sims, 4),
        "floor": round(floor, 12),
        "attainable": True,
        "exact": exact,
        "n_sims": n_sims,
    }


def required_n(
    effect: float,
    sd: float,
    alpha: float = 0.05,
    target: float = 0.8,
    max_n: int = MAX_N,
    n_sims: int = DEFAULT_SIMS,
    seed: int = 0,
) -> dict:
    """Smallest ``n`` reaching ``target`` power, plus the curve it was found on.

    Searched upward rather than solved, because the underlying test is discrete:
    power does not increase smoothly with ``n``, and inverting a formula would
    hide exactly the lattice effects that make small studies unreliable.
    """
    curve: List[dict] = []
    found: Optional[int] = None
    n = 2
    while n <= max_n:
        row = power_at_n(n, effect, sd, alpha=alpha, n_sims=n_sims, seed=seed)
        curve.append(row)
        if found is None and row["power"] >= target:
            found = n
            break
        # Coarse then fine: stepping one at a time to n = 200 would be slow and
        # the curve is smooth enough above ~20 that it would add nothing.
        n += 1 if n < 24 else 4
    return {"required_n": found, "target": target, "alpha": alpha,
            "effect": effect, "sd": sd, "curve": curve, "max_n": max_n}


def analyse(
    effect: float,
    sd: float,
    have_n: Optional[int] = None,
    alpha: float = 0.05,
    target: float = 0.8,
    n_sims: int = DEFAULT_SIMS,
    seed: int = 0,
) -> dict:
    """Full answer: power at the sample size you have, and the size you'd need."""
    if sd < 0:
        raise ValueError("'sd' must be non-negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("'alpha' must be between 0 and 1")
    if not 0.0 < target < 1.0:
        raise ValueError("'target' must be between 0 and 1")

    out = required_n(effect, sd, alpha=alpha, target=target, n_sims=n_sims, seed=seed)
    current = None
    if have_n:
        current = power_at_n(int(have_n), effect, sd, alpha=alpha,
                             n_sims=n_sims, seed=seed)
    out.update({
        "current": current,
        "test": "two-sided paired sign-flip permutation test",
        "method": (
            "Paired differences are drawn as Normal(effect, sd) and each simulated "
            "study is put through the real test. The permutation distribution is "
            f"enumerated exactly for n <= {EXACT_LIMIT} and sampled above it, so "
            "the reported power inherits the test's discreteness instead of "
            "smoothing it away."
        ),
        "live_computation": True,
        "verdict": _verdict(out, current, target),
    })
    return out


def _verdict(out: dict, current: Optional[dict], target: float) -> str:
    """State the answer without implying more precision than simulation gives."""
    need = out["required_n"]
    parts = []
    if current is not None:
        if not current["attainable"]:
            parts.append(
                f"At {current['n']} pairs the test cannot reach p <= {out['alpha']} at "
                f"all — the floor is {current['floor']:.4f} — so its power is zero "
                "whatever the effect size."
            )
        else:
            parts.append(
                f"At {current['n']} pairs this design would detect an effect that "
                f"size about {current['power']:.0%} of the time."
            )
    if need is None:
        parts.append(
            f"No sample size up to {out['max_n']} reached {target:.0%} power, which "
            "means the effect is small relative to its spread — more runs are not "
            "the cheap fix here."
        )
    else:
        parts.append(
            f"Reaching {target:.0%} power would take about **{need}** paired runs."
        )
    return " ".join(parts)
