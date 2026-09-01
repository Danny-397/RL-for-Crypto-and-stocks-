"""Serve the surrogate-data falsification test.

Every other experiment on this site shows the agent failing to beat buy-and-hold
on real markets. This one asks the question that actually matters and that the
others cannot answer: **is the agent weak, or is there nothing there to find?**

The construction (``tools/surrogate_test.py``) is borrowed from nonlinear
time-series analysis — surrogate-data testing, Theiler et al. 1992 — and applied
to a trading agent. Randomly permute a series' daily log returns and re-integrate
them. Because a sum is permutation-invariant, the surrogate ends at *exactly* the
same price, so buy-and-hold is byte-for-byte identical; but every scrap of
temporal structure a timing agent could exploit — autocorrelation, momentum,
volatility clustering — is gone. Train the same recipe on both and compare the
edge over buy-and-hold.

Why the positive control is the important half
-----------------------------------------------
A null result is only meaningful from a test that *could* have found something.
So the synthetic arm is run first, on data with a known AR(1) momentum signal:
there the agent must show a large structured edge that collapses under shuffling.
That is what licenses reading the real arm's null as "no exploitable structure"
rather than "the test cannot see anything."

Why this is served, not computed
--------------------------------
Both arms are full PPO training runs across seeds. This container has no PyTorch,
so the results come from the repository's committed artifacts, labelled with the
command that regenerates them — the same policy as every other training-derived
number here.

One honest limitation is surfaced rather than smoothed over: the committed
artifacts predate per-arm array recording, so they carry summary statistics only.
Live re-analysis is offered when the arrays are present and plainly declined when
they are not, instead of re-deriving a p-value from a mean.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS = os.path.join(_REPO_ROOT, "docs", "assets")

ARMS = {
    "synthetic": {
        "file": "surrogate_synthetic.json",
        "label": "Positive control — synthetic data with a known signal",
        "structured_label": "Structured (AR(1) momentum)",
        "generated_by": "python tools/surrogate_test.py --mode synthetic --seeds 5 --timesteps 60000",
        "axis": "held-out synthetic paths",
        "expectation": (
            "The agent should show a clear edge on the structured paths and lose it "
            "on their surrogates. If it does not, the test has no power and the real "
            "arm below cannot be interpreted at all."
        ),
    },
    "real": {
        "file": "surrogate_real.json",
        "label": "The real question — actual market history",
        "structured_label": "Real market history",
        "generated_by": "python tools/surrogate_test.py --mode real --seeds 3 --timesteps 120000",
        "axis": "held-out tickers",
        "expectation": (
            "If the real edge is indistinguishable from the surrogate edge, then the "
            "agent finds no more exploitable structure in real prices than in noise "
            "with the same distribution — the underperformance is signal absence, "
            "not a broken agent."
        ),
    },
}

METHOD = (
    "A surrogate is the same series with its daily log returns randomly permuted "
    "and re-integrated. The return multiset is preserved, so the final price — and "
    "therefore buy-and-hold — is identical; only the ordering is destroyed. Edge is "
    "the agent's total return minus buy-and-hold's, and the two arms are compared "
    "with the project's paired permutation test."
)

CAVEATS = [
    "Both arms are full training runs, so these are committed results, not live "
    "computation. The regeneration command is published beside each one.",
    "A non-significant difference is not proof of no structure. It means this "
    "design, at this sample size, could not distinguish real prices from shuffled "
    "ones — which is a weaker and more honest claim.",
    "Edge is unbounded: a leveraged agent that compounds badly on one ticker can "
    "produce a very large negative number, which is why the intervals matter more "
    "than the point estimates.",
]


def _load(name: str) -> Optional[dict]:
    path = os.path.join(ASSETS, ARMS[name]["file"])
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):  # pragma: no cover - a corrupt asset must not 500
        return None


def _interpret(arm: str, market: str, row: dict) -> str:
    """State what this row does and does not establish, in plain language."""
    diff, p = row["diff"], row["p"]
    significant = p < 0.05
    if arm == "synthetic":
        if significant and diff > 0:
            return (
                f"The control passes: on {market}, shuffling the returns costs the "
                f"agent {diff:+.1%} of edge (p = {p:.4f}). The test can detect "
                "structure when structure is there."
            )
        return (
            f"The control did not pass on {market} (difference {diff:+.1%}, "
            f"p = {p:.4f}). Without it, the real arm below cannot be read as "
            "evidence of anything."
        )
    if significant:
        return (
            f"On {market}, real prices differ from their surrogates by {diff:+.1%} "
            f"(p = {p:.4f}) — evidence that the agent is finding something in the "
            "ordering, not just in the distribution."
        )
    return (
        f"On {market}, the difference between real prices and shuffled ones is "
        f"{diff:+.1%} with p = {p:.4f} — not distinguishable. The agent extracts no "
        "more from real market history than from noise with the same distribution."
    )



def _robust_p(values_a, values_b) -> Optional[dict]:
    """Paired sign-flip test on the MEDIAN difference rather than the mean.

    The null is unchanged -- under it, structured and surrogate are exchangeable
    within a pair, so every combination of sign flips is equally likely. Only
    the statistic changes, and the median is what makes the test insensitive to
    a single blown-up surrogate path.

    Enumerated exactly: the sample sizes here (6 to 12 pairs) are far below the
    point where sampling would be needed, so there is no simulation error to
    report.
    """
    if not values_a or not values_b or len(values_a) != len(values_b):
        return None
    diffs = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    n = len(diffs)
    if n < 2 or n > 20:
        return None

    observed = abs(float(np.median(diffs)))
    # every combination of +1/-1 over the n pairs
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    signs = (1 - 2 * bits).astype(float)
    medians = np.abs(np.median(signs * diffs[None, :], axis=1))
    p = float((medians >= observed - 1e-12).mean())
    return {
        "statistic": "median paired difference",
        "median_diff": round(float(np.median(diffs)), 6),
        "mean_diff": round(float(np.mean(diffs)), 6),
        "p": round(p, 6),
        "significant_at_05": bool(p < 0.05),
        "exact": True,
        "floor": round(2.0 / (2 ** n), 6),
    }


def _row(arm: str, market: str, raw: dict) -> dict:
    """One market's result, with everything needed to read it honestly."""
    values_a = raw.get("values_structured")
    values_b = raw.get("values_surrogate")
    n_pairs = raw.get("n_pairs")
    if n_pairs is None and isinstance(values_a, list):
        n_pairs = len(values_a)
    return {
        "market": market,
        "edge_structured": round(float(raw["edge_structured"]), 6),
        "edge_surrogate": round(float(raw["edge_surrogate"]), 6),
        "structured_ci": [round(float(v), 6) for v in raw["structured_ci"]],
        "surrogate_ci": [round(float(v), 6) for v in raw["surrogate_ci"]],
        "diff": round(float(raw["diff"]), 6),
        "p": round(float(raw["p"]), 6),
        "significant_at_05": bool(raw["p"] < 0.05),
        "n_pairs": n_pairs,
        # Present only in artifacts generated after per-arm recording was added.
        # Their absence is what makes live re-analysis impossible, so it is stated
        # rather than worked around.
        "values_structured": values_a,
        "values_surrogate": values_b,
        "reanalysable": bool(values_a and values_b),
        # Computed here, not read from the artifact: this is the live
        # re-analysis the per-pair values make possible. The published
        # mean-based p-value above is left exactly as generated.
        "robust": _robust_p(values_a, values_b),
        "interpretation": _interpret(arm, market, raw),
    }



def _reanalysis_note(reanalysable: bool, partial: List[str]) -> str:
    """Say exactly which arms can be recomputed, including when only some can."""
    if reanalysable:
        return ("These artifacts carry per-arm values, so the comparison can be "
                "recomputed live at your own settings.")
    stale = ("These artifacts were generated before per-arm values were recorded, "
             "so they carry summary statistics only. The p-values shown are the "
             "ones computed at generation time; they are not re-derivable here, "
             "and no attempt is made to reconstruct them from the means.")
    if partial:
        return (f"Mixed: the {', '.join(partial)} arm carries per-arm values and "
                f"could be recomputed, but the rest cannot. " + stale)
    return stale


def results() -> Optional[dict]:
    """Both arms, or ``None`` when the committed artifacts are unavailable."""
    arms: List[dict] = []
    for name, spec in ARMS.items():
        payload = _load(name)
        if payload is None:
            continue
        rows = [_row(name, market, raw) for market, raw in sorted(payload.items())]
        arms.append({
            "arm": name,
            "label": spec["label"],
            "structured_label": spec["structured_label"],
            "expectation": spec["expectation"],
            "axis": spec["axis"],
            "source": f"docs/assets/{spec['file']}",
            "generated_by": spec["generated_by"],
            "markets": rows,
        })
    if not arms:
        return None

    # Per arm first, then the payload. ``any`` was wrong: while one arm had been
    # regenerated with per-arm values and the other had not, it told the caller
    # the whole comparison was re-analysable and handed them a note saying so.
    # A mixed state has to be reported as mixed.
    by_arm = {a["arm"]: all(r["reanalysable"] for r in a["markets"]) for a in arms}
    reanalysable = bool(by_arm) and all(by_arm.values())
    partial = sorted(k for k, v in by_arm.items() if v) if not reanalysable else []
    return {
        "arms": arms,
        "method": METHOD,
        "caveats": CAVEATS,
        "reference": "Theiler et al. (1992), Testing for nonlinearity in time series",
        "live_computation": False,
        "reanalysable": reanalysable,
        "reanalysable_by_arm": by_arm,
        "reanalysis_note": _reanalysis_note(reanalysable, partial),
        "verdict": _verdict(arms),
    }


def _verdict(arms: List[dict]) -> Optional[str]:
    """The two arms read together — which is the only way they mean anything."""
    control = next((a for a in arms if a["arm"] == "synthetic"), None)
    real = next((a for a in arms if a["arm"] == "real"), None)
    if control is None or real is None:
        return None

    control_ok = all(r["significant_at_05"] and r["diff"] > 0 for r in control["markets"])
    real_null = all(not r["significant_at_05"] for r in real["markets"])

    if control_ok and real_null:
        return (
            "The test has power — on synthetic data with a planted signal it detects "
            "the loss of that signal in both markets. Applied to real markets it "
            "finds nothing: the agent does no better on real price history than on "
            "the same returns in a random order. On this evidence the flat "
            "performance is the market's, not the agent's."
        )
    if not control_ok:
        return (
            "The positive control did not clear significance in both markets, so the "
            "real arm cannot be interpreted. A null from a test with unproven power "
            "says nothing."
        )
    return (
        "The control passes and at least one real market shows a difference that "
        "reaches significance — the agent may be reading genuine ordering there. "
        "Worth repeating with more seeds before believing it."
    )
