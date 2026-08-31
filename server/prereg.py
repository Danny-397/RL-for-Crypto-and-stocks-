"""Pre-registration: state the prediction before the result exists.

Every panel in this lab lets a visitor run an experiment and then read a number.
That order is exactly how researchers fool themselves — a result you have already
seen is very easy to have expected. Pre-registration is the standard defence:
commit to the prediction, and to how it will be judged, *before* the data comes
back.

How the commitment is actually enforced
---------------------------------------
The prediction is parsed and stamped at experiment-creation time, stored on the
:class:`~server.experiments.Experiment` beside its config, and scored by the
backend from the result. There is no endpoint that edits it. Changing your mind
after seeing the outcome means creating a new experiment with a new id, which the
notebook lists separately — so the record of what you actually predicted survives.

The judging rule is fixed here, not per experiment
--------------------------------------------------
"Beat the benchmark" needs a threshold, or every result can be read as a
near-miss after the fact. :data:`MATCH_BAND` is that threshold, declared once,
published in the payload, and applied identically to every prediction: an
outcome inside +/-2 percentage points of the benchmark counts as "about the
same", and outside it as a win or a loss. A visitor can disagree with the band,
but they cannot move it after the fact.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

DIRECTIONS = {
    "beats": "The agent will beat buy-and-hold",
    "matches": "The agent will land about level with buy-and-hold",
    "loses": "The agent will lose to buy-and-hold",
}

# An outcome within this band of the benchmark counts as "about the same".
MATCH_BAND = 0.02

RULE = (
    "Judged on the agent's total return minus buy-and-hold's over the same bars. "
    f"Inside +/-{MATCH_BAND:.0%} counts as 'about the same'; outside it counts as a "
    "win or a loss. The rule and the band are fixed before the run and are the "
    "same for every experiment."
)

# Kinds whose result carries a benchmark comparison this rule can be applied to.
SCORABLE_KINDS = ("rollout", "walk_forward", "distribution_shift")


def parse(payload: Dict[str, Any]) -> Optional[dict]:
    """Validate a pre-registration from untrusted JSON, or return ``None``.

    Returns ``None`` when no prediction was made. That absence is recorded
    honestly rather than filled in — an invented hypothesis is the notebook's
    version of a fabricated result.
    """
    raw = payload.get("prediction")
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"direction": raw}
    if not isinstance(raw, dict):
        raise ValueError("'prediction' must be an object or a direction string")

    direction = str(raw.get("direction", "")).lower().strip()
    if direction not in DIRECTIONS:
        raise ValueError(
            f"unknown prediction direction {direction!r}; expected one of "
            f"{sorted(DIRECTIONS)}"
        )
    note = raw.get("note")
    note = str(note).strip()[:400] if note else None
    return {
        "direction": direction,
        "statement": DIRECTIONS[direction],
        "note": note or None,
        "rule": RULE,
        "match_band": MATCH_BAND,
        # Stamped here, before the runner starts — the whole point.
        "registered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def observed_excess(kind: str, result: Optional[dict]) -> Optional[float]:
    """Pull the pre-declared quantity out of a finished result.

    Returns ``None`` when the experiment kind has no benchmark comparison, so the
    caller reports "not scorable" rather than scoring against something else.
    """
    if not result:
        return None
    if kind == "rollout":
        metrics, bench = result.get("metrics"), result.get("bench_metrics")
        if not metrics or not bench:
            return None
        return float(metrics["total_return"]) - float(bench["total_return"])
    if kind == "walk_forward":
        summary = result.get("summary") or {}
        value = summary.get("mean_excess_return")
        return float(value) if value is not None else None
    if kind == "distribution_shift":
        rows = result.get("regimes") or []
        values = [r.get("mean_excess_return") for r in rows]
        values = [float(v) for v in values if v is not None]
        return sum(values) / len(values) if values else None
    return None


def classify(excess: float) -> str:
    """Map a measured excess return onto the same three labels as a prediction."""
    if excess > MATCH_BAND:
        return "beats"
    if excess < -MATCH_BAND:
        return "loses"
    return "matches"


def evaluate(prediction: Optional[dict], kind: str, result: Optional[dict]) -> Optional[dict]:
    """Score a pre-registration against the result, or explain why it cannot be."""
    if prediction is None:
        return None
    excess = observed_excess(kind, result)
    if excess is None:
        return {
            "predicted": prediction["direction"],
            "observed": None,
            "matched": None,
            "scorable": False,
            "reason": (
                f"A {kind!r} experiment has no single agent-vs-benchmark number, "
                "so this prediction is recorded but not scored."
            ),
            "rule": RULE,
        }
    observed = classify(excess)
    matched = observed == prediction["direction"]
    return {
        "predicted": prediction["direction"],
        "observed": observed,
        "observed_excess": round(excess, 6),
        "matched": matched,
        "scorable": True,
        "rule": RULE,
        "match_band": MATCH_BAND,
        "verdict": _verdict(prediction["direction"], observed, excess, matched),
    }


def _verdict(predicted: str, observed: str, excess: float, matched: bool) -> str:
    """Say what happened without congratulating or scolding.

    One experiment confirming a prediction is one experiment. The wording keeps
    that in view, because "I called it" is the exact feeling this whole panel
    exists to make harder to trust.
    """
    outcome = f"{excess:+.1%} against buy-and-hold"
    if matched:
        return (
            f"You predicted the agent would {predicted.rstrip('s')}; it did "
            f"({outcome}). One run agreeing with one prediction is weak evidence — "
            "the seed and fold panels show how far a single run can move."
        )
    return (
        f"You predicted {predicted!r}; the run came out {observed!r} ({outcome}). "
        "Recorded as it stands. A prediction that misses is the useful kind: it is "
        "the only sort that could have been wrong."
    )


def describe() -> dict:
    """The options and the judging rule, for the frontend to render verbatim."""
    return {
        "directions": [{"key": k, "statement": v} for k, v in DIRECTIONS.items()],
        "rule": RULE,
        "match_band": MATCH_BAND,
        "scorable_kinds": list(SCORABLE_KINDS),
    }
