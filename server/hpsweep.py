"""Serve the hyper-parameter sensitivity sweep.

The seed study asks whether the result survives a different random seed. This
asks the question a reader raises next and that nothing else here answered: does
it survive a different **recipe**? Flat performance from one learning rate is a
much weaker claim than flat performance from nine configurations spanning four
hyper-parameters.

Served, not computed: every cell is a training run. The artifact is produced by
``tools/hyperparameter_sweep.py`` and labelled with the command that regenerates
it, exactly like the ablation and the surrogate test.

The one thing this module refuses to do
----------------------------------------
It never reports a "best" configuration as a finding. The sweep is a fragility
check, and ranking nine recipes by outcome and then quoting the winner is the
p-hacking the rest of this project argues against — with a handful of seeds each,
the top of that ranking is mostly noise. Rows are returned in their published
order, the summary leads with how many configurations produced a positive edge
(usually none), and the resolution floor travels with it.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSET = os.path.join(_REPO_ROOT, "docs", "assets", "hyperparameter_sweep.json")

GENERATED_BY = "python tools/hyperparameter_sweep.py --seeds 3 --timesteps 60000"

DESIGN_NOTE = (
    "One factor at a time: each hyper-parameter is moved either side of its "
    "published default while everything else is held fixed, so a change in "
    "outcome is attributable to that knob alone. Deliberately not a grid search "
    "— a grid would turn a fragility check into a search for the best recipe, "
    "which is the opposite of the question being asked."
)

CAVEATS = [
    "Every configuration is a full training run, so these are committed results "
    "rather than live computation.",
    "A handful of seeds per configuration cannot establish significance for any "
    "single one. The useful reading is whether ANY recipe produced a positive "
    "edge, not which recipe came top.",
    "The evaluation is identical to the seed study: agents are domain-randomized "
    "across the basket and scored on every ticker's held-out split, so the two "
    "results are directly comparable.",
]


def _load() -> Optional[dict]:
    if not os.path.exists(ASSET):
        return None
    try:
        with open(ASSET, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):  # pragma: no cover - a corrupt asset must not 500
        return None


def _row(raw: dict) -> dict:
    """One configuration, with its seed spread kept visible."""
    return {
        "config": raw["config"],
        "knob": raw.get("knob"),
        "value": raw.get("value"),
        "is_baseline": raw.get("knob") is None,
        "n_seeds": raw.get("n_seeds"),
        "mean_return": raw.get("mean_return"),
        "benchmark_return": raw.get("benchmark_return"),
        "mean_edge": raw["mean_edge"],
        "edge_ci": raw.get("edge_ci"),
        "worst_seed_edge": raw.get("worst_seed_edge"),
        "best_seed_edge": raw.get("best_seed_edge"),
        "seed_edges": raw.get("seed_edges"),
        "positive_edge": bool(raw["mean_edge"] > 0),
    }


def results() -> Optional[dict]:
    """The sweep as served, or ``None`` when the artifact is unavailable."""
    payload = _load()
    if payload is None:
        return None

    markets = []
    for market, block in sorted(payload.get("markets", {}).items()):
        # Published order, never sorted by outcome — see the module docstring.
        rows = [_row(r) for r in block.get("rows", [])]
        markets.append({
            "market": market,
            "rows": rows,
            "summary": block.get("summary", {}),
        })
    if not markets:
        return None

    return {
        "generated": payload.get("generated"),
        "timesteps": payload.get("timesteps"),
        "seeds_per_config": payload.get("seeds_per_config"),
        "knobs": payload.get("knobs", {}),
        "markets": markets,
        "design": DESIGN_NOTE,
        "caveats": CAVEATS,
        "source": "docs/assets/hyperparameter_sweep.json",
        "generated_by": GENERATED_BY,
        "live_computation": False,
        "headline": _headline(markets),
    }


def _headline(markets: List[dict]) -> str:
    """Read every market together, leading with the least flattering true claim."""
    totals = [(m["market"], m["summary"].get("n_positive", 0),
               m["summary"].get("n_configs", 0)) for m in markets]
    positive = sum(t[1] for t in totals)
    configs = sum(t[2] for t in totals)
    if configs == 0:
        return "The sweep artifact carries no configurations."
    if positive == 0:
        return (
            f"None of the {configs} configurations tested produced a positive edge "
            "over buy-and-hold. The flat result is a property of the problem, not "
            "of one unlucky hyper-parameter choice."
        )
    parts = ", ".join(f"{n}/{c} on {m}" for m, n, c in totals if c)
    return (
        f"{positive} of {configs} configurations produced a positive mean edge "
        f"({parts}). At this many seeds each that is descriptive rather than "
        "significant, and the honest next step is many more seeds on those "
        "settings — not quoting them as a result."
    )
