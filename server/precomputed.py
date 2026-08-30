"""Catalog of the project's real, committed experiment results.

Some questions this lab asks cannot be answered live, and pretending otherwise
would be the one unforgivable thing a research site can do.

**Why multi-seed cannot be a live button.** The deployed policy is evaluated
deterministically (policy mean, no sampling), so re-running a rollout under a
different *evaluation* seed returns a bit-identical result. The seed variation
that matters in this project is the **training** seed — and each of those points
is a full PPO training run (60k-200k steps). A "run 5 seeds" button that
finished in two seconds would be a lie.

So the seed-level numbers here are the **real** ones produced by the repo's own
experiment scripts and committed to the repository:

* ``docs/significance.js``            <- ``tools/real_significance.py``
* ``docs/assets/ablation_multiseed.json`` <- ``tools/ablation_multiseed.py``

What *is* live is the statistics. The bootstrap and permutation machinery in
:mod:`rl_trader.evaluation.statistics` is pure NumPy and runs in milliseconds, so
a visitor can genuinely re-run the inference — change the confidence level, the
resample count, the permutation count — and watch real p-values recompute on
real data. That is the honest and more instructive half of the experiment.

Every series returned from here carries a ``source`` and ``generated_by`` field
so the UI can always say where the number came from and how to regenerate it.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DOCS = os.path.join(_REPO_ROOT, "docs")


def _load_js_global(path: str) -> Optional[dict]:
    """Parse a ``window.X = {...};`` file into a dict.

    The site ships data as JS globals so it renders with no web server at all;
    the API reads the same files so the two can never disagree.
    """
    try:
        src = open(path, encoding="utf-8").read()
        body = src[src.index("{"): src.rstrip().rstrip(";").rindex("}") + 1]
        return json.loads(body)
    except Exception:
        return None


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Real seed-level datasets                                                     #
# --------------------------------------------------------------------------- #
def seed_datasets() -> Dict[str, dict]:
    """Every real per-training-seed series the repository ships.

    Each entry is one *arm* of a real multi-seed experiment: a list of outcomes,
    one per independent training seed, plus the benchmark it should be judged
    against where the source experiment defines one.
    """
    out: Dict[str, dict] = {}

    sig = _load_js_global(os.path.join(_DOCS, "significance.js"))
    if sig:
        for market, blob in sig.items():
            returns = blob.get("seed_returns") or []
            if not returns:
                continue
            key = f"real:{market}"
            out[key] = {
                "key": key,
                "label": f"Real market — {market} agent, held-out return per training seed",
                "market": market,
                "experiment": "real_significance",
                "values": [float(v) for v in returns],
                "seeds": blob.get("seeds", len(returns)),
                "benchmark": blob.get("bh"),
                "benchmark_label": "buy & hold",
                "units": "total return (fraction)",
                "source": "docs/significance.js",
                "generated_by": "python tools/real_significance.py",
                "published": {
                    "mean": blob.get("mean"),
                    "ci_low": blob.get("ci_low"),
                    "ci_high": blob.get("ci_high"),
                    "p": blob.get("p"),
                },
            }

    abl = _load_json(os.path.join(_DOCS, "assets", "ablation_multiseed.json"))
    if abl:
        summary = abl.get("summary", abl)
        timesteps = summary.get("timesteps")
        for market, arms in (summary.get("markets") or {}).items():
            for arm, blob in arms.items():
                arm_label = "single-path" if arm == "single" else "domain-randomized"
                for split in ("in", "oos"):
                    values = blob.get(f"{split}_per_seed") or []
                    if not values:
                        continue
                    split_label = "in-sample" if split == "in" else "held-out"
                    key = f"ablation:{market}:{arm}:{split}"
                    out[key] = {
                        "key": key,
                        "label": (
                            f"Ablation — {market}, {arm_label} agent, "
                            f"{split_label} return per training seed"
                        ),
                        "market": market,
                        "experiment": "ablation_multiseed",
                        "arm": arm,
                        "split": split,
                        "values": [float(v) for v in values],
                        "seeds": len(values),
                        "benchmark": None,
                        "units": "total return (fraction)",
                        "timesteps": timesteps,
                        "source": "docs/assets/ablation_multiseed.json",
                        "generated_by": (
                            "python tools/ablation_multiseed.py "
                            "--seeds 42 43 44 45 46 --timesteps 60000"
                        ),
                        "published": {
                            "mean": blob.get(f"{split}_mean"),
                            "ci": blob.get(f"{split}_ci"),
                        },
                    }
    return out


def paired_asset_datasets() -> Dict[str, dict]:
    """Per-ticker agent vs. buy-&-hold returns — the paper's significance axis.

    ``tools/real_significance.py`` pairs *across held-out tickers*, not across
    seeds, and that is where the published p-value comes from. The committed
    ``docs/results.js`` carries exactly this pairing for the published run, so a
    visitor can re-run the correctly-paired test live.

    One caveat is stated on every response: these per-ticker returns come from
    the single published training seed, whereas the paper's p-value pairs the
    *seed-averaged* per-ticker returns over 5 seeds. The axis is the same; the
    seed averaging is not, so the two p-values need not match exactly.
    """
    blob = _load_js_global(os.path.join(_DOCS, "results.js"))
    if not blob:
        return {}
    out: Dict[str, dict] = {}
    for market, mblob in (blob.get("markets") or {}).items():
        rows = mblob.get("per_ticker") or []
        pairs = [
            (
                r.get("ticker"),
                (r.get("metrics") or {}).get("total_return"),
                (r.get("bench_metrics") or {}).get("total_return"),
            )
            for r in rows
        ]
        pairs = [(t, a, b) for t, a, b in pairs if a is not None and b is not None]
        if not pairs:
            continue
        key = f"assets:{market}"
        out[key] = {
            "key": key,
            "label": f"{market} — agent vs buy-&-hold, per held-out ticker",
            "market": market,
            "experiment": "per_ticker_significance",
            "labels": [t for t, _a, _b in pairs],
            "agent": [float(a) for _t, a, _b in pairs],
            "benchmark": [float(b) for _t, _a, b in pairs],
            "n_pairs": len(pairs),
            "units": "total return (fraction)",
            "axis": "held_out_ticker",
            "source": "docs/results.js",
            "generated_by": (
                f"python tools/build_site_data.py --real "
                f"--timesteps {blob.get('timesteps')} --seed {blob.get('seed')}"
            ),
            "caveat": (
                "Single published training seed. The paper's p-value pairs the "
                "seed-averaged per-ticker returns across 5 seeds "
                "(tools/real_significance.py), so values need not match exactly."
            ),
        }
    return out


def dataset_catalog() -> List[dict]:
    """Describe available datasets without shipping every value twice."""
    return [
        {
            "key": d["key"],
            "label": d["label"],
            "market": d.get("market"),
            "experiment": d["experiment"],
            "n_seeds": d["seeds"],
            "has_benchmark": d.get("benchmark") is not None,
            "source": d["source"],
            "generated_by": d["generated_by"],
        }
        for d in seed_datasets().values()
    ]


# --------------------------------------------------------------------------- #
# Generalization (single-path vs domain-randomized)                            #
# --------------------------------------------------------------------------- #
def generalization_results() -> Optional[dict]:
    """The real domain-randomization ablation, shaped for the A/B panel.

    Reports, per market and per arm, the in-sample and held-out means with their
    bootstrap CIs and the per-seed values behind them, plus the *generalization
    gap* — the whole point of the experiment.
    """
    abl = _load_json(os.path.join(_DOCS, "assets", "ablation_multiseed.json"))
    if not abl:
        return None
    summary = abl.get("summary", abl)
    markets = summary.get("markets") or {}

    out: Dict[str, dict] = {}
    for market, arms in markets.items():
        arm_out = {}
        for arm, blob in arms.items():
            in_mean = blob.get("in_mean")
            oos_mean = blob.get("oos_mean")
            arm_out[arm] = {
                "label": "Agent A — single price path" if arm == "single"
                else "Agent B — domain randomized",
                "in_sample": {
                    "mean": in_mean,
                    "ci": blob.get("in_ci"),
                    "per_seed": blob.get("in_per_seed"),
                },
                "held_out": {
                    "mean": oos_mean,
                    "ci": blob.get("oos_ci"),
                    "per_seed": blob.get("oos_per_seed"),
                },
                # The gap is the headline: how much of the in-sample result survives.
                "generalization_gap": (
                    round(float(in_mean) - float(oos_mean), 6)
                    if in_mean is not None and oos_mean is not None
                    else None
                ),
            }
        out[market] = arm_out

    return {
        "markets": out,
        "seeds": summary.get("seeds"),
        "timesteps": summary.get("timesteps"),
        "units": "total return (fraction); 1.0 = +100%",
        "source": "docs/assets/ablation_multiseed.json",
        "generated_by": (
            "python tools/ablation_multiseed.py --seeds 42 43 44 45 46 --timesteps 60000"
        ),
        "live": False,
        "why_not_live": (
            "Each point is a full PPO training run (60k steps). These are the real "
            "committed results; the statistics computed over them run live."
        ),
    }


# --------------------------------------------------------------------------- #
# Provenance of the baked dashboard results                                    #
# --------------------------------------------------------------------------- #
def results_provenance() -> dict:
    """When the baked dashboard results were generated, and by what."""
    path = os.path.join(_DOCS, "results.js")
    blob = _load_js_global(path)
    if not blob:
        return {"available": False}
    return {
        "available": True,
        "generated": blob.get("generated"),
        "data_source": re.sub(r"\s+", " ", str(blob.get("data_source", ""))).strip(),
        "timesteps": blob.get("timesteps"),
        "seed": blob.get("seed"),
        "source": "docs/results.js",
        "generated_by": (
            f"python tools/build_site_data.py --real "
            f"--timesteps {blob.get('timesteps')} --seed {blob.get('seed')}"
        ),
    }
