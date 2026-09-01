"""Hyper-parameter sensitivity: is the negative result seed luck, or recipe luck?

``tools/real_significance.py`` asks whether the out-of-sample edge survives a
change of **random seed**. This asks the sibling question nothing else in the
repo asks: does it survive a change of **hyper-parameters**? A reader is entitled
to suspect that the flat performance is an artifact of one unlucky learning rate.

Design
------
One factor at a time, around the published defaults. Each knob is moved to a
value either side of its default while everything else is held fixed, so any
change in outcome is attributable to that knob alone. The baseline configuration
is trained too, under the same seeds, as the comparison point.

This is deliberately *not* a grid search. A grid over four knobs would multiply
the compute by an order of magnitude and — far worse — would turn the study into
a search for the best configuration, which is exactly the p-hacking this project
exists to argue against. The question here is "is the conclusion fragile?", not
"what is the best recipe?", and only the first question is asked.

Evaluation mirrors ``real_significance.py`` exactly: agents are trained on the
real basket, domain-randomized across tickers, then scored on every ticker's
held-out test split against buy-and-hold. Checkpoints go to a throwaway
directory so a sweep can never overwrite a deployed model.

What it can and cannot conclude
-------------------------------
With a handful of seeds per configuration this cannot establish significance for
any single one — a sign test over ``n`` seeds cannot go below ``2 / 2**n``, which
is 0.25 at three seeds. What it *can* say is descriptive and still worth a lot:
whether **any** configuration produced a positive edge at all. A sweep in which
none does is much harder to dismiss as an unlucky recipe.

Run from the repo root (after tools/fetch_data.py):

    python tools/hyperparameter_sweep.py --seeds 3 --timesteps 60000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - older interpreters / redirected streams
    pass

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.evaluation.evaluate_agent import ANNUALISATION
from rl_trader.evaluation.statistics import bootstrap_ci
from rl_trader.training.utils import get_logger, run_ppo_training

# Reuse the real-basket loading and per-ticker evaluation rather than restating
# them: any drift between this study and the significance study would make the
# two incomparable, which would defeat the purpose of running it.
from tools.real_significance import _evaluate_basket, load_real_basket

CFG = {"stock": stock_config, "crypto": crypto_config}

# One factor at a time: each knob moved either side of its published default,
# everything else held at the default. Values span the range these settings
# plausibly take in published PPO work rather than being tuned for an outcome.
KNOBS = {
    "learning_rate": [1e-4, 1e-3],
    "clip_ratio": [0.1, 0.3],
    "gae_lambda": [0.90, 0.99],
    "entropy_coef": [0.0, 0.05],
}


def variants() -> list:
    """The baseline plus one variant per (knob, alternative value)."""
    out = [{"key": "baseline", "knob": None, "value": None}]
    for knob, values in KNOBS.items():
        for value in values:
            out.append({"key": f"{knob}={value:g}", "knob": knob, "value": value})
    return out


def run_variant(market: str, spec: dict, basket, seeds: int, timesteps: int, log) -> dict:
    """Train ``seeds`` agents under one configuration and score the whole basket."""
    tickers = list(basket)
    train_slices = [basket[t]["train"] for t in tickers]
    periods = ANNUALISATION.get(market, 252)

    seed_edges, seed_returns = [], []
    bh_mean = None
    for s in range(seeds):
        cfg = CFG[market]()
        cfg.market = market
        cfg.train.total_timesteps = timesteps
        cfg.train.eval_interval = 0
        cfg.train.seed = 200 + s
        # Throwaway: a sweep must never be able to overwrite a deployed model.
        cfg.train.checkpoint_dir = os.path.join("checkpoints", "_hpsweep")
        if spec["knob"] is not None:
            setattr(cfg.ppo, spec["knob"], spec["value"])
        random.seed(200 + s)

        def factory():
            return random.choice(train_slices)

        agent, _ = run_ppo_training(cfg, train_series_factory=factory)
        agent_rets, bh_rets = _evaluate_basket(agent, basket, market, cfg, periods)
        bh_mean = float(bh_rets.mean())
        seed_returns.append(float(agent_rets.mean()))
        seed_edges.append(float((agent_rets - bh_rets).mean()))
        log.info(
            "[%s] %-22s seed %d/%d | agent %+.2f%% | edge vs B&H %+.2f%%",
            market, spec["key"], s + 1, seeds,
            100 * seed_returns[-1], 100 * seed_edges[-1],
        )

    edges = np.array(seed_edges, dtype=float)
    ci = bootstrap_ci(seed_edges)
    return {
        "config": spec["key"],
        "knob": spec["knob"],
        "value": spec["value"],
        "n_seeds": seeds,
        "mean_return": round(float(np.mean(seed_returns)), 6),
        "benchmark_return": round(bh_mean, 6) if bh_mean is not None else None,
        "mean_edge": round(float(edges.mean()), 6),
        "edge_ci": [round(ci.mean, 6), round(ci.low, 6), round(ci.high, 6)],
        "worst_seed_edge": round(float(edges.min()), 6),
        "best_seed_edge": round(float(edges.max()), 6),
        "seed_edges": [round(float(v), 6) for v in edges],
        "positive_edge": bool(edges.mean() > 0),
        # With a handful of seeds an interval that excludes zero is the strongest
        # statement available, and it is still weak. Recorded, never headlined.
        "ci_excludes_zero": bool(ci.low > 0 or ci.high < 0),
    }


def summarise(market: str, rows: list, seeds: int) -> dict:
    """What the sweep as a whole does and does not establish."""
    positive = [r for r in rows if r["mean_edge"] > 0]
    best = max(rows, key=lambda r: r["mean_edge"])
    worst = min(rows, key=lambda r: r["mean_edge"])
    baseline = next((r for r in rows if r["config"] == "baseline"), None)
    floor = 2.0 / (2 ** seeds) if 0 < seeds <= 30 else 0.0

    if not positive:
        verdict = (
            f"No configuration produced a positive edge on {market}. Across "
            f"{len(rows)} recipes spanning four hyper-parameters, the best still "
            f"trailed buy-and-hold by {abs(best['mean_edge']):.1%}. The flat result "
            "is not an artifact of one unlucky setting."
        )
    else:
        names = ", ".join(r["config"] for r in positive)
        verdict = (
            f"{len(positive)} of {len(rows)} configurations produced a positive mean "
            f"edge on {market} ({names}). At {seeds} seeds each this is descriptive, "
            "not significant — the next step would be re-running those settings with "
            "many more seeds before believing them."
        )

    return {
        "market": market,
        "n_configs": len(rows),
        "n_seeds": seeds,
        "n_positive": len(positive),
        "baseline_edge": baseline["mean_edge"] if baseline else None,
        "best_config": best["config"],
        "best_edge": best["mean_edge"],
        "worst_config": worst["config"],
        "worst_edge": worst["mean_edge"],
        "spread": round(best["mean_edge"] - worst["mean_edge"], 6),
        "sign_test_floor": round(floor, 12),
        "verdict": verdict,
        "power_note": (
            f"Each configuration is {seeds} seeds, so a sign test over them cannot "
            f"produce a p-value below {floor:.4f}. Nothing here is a significance "
            "claim; the sweep answers the descriptive question of whether the "
            "conclusion moves when the recipe does."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", type=int, default=3, help="Seeds per configuration.")
    ap.add_argument("--timesteps", type=int, default=60_000)
    ap.add_argument("--markets", nargs="+", default=["stock", "crypto"])
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default=os.path.join("docs", "assets", "hyperparameter_sweep.json"))
    args = ap.parse_args()

    log = get_logger("hpsweep")
    specs = variants()
    total = len(specs) * args.seeds * len(args.markets)
    log.info("sweep: %d configs x %d seeds x %d markets = %d runs of %d steps",
             len(specs), args.seeds, len(args.markets), total, args.timesteps)

    started = time.time()
    payload = {
        "generated": time.strftime("%Y-%m-%d"),
        "timesteps": args.timesteps,
        "seeds_per_config": args.seeds,
        "design": "one factor at a time around the published defaults",
        "knobs": {k: v for k, v in KNOBS.items()},
        "markets": {},
    }

    for market in args.markets:
        basket = load_real_basket(args.data_dir, market)
        if not basket:
            raise SystemExit(f"No CSVs in {os.path.join(args.data_dir, market)}.")
        rows = [run_variant(market, spec, basket, args.seeds, args.timesteps, log)
                for spec in specs]
        payload["markets"][market] = {
            "rows": rows,
            "summary": summarise(market, rows, args.seeds),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")

    print(f"\nWrote {args.out}  ({time.time() - started:.0f}s)")
    for market, block in payload["markets"].items():
        s = block["summary"]
        print(f"\n{market}: {s['n_positive']}/{s['n_configs']} configurations beat buy-&-hold")
        print(f"  {s['verdict']}")
        for row in sorted(block["rows"], key=lambda r: -r["mean_edge"]):
            print(f"    {row['config']:<24}{row['mean_edge']:>+9.1%}"
                  f"   [{row['edge_ci'][1]:+.1%}, {row['edge_ci'][2]:+.1%}]")


if __name__ == "__main__":
    main()
