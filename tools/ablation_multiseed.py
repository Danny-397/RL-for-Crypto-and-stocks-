"""Run the domain-randomization ablation across multiple seeds.

The repo's tools/ablation.py runs ONE seed (default 42) and writes
docs/assets/ablation.json. The paper's Table 1 quotes that single run.

That is inconsistent with the paper's own central claim -- that single-run RL
numbers are not evidence (RQ3, Henderson et al. 2018). It is also the likely
cause of the discrepancy between the numbers in the paper (+5821% / +18709%)
and the ones in docs/assets/ablation.json (+14581% / +1956390%): two different
runs of a quantity that is not stable across seeds.

This driver reuses ablation.run() unmodified, sweeps seeds, and reports
mean and bootstrap CI so Table 1 can be held to the same standard as Table 2.

Run from the repo root:
    python tools/ablation_multiseed.py --seeds 42 43 44 45 46 --timesteps 60000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from tools.ablation import run  # noqa: E402


def boot_ci(vals: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple:
    """Percentile bootstrap 95% CI of the mean."""
    rng = np.random.default_rng(seed)
    if len(vals) < 2:
        return float(vals.mean()), float(vals.mean()), float(vals.mean())
    draws = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--timesteps", type=int, default=60_000)
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--markets", nargs="+", default=["stock", "crypto"])
    ap.add_argument("--out", default="docs/assets/ablation_multiseed.json")
    ap.add_argument(
        "--from-raw", action="store_true",
        help="Skip training; summarize the existing *_raw.json checkpoint. Lets a "
             "partially-complete sweep be written up at whatever n it reached.",
    )
    args = ap.parse_args()

    per_seed: dict = {m: [] for m in args.markets}
    raw_path = args.out.replace(".json", "_raw.json")

    if args.from_raw:
        with open(raw_path, encoding="utf-8") as fh:
            per_seed = json.load(fh)
        per_seed = {m: v for m, v in per_seed.items() if v}
        # An interrupted sweep can leave one market a seed ahead of the other.
        # Truncate to the seeds every market finished, so the n we report is the
        # n every row actually has.
        common = set.intersection(*({r["seed"] for r in v} for v in per_seed.values()))
        per_seed = {m: [r for r in v if r["seed"] in common] for m, v in per_seed.items()}
        args.markets = list(per_seed)
        args.seeds = sorted(common)
        print(f"Summarizing {raw_path}: markets={args.markets} seeds={args.seeds}")

    for seed in ([] if args.from_raw else args.seeds):
        for market in args.markets:
            print(f"\n===== market={market} seed={seed} =====", flush=True)
            res = run(market, args.timesteps, seed, args.n_eval)
            res["seed"] = seed
            per_seed[market].append(res)
            print(f"  single  in={res['single']['in']:+.1%}  oos={res['single']['oos_mean']:+.1%}")
            print(f"  domain  in={res['domain']['in']:+.1%}  oos={res['domain']['oos_mean']:+.1%}", flush=True)

            # Checkpoint after every (market, seed). These runs take hours, and a
            # driver that only writes at the end throws away everything if it is
            # interrupted -- which is exactly what happened the first time.
            os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as fh:
                json.dump(per_seed, fh, indent=2)

    summary: dict = {"seeds": args.seeds, "timesteps": args.timesteps, "markets": {}}

    print("\n\n================ MULTI-SEED ABLATION ================")
    hdr = f"{'Market':<8}{'Training':<15}{'In-sample (mean)':>34}{'Out-of-sample (mean)':>34}"
    print(hdr)
    print("-" * len(hdr))

    for market in args.markets:
        summary["markets"][market] = {}
        for key, label in (("single", "single-path"), ("domain", "domain-random")):
            ins = np.array([r[key]["in"] for r in per_seed[market]])
            oos = np.array([r[key]["oos_mean"] for r in per_seed[market]])
            i_m, i_lo, i_hi = boot_ci(ins)
            o_m, o_lo, o_hi = boot_ci(oos)
            summary["markets"][market][key] = {
                "in_per_seed": ins.tolist(), "oos_per_seed": oos.tolist(),
                "in_mean": i_m, "in_ci": [i_lo, i_hi],
                "oos_mean": o_m, "oos_ci": [o_lo, o_hi],
                "in_min": float(ins.min()), "in_max": float(ins.max()),
            }
            istr = f"{i_m:+,.0%} [{i_lo:+,.0%}, {i_hi:+,.0%}]"
            ostr = f"{o_m:+.0%} [{o_lo:+.0%}, {o_hi:+.0%}]"
            print(f"{market:<8}{label:<15}{istr:>34}{ostr:>34}")
        print("-" * len(hdr))

    # The headline: how unstable is the single-path in-sample number across seeds?
    print("\nSingle-path in-sample spread across seeds (the memorization artifact):")
    for market in args.markets:
        s = summary["markets"][market]["single"]
        print(f"  {market:<8} min {s['in_min']:+,.0%}   max {s['in_max']:+,.0%}   "
              f"per-seed {[f'{v:+,.0%}' for v in s['in_per_seed']]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "per_seed": per_seed}, fh, indent=2)
    print(f"\nWrote {args.out}")

    write_site_shim(summary, per_seed)


def write_site_shim(summary: dict, per_seed: dict,
                    path: str = os.path.join("docs", "ablation.js")) -> None:
    """Publish the same summary as a JS global for the website.

    The site used to carry these four rows as hand-typed HTML. That is exactly
    the failure mode this project exists to complain about: a number in a
    document drifts away from the run that produced it, and nothing catches the
    drift. Re-running the sweep now rewrites the page's copy of the result, so
    the table cannot quietly disagree with the artifact that backs it.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("/* Auto-generated by tools/ablation_multiseed.py"
                 " -- domain-randomization ablation. */\n")
        fh.write("window.RL_ABLATION = ")
        json.dump({"summary": summary, "per_seed": per_seed}, fh, indent=1)
        fh.write(";\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
