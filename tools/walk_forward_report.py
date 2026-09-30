"""Multi-fold walk-forward evaluation on real data.

§5 of RESULTS.md answers "does the edge survive a different *seed*?" — it repeats
one 60/40 chronological split across independent seeds. It does not answer "does
it survive a different *slice of history*?", and the limitations section names
that gap:

    Real-data walk-forward could be multi-fold (the evaluation/walk_forward.py
    splitter is built for this) ... rolling re-training folds would add a second
    axis of robustness.

This tool closes it. For each seed it retrains from scratch on every rolling fold
and scores the agent against buy-&-hold on the *following*, unseen block. That
gives two axes at once: seed variance within a period, and period variance across
history.

The splitter (``generate_folds``) was already unit-tested; ``rolling_walk_forward``
had no caller until this script.

Run from the repo root, after ``tools/fetch_data.py``:

    python tools/walk_forward_report.py --market crypto --tickers BTC-USD \
        --folds 4 --seeds 5 --timesteps 60000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_trader.config.training_config import crypto_config, stock_config  # noqa: E402
from rl_trader.data.data_loader import attach_market_index, load_ohlcv_csv  # noqa: E402
from rl_trader.evaluation.walk_forward import generate_folds, rolling_walk_forward  # noqa: E402

CFG_FNS = {"crypto": crypto_config, "stock": stock_config}


def bootstrap_ci(xs, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for the mean. Matches the protocol used in §5."""
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(xs, size=(n_boot, len(xs)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def fold_date_ranges(df, n_folds: int, scheme: str):
    """Reproduce the splitter's fold boundaries as calendar dates, for reporting."""
    from rl_trader.data.data_loader import add_technical_indicators
    featured = add_technical_indicators(df)
    dates = featured.index if hasattr(featured.index, "date") else featured.iloc[:, 0]
    out = []
    for f in generate_folds(len(featured), n_folds, scheme=scheme):
        try:
            lo, hi = dates[f.test.start], dates[f.test.stop - 1]
            out.append((str(lo)[:10], str(hi)[:10]))
        except Exception:  # noqa: BLE001 - reporting only; never fail the run
            out.append(("?", "?"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", choices=["crypto", "stock"], default="crypto")
    ap.add_argument("--tickers", nargs="+", default=["BTC-USD"])
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=60_000)
    ap.add_argument("--scheme", choices=["expanding", "sliding"], default="expanding")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="docs/assets/walk_forward_multifold.json")
    args = ap.parse_args()

    print("MULTI-FOLD WALK-FORWARD  (%s, %d folds x %d seeds, %s steps, %s)"
          % (args.market, args.folds, args.seeds, f"{args.timesteps:,}", args.scheme), flush=True)

    payload = {
        "market": args.market, "folds": args.folds, "seeds": args.seeds,
        "timesteps": args.timesteps, "scheme": args.scheme,
        "tickers": args.tickers, "results": {},
    }
    t_start = time.time()

    for ticker in args.tickers:
        path = os.path.join(args.data_dir, args.market, ticker + ".csv")
        if not os.path.exists(path):
            print("  skip %s (no CSV at %s)" % (ticker, path), flush=True)
            continue
        df = attach_market_index(load_ohlcv_csv(path), args.data_dir, args.market)
        ranges = fold_date_ranges(df, args.folds, args.scheme)
        print("\n== %s  (%d rows)" % (ticker, len(df)), flush=True)
        for i, (lo, hi) in enumerate(ranges):
            print("   fold %d test window: %s .. %s" % (i, lo, hi), flush=True)

        per_seed = []
        for s in range(args.seeds):
            cfg = CFG_FNS[args.market]()
            cfg.market = args.market
            cfg.train.total_timesteps = args.timesteps
            cfg.train.eval_interval = 0
            cfg.train.seed = 100 + s          # same seed convention as §5
            cfg.train.checkpoint_dir = os.path.join("checkpoints", "_walkfwd")
            random.seed(100 + s)
            np.random.seed(100 + s)

            t0 = time.time()
            res = rolling_walk_forward(df, cfg, n_folds=args.folds,
                                       scheme=args.scheme, timesteps=args.timesteps)
            per_seed.append(res)
            print("   seed %d done in %.1f min | agent per fold: %s"
                  % (100 + s, (time.time() - t0) / 60,
                     ", ".join("%+.1f%%" % (r["agent_return"] * 100) for r in res)), flush=True)

        # aggregate per fold across seeds
        folds_out = []
        for fi in range(args.folds):
            agent = [ps[fi]["agent_return"] for ps in per_seed if fi < len(ps)]
            bh = [ps[fi]["bh_return"] for ps in per_seed if fi < len(ps)]
            lo, hi = bootstrap_ci(agent)
            folds_out.append({
                "fold": fi,
                "test_start": ranges[fi][0] if fi < len(ranges) else "?",
                "test_end": ranges[fi][1] if fi < len(ranges) else "?",
                "agent_mean": float(np.mean(agent)), "agent_ci": [lo, hi],
                "agent_per_seed": [float(a) for a in agent],
                "bh_return": float(np.mean(bh)),
                "beats_bh": bool(np.mean(agent) > np.mean(bh)),
            })
        payload["results"][ticker] = folds_out

        print("\n   | Fold | Test window | Agent (mean, 95%% CI over %d seeds) | Buy & hold |" % args.seeds, flush=True)
        for f in folds_out:
            print("   | %d | %s .. %s | %+.1f%% [%+.1f%%, %+.1f%%] | %+.1f%% |"
                  % (f["fold"], f["test_start"], f["test_end"], f["agent_mean"] * 100,
                     f["agent_ci"][0] * 100, f["agent_ci"][1] * 100, f["bh_return"] * 100), flush=True)

        allf = [a for f in folds_out for a in f["agent_per_seed"]]
        lo, hi = bootstrap_ci(allf)
        payload["results"][ticker + "__pooled"] = {
            "n": len(allf), "mean": float(np.mean(allf)), "ci": [lo, hi],
            "folds_beating_bh": sum(1 for f in folds_out if f["beats_bh"]),
        }
        print("   pooled over folds x seeds (n=%d): %+.1f%% [%+.1f%%, %+.1f%%] | folds beating B&H: %d/%d"
              % (len(allf), np.mean(allf) * 100, lo * 100, hi * 100,
                 payload["results"][ticker + "__pooled"]["folds_beating_bh"], len(folds_out)), flush=True)

    payload["runtime_min"] = (time.time() - t_start) / 60
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote %s (%.1f min total)" % (args.out, payload["runtime_min"]), flush=True)


if __name__ == "__main__":
    main()
