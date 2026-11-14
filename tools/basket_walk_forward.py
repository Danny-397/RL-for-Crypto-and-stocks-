"""Walk-forward over the whole basket: does the result hold in every period?

§5 of RESULTS.md trains on the first 60% of history and tests on the last 40%,
once. A single cut cannot tell a property of the agent from a property of the
months that happened to fall in the test block. §5c ran four rolling folds, but
on BTC-USD alone and with the PPO before its repairs (§10a).

This runs the study itself, the whole basket, across history: four folds on one
calendar for every ticker. For fold k the agent is trained from scratch, on
every ticker's history before the fold's first test date (domain-randomised
across tickers, exactly as §5 trains), and scored on every ticker over the
fold's test block against buy-and-hold on the same block. The training dates of
every ticker precede the test dates of every ticker, so there is no cross-ticker
look-ahead (§10c), and each fold's feature scaler is fit on its training rows.

Folds are laid out on the window every ticker trades in: its first 40% is the
first fold's training period, and the remaining 60% is cut into ``--folds``
equal test blocks (expanding window, as ``rl_trader.evaluation.walk_forward``).

Run one fold of one market (CI runs every fold as its own job):

    python tools/basket_walk_forward.py --market crypto --fold 0 --seeds 10
    python tools/basket_walk_forward.py --summarise results/depth/wf-*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_trader.config.training_config import crypto_config, stock_config  # noqa: E402

CFG = {"stock": stock_config, "crypto": crypto_config}
TRAIN_MIN_FRAC = 0.4


def fold_dates(date_lists, n_folds: int, train_min_frac: float = TRAIN_MIN_FRAC):
    """Calendar boundaries of the test blocks, on the window every series covers.

    ``date_lists`` is one sorted sequence of dates per ticker. Returns
    ``n_folds`` ``(start, end)`` pairs: the test block of fold k is
    ``start <= date < end``, and the last block runs to the common end
    (``end`` is ``None``). Pure, so it is unit-tested without training.
    """
    import pandas as pd

    series = [pd.Series(pd.to_datetime(list(d))) for d in date_lists]
    start = max(s.iloc[0] for s in series)
    end = min(s.iloc[-1] for s in series)
    common = min((s[(s >= start) & (s <= end)] for s in series), key=len).reset_index(drop=True)
    n = len(common)
    first = int(n * train_min_frac)
    block = (n - first) // n_folds
    if block < 2:
        raise ValueError("not enough common history for the requested folds")
    out = []
    for k in range(n_folds):
        lo = common.iloc[first + k * block]
        hi = None if k == n_folds - 1 else common.iloc[first + (k + 1) * block]
        out.append((lo, hi))
    return out


def _slices(featured, lo, hi):
    """Row masks of one ticker's training rows (before ``lo``) and test rows."""
    d = featured["date"]
    train = (d < lo).to_numpy()
    test = ((d >= lo) & (d < hi)).to_numpy() if hi is not None else (d >= lo).to_numpy()
    return train, test


def load_fold(data_dir: str, market: str, n_folds: int, fold: int):
    """Train/test ``MarketData`` per ticker for one fold, plus its test window."""
    from rl_trader.data.data_loader import (
        FEATURE_COLUMNS,
        MarketData,
        _fit_scaler,
        add_technical_indicators,
        attach_market_index,
        load_ohlcv_csv,
    )

    featured = {}
    for path in sorted(glob.glob(os.path.join(data_dir, market, "*.csv"))):
        ticker = os.path.splitext(os.path.basename(path))[0]
        df = attach_market_index(load_ohlcv_csv(path), data_dir, market)
        featured[ticker] = add_technical_indicators(df)
    if not featured:
        raise SystemExit(f"No CSVs in {os.path.join(data_dir, market)}; run tools/fetch_data.py.")
    bounds = fold_dates([f["date"] for f in featured.values()], n_folds)
    lo, hi = bounds[fold]
    basket = {}
    for ticker, f in featured.items():
        tr, te = _slices(f, lo, hi)
        feat = f[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        prices = f["close"].to_numpy(dtype=np.float32)
        if tr.sum() <= 60 or te.sum() <= 60:
            continue
        mean, std = _fit_scaler(feat[tr])

        def scale(a, mean=mean, std=std):
            return ((a - mean) / std).astype(np.float32)

        basket[ticker] = {
            "train": MarketData(scale(feat[tr]), prices[tr], FEATURE_COLUMNS),
            "test": MarketData(scale(feat[te]), prices[te], FEATURE_COLUMNS),
        }
    test_dates = [f["date"][_slices(f, lo, hi)[1]] for f in featured.values()]
    window = (str(min(d.iloc[0] for d in test_dates))[:10],
              str(max(d.iloc[-1] for d in test_dates))[:10])
    return basket, window


def run_fold(market: str, fold: int, n_folds: int, seeds: int, seed_base: int,
             timesteps: int, data_dir: str) -> dict:
    from rl_trader.evaluation.evaluate_agent import ANNUALISATION
    from rl_trader.training.utils import get_logger, run_ppo_training
    from tools.real_significance import _evaluate_basket

    log = get_logger("basket_walk_forward")
    basket, window = load_fold(data_dir, market, n_folds, fold)
    tickers = list(basket)
    train_slices = [basket[t]["train"] for t in tickers]
    periods = ANNUALISATION.get(market, 252)
    per_seed, bh, betas, alphas = [], None, [], []
    for s in range(seeds):
        cfg = CFG[market]()
        cfg.market = market
        cfg.train.total_timesteps = timesteps
        cfg.train.eval_interval = 0
        cfg.train.seed = seed_base + s
        cfg.train.checkpoint_dir = os.path.join("checkpoints", "_basketwf")
        random.seed(seed_base + s)

        def factory():
            return random.choice(train_slices)

        agent, _ = run_ppo_training(cfg, train_series_factory=factory)
        a, b, risk = _evaluate_basket(agent, basket, market, cfg, periods)
        per_seed.append([round(float(v), 4) for v in a])
        bh = [round(float(v), 4) for v in b]
        betas.append(float(risk["beta"].mean()))
        alphas.append(float(risk["alpha"].mean()))
        log.info("[%s fold %d] seed %d/%d | basket agent %+.1f%% | B&H %+.1f%%",
                 market, fold, s + 1, seeds, 100 * float(np.mean(a)), 100 * float(np.mean(b)))
    return {
        "market": market, "fold": fold, "folds": n_folds, "test_start": window[0],
        "test_end": window[1], "tickers": tickers, "seed_base": seed_base,
        "seeds": seeds, "timesteps": timesteps, "per_seed_ticker": per_seed,
        "bh_by_ticker": bh, "beta": round(float(np.mean(betas)), 3),
        "alpha": round(float(np.mean(alphas)), 4),
    }


def summarise(paths) -> dict:
    """Combine fold files into the per-fold and pooled verdicts."""
    from rl_trader.evaluation.rigor import aggregate_scores
    from rl_trader.evaluation.statistics import paired_permutation_test

    by_market: dict = {}
    for p in sorted(paths):
        with open(p, encoding="utf-8") as fh:
            r = json.load(fh)
        by_market.setdefault(r["market"], []).append(r)
    out = {}
    for market, folds in by_market.items():
        folds.sort(key=lambda r: r["fold"])
        rows, pooled_a, pooled_b = [], [], []
        for r in folds:
            m = np.array(r["per_seed_ticker"], dtype=float)
            bh = np.array(r["bh_by_ticker"], dtype=float)
            agent_by_ticker = m.mean(axis=0)
            diff, p = paired_permutation_test(agent_by_ticker, bh)
            pooled_a.extend(agent_by_ticker)
            pooled_b.extend(bh)
            rows.append({
                "fold": r["fold"], "test_start": r["test_start"], "test_end": r["test_end"],
                "n_tickers": len(bh), "agent": aggregate_scores(m.mean(axis=1)),
                "bh": round(float(bh.mean()), 4), "diff": round(float(diff), 4),
                "p": round(float(p), 4),
                "tickers_behind": int((agent_by_ticker < bh).sum()),
                "beta": r["beta"], "alpha": r["alpha"],
            })
        diff, p = paired_permutation_test(np.array(pooled_a), np.array(pooled_b))
        a, b = np.array(pooled_a), np.array(pooled_b)
        out[market] = {
            "folds": rows, "seeds": folds[0]["seeds"], "seed_base": folds[0]["seed_base"],
            "timesteps": folds[0]["timesteps"],
            "pooled": {
                "n_pairs": len(a), "diff": round(float(diff), 4), "p": round(float(p), 4),
                "pairs_behind": int((a < b).sum()),
                "folds_behind": sum(1 for r in rows if r["diff"] < 0),
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", choices=["stock", "crypto"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out-dir", default=os.path.join("results", "depth"))
    ap.add_argument("--summarise", nargs="+", metavar="FOLD_JSON",
                    help="combine fold files instead of training")
    ap.add_argument("--out", default=os.path.join("docs", "assets", "basket_walk_forward.json"))
    args = ap.parse_args()

    if args.summarise:
        res = summarise(args.summarise)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
            fh.write("\n")
        for market, r in res.items():
            print(f"\n{market}: pooled agent - B&H {r['pooled']['diff']:+.1%} "
                  f"(p = {r['pooled']['p']:.4f}, {r['pooled']['pairs_behind']}/"
                  f"{r['pooled']['n_pairs']} ticker-folds behind)")
            for f in r["folds"]:
                print(f"  fold {f['fold']} {f['test_start']}..{f['test_end']}: agent "
                      f"{f['agent']['mean']:+.1%}  B&H {f['bh']:+.1%}  p = {f['p']:.3f}")
        print(f"wrote {args.out}")
        return

    if args.market is None or args.fold is None:
        ap.error("--market and --fold are required unless --summarise is given")
    res = run_fold(args.market, args.fold, args.folds, args.seeds, args.seed_base,
                   args.timesteps, args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"wf-{args.market}-fold{args.fold}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
        fh.write("\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
