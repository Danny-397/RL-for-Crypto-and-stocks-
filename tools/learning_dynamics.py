"""Learning dynamics: does more training actually buy better out-of-sample return?

Retrains the real-basket agent from scratch at several compute budgets and records
the held-out (and in-sample) basket-mean return at each. The point isn't a bigger
number — it's the *shape*: in-sample return keeps climbing with compute while the
held-out return plateaus and wobbles with noise. That gap is the honest answer to
"shouldn't it just get better and better?": the ceiling is the market's
predictability, not the number of training steps.

Reproducible (fixed seed, domain-randomized across tickers, sandboxed checkpoints).
Run from the repo root after tools/fetch_data.py:

    python tools/learning_dynamics.py --budgets 20000 60000 120000 200000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import numpy as np

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.data.data_loader import attach_market_index, load_ohlcv_csv, prepare_market_data
from rl_trader.envs import make_env
from rl_trader.evaluation.evaluate_agent import ANNUALISATION, backtest, compute_metrics
from rl_trader.training.utils import get_logger, run_ppo_training


def load_real_basket(data_dir: str, market: str, train_frac: float = 0.6) -> dict:
    """Every ticker CSV under data_dir/market as train/test splits (no leakage)."""
    basket = {}
    for path in sorted(glob.glob(os.path.join(data_dir, market, "*.csv"))):
        ticker = os.path.splitext(os.path.basename(path))[0]
        df = attach_market_index(load_ohlcv_csv(path), data_dir, market)
        splits = prepare_market_data(df, market=market, train_frac=train_frac, val_frac=0.0)
        if len(splits["train"]) > 60 and len(splits["test"]) > 60:
            basket[ticker] = splits
    return basket


def _basket_mean(agent, basket, market, cfg, split: str) -> float:
    """Mean agent total return across the basket on a given split."""
    rets = []
    for sp in basket.values():
        env = make_env(market, sp[split], cfg.env, cfg.reward, random_start=False)
        rets.append(backtest(agent, env, market=market).metrics["total_return"])
    return float(np.mean(rets))


def _bh_mean(basket, cfg, periods) -> float:
    w = cfg.env.window_size
    vals = []
    for sp in basket.values():
        prices = sp["test"].prices[w - 1:]
        vals.append(compute_metrics(cfg.env.initial_balance * (prices / prices[0]), periods)["total_return"])
    return float(np.mean(vals))


def run_market(market, cfg_fn, budgets, data_dir, seed, log):
    basket = load_real_basket(data_dir, market)
    if not basket:
        raise SystemExit(f"No CSVs in {os.path.join(data_dir, market)} — run tools/fetch_data.py.")
    train_slices = [sp["train"] for sp in basket.values()]
    periods = ANNUALISATION.get(market, 252)

    oos, insample = [], []
    cfg = None
    for b in budgets:
        cfg = cfg_fn()
        cfg.market = market
        cfg.train.total_timesteps = b
        cfg.train.eval_interval = 0
        cfg.train.seed = seed
        cfg.train.checkpoint_dir = os.path.join("checkpoints", "_learn")
        random.seed(seed)

        def factory():
            return random.choice(train_slices)

        agent, _ = run_ppo_training(cfg, train_series_factory=factory)
        o = _basket_mean(agent, basket, market, cfg, "test")
        i = _basket_mean(agent, basket, market, cfg, "train")
        oos.append(round(o, 4))
        insample.append(round(i, 4))
        log.info("[%s] budget %6d | in-sample %+.1f%% | held-out %+.1f%%", market, b, 100 * i, 100 * o)

    return {
        "budgets": list(budgets),
        "oos": oos,
        "insample": insample,
        "bh": round(_bh_mean(basket, cfg, periods), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budgets", type=int, nargs="+", default=[20000, 60000, 120000, 200000])
    p.add_argument("--data-dir", type=str, default="data/raw")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--market", choices=["stock", "crypto", "both"], default="both")
    args = p.parse_args()

    log = get_logger("learning_dynamics")
    markets = ["stock", "crypto"] if args.market == "both" else [args.market]
    fns = {"stock": stock_config, "crypto": crypto_config}

    print("\n" + "=" * 70)
    print(f"LEARNING DYNAMICS  (budgets: {', '.join(f'{b // 1000}k' for b in args.budgets)})")
    print("=" * 70)
    site = {}
    for m in markets:
        site[m] = run_market(m, fns[m], args.budgets, args.data_dir, args.seed, log)
        r = site[m]
        print(f"\n{m.upper()}  (buy-&-hold {r['bh']:+.1%})")
        for b, i, o in zip(r["budgets"], r["insample"], r["oos"]):
            print(f"  {b // 1000:>4}k steps | in-sample {i:+8.1%} | held-out {o:+8.1%}")
    print("=" * 70)

    out = os.path.join("docs", "learning.js")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("/* Auto-generated by tools/learning_dynamics.py — held-out return vs compute. */\n")
        fh.write("window.RL_LEARNING = ")
        json.dump(site, fh, indent=0)
        fh.write(";\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
