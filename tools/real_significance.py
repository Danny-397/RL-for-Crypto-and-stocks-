"""Real-data significance: is the out-of-sample crypto/stock edge statistically real?

The headline dashboard number comes from one training seed on one walk-forward
split. This script closes that gap with the same rigor the synthetic study uses,
but on **real markets**:

1. Train ``--seeds`` independent agents on the real basket (domain-randomized
   across tickers), each on a throwaway checkpoint so deployment models are safe.
2. For every seed, backtest on each ticker's held-out test period and record the
   agent's return vs. buy-&-hold's.
3. Report (a) a bootstrap 95% CI on the basket-mean agent return *across seeds*
   — how repeatable is it? — and (b) a paired permutation test of the agent vs.
   buy-&-hold *across the held-out tickers* — is the cross-sectional edge
   distinguishable from noise?

Run from the repo root (after tools/fetch_data.py):

    python tools/real_significance.py --seeds 5 --timesteps 150000
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.data.data_loader import load_ohlcv_csv, prepare_market_data
from rl_trader.envs import make_env
from rl_trader.evaluation.evaluate_agent import ANNUALISATION, backtest, compute_metrics
from rl_trader.evaluation.statistics import bootstrap_ci, paired_permutation_test
from rl_trader.training.utils import get_logger, run_ppo_training


def load_real_basket(data_dir: str, market: str, train_frac: float = 0.6) -> dict:
    """Load every ticker CSV under ``data_dir/market`` into train/test splits.

    Chronological split with the scaler fit on the training slice only — a clean
    walk-forward setup with no look-ahead leakage (mirrors build_site_data.py).
    """
    basket = {}
    for path in sorted(glob.glob(os.path.join(data_dir, market, "*.csv"))):
        ticker = os.path.splitext(os.path.basename(path))[0]
        splits = prepare_market_data(load_ohlcv_csv(path), market=market,
                                     train_frac=train_frac, val_frac=0.0)
        if len(splits["train"]) > 60 and len(splits["test"]) > 60:
            basket[ticker] = splits
    return basket


def _evaluate_basket(agent, basket, market, cfg, periods):
    """Per-ticker agent and buy-&-hold total returns over the held-out test split."""
    agent_rets, bh_rets = [], []
    w = cfg.env.window_size
    for sp in basket.values():
        test = sp["test"]
        env = make_env(market, test, cfg.env, cfg.reward, random_start=False)
        agent_rets.append(backtest(agent, env, market=market).metrics["total_return"])
        prices = test.prices[w - 1 :]
        bh = compute_metrics(cfg.env.initial_balance * (prices / prices[0]), periods)
        bh_rets.append(bh["total_return"])
    return np.array(agent_rets), np.array(bh_rets)


def run_market(market: str, cfg_fn, seeds: int, timesteps: int, data_dir: str, log):
    basket = load_real_basket(data_dir, market)
    if not basket:
        raise SystemExit(f"No CSVs in {os.path.join(data_dir, market)} — run tools/fetch_data.py.")
    tickers = list(basket)
    train_slices = [basket[t]["train"] for t in tickers]
    periods = ANNUALISATION.get(market, 252)

    seed_mean_returns, seed_win_rates = [], []
    agent_by_asset = np.zeros(len(tickers))
    bh_by_asset = None

    for s in range(seeds):
        cfg = cfg_fn()
        cfg.market = market
        cfg.train.total_timesteps = timesteps
        cfg.train.eval_interval = 0
        cfg.train.seed = 100 + s
        cfg.train.checkpoint_dir = os.path.join("checkpoints", "_realsig")
        random.seed(100 + s)

        def factory():
            return random.choice(train_slices)

        agent, _ = run_ppo_training(cfg, train_series_factory=factory)
        a, b = _evaluate_basket(agent, basket, market, cfg, periods)
        bh_by_asset = b
        agent_by_asset += a
        seed_mean_returns.append(float(a.mean()))
        seed_win_rates.append(float((a > b).mean()))
        log.info("[%s] seed %d/%d | basket-mean agent %+.2f%% | win-rate %.0f%%",
                 market, s + 1, seeds, 100 * seed_mean_returns[-1], 100 * seed_win_rates[-1])

    agent_by_asset /= seeds
    ret_ci = bootstrap_ci(seed_mean_returns)
    obs_diff, p_value = paired_permutation_test(agent_by_asset, bh_by_asset)
    return {
        "tickers": tickers,
        "ret_ci": ret_ci,
        "mean_win_rate": float(np.mean(seed_win_rates)),
        "bh_mean": float(bh_by_asset.mean()),
        "diff": obs_diff,
        "p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--market", choices=["stock", "crypto", "both"], default="both")
    args = parser.parse_args()

    log = get_logger("real_significance")
    markets = ["stock", "crypto"] if args.market == "both" else [args.market]
    cfg_fns = {"stock": stock_config, "crypto": crypto_config}

    print("\n" + "=" * 70)
    print(f"REAL-DATA SIGNIFICANCE  ({args.seeds} seeds, {args.timesteps:,} timesteps)")
    print("=" * 70)
    for market in markets:
        r = run_market(market, cfg_fns[market], args.seeds, args.timesteps, args.data_dir, log)
        ci = r["ret_ci"]
        verdict = "DISTINGUISHABLE" if r["p"] < 0.05 else "not distinguishable"
        print(f"\n{market.upper()}  ({len(r['tickers'])} held-out tickers)")
        print(f"  Agent basket return : mean {ci.mean:+.2%}  95% CI [{ci.low:+.2%}, {ci.high:+.2%}]")
        print(f"  Buy & hold (mean)   : {r['bh_mean']:+.2%}")
        print(f"  Mean win-rate       : {r['mean_win_rate']:.0%} of tickers beat buy-&-hold")
        print(f"  Agent vs B&H        : {r['diff']:+.2%}  (paired permutation p = {r['p']:.4f})")
        print(f"  Verdict             : cross-sectional edge is {verdict} at alpha=0.05")
    print("=" * 70)


if __name__ == "__main__":
    main()
