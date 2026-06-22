"""Compare the trained PPO agent against every baseline on the real test data.

Loads the checkpoints written by ``build_site_data.py --real`` and, for each
market, reports the mean (across the basket's held-out test periods) total
return, Sharpe, and max drawdown for the agent and for every baseline strategy.

Run from the repo root (after a --real build):

    python tools/baseline_report.py
"""

from __future__ import annotations

import glob
import os

import numpy as np

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.data.data_loader import attach_market_index, load_ohlcv_csv, prepare_market_data
from rl_trader.envs import make_env
from rl_trader.evaluation.baselines import evaluate_baselines
from rl_trader.evaluation.evaluate_agent import backtest
from rl_trader.models.ppo_agent import PPOAgent


def _basket(market: str, data_dir: str = "data/raw"):
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, market, "*.csv"))):
        ticker = os.path.splitext(os.path.basename(path))[0]
        # merge the market index so the cross-asset features match how the agent trained
        df = attach_market_index(load_ohlcv_csv(path), data_dir, market)
        splits = prepare_market_data(df, market=market, train_frac=0.6, val_frac=0.0)
        if len(splits["test"]) > 60:
            out[ticker] = splits
    return out


def main() -> None:
    for market in ("stock", "crypto"):
        cfg = stock_config() if market == "stock" else crypto_config()
        ckpt = os.path.join("checkpoints", f"ppo_{market}.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] {market}: no checkpoint ({ckpt}) — run build_site_data.py --real first.")
            continue
        agent = PPOAgent.from_checkpoint(ckpt)
        basket = _basket(market)

        # Collect per-ticker metrics for the agent and each baseline.
        rows = {"PPO agent": []}
        for ticker, splits in basket.items():
            test = splits["test"]
            env = make_env(market, test, cfg.env, cfg.reward, random_start=False)
            rows["PPO agent"].append(backtest(agent, env, market=market).metrics)
            for name, m in evaluate_baselines(test, cfg.env, cfg.reward, market=market).items():
                rows.setdefault(name, []).append(m)

        print(f"\n### {market.upper()}  (mean over {len(basket)} held-out tickers)\n")
        print(f"| {'Strategy':<14} | {'Return':>9} | {'Sharpe':>7} | {'Max DD':>7} |")
        print(f"|{'-'*16}|{'-'*11}|{'-'*9}|{'-'*9}|")
        for name, ms in rows.items():
            ret = np.mean([m["total_return"] for m in ms])
            shp = np.mean([m["sharpe"] for m in ms])
            dd = np.mean([m["max_drawdown"] for m in ms])
            print(f"| {name:<14} | {ret:>+8.1%} | {shp:>7.2f} | {dd:>7.1%} |")


if __name__ == "__main__":
    main()
