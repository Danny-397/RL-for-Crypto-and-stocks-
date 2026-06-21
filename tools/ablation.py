"""Ablation: does domain randomization actually prevent overfitting?

This experiment quantifies the project's central methodological claim. We train
two otherwise-identical PPO agents:

    1. SINGLE-PATH   — trained on one fixed price series (the naive setup).
    2. DOMAIN-RANDOM — trained on a fresh random series every episode.

We then measure each agent's return on (a) the path it trained on ("in-sample")
and (b) many unseen held-out paths ("out-of-sample"). A large in-sample /
out-of-sample gap is the signature of overfitting.

Run from the repo root:

    python tools/ablation.py --timesteps 60000 --market stock
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.data.data_loader import synthetic_market_data
from rl_trader.envs import make_env
from rl_trader.evaluation.evaluate_agent import backtest
from rl_trader.training.utils import run_ppo_training


def _config_for(market: str):
    return crypto_config() if market == "crypto" else stock_config()


def _train(market: str, cfg, timesteps: int, seed: int, domain_random: bool):
    """Train one agent; return (agent, fixed_training_data_or_None)."""
    cfg.market = market
    cfg.train.total_timesteps = timesteps
    cfg.train.seed = seed
    cfg.train.eval_interval = 0
    # Write to a throwaway dir so the ablation never clobbers the deployment
    # checkpoints produced by build_site_data.py --real (which baseline_report
    # and export_policy load).
    cfg.train.checkpoint_dir = os.path.join("checkpoints", "_ablation")

    if domain_random:
        factory = lambda: synthetic_market_data(market)  # noqa: E731
        fixed = None
    else:
        fixed = synthetic_market_data(market, seed=seed)  # ONE fixed path
        factory = lambda: fixed  # noqa: E731

    agent, _ = run_ppo_training(cfg, train_series_factory=factory)
    return agent, fixed


def _return_on(agent, data, market, cfg) -> float:
    env = make_env(market, data, cfg.env, cfg.reward, random_start=False)
    return backtest(agent, env, market=market).metrics["total_return"]


def _oos_returns(agent, market, cfg, n: int = 30) -> np.ndarray:
    return np.array([
        _return_on(agent, synthetic_market_data(market, seed=20_000 + k), market, cfg)
        for k in range(n)
    ])


def run(market: str, timesteps: int, seed: int, n_eval: int) -> dict:
    cfg = _config_for(market)

    # --- Single-path agent (expected to overfit) ---
    sp_agent, sp_fixed = _train(market, _config_for(market), timesteps, seed, False)
    sp_in = _return_on(sp_agent, sp_fixed, market, cfg)
    sp_oos = _oos_returns(sp_agent, market, cfg, n_eval)

    # --- Domain-randomized agent ---
    dr_agent, _ = _train(market, _config_for(market), timesteps, seed, True)
    # "In-sample" reference path for the DR agent: a fresh training draw.
    dr_ref = synthetic_market_data(market, seed=seed)
    dr_in = _return_on(dr_agent, dr_ref, market, cfg)
    dr_oos = _oos_returns(dr_agent, market, cfg, n_eval)

    return {
        "single": {"in": sp_in, "oos_mean": sp_oos.mean(), "oos_std": sp_oos.std()},
        "domain": {"in": dr_in, "oos_mean": dr_oos.mean(), "oos_std": dr_oos.std()},
    }


def _fmt(v: float) -> str:
    return f"{v:+.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-randomization ablation.")
    parser.add_argument("--timesteps", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--market", choices=["stock", "crypto", "both"], default="both")
    args = parser.parse_args()

    markets = ["stock", "crypto"] if args.market == "both" else [args.market]
    print(f"\nDomain-randomization ablation  (timesteps={args.timesteps}, "
          f"{args.n_eval} held-out paths)\n")
    print(f"{'Market':<8}{'Training':<16}{'In-sample':>12}{'Out-of-sample':>22}{'Gap':>12}")
    print("-" * 70)
    all_results = {}
    for market in markets:
        res = run(market, args.timesteps, args.seed, args.n_eval)
        all_results[market] = res
        for label, key in (("single-path", "single"), ("domain-random", "domain")):
            r = res[key]
            oos = f"{_fmt(r['oos_mean'])} +/- {abs(r['oos_std']):.0%}"
            gap = r["in"] - r["oos_mean"]
            print(f"{market:<8}{label:<16}{_fmt(r['in']):>12}{oos:>22}{_fmt(gap):>12}")
        print("-" * 70)
    print("\nA large in-sample/out-of-sample gap = overfitting. Domain randomization\n"
          "should collapse that gap and lift out-of-sample performance.\n")

    # Persist so tools/make_figures.py can render the chart without re-training.
    out_path = os.path.join("docs", "assets", "ablation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
