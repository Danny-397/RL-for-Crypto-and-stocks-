"""CLI entry point: train the *same* PPO architecture on both markets and
compare how it behaves.

This is the project's headline experiment. The agent architecture is held
constant; only the environment (stock vs crypto) and its preset change. We then
backtest each trained agent on its held-out test split and print a side-by-side
metrics table — letting us ask whether a single RL recipe adapts differently to
the two regimes.

Examples
--------
    python -m rl_trader.scripts.compare_markets
    python -m rl_trader.scripts.compare_markets --timesteps 40000 --plot
"""

from __future__ import annotations

import argparse
from typing import Dict

from ..config.training_config import crypto_config, stock_config
from ..data.data_loader import prepare_market_data
from ..envs import make_env
from ..evaluation.evaluate_agent import BacktestResult, backtest
from ..training.utils import run_ppo_training


def _train_and_backtest(config) -> BacktestResult:
    """Train on the train split, then backtest on the held-out test split."""
    agent, _ = run_ppo_training(config)
    splits = prepare_market_data(None, market=config.market, seed=config.train.seed)
    test_env = make_env(
        config.market, splits["test"], config.env, config.reward, random_start=False
    )
    return backtest(agent, test_env, market=config.market)


def _print_table(results: Dict[str, BacktestResult]) -> None:
    """Print a clean side-by-side metrics comparison."""
    cols = list(results.keys())
    rows = [
        ("Total return", "total_return", "{:+.2%}"),
        ("Sharpe ratio", "sharpe", "{:.2f}"),
        ("Max drawdown", "max_drawdown", "{:.2%}"),
        ("Final equity", "final_equity", "${:,.0f}"),
    ]
    header = f"{'Metric':<16}" + "".join(f"{c.capitalize():>16}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for label, key, fmt in rows:
        line = f"{label:<16}"
        for c in cols:
            line += f"{fmt.format(results[c].metrics[key]):>16}"
        print(line)
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PPO across stock & crypto.")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total steps for EACH market (default: config value).")
    parser.add_argument("--seed", type=int, default=42, help="Shared random seed.")
    parser.add_argument("--plot", action="store_true",
                        help="Save equity-curve plots to the current directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    configs = {"stock": stock_config(), "crypto": crypto_config()}
    results: Dict[str, BacktestResult] = {}
    for market, cfg in configs.items():
        cfg.train.seed = args.seed
        if args.timesteps is not None:
            cfg.train.total_timesteps = args.timesteps
        print(f"\n=== Training PPO on {market} ===")
        results[market] = _train_and_backtest(cfg)

    _print_table(results)

    if args.plot:
        from ..evaluation.plots import plot_equity_curve
        for market, res in results.items():
            plot_equity_curve(
                res.equity_curve, title=f"{market.capitalize()} — test equity",
                save_path=f"equity_{market}.png",
            )
            print(f"Saved equity_{market}.png")


if __name__ == "__main__":
    main()
