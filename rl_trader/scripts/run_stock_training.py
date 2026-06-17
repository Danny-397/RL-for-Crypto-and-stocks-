"""CLI entry point: train a PPO agent on the stock environment.

Examples
--------
    python -m rl_trader.scripts.run_stock_training
    python -m rl_trader.scripts.run_stock_training --timesteps 50000 --seed 7
    python -m rl_trader.scripts.run_stock_training --data data/raw/AAPL.csv
"""

from __future__ import annotations

import argparse

from ..config.training_config import stock_config
from ..data.data_loader import load_ohlcv_csv
from ..training.train_stock import train_stock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO agent on stocks.")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total environment steps (default: config value).")
    parser.add_argument("--rollout", type=int, default=None,
                        help="Steps collected per PPO update.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to an OHLCV CSV. Omit to use synthetic data.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Where to save the trained model.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = stock_config()
    if args.timesteps is not None:
        config.train.total_timesteps = args.timesteps
    if args.rollout is not None:
        config.train.rollout_length = args.rollout
    if args.lr is not None:
        config.ppo.learning_rate = args.lr
    if args.seed is not None:
        config.train.seed = args.seed
    if args.checkpoint_dir is not None:
        config.train.checkpoint_dir = args.checkpoint_dir

    df = load_ohlcv_csv(args.data) if args.data else None
    train_stock(config=config, df=df)


if __name__ == "__main__":
    main()
