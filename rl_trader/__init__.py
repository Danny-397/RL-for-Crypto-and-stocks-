"""rl_trader — a modular PPO reinforcement-learning framework for trading.

The package is organised into clearly separated concerns so that each piece can
be understood, tested, and extended in isolation:

    config/      hyper-parameters and run settings (dataclass-based)
    data/        loading, feature engineering, scaling, train/val/test splits
    envs/        Gymnasium trading environments (stock + crypto)
    models/      PPO agent and neural-network architectures
    training/    rollout collection, PPO update loop, logging utilities
    evaluation/  backtesting metrics and plotting
    scripts/     command-line entry points
"""

__version__ = "0.1.0"
