"""Dataclass-based configuration.

Everything that controls a run lives here so experiments are reproducible and
self-documenting. Grouping settings into small dataclasses keeps each concern
(environment, reward shaping, PPO, training loop) independent and easy to tune.

Factory helpers (:func:`stock_config`, :func:`crypto_config`) return sensible
market-specific presets: crypto trades 24/7, is more volatile, and carries
higher effective costs, so its defaults differ from equities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Tuple


@dataclass
class EnvConfig:
    """Trading-environment mechanics (account, costs, position limits)."""

    window_size: int = 20            # number of past bars in each observation
    initial_balance: float = 100_000.0
    transaction_cost: float = 0.0005  # proportional cost per unit traded (5 bps)
    slippage: float = 0.0005          # proportional price impact per trade
    allow_short: bool = True          # permit negative (short) positions
    max_position: float = 1.0         # max |position value| as a fraction of equity
    bankruptcy_threshold: float = 0.2  # terminate if equity < threshold * initial


@dataclass
class RewardConfig:
    """Weights for the shared reward function (see :class:`BaseTradingEnv`)."""

    use_log_return: bool = True       # log return is additive and well-scaled
    drawdown_penalty: float = 0.10    # penalise depth below the equity high-water mark
    turnover_penalty: float = 0.001   # penalise churn (transaction friction)


@dataclass
class PPOConfig:
    """Proximal Policy Optimization hyper-parameters."""

    learning_rate: float = 3e-4
    gamma: float = 0.99               # discount factor
    gae_lambda: float = 0.95          # GAE bias/variance trade-off
    clip_ratio: float = 0.2           # PPO surrogate clipping
    value_coef: float = 0.5           # value-loss weight
    entropy_coef: float = 0.01        # exploration bonus
    max_grad_norm: float = 0.5        # gradient clipping
    update_epochs: int = 10           # passes over each rollout
    minibatch_size: int = 256
    hidden_sizes: Tuple[int, ...] = (128, 128)
    use_lstm: bool = False            # experimental; see models/networks.py
    init_log_std: float = -0.5        # initial policy std = exp(-0.5) ~= 0.61


@dataclass
class TrainConfig:
    """Outer training-loop settings and I/O locations."""

    total_timesteps: int = 200_000
    rollout_length: int = 2048        # environment steps collected per PPO update
    seed: int = 42
    log_interval: int = 1             # log every N updates
    eval_interval: int = 10           # evaluate every N updates (0 disables)
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    device: str = "auto"              # "auto" | "cpu" | "cuda"


@dataclass
class Config:
    """Top-level container bundling every sub-config plus the market name."""

    market: str = "stock"
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        """Return a plain nested dict (handy for logging / serialisation)."""
        return asdict(self)


def default_config() -> Config:
    """A neutral baseline configuration."""
    return Config()


def stock_config() -> Config:
    """Preset tuned for equities: low costs, modest volatility expectations."""
    cfg = Config(market="stock")
    cfg.env.transaction_cost = 0.0005
    cfg.env.slippage = 0.0003
    cfg.env.allow_short = True
    return cfg


def crypto_config() -> Config:
    """Preset tuned for crypto: higher costs/slippage, fatter tails, 24/7.

    The wider initial policy std encourages more exploration in the noisier
    crypto regime, and the larger drawdown penalty reflects harsher tail risk.
    """
    cfg = Config(market="crypto")
    cfg.env.transaction_cost = 0.0010
    cfg.env.slippage = 0.0010
    cfg.env.allow_short = True
    cfg.reward.drawdown_penalty = 0.15
    cfg.ppo.init_log_std = -0.25
    return cfg
