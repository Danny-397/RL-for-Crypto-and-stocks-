"""Dataclass-based configuration.

Everything that controls a run lives here so experiments are reproducible and
self-documenting. Grouping settings into small dataclasses keeps each concern
(environment, reward shaping, PPO, training loop) independent and easy to tune.

Factory helpers (:func:`stock_config`, :func:`crypto_config`) return sensible
market-specific presets: crypto trades 24/7, is more volatile, and carries
higher effective costs, so its defaults differ from equities.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Tuple

# The seven PPO implementation details RESULTS.md §10a found to matter. The
# repaired values are the defaults from v2.3.0; the published values are what
# every result up to v2.2.1 was produced with.
REPAIRED_PPO = {
    "bootstrap_truncation": True, "shared_trunk": False, "normalize_at_collection": True,
    "squash_mean": False, "value_loss": "clipped", "adam_eps": 1e-5, "advantage_norm": "minibatch",
}
PUBLISHED_PPO = {
    "bootstrap_truncation": False, "shared_trunk": True, "normalize_at_collection": False,
    "squash_mean": True, "value_loss": "max", "adam_eps": 1e-8, "advantage_norm": "rollout",
}


def published_mode() -> bool:
    """True when ``RL_TRADER_PUBLISHED=1``: reproduce the v2.2.1 numbers.

    Restores the published PPO (every ``PPOConfig`` built while it is set) and the
    published per-ticker real-data split (``tools/real_significance.load_real_basket``).
    """
    return os.environ.get("RL_TRADER_PUBLISHED", "").strip().lower() in ("1", "true", "yes")


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
    """Weights for the shared reward function (see :class:`BaseTradingEnv`).

    Two reward formulations are supported via ``kind``:

    * ``"return"`` (default) — risk-aware net return: scaled per-step return
      minus drawdown and turnover penalties. ``return_scale`` keeps the (tiny)
      per-step return the dominant learning signal; the penalties are
      deliberately an order of magnitude smaller so they *shape* behaviour
      without overwhelming the profit objective.
    * ``"dsr"`` — the **Differential Sharpe Ratio** (Moody & Saffell, 1998): an
      online, per-step approximation of the change in the Sharpe ratio. Rewarding
      DSR makes the agent optimise *risk-adjusted* return directly rather than
      raw PnL, which empirically yields steadier equity curves. Turnover is still
      penalised; the drawdown term is subsumed by the volatility normalisation.
    """

    kind: str = "return"              # "return" | "dsr"
    use_log_return: bool = True       # log return is additive and well-scaled
    return_scale: float = 100.0       # lift ~1e-3 returns into a PPO-learnable range
    drawdown_penalty: float = 0.05    # penalise depth below the equity high-water mark
    turnover_penalty: float = 0.0005  # penalise churn (transaction friction)
    dsr_eta: float = 0.04             # DSR adaptation rate (EMA horizon ~1/eta bars)
    dsr_scale: float = 1.0            # lift the (small) DSR signal into PPO's range


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
    use_lstm: bool = False            # train the recurrent (LSTM) policy instead
    recurrent_seq_len: int = 64       # truncated-BPTT sequence length (LSTM only)
    init_log_std: float = -0.5        # initial policy std = exp(-0.5) ~= 0.61
    normalize_obs: bool = True        # running mean/std normalisation of observations
    obs_clip: float = 10.0            # clip normalised observations to +/- this many std
    normalize_reward: bool = False    # running-std reward scaling (off: return_scale handles it)
    # The repaired update (RESULTS.md §10a). Each line notes the v2.2.1 value;
    # RL_TRADER_PUBLISHED=1 restores all of them at once.
    bootstrap_truncation: bool = True  # bootstrap V(s) when data runs out (published: False)
    shared_trunk: bool = False        # separate critic trunk (published: True, shared)
    squash_mean: bool = False         # tanh on the policy mean (published: True)
    normalize_at_collection: bool = True  # store obs normalised as seen (published: False)
    value_loss: str = "clipped"       # SB3's value loss (published: "max")
    adam_eps: float = 1e-5            # Adam epsilon (published: 1e-8)
    advantage_norm: str = "minibatch"  # per-minibatch advantages (published: "rollout")

    def __post_init__(self) -> None:
        if published_mode():
            for k, v in PUBLISHED_PPO.items():
                setattr(self, k, v)


def repaired_ppo(ppo: "PPOConfig") -> "PPOConfig":
    """Switch a PPOConfig to the repaired update (RESULTS.md §10a).

    The default since v2.3.0. Benchmarked against Stable-Baselines3 under the same
    hyper-parameters, the published PPO trails it on HalfCheetah and Pendulum;
    with these seven changes it matches SB3 on InvertedPendulum, Hopper and
    HalfCheetah.
    """
    for k, v in REPAIRED_PPO.items():
        setattr(ppo, k, v)
    return ppo


def published_ppo(ppo: "PPOConfig") -> "PPOConfig":
    """Switch a PPOConfig to the update every v2.2.1 result was produced with."""
    for k, v in PUBLISHED_PPO.items():
        setattr(ppo, k, v)
    return ppo


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
    deterministic: bool = False       # pin Torch to deterministic kernels (slower)


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
    cfg.reward.drawdown_penalty = 0.08
    return cfg
