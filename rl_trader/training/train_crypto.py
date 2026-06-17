"""Crypto-market training entry point.

A thin wrapper that selects the crypto preset and delegates to the shared PPO
engine in :func:`rl_trader.training.utils.run_ppo_training`. The agent
architecture is identical to the stock agent — only the environment dynamics
and a few hyper-parameters change.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config.training_config import Config, crypto_config
from .utils import run_ppo_training


def train_crypto(config: Optional[Config] = None, df: Optional[pd.DataFrame] = None):
    """Train a PPO agent on the crypto environment.

    Parameters
    ----------
    config:
        Optional override. Defaults to :func:`crypto_config`.
    df:
        Optional raw OHLCV data. If ``None``, synthetic crypto data is used.
    """
    config = config or crypto_config()
    config.market = "crypto"
    return run_ppo_training(config, df=df)
