"""Stock-market training entry point.

A thin wrapper that selects the equities preset and delegates to the shared
PPO engine in :func:`rl_trader.training.utils.run_ppo_training`. Keeping the
loop logic in one place guarantees stock and crypto agents train identically —
only their configuration differs.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config.training_config import Config, stock_config
from .utils import run_ppo_training


def train_stock(config: Optional[Config] = None, df: Optional[pd.DataFrame] = None):
    """Train a PPO agent on the stock environment.

    Parameters
    ----------
    config:
        Optional override. Defaults to :func:`stock_config`.
    df:
        Optional raw OHLCV data. If ``None``, synthetic equity data is used.
    """
    config = config or stock_config()
    config.market = "stock"
    return run_ppo_training(config, df=df)
