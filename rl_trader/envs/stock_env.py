"""Equity-market trading environment.

Equities trade on a fixed session calendar with relatively low transaction
costs and (typically) lower volatility than crypto. The mechanics are inherited
wholesale from :class:`BaseTradingEnv`; this subclass exists to give the market
a clear identity and a natural home for equity-specific behaviour (e.g. borrow
fees on shorts, session gaps) as the framework grows.
"""

from __future__ import annotations

from ..config.training_config import EnvConfig, RewardConfig
from ..data.data_loader import MarketData
from .base_env import BaseTradingEnv


class StockTradingEnv(BaseTradingEnv):
    """Trading environment specialised for equities."""

    market_name = "stock"

    def __init__(
        self,
        data: MarketData,
        env_config: EnvConfig,
        reward_config: RewardConfig,
        random_start: bool = True,
    ) -> None:
        super().__init__(data, env_config, reward_config, random_start=random_start)
        # Extension point: model overnight gaps, borrow costs on shorts, or a
        # session calendar here without touching the shared base logic.
