"""Crypto-market trading environment.

Crypto markets trade 24/7, exhibit fatter tails and higher realised volatility,
and generally carry higher effective costs (fees + slippage) than equities. The
preset in :func:`rl_trader.config.crypto_config` captures those differences;
this subclass marks the market identity and is the place to add crypto-specific
behaviour (funding rates on perps, exchange-specific fee tiers) later on.
"""

from __future__ import annotations

from ..config.training_config import EnvConfig, RewardConfig
from ..data.data_loader import MarketData
from .base_env import BaseTradingEnv


class CryptoTradingEnv(BaseTradingEnv):
    """Trading environment specialised for crypto assets."""

    market_name = "crypto"

    def __init__(
        self,
        data: MarketData,
        env_config: EnvConfig,
        reward_config: RewardConfig,
        random_start: bool = True,
    ) -> None:
        super().__init__(data, env_config, reward_config, random_start=random_start)
        # Extension point: add perpetual-swap funding payments, tiered maker/
        # taker fees, or 24/7 session handling here.
