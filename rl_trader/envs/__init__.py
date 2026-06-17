"""Gymnasium trading environments."""

from .base_env import BaseTradingEnv
from .crypto_env import CryptoTradingEnv
from .stock_env import StockTradingEnv


def make_env(market: str, data, env_config, reward_config, random_start: bool = True):
    """Factory: build the right environment for a market name.

    Centralising construction here means training/evaluation code never needs
    to branch on the market — adding a new market (e.g. FX) is a one-line edit.
    """
    market = market.lower()
    if market == "stock":
        return StockTradingEnv(data, env_config, reward_config, random_start=random_start)
    if market == "crypto":
        return CryptoTradingEnv(data, env_config, reward_config, random_start=random_start)
    raise ValueError(f"Unknown market {market!r}; expected 'stock' or 'crypto'.")


__all__ = ["BaseTradingEnv", "StockTradingEnv", "CryptoTradingEnv", "make_env"]
