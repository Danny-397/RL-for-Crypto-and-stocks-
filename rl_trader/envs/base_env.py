"""Shared trading-environment logic.

:class:`BaseTradingEnv` implements all market-agnostic mechanics — observation
construction, order execution with costs/slippage, portfolio accounting, the
shared reward function, and episode termination. Market-specific subclasses
(:class:`StockTradingEnv`, :class:`CryptoTradingEnv`) only customise defaults
and metadata, which keeps behaviour consistent and bugs in one place.

Action semantics
----------------
A single continuous action ``a in [-max_position, max_position]`` is the
*target* position expressed as a fraction of current equity:

    a = +1.0  -> go fully long  (100% of equity in the asset)
    a =  0.0  -> hold only cash (flat)
    a = -1.0  -> go fully short (if shorting is permitted)

Targeting a *position* (rather than incremental buy/sell actions) gives the
agent direct, stable control over exposure and makes position sizing explicit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config.training_config import EnvConfig, RewardConfig
from ..data.data_loader import MarketData


class BaseTradingEnv(gym.Env):
    """A single-asset long/short trading environment with continuous sizing."""

    metadata = {"render_modes": ["human"]}
    market_name: str = "base"

    def __init__(
        self,
        data: MarketData,
        env_config: EnvConfig,
        reward_config: RewardConfig,
        random_start: bool = True,
    ) -> None:
        super().__init__()
        if len(data) <= env_config.window_size + 1:
            raise ValueError("Not enough data for even one full observation window.")

        self.data = data
        self.cfg = env_config
        self.rcfg = reward_config
        self.random_start = random_start

        self.n_features = data.features.shape[1]
        self.window = env_config.window_size
        self._max_t = len(data) - 1  # last valid index into the price series

        # Observation = flattened feature window + 3 account-state scalars
        # (position fraction, cash fraction, normalised equity).
        obs_dim = self.window * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        low = -self.cfg.max_position if self.cfg.allow_short else 0.0
        self.action_space = spaces.Box(
            low=low, high=self.cfg.max_position, shape=(1,), dtype=np.float32
        )

        # Episode state (initialised in reset()).
        self.t: int = 0
        self.cash: float = 0.0
        self.units: float = 0.0  # asset units held (negative => short)
        self.equity: float = 0.0
        self.peak_equity: float = 0.0

    # ------------------------------------------------------------------ #
    # Core gym API                                                       #
    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Start far enough in that a full window exists behind the pointer.
        first = self.window - 1
        if self.random_start and self._max_t - first > 10:
            # Leave room so episodes are not trivially short.
            self.t = int(self.np_random.integers(first, self._max_t - 1))
        else:
            self.t = first

        self.cash = self.cfg.initial_balance
        self.units = 0.0
        self.equity = self.cfg.initial_balance
        self.peak_equity = self.cfg.initial_balance
        return self._get_observation(), self._info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = float(np.clip(action, self.action_space.low, self.action_space.high)[0])

        price = float(self.data.prices[self.t])
        prev_equity = self.equity

        # --- Rebalance to the target position, paying costs on the traded notional.
        target_value = action * prev_equity
        target_units = target_value / price
        traded_units = target_units - self.units
        trade_notional = abs(traded_units) * price
        cost = trade_notional * (self.cfg.transaction_cost + self.cfg.slippage)

        self.cash -= traded_units * price + cost
        self.units = target_units

        # --- Advance one bar and mark the portfolio to the new price.
        self.t += 1
        next_price = float(self.data.prices[self.t])
        self.equity = self.cash + self.units * next_price
        self.peak_equity = max(self.peak_equity, self.equity)

        reward = self._compute_reward(prev_equity, self.equity, cost, trade_notional)

        terminated = self.equity <= self.cfg.bankruptcy_threshold * self.cfg.initial_balance
        truncated = self.t >= self._max_t

        return self._get_observation(), reward, terminated, truncated, self._info(cost)

    # ------------------------------------------------------------------ #
    # Reward                                                             #
    # ------------------------------------------------------------------ #
    def _compute_reward(
        self, prev_equity: float, new_equity: float, cost: float, trade_notional: float
    ) -> float:
        """Shared reward: risk-aware return net of friction.

        reward = return  -  drawdown_penalty * drawdown  -  turnover_penalty * turnover

        * return is log or simple, per config (log returns are additive across
          time, which pairs naturally with discounted RL objectives).
        * drawdown is the current depth below the equity high-water mark.
        * turnover penalises churn beyond the explicit cash cost, nudging the
          agent away from noise-trading.
        """
        eps = 1e-8
        if self.rcfg.use_log_return:
            ret = float(np.log((new_equity + eps) / (prev_equity + eps)))
        else:
            ret = (new_equity - prev_equity) / (prev_equity + eps)

        drawdown = (self.peak_equity - new_equity) / (self.peak_equity + eps)
        turnover = trade_notional / (prev_equity + eps)

        return (
            ret
            - self.rcfg.drawdown_penalty * max(drawdown, 0.0)
            - self.rcfg.turnover_penalty * turnover
        )

    # ------------------------------------------------------------------ #
    # Observation + info                                                 #
    # ------------------------------------------------------------------ #
    def _get_observation(self) -> np.ndarray:
        window = self.data.features[self.t - self.window + 1 : self.t + 1]
        price = float(self.data.prices[self.t])
        position_fraction = (self.units * price) / (self.equity + 1e-8)
        cash_fraction = self.cash / (self.cfg.initial_balance + 1e-8)
        equity_norm = self.equity / (self.cfg.initial_balance + 1e-8)
        account_state = np.array(
            [position_fraction, cash_fraction, equity_norm], dtype=np.float32
        )
        return np.concatenate([window.flatten(), account_state]).astype(np.float32)

    def _info(self, cost: float = 0.0) -> dict:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "units": self.units,
            "price": float(self.data.prices[self.t]),
            "drawdown": (self.peak_equity - self.equity) / (self.peak_equity + 1e-8),
            "cost": cost,
            "t": self.t,
        }

    def render(self) -> None:  # pragma: no cover - convenience only
        print(
            f"[{self.market_name}] t={self.t} equity={self.equity:,.2f} "
            f"units={self.units:.4f} cash={self.cash:,.2f}"
        )
