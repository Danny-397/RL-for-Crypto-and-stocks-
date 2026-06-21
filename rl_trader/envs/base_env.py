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

    def reload(self, data: MarketData) -> None:
        """Swap in a fresh price/feature series (same feature dimension).

        Used for **domain randomization** during training: by resampling a new
        synthetic path between episodes, the agent is forced to learn a
        *generalizable* policy rather than memorising one historical sequence —
        the single most effective overfitting control for RL trading.
        """
        if data.features.shape[1] != self.n_features:
            raise ValueError("reload() requires the same number of features.")
        if len(data) <= self.window + 1:
            raise ValueError("Reloaded series is too short for one window.")
        self.data = data
        self._max_t = len(data) - 1

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
        # Differential Sharpe Ratio running moments (first/second moment of the
        # per-step return). Reset each episode so the estimate is path-local.
        self._dsr_a = 0.0
        self._dsr_b = 0.0
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
        """Shared reward. Dispatches on ``reward.kind`` (see :class:`RewardConfig`).

        Both formulations subtract a turnover term so churn beyond the explicit
        cash cost is discouraged.
        """
        eps = 1e-8
        if self.rcfg.use_log_return:
            ret = float(np.log((new_equity + eps) / (prev_equity + eps)))
        else:
            ret = (new_equity - prev_equity) / (prev_equity + eps)
        turnover = trade_notional / (prev_equity + eps)

        if self.rcfg.kind == "dsr":
            base = self._differential_sharpe(ret)
        else:
            # Risk-aware net return: scaled return minus drawdown depth below the
            # equity high-water mark. ``return_scale`` keeps return the dominant
            # term so the agent optimises for profit, not merely avoiding losses.
            drawdown = (self.peak_equity - new_equity) / (self.peak_equity + eps)
            base = (
                self.rcfg.return_scale * ret
                - self.rcfg.drawdown_penalty * max(drawdown, 0.0)
            )

        return base - self.rcfg.turnover_penalty * turnover

    def _differential_sharpe(self, ret: float) -> float:
        """Differential Sharpe Ratio reward (Moody & Saffell, 1998).

        Maintains exponentially-weighted estimates of the first moment ``A`` and
        second moment ``B`` of the per-step return, then returns the marginal
        contribution of the latest return to the Sharpe ratio:

            D_t = (B_{t-1}·ΔA − ½·A_{t-1}·ΔB) / (B_{t-1} − A_{t-1}²)^{3/2}

        with ΔA = R_t − A_{t-1}, ΔB = R_t² − B_{t-1}. Rewarding ``D_t`` each step
        makes the *cumulative* reward track the Sharpe ratio, so the policy is
        trained to maximise risk-adjusted — not raw — return.
        """
        eta = self.rcfg.dsr_eta
        a_prev, b_prev = self._dsr_a, self._dsr_b
        delta_a = ret - a_prev
        delta_b = ret * ret - b_prev

        variance = b_prev - a_prev * a_prev
        if variance <= 1e-12:
            dsr = 0.0  # not enough dispersion yet to define a Sharpe ratio
        else:
            dsr = (b_prev * delta_a - 0.5 * a_prev * delta_b) / (variance ** 1.5)

        # Update the running moments for the next step.
        self._dsr_a = a_prev + eta * delta_a
        self._dsr_b = b_prev + eta * delta_b
        return self.rcfg.dsr_scale * float(dsr)

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
