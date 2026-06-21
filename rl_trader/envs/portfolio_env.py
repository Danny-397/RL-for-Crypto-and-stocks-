"""Cross-sectional portfolio trading environment.

Where :class:`~rl_trader.envs.base_env.BaseTradingEnv` sizes a position in *one*
asset, this environment lets the agent allocate across a **whole basket at once**:
at each step it sees every asset's features simultaneously and outputs a vector of
target weights. That is the cross-sectional setup real quant strategies use — go
long the assets that look strong, short the ones that look weak, size by conviction
— and it is a strict generalisation of the single-asset env (with ``N = 1`` the two
coincide).

Action semantics
----------------
The policy emits a raw vector ``a in [-1, 1]^N``. It is interpreted as target
**dollar weights** (fraction of equity per asset), subject to a gross-exposure
budget: if ``sum |a| > max_gross`` the vector is scaled down so leverage is capped;
otherwise it is used as-is and the unallocated remainder sits in cash. Long-only
baskets clamp the lower bound to 0. Keeping the weight map this simple (a linear
rescale, no softmax) preserves an exact Gaussian log-probability for PPO — the
same design choice the single-asset env makes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config.training_config import EnvConfig, RewardConfig
from ..data.portfolio_data import PortfolioData


class PortfolioTradingEnv(gym.Env):
    """A long/short, multi-asset portfolio environment with continuous weights."""

    metadata = {"render_modes": ["human"]}
    market_name = "portfolio"

    def __init__(
        self,
        data: PortfolioData,
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

        self.n_assets = data.n_assets
        self.n_features = data.n_features
        self.window = env_config.window_size
        self._max_t = len(data) - 1

        # Observation = flattened feature window for every asset, plus per-asset
        # current weights and two account scalars (cash fraction, normalised equity).
        obs_dim = self.window * self.n_assets * self.n_features + self.n_assets + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        low = -self.cfg.max_position if self.cfg.allow_short else 0.0
        self.action_space = spaces.Box(
            low=low, high=self.cfg.max_position, shape=(self.n_assets,), dtype=np.float32
        )

        self.t = 0
        self.cash = 0.0
        self.units = np.zeros(self.n_assets, dtype=np.float64)
        self.equity = 0.0
        self.peak_equity = 0.0

    def reload(self, data: PortfolioData) -> None:
        """Swap in a fresh basket of the same shape (domain randomization)."""
        if data.n_assets != self.n_assets or data.n_features != self.n_features:
            raise ValueError("reload() requires the same asset/feature dimensions.")
        if len(data) <= self.window + 1:
            raise ValueError("Reloaded series is too short for one window.")
        self.data = data
        self._max_t = len(data) - 1

    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        first = self.window - 1
        if self.random_start and self._max_t - first > 10:
            self.t = int(self.np_random.integers(first, self._max_t - 1))
        else:
            self.t = first
        self.cash = self.cfg.initial_balance
        self.units = np.zeros(self.n_assets, dtype=np.float64)
        self.equity = self.cfg.initial_balance
        self.peak_equity = self.cfg.initial_balance
        self._dsr_a = 0.0
        self._dsr_b = 0.0
        return self._get_observation(), self._info()

    def _target_weights(self, action: np.ndarray) -> np.ndarray:
        """Map a raw action vector to gross-exposure-capped target weights."""
        a = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float64)
        gross = np.abs(a).sum()
        if gross > self.cfg.max_position and gross > 1e-12:
            a = a * (self.cfg.max_position / gross)
        return a

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        prices = self.data.prices[self.t].astype(np.float64)
        prev_equity = self.equity

        weights = self._target_weights(np.asarray(action, dtype=np.float64))
        target_units = (weights * prev_equity) / prices
        traded_units = target_units - self.units
        trade_notional = float(np.abs(traded_units) @ prices)
        cost = trade_notional * (self.cfg.transaction_cost + self.cfg.slippage)

        self.cash -= float(traded_units @ prices) + cost
        self.units = target_units

        self.t += 1
        next_prices = self.data.prices[self.t].astype(np.float64)
        self.equity = self.cash + float(self.units @ next_prices)
        self.peak_equity = max(self.peak_equity, self.equity)

        reward = self._compute_reward(prev_equity, self.equity, trade_notional)
        terminated = self.equity <= self.cfg.bankruptcy_threshold * self.cfg.initial_balance
        truncated = self.t >= self._max_t
        return self._get_observation(), reward, terminated, truncated, self._info(cost)

    # ------------------------------------------------------------------ #
    def _compute_reward(self, prev_equity: float, new_equity: float, trade_notional: float) -> float:
        """Portfolio reward — identical formulation to the single-asset env."""
        eps = 1e-8
        if self.rcfg.use_log_return:
            ret = float(np.log((new_equity + eps) / (prev_equity + eps)))
        else:
            ret = (new_equity - prev_equity) / (prev_equity + eps)
        turnover = trade_notional / (prev_equity + eps)

        if self.rcfg.kind == "dsr":
            base = self._differential_sharpe(ret)
        else:
            drawdown = (self.peak_equity - new_equity) / (self.peak_equity + eps)
            base = self.rcfg.return_scale * ret - self.rcfg.drawdown_penalty * max(drawdown, 0.0)
        return base - self.rcfg.turnover_penalty * turnover

    def _differential_sharpe(self, ret: float) -> float:
        """Online Differential Sharpe Ratio (Moody & Saffell, 1998)."""
        eta = self.rcfg.dsr_eta
        a_prev, b_prev = self._dsr_a, self._dsr_b
        delta_a = ret - a_prev
        delta_b = ret * ret - b_prev
        variance = b_prev - a_prev * a_prev
        dsr = 0.0 if variance <= 1e-12 else (b_prev * delta_a - 0.5 * a_prev * delta_b) / (variance ** 1.5)
        self._dsr_a = a_prev + eta * delta_a
        self._dsr_b = b_prev + eta * delta_b
        return self.rcfg.dsr_scale * float(dsr)

    # ------------------------------------------------------------------ #
    def _get_observation(self) -> np.ndarray:
        window = self.data.features[self.t - self.window + 1 : self.t + 1]  # [w, N, F]
        prices = self.data.prices[self.t].astype(np.float64)
        weights = (self.units * prices) / (self.equity + 1e-8)  # current per-asset weights
        cash_fraction = self.cash / (self.cfg.initial_balance + 1e-8)
        equity_norm = self.equity / (self.cfg.initial_balance + 1e-8)
        account = np.concatenate([weights, [cash_fraction, equity_norm]]).astype(np.float32)
        return np.concatenate([window.flatten(), account]).astype(np.float32)

    def _info(self, cost: float = 0.0) -> dict:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "drawdown": (self.peak_equity - self.equity) / (self.peak_equity + 1e-8),
            "cost": cost,
            "t": self.t,
            "gross_exposure": float(
                np.abs(self.units * self.data.prices[self.t]).sum() / (self.equity + 1e-8)
            ),
        }

    def render(self) -> None:  # pragma: no cover - convenience only
        print(f"[portfolio] t={self.t} equity={self.equity:,.2f} cash={self.cash:,.2f}")
