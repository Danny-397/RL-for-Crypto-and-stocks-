"""Hedging a short option with trading costs: pricing, paths, a batched environment, baselines.

The problem
-----------
You have sold one European call (strike ``K``, ``n_steps`` trading days to expiry)
and collected its price. Each day you choose how many shares to hold against it,
``h`` in [0, 1] per option. Holding the Black-Scholes delta makes the position
risk-free in continuous time with free trading; with a proportional trading cost
``c`` (a share of the notional traded), rebalancing every day is expensive and the
best policy is unknown in closed form. Theory says it is a *no-trade band* around
delta that widens as costs rise (Whalley & Wilmott 1997). That is the problem deep
hedging (Buehler et al. 2019) and RL hedging (Kolm & Ritter 2019) attack, and it
has two built-in checks: with ``c = 0`` the best hedge is delta, and with ``c > 0``
the best hedge trades less.

Units: the underlying starts at 1.0, so P&L is a fraction of the notional. Every
array is batched over paths, so many episodes step together. NumPy only, so the
same code runs on the server, which has no PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

TRADING_DAYS = 252


# ── Black-Scholes, vectorized ─────────────────────────────────────────────────

def norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via erf (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7)."""
    x = np.asarray(x, dtype=np.float64)
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def _d1(S, K, tau, sigma, r):
    sq = sigma * np.sqrt(tau)
    return (np.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / sq, sq


def _prep(S, K, tau, sigma):
    arrs = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (S, K, tau, sigma)))
    shape = arrs[0].shape
    return shape, [np.atleast_1d(a).ravel() for a in arrs]


def bs_call(S, K, tau, sigma, r: float = 0.0) -> np.ndarray:
    """European call value; intrinsic value at ``tau <= 0``."""
    shape, (S, K, tau, sigma) = _prep(S, K, tau, sigma)
    out = np.maximum(S - K, 0.0)
    live = tau > 0
    if np.any(live):
        d1, sq = _d1(S[live], K[live], tau[live], sigma[live], r)
        out[live] = S[live] * norm_cdf(d1) - K[live] * np.exp(-r * tau[live]) * norm_cdf(d1 - sq)
    return out.reshape(shape)


def bs_delta(S, K, tau, sigma, r: float = 0.0) -> np.ndarray:
    """Call delta; at ``tau <= 0``, 1 in the money and 0 out of it."""
    shape, (S, K, tau, sigma) = _prep(S, K, tau, sigma)
    out = (S > K).astype(np.float64)
    live = tau > 0
    if np.any(live):
        d1, _ = _d1(S[live], K[live], tau[live], sigma[live], r)
        out[live] = norm_cdf(d1)
    return out.reshape(shape)


# ── The specification ─────────────────────────────────────────────────────────

@dataclass
class HedgeSpec:
    """One hedging problem. ``cost`` is the proportional cost per unit of notional traded."""

    n_steps: int = 21                 # trading days to expiry (one month)
    strike: float = 1.0               # at the money
    cost: float = 0.0                 # e.g. 0.002 = 0.2% of the notional traded
    risk_aversion: float = 1.0        # kappa in reward = x - kappa/2 * x^2, x in % of notional
    vol_range: Tuple[float, float] = (0.10, 0.45)   # training paths draw a volatility from this
    drift: float = 0.0                # annual drift of simulated paths
    rate: float = 0.0
    close_at_expiry: bool = True      # the hedge is sold at expiry (pays the cost once more)

    @property
    def dt(self) -> float:
        return 1.0 / TRADING_DAYS


# ── Paths ─────────────────────────────────────────────────────────────────────

def simulate_paths(spec: HedgeSpec, n_paths: int, rng: np.random.Generator,
                   vol: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Geometric Brownian paths from 1.0, each with its own volatility.

    Returns ``(S, sigma)`` with shape ``(n_paths, n_steps + 1)``: the prices, and the
    volatility the option is marked at each day (the path's own, constant).
    Drawing a new volatility per path is domain randomization: the policy has to
    work across regimes, not memorise one.
    """
    if vol is None:
        vol = rng.uniform(*spec.vol_range, size=n_paths)
    vol = np.asarray(vol, dtype=np.float64).reshape(-1)
    z = rng.standard_normal((n_paths, spec.n_steps))
    step = (spec.drift - 0.5 * vol[:, None] ** 2) * spec.dt + vol[:, None] * np.sqrt(spec.dt) * z
    S = np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(step, axis=1)], axis=1))
    sigma = np.repeat(vol[:, None], spec.n_steps + 1, axis=1)
    return S, sigma


# ── The environment ───────────────────────────────────────────────────────────

OBS_NAMES = ('log moneyness ln(S/K)', 'time left (fraction of the option life)', 'implied volatility',
             'current hedge')
OBS_DIM = len(OBS_NAMES)


class VecHedgeEnv:
    """A batch of hedging episodes stepping together.

    Observation per path: ``[ln(S/K), time left / life, sigma, current hedge]``; it
    does *not* include delta, so a policy that hedges at delta has learned it.
    Action: the new hedge in [0, 1] (the policy's [-1, 1] output is mapped by
    :func:`action_to_hedge`). Each step books, per path,

        dw = h * (S[t+1] - S[t]) - (V[t+1] - V[t]) - cost * |h - h_prev| * S[t]

    with ``V`` the call marked at that day's volatility (its payoff at expiry), plus
    the cost of selling the hedge at expiry. The episode's sum of ``dw`` is the
    hedged P&L beyond the premium already collected. The reward is
    ``x - risk_aversion / 2 * x**2`` with ``x = 100 * dw`` (percent of notional).
    """

    def __init__(self, S: np.ndarray, sigma: np.ndarray, spec: HedgeSpec, V: Optional[np.ndarray] = None):
        S = np.asarray(S, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        if S.shape != sigma.shape or S.shape[1] != spec.n_steps + 1:
            raise ValueError('S and sigma must both be (n_paths, n_steps + 1)')
        self.S, self.sigma, self.spec = S, sigma, spec
        self.n = S.shape[0]
        tau = (spec.n_steps - np.arange(spec.n_steps + 1)) * spec.dt
        self.tau = np.broadcast_to(tau, S.shape)
        # The option is marked at Black-Scholes unless real prices are supplied
        # (tools/hedging_quotes.py passes market mid quotes, in units of the strike).
        if V is None:
            self.V = bs_call(S, spec.strike, self.tau, sigma, spec.rate)
        else:
            self.V = np.asarray(V, dtype=np.float64)
            if self.V.shape != S.shape:
                raise ValueError('V must have the shape of S')
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.h = np.zeros(self.n)
        self.pnl = np.zeros(self.n)
        self.costs = np.zeros(self.n)
        self.turnover = np.zeros(self.n)
        return self.obs()

    def obs(self) -> np.ndarray:
        t = self.t
        return np.stack([np.log(self.S[:, t] / self.spec.strike),
                         np.full(self.n, (self.spec.n_steps - t) / self.spec.n_steps),
                         self.sigma[:, t], self.h], axis=1).astype(np.float32)

    @property
    def delta(self) -> np.ndarray:
        """Black-Scholes delta now (for the baselines, not the policy)."""
        t = self.t
        return bs_delta(self.S[:, t], self.spec.strike, self.tau[:, t], self.sigma[:, t], self.spec.rate)

    def step(self, hedge: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        spec, t = self.spec, self.t
        h = np.clip(np.asarray(hedge, dtype=np.float64).reshape(-1), 0.0, 1.0)
        traded = np.abs(h - self.h)
        cost = spec.cost * traded * self.S[:, t]
        dw = h * (self.S[:, t + 1] - self.S[:, t]) - (self.V[:, t + 1] - self.V[:, t]) - cost
        self.turnover += traded
        self.h = h
        self.t += 1
        done = self.t >= spec.n_steps
        if done and spec.close_at_expiry:
            close = spec.cost * self.h * self.S[:, self.t]
            dw = dw - close
            cost = cost + close
            self.turnover += self.h
        self.costs += cost
        self.pnl += dw
        x = 100.0 * dw
        reward = x - 0.5 * spec.risk_aversion * x * x
        return (self.obs() if not done else np.zeros((self.n, OBS_DIM), np.float32)), reward, done, \
            {'dw': dw, 'cost': cost}


def action_to_hedge(a: np.ndarray) -> np.ndarray:
    """Map the policy's [-1, 1] output to a hedge in [0, 1]."""
    return np.clip((np.asarray(a, dtype=np.float64) + 1.0) / 2.0, 0.0, 1.0)


def hedge_to_action(h: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(h, dtype=np.float64) - 1.0


# ── Baselines ─────────────────────────────────────────────────────────────────

def run_policy(env: VecHedgeEnv, policy) -> Dict[str, np.ndarray]:
    """Run ``policy(env) -> hedges`` over every path; return per-path results."""
    env.reset()
    rewards = np.zeros(env.n)
    hedges = []
    done = False
    while not done:
        h = np.clip(policy(env), 0.0, 1.0)
        hedges.append(h)
        _, r, done, _ = env.step(h)
        rewards += r
    return {'pnl': env.pnl.copy(), 'cost': env.costs.copy(), 'turnover': env.turnover.copy(),
            'reward': rewards, 'hedges': np.stack(hedges, axis=1)}


def no_hedge(env: VecHedgeEnv) -> np.ndarray:
    return np.zeros(env.n)


def delta_hedge(env: VecHedgeEnv) -> np.ndarray:
    return env.delta


def band_hedge(width: float):
    """Delta with a no-trade band: trade back to the band's edge only when the hedge leaves it.

    ``width`` is the half-width in units of delta. ``width = 0`` is daily delta hedging.
    """
    def policy(env: VecHedgeEnv) -> np.ndarray:
        d = env.delta
        if env.t == 0:
            return d
        return np.clip(env.h, d - width, d + width)
    policy.width = width
    return policy


def summarize(res: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Risk and cost of a set of hedged episodes, in percent of notional."""
    pnl = 100.0 * res['pnl']
    loss = -pnl
    tail = np.sort(loss)[-max(1, int(np.ceil(0.05 * len(loss)))):]
    return {'mean': float(pnl.mean()), 'std': float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
            'cvar95': float(tail.mean()),              # average of the worst 5% of losses
            'cost': float(100.0 * res['cost'].mean()), 'turnover': float(res['turnover'].mean()),
            'objective': float(res['reward'].mean()), 'n': int(len(pnl))}
