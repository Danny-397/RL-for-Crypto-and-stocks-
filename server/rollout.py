"""Full-trace rollouts: run a policy through an environment and record everything.

The dashboard's older ``/api/live`` endpoint returned only an equity curve. The
lab needs the whole causal chain at every bar — *what the agent saw, what it
decided, what the environment paid it, and where that left the portfolio* — so
a visitor can scrub to any timestep and inspect the actual decision.

Every number here comes from stepping the real
:class:`~rl_trader.envs.base_env.BaseTradingEnv`. Nothing is simulated a second
time in the frontend and nothing is interpolated.

Two subtleties worth stating, because they determine whether the X-Ray panel is
honest:

**Decision timing.** ``BaseTradingEnv.step`` advances ``t`` *before* returning
its observation, so the observation handed back is already the *next* bar's.
Each record therefore snapshots the pre-step state (the bar the decision was
actually made on) before calling ``step``.

**Counterfactuals.** :func:`counterfactual` restores the exact environment state
at a chosen bar and replays it under a different action. That is a genuine
environment re-evaluation, not a prediction: the alternative action is scored on
price movement that had already happened, which is exactly what makes it a
*counterfactual* rather than a forecast. Callers must label it as such.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from rl_trader.evaluation.evaluate_agent import ANNUALISATION, compute_metrics


@dataclass
class StepRecord:
    """One bar of the causal chain: observation -> policy -> action -> reward."""

    t: int                          # index into the price/feature series
    step: int                       # 0-based position within this episode
    date: Optional[str]             # calendar date when the series carries one
    price: float
    action: float                   # policy's target position, in [-1, 1]
    position_before: float          # exposure as a fraction of equity, pre-trade
    position_after: float           # exposure as a fraction of equity, post-trade
    equity: float                   # portfolio value after the bar resolves
    cash: float
    units: float
    reward: float                   # reward the environment actually paid
    cost: float                     # transaction cost charged this bar
    drawdown: float                 # depth below the running equity peak
    value: Optional[float]          # critic estimate, when the head was exported


@dataclass
class Trace:
    """A complete episode: per-bar records plus headline scoring."""

    market: str
    steps: List[StepRecord] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    bench_curve: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    bench_metrics: Dict[str, float] = field(default_factory=dict)
    value_available: bool = False

    def to_dict(self) -> dict:
        return {
            "market": self.market,
            "steps": [asdict(s) for s in self.steps],
            "equity_curve": self.equity_curve,
            "bench_curve": self.bench_curve,
            "dates": self.dates,
            "metrics": self.metrics,
            "bench_metrics": self.bench_metrics,
            "value_available": self.value_available,
            "n_steps": len(self.steps),
        }


def _position_fraction(units: float, price: float, equity: float) -> float:
    return (units * price) / (equity + 1e-8)


def _snapshot(env) -> dict:
    """Capture the mutable episode state needed to rewind ``env`` exactly."""
    return {
        "t": env.t,
        "cash": env.cash,
        "units": env.units,
        "equity": env.equity,
        "peak_equity": env.peak_equity,
        "dsr_a": getattr(env, "_dsr_a", 0.0),
        "dsr_b": getattr(env, "_dsr_b", 0.0),
    }


def _restore(env, state: dict) -> None:
    """Rewind ``env`` to a state captured by :func:`_snapshot`."""
    env.t = state["t"]
    env.cash = state["cash"]
    env.units = state["units"]
    env.equity = state["equity"]
    env.peak_equity = state["peak_equity"]
    env._dsr_a = state["dsr_a"]
    env._dsr_b = state["dsr_b"]


def run_trace(
    policy,
    env,
    market: str = "stock",
    dates: Optional[Sequence[str]] = None,
    initial_balance: Optional[float] = None,
) -> Trace:
    """Run ``policy`` deterministically through ``env``, recording every bar.

    ``dates`` (optional) should align with the environment's *full* price series;
    it is indexed by each record's ``t`` so labels stay attached to the right bar
    after any downsampling the frontend applies.
    """
    obs, info = env.reset()
    trace = Trace(market=market, value_available=bool(getattr(policy, "has_value", False)))
    equity_curve: List[float] = [info["equity"]]
    start_balance = initial_balance if initial_balance is not None else info["equity"]

    def _date_at(idx: int) -> Optional[str]:
        if dates is None or idx >= len(dates):
            return None
        return str(dates[idx])

    step_i = 0
    done = False
    first_t = env.t
    while not done:
        # Snapshot the bar the decision is actually made on, before step() moves t.
        t_decision = env.t
        price_at_decision = float(env.data.prices[t_decision])
        equity_before = env.equity
        position_before = _position_fraction(env.units, price_at_decision, equity_before)

        out = policy.evaluate(obs)
        obs, reward, terminated, truncated, info = env.step(
            np.array([out.action], dtype=np.float32)
        )

        trace.steps.append(
            StepRecord(
                t=t_decision,
                step=step_i,
                date=_date_at(t_decision),
                price=round(price_at_decision, 6),
                action=round(out.action, 6),
                position_before=round(position_before, 6),
                # After rebalancing, exposure equals the target the policy asked for.
                position_after=round(out.action, 6),
                equity=round(float(info["equity"]), 4),
                cash=round(float(info["cash"]), 4),
                units=round(float(info["units"]), 8),
                reward=round(float(reward), 8),
                cost=round(float(info["cost"]), 6),
                drawdown=round(float(info["drawdown"]), 6),
                value=round(out.value, 6) if out.value is not None else None,
            )
        )
        equity_curve.append(float(info["equity"]))
        step_i += 1
        done = terminated or truncated

    # Buy-&-hold over the identical window, from the same starting capital.
    last_t = trace.steps[-1].t + 1 if trace.steps else first_t
    prices = np.asarray(env.data.prices[first_t: last_t + 1], dtype=float)
    bench = start_balance * (prices / prices[0]) if len(prices) else np.array([start_balance])

    periods = ANNUALISATION.get(market, 252)
    equity_arr = np.asarray(equity_curve, dtype=float)

    trace.equity_curve = [round(float(v), 4) for v in equity_arr]
    trace.bench_curve = [round(float(v), 4) for v in bench]
    trace.dates = [d for d in (_date_at(i) for i in range(first_t, last_t + 1)) if d is not None]
    trace.metrics = {k: round(float(v), 6) for k, v in compute_metrics(equity_arr, periods).items()}
    trace.bench_metrics = {
        k: round(float(v), 6) for k, v in compute_metrics(bench, periods).items()
    }
    return trace


def observation_detail(env, t: int, feature_names: Sequence[str], window: int) -> dict:
    """Reconstruct the full observation the agent saw at bar ``t``.

    The policy consumes ``window x n_features`` scaled market features flattened
    together with three account scalars. The X-Ray panel shows the *newest* bar's
    features by default (the row that changed); this returns the entire window so
    a visitor can inspect the real 563-dimensional input rather than a summary.
    """
    lo = max(0, t - window + 1)
    win = np.asarray(env.data.features[lo: t + 1], dtype=float)
    return {
        "t": int(t),
        "window": int(window),
        "feature_names": list(feature_names),
        # newest bar first is how a trader reads it
        "current": [round(float(v), 6) for v in win[-1]] if len(win) else [],
        "window_values": [[round(float(v), 6) for v in row] for row in win],
        "obs_dim": int(window * len(feature_names) + 3),
    }


def counterfactual(
    policy,
    env,
    target_step: int,
    actions: Sequence[float],
    market: str = "stock",
    horizon: int = 1,
    dates: Optional[Sequence[str]] = None,
) -> dict:
    """Re-evaluate one bar under alternative actions, holding each for ``horizon`` bars.

    The episode is replayed under the policy up to ``target_step``; the state is
    then snapshotted and each candidate action is applied to that identical
    state, held for ``horizon`` bars, and scored on the price path that actually
    occurred. This measures *what the environment would have paid* for a
    different choice — it does not imply the agent could have known the outcome.
    """
    obs, info = env.reset()
    # Replay the policy's own decisions up to the bar of interest.
    for _ in range(target_step):
        out = policy.evaluate(obs)
        obs, _r, term, trunc, info = env.step(np.array([out.action], dtype=np.float32))
        if term or trunc:
            break

    state = _snapshot(env)
    obs_at_target = obs
    t_decision = env.t
    agent_out = policy.evaluate(obs_at_target)
    equity_before = env.equity
    horizon = max(1, int(horizon))

    results = []
    for a in actions:
        _restore(env, state)
        obs_local = obs_at_target
        total_reward = 0.0
        peak = env.equity
        trough_dd = 0.0
        steps_taken = 0
        for _ in range(horizon):
            obs_local, reward, term, trunc, info_local = env.step(
                np.array([float(a)], dtype=np.float32)
            )
            total_reward += float(reward)
            peak = max(peak, float(info_local["equity"]))
            trough_dd = max(trough_dd, (peak - float(info_local["equity"])) / (peak + 1e-8))
            steps_taken += 1
            if term or trunc:
                break
        end_equity = float(env.equity)
        results.append(
            {
                "action": round(float(a), 6),
                # Compared at the precision the API actually reports actions to,
                # so a candidate echoing a value read back from a trace still matches.
                "is_agent_action": bool(abs(float(a) - agent_out.action) < 1e-6),
                "end_equity": round(end_equity, 4),
                "equity_change": round(end_equity - equity_before, 4),
                "return": round((end_equity - equity_before) / (equity_before + 1e-8), 6),
                "reward": round(total_reward, 8),
                "max_drawdown": round(trough_dd, 6),
                "steps": steps_taken,
            }
        )

    _restore(env, state)
    return {
        "market": market,
        "step": int(target_step),
        "t": int(t_decision),
        "date": (str(dates[t_decision]) if dates is not None and t_decision < len(dates) else None),
        "price": round(float(env.data.prices[t_decision]), 6),
        "equity_before": round(equity_before, 4),
        "agent_action": round(agent_out.action, 6),
        "agent_value": round(agent_out.value, 6) if agent_out.value is not None else None,
        "horizon": horizon,
        "candidates": results,
        "note": (
            "Environment counterfactual: each action is replayed from the identical "
            "state on price movement that already occurred. It does not imply the "
            "agent could have known the outcome."
        ),
    }
