"""Trade the same bars the agent trades — as a *baseline*, not a game.

This exists for one reason: the site keeps saying that beating buy-and-hold is
hard and that a good-looking backtest is usually luck. A visitor can read that
and still assume they personally would have done better. So let them try, on the
identical series, through the identical environment, paying the identical costs.

No lookahead, actually enforced
-------------------------------
The series lives on the server. A session hands out the warm-up window at the
start and then exactly **one new bar per decision** — the client is never sent a
price it has not already traded through. That is a property of the protocol, not
a promise in the copy: there is nothing in the page to read ahead in.

What is and is not being compared
---------------------------------
The human sees a price chart and their own account. The agent sees 28
standardised indicators over a 20-bar window plus three account scalars — 563
numbers. **These are not the same information, and this is not a like-for-like
contest.** The response says so every time.

The comparison that *is* fair is the third line: **buy-and-hold**, over the same
bars, through the same cost model, for both. That is the benchmark the whole
project is about, and the usual outcome — that neither the visitor nor the agent
clears it — is the finding, not a disappointment.

Ephemeral by construction
-------------------------
Sessions live in memory, capped and time-limited, exactly like experiments. A
restart loses them, and the API says so rather than implying anything durable.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from rl_trader.evaluation.evaluate_agent import ANNUALISATION, compute_metrics

MAX_SESSIONS = 60
SESSION_TTL = 1800.0        # 30 minutes of inactivity
DEFAULT_STEPS = 60
MIN_STEPS, MAX_STEPS = 10, 150

INFORMATION_NOTE = (
    "You see the price and your own account. The agent sees 28 standardised "
    "indicators over a 20-bar window plus three account scalars — 563 numbers. "
    "This is deliberately not a like-for-like comparison of skill. The line worth "
    "watching is buy-and-hold: it is the benchmark both of you are measured "
    "against, through the same transaction costs."
)

STORAGE_NOTE = (
    "ephemeral — sessions live in memory, expire after 30 minutes of inactivity, "
    "and are lost on restart"
)


@dataclass
class Session:
    """One person's run through a fixed series, with the environment as referee."""

    id: str
    config: Dict[str, Any]
    env: Any
    cfg_obj: Any
    dates: Optional[List[str]]
    meta: Dict[str, Any]
    max_steps: int
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    actions: List[float] = field(default_factory=list)
    equity: List[float] = field(default_factory=list)
    revealed: int = 0            # how many bars the client has been shown
    done: bool = False
    result: Optional[Dict[str, Any]] = None

    @property
    def market(self) -> str:
        return self.config["market"]

    def touch(self) -> None:
        self.last_seen = time.time()

    def prices_upto(self, n: int) -> List[float]:
        arr = np.asarray(self.env.data.prices[:n], dtype=float)
        return [round(float(v), 6) for v in arr]

    def account(self, info: Dict[str, Any]) -> Dict[str, Any]:
        price = float(self.env.data.prices[min(self.env.t, len(self.env.data.prices) - 1)])
        equity = float(info.get("equity", self.env.equity))
        return {
            "equity": round(equity, 2),
            "cash": round(float(self.env.cash), 2),
            "units": round(float(self.env.units), 6),
            "position_fraction": round((float(self.env.units) * price) / (equity + 1e-8), 4),
            "return_so_far": round(equity / self.cfg_obj.env.initial_balance - 1.0, 6),
        }


class SessionStore:
    """Thread-safe, bounded, self-expiring registry of human sessions."""

    def __init__(self, max_sessions: int = MAX_SESSIONS, ttl: float = SESSION_TTL) -> None:
        self._lock = threading.Lock()
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._max = max_sessions
        self._ttl = ttl

    def _new_id(self) -> str:
        while True:
            candidate = "HUM-" + secrets.token_hex(3).upper()[:5]
            if candidate not in self._sessions:
                return candidate

    def _evict_locked(self) -> None:
        now = time.time()
        for key in [k for k, s in self._sessions.items() if now - s.last_seen > self._ttl]:
            del self._sessions[key]
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)

    def add(self, session: Session) -> Session:
        with self._lock:
            session.id = self._new_id()
            self._sessions[session.id] = session
            self._evict_locked()
        return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session.last_seen > self._ttl:
                del self._sessions[session_id]
                return None
            session.touch()
            return session

    def stats(self) -> dict:
        with self._lock:
            return {"active": len(self._sessions), "capacity": self._max,
                    "ttl_seconds": int(self._ttl), "storage": STORAGE_NOTE}


def start(env, cfg_obj, dates, meta, config: Dict[str, Any], max_steps: int) -> Session:
    """Open a session at the first tradeable bar and reveal only the warm-up."""
    max_steps = max(MIN_STEPS, min(MAX_STEPS, int(max_steps)))
    _obs, info = env.reset()
    session = Session(
        id="", config=config, env=env, cfg_obj=cfg_obj, dates=dates, meta=meta,
        max_steps=max_steps,
    )
    # After reset the environment sits at t = window - 1, so the bars up to and
    # including it are already "history" — that, and nothing beyond it, is what
    # the client may see.
    session.revealed = env.t + 1
    session.equity = [float(info["equity"])]
    available = len(env.data.prices) - session.revealed
    if available < MIN_STEPS:
        raise ValueError(
            f"only {available} tradeable bars in this series — need at least {MIN_STEPS}"
        )
    session.max_steps = min(max_steps, available)
    return session


def opening(session: Session) -> dict:
    """The payload that starts a run: history so far, and nothing after it."""
    return {
        "session_id": session.id,
        "market": session.market,
        "config": session.config,
        "step": 0,
        "max_steps": session.max_steps,
        "prices": session.prices_upto(session.revealed),
        "dates": session.dates[:session.revealed] if session.dates else None,
        "initial_balance": session.cfg_obj.env.initial_balance,
        "transaction_cost": session.cfg_obj.env.transaction_cost,
        "slippage": session.cfg_obj.env.slippage,
        "allow_short": session.cfg_obj.env.allow_short,
        "account": session.account({"equity": session.equity[0]}),
        "meta": session.meta,
        "information_note": INFORMATION_NOTE,
        "storage": STORAGE_NOTE,
        "lookahead_note": (
            "The price series is held on the server. You are sent exactly one new "
            "bar per decision, so there is nothing here to read ahead in."
        ),
    }


def step(session: Session, action: float) -> dict:
    """Apply one decision and reveal exactly one new bar."""
    if session.done:
        raise ValueError("this session has already finished")
    action = float(max(-1.0, min(1.0, float(action))))
    if not session.cfg_obj.env.allow_short:
        action = max(0.0, action)

    _obs, reward, term, trunc, info = session.env.step(
        np.array([action], dtype=np.float32)
    )
    session.actions.append(action)
    session.equity.append(float(info["equity"]))
    session.revealed = min(session.env.t + 1, len(session.env.data.prices))

    at_limit = len(session.actions) >= session.max_steps
    session.done = bool(term or trunc or at_limit)

    idx = session.revealed - 1
    return {
        "step": len(session.actions),
        "max_steps": session.max_steps,
        "action": round(action, 4),
        "price": round(float(session.env.data.prices[idx]), 6),
        "date": session.dates[idx] if session.dates and idx < len(session.dates) else None,
        "reward": round(float(reward), 6),
        "cost_paid": round(float(info.get("cost", 0.0)), 4),
        "account": session.account(info),
        "done": session.done,
        "reason": ("episode ended" if (term or trunc) else
                   "reached the decision limit" if at_limit else None),
    }


def _metrics(equity: List[float], market: str) -> dict:
    periods = ANNUALISATION.get(market, 252)
    out = compute_metrics(np.asarray(equity, dtype=float), periods)
    return {k: round(float(v), 6) for k, v in out.items()}


def finish(session: Session, policy, build_env) -> dict:
    """Score the run against the deployed agent and buy-and-hold on the same bars.

    ``build_env`` rebuilds an identical environment from the session's config, so
    the agent trades the *same* prices under the *same* costs. It is run for
    exactly as many bars as the human traded — comparing a 60-decision run against
    a 600-bar one would be measuring the series, not the decisions.
    """
    n = len(session.actions)
    if n == 0:
        raise ValueError("no decisions were made in this session")

    env = build_env()
    _obs, info = env.reset()
    agent_equity = [float(info["equity"])]
    agent_actions: List[float] = []
    for _ in range(n):
        a = float(policy.act(_obs))
        _obs, _r, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        agent_equity.append(float(info["equity"]))
        agent_actions.append(a)
        if term or trunc:
            break

    # Buy-and-hold over exactly the bars that were traded, from the same capital.
    # ``revealed`` is one past the current bar, so backing out n decisions lands
    # on the bar where the first decision was made.
    start_idx = int(max(0, session.revealed - n - 1))
    prices = np.asarray(
        session.env.data.prices[start_idx: start_idx + len(agent_equity)], dtype=float
    )
    bench = session.cfg_obj.env.initial_balance * (prices / prices[0])

    human = _metrics(session.equity, session.market)
    agent = _metrics(agent_equity, session.market)
    benchmark = _metrics(list(bench), session.market)

    session.done = True
    session.result = {
        "session_id": session.id,
        "bars_traded": n,
        "market": session.market,
        "config": session.config,
        "you": {
            "metrics": human,
            "equity_curve": [round(v, 2) for v in session.equity],
            "mean_position": round(float(np.mean(session.actions)), 4),
            "trades": int(np.count_nonzero(np.diff([0.0] + session.actions))),
        },
        "agent": {
            "metrics": agent,
            "equity_curve": [round(v, 2) for v in agent_equity],
            "mean_position": round(float(np.mean(agent_actions)), 4) if agent_actions else 0.0,
            "policy": policy.describe(),
        },
        "benchmark": {
            "metrics": benchmark,
            "equity_curve": [round(float(v), 2) for v in bench],
            "label": "buy & hold, same bars, same costs on entry",
        },
        "first_bar_index": start_idx,
        "you_beat_agent": human["total_return"] > agent["total_return"],
        "you_beat_benchmark": human["total_return"] > benchmark["total_return"],
        "agent_beat_benchmark": agent["total_return"] > benchmark["total_return"],
        "information_note": INFORMATION_NOTE,
        "sample_note": (
            f"One run of {n} decisions on one price path. That is a single sample: "
            "the seed and walk-forward panels exist because results this small move "
            "a long way on a different draw."
        ),
        "verdict": _verdict(human, agent, benchmark, n),
        "live_computation": True,
    }
    return session.result


def _verdict(human: dict, agent: dict, benchmark: dict, n: int) -> str:
    """Describe the outcome without turning it into a score to beat.

    Deliberately leads with the benchmark rather than the head-to-head, because
    the head-to-head is the comparison this module has just finished explaining
    is not like-for-like.
    """
    h, a, b = human["total_return"], agent["total_return"], benchmark["total_return"]
    parts = [
        f"Over {n} decisions: you {h:+.1%}, the agent {a:+.1%}, buy-and-hold {b:+.1%}."
    ]
    beat = [name for name, v in (("you", h), ("the agent", a)) if v > b]
    if not beat:
        parts.append(
            "Neither of you cleared buy-and-hold — which is the project's whole "
            "finding, arrived at the hard way."
        )
    elif len(beat) == 2:
        parts.append(
            "Both of you cleared buy-and-hold on this path. On a single path that "
            "is not yet evidence of anything; try another seed."
        )
    else:
        parts.append(
            f"Only {beat[0]} cleared buy-and-hold here, on one path. Run it again "
            "on a different seed before reading much into that."
        )
    return " ".join(parts)
