"""Tests for the human-baseline sessions.

The panel makes one hard claim — you cannot see a bar before you have traded
through it — and one soft one — the comparison against the agent is not
like-for-like. The first is a property of the protocol and is tested as such.
The second only holds if the note saying so is on every payload, so that is
tested too.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from rl_trader.config.training_config import stock_config
from rl_trader.data.data_loader import synthetic_market_data
from rl_trader.envs import make_env
from server import human


class FlatPolicy:
    """Always fully long — deterministic, so comparisons are reproducible."""

    def act(self, obs) -> float:
        return 1.0

    def describe(self) -> dict:
        return {"name": "flat-long"}


def _build(n_steps: int = 400, seed: int = 3, allow_short: bool = True):
    cfg = stock_config()
    cfg.env.allow_short = allow_short
    data = synthetic_market_data("stock", seed=seed, n_steps=n_steps)
    env = make_env("stock", data, cfg.env, cfg.reward, random_start=False)
    return env, cfg


def _session(max_steps: int = 20, **kw):
    env, cfg = _build(**kw)
    config = {"market": "stock", "mode": "synthetic", "regime": "momentum",
              "seed": 3, "n_steps": 400}
    session = human.start(env, cfg, None, {"synthetic": True}, config, max_steps)
    session.id = "HUM-TEST1"
    return session, cfg


# --------------------------------------------------------------------------- #
# No lookahead                                                                 #
# --------------------------------------------------------------------------- #
def test_the_opening_reveals_only_the_warm_up_window():
    session, cfg = _session()
    opening = human.opening(session)
    assert len(opening["prices"]) == cfg.env.window_size
    assert opening["step"] == 0
    # Everything after the current bar stays on the server.
    assert len(opening["prices"]) < len(session.env.data.prices)


def test_exactly_one_bar_is_released_per_decision():
    """The claim the panel makes about itself, checked step by step."""
    session, cfg = _session(max_steps=15)
    revealed = session.revealed
    for i in range(15):
        out = human.step(session, 0.5)
        revealed += 1
        assert session.revealed == revealed
        assert out["step"] == i + 1
        # The bar returned is the newest one, never anything beyond it.
        assert out["price"] == pytest.approx(
            float(session.env.data.prices[session.revealed - 1]), abs=1e-5
        )


def test_a_step_payload_carries_one_price_not_a_series():
    session, _cfg = _session()
    out = human.step(session, 1.0)
    assert isinstance(out["price"], float)
    assert "prices" not in out


# --------------------------------------------------------------------------- #
# Stepping                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sent, expected", [(5.0, 1.0), (-9.0, -1.0), (0.25, 0.25)])
def test_actions_are_clamped_to_the_action_space(sent, expected):
    session, _cfg = _session()
    assert human.step(session, sent)["action"] == pytest.approx(expected)


def test_shorting_is_refused_when_the_environment_forbids_it():
    session, _cfg = _session(allow_short=False)
    assert human.step(session, -1.0)["action"] == 0.0


def test_the_session_ends_at_the_decision_limit():
    session, _cfg = _session(max_steps=10)
    for _ in range(9):
        assert human.step(session, 0.0)["done"] is False
    last = human.step(session, 0.0)
    assert last["done"] is True
    assert "decision limit" in last["reason"]


def test_stepping_a_finished_session_is_refused():
    session, _cfg = _session(max_steps=10)
    for _ in range(10):
        human.step(session, 0.0)
    with pytest.raises(ValueError, match="already finished"):
        human.step(session, 0.0)


def test_a_series_too_short_to_trade_is_rejected():
    env, cfg = _build(n_steps=145)
    with pytest.raises(ValueError, match="tradeable bars"):
        human.start(env, cfg, None, {}, {"market": "stock"}, max_steps=60)


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def test_all_three_curves_cover_exactly_the_same_bars():
    """A 20-decision run must not be scored against a 600-bar benchmark."""
    session, _cfg = _session(max_steps=20)
    for _ in range(20):
        human.step(session, 0.5)
    out = human.finish(session, FlatPolicy(), lambda: _build()[0])

    n = len(out["you"]["equity_curve"])
    assert out["bars_traded"] == 20
    assert n == 21  # the opening balance plus one point per decision
    assert len(out["agent"]["equity_curve"]) == n
    assert len(out["benchmark"]["equity_curve"]) == n


def test_the_benchmark_starts_from_the_same_capital():
    session, cfg = _session(max_steps=15)
    for _ in range(15):
        human.step(session, 1.0)
    out = human.finish(session, FlatPolicy(), lambda: _build()[0])
    assert out["benchmark"]["equity_curve"][0] == pytest.approx(
        cfg.env.initial_balance, rel=1e-6
    )
    assert out["you"]["equity_curve"][0] == pytest.approx(cfg.env.initial_balance, rel=1e-6)


def test_the_headline_flags_agree_with_the_numbers():
    session, _cfg = _session(max_steps=15)
    for _ in range(15):
        human.step(session, 0.2)
    out = human.finish(session, FlatPolicy(), lambda: _build()[0])
    you = out["you"]["metrics"]["total_return"]
    agent = out["agent"]["metrics"]["total_return"]
    bench = out["benchmark"]["metrics"]["total_return"]
    assert out["you_beat_agent"] is (you > agent)
    assert out["you_beat_benchmark"] is (you > bench)
    assert out["agent_beat_benchmark"] is (agent > bench)


def test_identical_play_produces_identical_scores():
    results = []
    for _ in range(2):
        session, _cfg = _session(max_steps=12)
        for a in np.linspace(-1, 1, 12):
            human.step(session, float(a))
        results.append(human.finish(session, FlatPolicy(), lambda: _build()[0]))
    assert results[0]["you"]["metrics"] == results[1]["you"]["metrics"]
    assert results[0]["agent"]["metrics"] == results[1]["agent"]["metrics"]


def test_finishing_without_deciding_anything_is_refused():
    session, _cfg = _session()
    with pytest.raises(ValueError, match="no decisions"):
        human.finish(session, FlatPolicy(), lambda: _build()[0])


def test_the_asymmetry_is_stated_on_every_payload():
    """The comparison is not like-for-like, so it must never be presented as one."""
    session, _cfg = _session(max_steps=10)
    assert "not a like-for-like" in human.opening(session)["information_note"]
    for _ in range(10):
        human.step(session, 0.0)
    out = human.finish(session, FlatPolicy(), lambda: _build()[0])
    assert "not a like-for-like" in out["information_note"]
    assert "single sample" in out["sample_note"]
    # The verdict leads with the benchmark, not the head-to-head.
    assert "buy-and-hold" in out["verdict"]


# --------------------------------------------------------------------------- #
# Session store                                                                #
# --------------------------------------------------------------------------- #
def test_sessions_are_addressable_and_bounded():
    store = human.SessionStore(max_sessions=3)
    ids = []
    for _ in range(5):
        session, _cfg = _session(max_steps=10)
        ids.append(store.add(session).id)
    assert len(set(ids)) == 5              # ids are unique
    assert store.stats()["active"] == 3    # oldest evicted
    assert store.get(ids[0]) is None
    assert store.get(ids[-1]) is not None


def test_sessions_expire_and_say_so():
    store = human.SessionStore(ttl=0.05)
    session, _cfg = _session(max_steps=10)
    sid = store.add(session).id
    assert store.get(sid) is not None
    time.sleep(0.1)
    assert store.get(sid) is None
    assert "ephemeral" in store.stats()["storage"]


def test_activity_keeps_a_session_alive():
    store = human.SessionStore(ttl=0.2)
    session, _cfg = _session(max_steps=10)
    sid = store.add(session).id
    for _ in range(3):
        time.sleep(0.1)
        assert store.get(sid) is not None
