"""Tests for the live agent-vs-baselines comparison.

The whole value of this table is that it can embarrass the agent. So the tests
are mostly about refusing to flatter it: the random arm must be a distribution
rather than one draw, "beats random" must mean clearing the whole observed range,
and the verdict must name every strategy that beat it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl_trader.config.training_config import stock_config
from rl_trader.data.data_loader import synthetic_market_data
from server import baselines_api


@pytest.fixture(scope="module")
def series():
    cfg = stock_config()
    return synthetic_market_data("stock", seed=7, n_steps=400), cfg


def _compare(series, agent_return, n_random=8, **kw):
    data, cfg = series
    metrics = {"total_return": agent_return, "sharpe": 0.5, "max_drawdown": 0.1}
    return baselines_api.compare(
        data, cfg.env, cfg.reward, "stock", metrics, n_random=n_random, **kw
    )


# --------------------------------------------------------------------------- #
# The random arm is a distribution                                             #
# --------------------------------------------------------------------------- #
def test_the_random_arm_reports_a_range_not_a_point(series):
    out = _compare(series, 0.0, n_random=12)
    random_row = next(r for r in out["rows"] if r["key"] == "random")
    assert random_row["n_seeds"] == 12
    assert random_row["worst"] < random_row["best"]
    assert random_row["worst"] <= random_row["total_return"] <= random_row["best"]
    assert out["random_range"] == [random_row["worst"], random_row["best"]]


def test_more_seeds_widen_the_observed_range(series):
    """More draws can only reveal more of the distribution, never less."""
    few = _compare(series, 0.0, n_random=4)["random_range"]
    many = _compare(series, 0.0, n_random=20)["random_range"]
    assert many[0] <= few[0] and many[1] >= few[1]


def test_the_random_note_explains_why_one_draw_would_not_do(series):
    note = _compare(series, 0.0, n_random=6)["random_note"]
    assert "single sample" in note
    assert "independent draws" in note


# --------------------------------------------------------------------------- #
# "Beating random" is the strict claim                                         #
# --------------------------------------------------------------------------- #
def test_beating_the_mean_is_not_the_same_as_beating_the_range(series):
    """Clearing the average of a wide distribution is not evidence of skill."""
    out = _compare(series, 0.0, n_random=12)
    lo, hi = out["random_range"]
    mean = next(r for r in out["rows"] if r["key"] == "random")["total_return"]
    # An agent placed between the random mean and the random best.
    between = _compare(series, (mean + hi) / 2, n_random=12)
    assert between["beats_random_mean"] is True
    assert between["beats_every_random_draw"] is False
    assert lo < hi   # the fixture really does produce a spread


def test_clearing_every_draw_is_reported_as_such(series):
    out = _compare(series, 0.0, n_random=12)
    strong = _compare(series, out["random_range"][1] + 1.0, n_random=12)
    assert strong["beats_every_random_draw"] is True
    assert "beat all" in strong["verdict"]


# --------------------------------------------------------------------------- #
# Ranking and honesty                                                          #
# --------------------------------------------------------------------------- #
def test_every_baseline_is_scored_on_the_same_series(series):
    out = _compare(series, 0.1)
    keys = {r["key"] for r in out["rows"]}
    assert keys == {"agent", "buy_and_hold", "ma_crossover", "flat", "random"}
    assert out["n_strategies"] == 5
    assert "same transaction cost" in out["note"]


def test_a_losing_agent_is_ranked_last_and_told_so(series):
    out = _compare(series, -5.0)
    assert out["agent_rank"] == out["n_strategies"]
    assert out["beaten"] == []
    assert out["ranked"][-1] == "agent"
    assert "beaten by" in out["verdict"]
    assert "not distinguishable from luck" in out["verdict"]


def test_a_winning_agent_names_what_it_beat(series):
    out = _compare(series, 50.0)
    assert out["agent_rank"] == 1
    assert len(out["beaten"]) == 4
    assert "came top of every baseline here — on one path" in out["verdict"]


def test_the_flat_strategy_really_does_nothing(series):
    """The do-nothing control anchors the table; if it drifts, costs are leaking."""
    flat = next(r for r in _compare(series, 0.0)["rows"] if r["key"] == "flat")
    assert flat["total_return"] == pytest.approx(0.0, abs=1e-9)
    assert flat["max_drawdown"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# The two buy-and-holds                                                        #
# --------------------------------------------------------------------------- #
def test_the_cost_free_benchmark_is_labelled_separately(series):
    """The chart's benchmark pays no costs; the strategy does. Both are shown."""
    out = _compare(series, 0.0, cost_free_benchmark={"total_return": 0.42})
    ref = out["cost_free_benchmark"]
    assert ref["total_return"] == 0.42
    assert "paying no costs" in ref["note"]
    # And it is *not* smuggled into the ranked table as a strategy.
    assert "cost_free" not in {r["key"] for r in out["rows"]}


def test_the_costed_buy_and_hold_is_not_better_than_the_cost_free_one(series):
    """Paying an entry cost cannot improve the result — a sanity check on wiring."""
    data, cfg = series
    out = _compare(series, 0.0)
    costed = next(r for r in out["rows"] if r["key"] == "buy_and_hold")["total_return"]
    prices = np.asarray(data.prices, dtype=float)
    free = float(prices[-1] / prices[cfg.env.window_size - 1] - 1.0)
    assert costed <= free + 1e-9


def test_the_comparison_declares_itself_live(series):
    assert _compare(series, 0.0)["live_computation"] is True
