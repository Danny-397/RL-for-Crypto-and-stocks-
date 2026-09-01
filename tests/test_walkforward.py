"""Tests for the walk-forward panel's server side.

The panel's entire claim is methodological — "these blocks are disjoint, and the
scaler for each one was fit without looking at it" — so the properties tested
here are exactly the ones a reader is being asked to take on trust.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl_trader.config.training_config import stock_config
from rl_trader.data.data_loader import generate_synthetic_ohlcv
from server import walkforward


class SumPolicy:
    """Deterministic and sensitive to the whole observation.

    Any change in the scaling shows up as a change in behaviour, which is what
    makes the leakage comparison a real measurement rather than a coincidence.
    """

    def act(self, obs) -> float:
        return float(np.tanh(np.asarray(obs, dtype=float).sum() * 0.01))

    def describe(self) -> dict:
        return {"name": "sum-policy"}


@pytest.fixture
def frame():
    return generate_synthetic_ohlcv(n_steps=900, seed=5, momentum=0.5)


# --------------------------------------------------------------------------- #
# Fold geometry                                                                #
# --------------------------------------------------------------------------- #
def test_folds_are_disjoint_and_chronological():
    """The property the whole panel exists to show."""
    plan = walkforward.fold_plan(1000, n_folds=4)
    assert len(plan) == 4
    for row in plan:
        assert row["disjoint"] is True
        assert row["train_end"] <= row["test_start"]
        assert row["test_start"] < row["test_end"]
    # Test blocks tile the out-of-sample region without overlapping.
    for a, b in zip(plan, plan[1:]):
        assert a["test_end"] == b["test_start"]


def test_test_blocks_cover_everything_after_the_first_training_window():
    plan = walkforward.fold_plan(1000, n_folds=5, train_min_frac=0.4)
    assert plan[0]["test_start"] == 400
    assert plan[-1]["test_end"] == 1000


def test_expanding_grows_the_training_window_and_sliding_does_not():
    expanding = walkforward.fold_plan(1000, n_folds=4, scheme="expanding")
    sliding = walkforward.fold_plan(1000, n_folds=4, scheme="sliding")
    exp_sizes = [r["train_end"] - r["train_start"] for r in expanding]
    sli_sizes = [r["train_end"] - r["train_start"] for r in sliding]
    assert exp_sizes == sorted(exp_sizes) and exp_sizes[0] < exp_sizes[-1]
    assert len(set(sli_sizes)) == 1
    # A sliding fold drops stale history; an expanding one never does.
    assert all(r["train_start"] == 0 for r in expanding)
    assert sliding[-1]["train_start"] > 0


def test_fold_plan_labels_calendar_ranges_when_dates_are_known():
    dates = [f"2020-01-{d:02d}" for d in range(1, 11)] * 100
    plan = walkforward.fold_plan(1000, n_folds=3, dates=dates)
    for row in plan:
        assert row["test_from"] == dates[row["test_start"]]
        assert row["test_to"] == dates[row["test_end"] - 1]


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
def test_every_fold_is_evaluated_out_of_sample(frame):
    rows = walkforward.evaluate_folds(
        SumPolicy(), frame, "stock", stock_config(), n_folds=4
    )
    assert len(rows) == 4
    for row in rows:
        assert row["test_bars"] > 0
        assert row["train_end"] <= row["test_start"]
        assert row["excess_return"] == pytest.approx(
            row["agent_return"] - row["benchmark_return"], abs=1e-6
        )


def test_evaluation_is_deterministic(frame):
    cfg = stock_config()
    a = walkforward.evaluate_folds(SumPolicy(), frame, "stock", cfg, n_folds=3)
    b = walkforward.evaluate_folds(SumPolicy(), frame, "stock", cfg, n_folds=3)
    assert a == b


def test_leaked_scaling_really_does_change_the_answer(frame):
    """Fitting the scaler on the test block is not a cosmetic difference.

    Both arms run the same policy over the same prices; only the scaler's fitting
    window differs. If these ever came back identical, the panel's leakage
    comparison would be showing nothing and should not be displayed.
    """
    cfg = stock_config()
    correct = walkforward.evaluate_folds(
        SumPolicy(), frame, "stock", cfg, n_folds=4, scaling="train_only"
    )
    leaked = walkforward.evaluate_folds(
        SumPolicy(), frame, "stock", cfg, n_folds=4, scaling="full_sample"
    )
    delta = walkforward.leakage_delta(correct, leaked)
    assert delta["identical"] is False
    assert delta["max_abs_delta"] > 0.0
    assert len(delta["per_fold"]) == 4


def test_leakage_delta_reports_no_difference_honestly():
    rows = [{"fold": 0, "agent_return": 0.1}, {"fold": 1, "agent_return": -0.2}]
    delta = walkforward.leakage_delta(rows, rows)
    assert delta["identical"] is True
    assert delta["mean_delta"] == 0.0


@pytest.mark.parametrize("kwargs", [{"scheme": "diagonal"}, {"scaling": "psychic"}])
def test_unknown_options_are_rejected(frame, kwargs):
    with pytest.raises(ValueError):
        walkforward.evaluate_folds(
            SumPolicy(), frame, "stock", stock_config(), n_folds=3, **kwargs
        )


def test_folds_too_short_to_step_are_skipped_not_faked():
    """A fold shorter than the observation window cannot be evaluated at all.

    It is dropped, so the summary counts only folds that really ran — rather
    than emitting a zero that would read as "the agent made nothing here".
    """
    short = generate_synthetic_ohlcv(n_steps=340, seed=1)
    rows = walkforward.evaluate_folds(
        SumPolicy(), short, "stock", stock_config(), n_folds=8
    )
    assert len(rows) < 8
    assert all(r["test_bars"] > 0 for r in rows)


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #
def test_summary_reports_the_sign_test_floor():
    rows = [{"fold": i, "excess_return": e, "agent_return": e, "benchmark_return": 0.0}
            for i, e in enumerate([0.1, -0.2, 0.3, -0.05])]
    out = walkforward.summarise(rows)
    assert out["n_folds"] == 4
    assert out["folds_beaten"] == 2
    assert out["sign_test_floor"] == pytest.approx(2 / 2 ** 4)
    assert out["sign_test_can_reach_05"] is False
    assert "cannot reach significance" in out["spread_note"]
    assert out["worst_fold_excess"] == -0.2
    assert out["best_fold_excess"] == 0.3


def test_summary_allows_significance_when_the_design_can_reach_it():
    rows = [{"fold": i, "excess_return": 0.1, "agent_return": 0.1, "benchmark_return": 0.0}
            for i in range(8)]
    out = walkforward.summarise(rows)
    assert out["sign_test_floor"] == pytest.approx(2 / 2 ** 8)
    assert out["sign_test_can_reach_05"] is True
    assert "can reach significance" in out["spread_note"]


def test_summary_of_nothing_says_nothing():
    assert walkforward.summarise([]) == {"n_folds": 0}


def test_describe_lists_both_schemes_and_the_fixed_policy_caveat():
    d = walkforward.describe()
    assert {s["key"] for s in d["schemes"]} == {"expanding", "sliding"}
    assert {s["key"] for s in d["scalings"]} == {"train_only", "full_sample"}
    assert "not a retrained walk-forward" in d["fixed_policy_note"]
