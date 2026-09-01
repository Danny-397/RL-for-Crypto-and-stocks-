"""Tests for the power calculator.

This module tells visitors how many runs an experiment would need, which is a
claim about a test's behaviour rather than about data. The load-bearing test is
therefore calibration: with no true effect, the procedure must reject at close to
alpha. If that ever drifts, every power number it reports is wrong in a way no
amount of UI copy would reveal.
"""

from __future__ import annotations

import pytest

from server import power


# --------------------------------------------------------------------------- #
# The p-value itself                                                           #
# --------------------------------------------------------------------------- #
def test_identical_differences_land_exactly_on_the_floor():
    """All-same-sign differences give the smallest p the design can produce."""
    assert power.sign_flip_p([1.0, 1.0, 1.0]) == pytest.approx(2 / 2 ** 3)
    assert power.sign_flip_p([1.0] * 5) == pytest.approx(2 / 2 ** 5)
    assert power.sign_flip_p([-2.0] * 6) == pytest.approx(2 / 2 ** 6)


def test_a_zero_mean_difference_is_maximally_unsurprising():
    assert power.sign_flip_p([1.0, -1.0, 1.0, -1.0]) == pytest.approx(1.0)


def test_the_p_value_is_symmetric_under_negation():
    d = [0.3, -0.1, 0.4, 0.2, -0.05, 0.6]
    assert power.sign_flip_p(d) == pytest.approx(power.sign_flip_p([-x for x in d]))


def test_an_empty_comparison_is_p_one():
    assert power.sign_flip_p([]) == 1.0


def test_exact_enumeration_matches_a_large_sample(monkeypatch):
    """Sampling above the exact limit must approximate what enumeration gives."""
    d = [0.4, 0.1, 0.35, -0.05, 0.2, 0.3, 0.15, 0.25]
    exact = power.sign_flip_p(d)
    monkeypatch.setattr(power, "EXACT_LIMIT", 2)   # force the sampled path
    sampled = power.sign_flip_p(d, n_perm=60000, seed=1)
    assert sampled == pytest.approx(exact, abs=0.01)


# --------------------------------------------------------------------------- #
# Calibration — the test that makes every other number trustworthy             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [8, 12])
def test_with_no_true_effect_the_rejection_rate_matches_alpha(n):
    out = power.power_at_n(n, effect=0.0, sd=0.3, alpha=0.05, n_sims=4000, seed=3)
    # Discreteness means it cannot land exactly on alpha; it must not exceed it
    # by much, and must certainly not sit near zero or near one.
    assert 0.01 < out["power"] < 0.09


def test_calibration_holds_at_a_different_alpha():
    out = power.power_at_n(12, effect=0.0, sd=0.3, alpha=0.20, n_sims=4000, seed=5)
    assert 0.14 < out["power"] < 0.26


# --------------------------------------------------------------------------- #
# The floor dominates at small n                                               #
# --------------------------------------------------------------------------- #
def test_five_pairs_have_zero_power_whatever_the_effect():
    """The project's own seed count cannot reach 0.05, so its power is zero."""
    out = power.power_at_n(5, effect=100.0, sd=0.01, alpha=0.05, n_sims=200)
    assert out["attainable"] is False
    assert out["power"] == 0.0
    assert out["floor"] == pytest.approx(2 / 2 ** 5)
    assert "cannot produce" in out["note"]


def test_six_pairs_clear_the_floor_and_can_detect_a_large_effect():
    out = power.power_at_n(6, effect=1.0, sd=0.05, alpha=0.05, n_sims=400)
    assert out["attainable"] is True
    assert out["power"] > 0.95


def test_a_tiny_n_is_handled_rather_than_crashing():
    assert power.power_at_n(1, effect=1.0, sd=0.1)["power"] == 0.0


# --------------------------------------------------------------------------- #
# Monotonicity                                                                 #
# --------------------------------------------------------------------------- #
def test_power_rises_with_sample_size():
    curve = [power.power_at_n(n, effect=0.2, sd=0.35, n_sims=1500, seed=7)["power"]
             for n in (8, 14, 24)]
    assert curve[0] < curve[1] < curve[2]


def test_power_rises_with_effect_size():
    curve = [power.power_at_n(12, effect=e, sd=0.3, n_sims=1500, seed=9)["power"]
             for e in (0.05, 0.2, 0.5)]
    assert curve[0] < curve[1] < curve[2]


def test_power_falls_as_the_spread_widens():
    tight = power.power_at_n(12, effect=0.2, sd=0.1, n_sims=1500, seed=11)["power"]
    loose = power.power_at_n(12, effect=0.2, sd=0.9, n_sims=1500, seed=11)["power"]
    assert tight > loose


# --------------------------------------------------------------------------- #
# Required sample size                                                         #
# --------------------------------------------------------------------------- #
def test_required_n_finds_a_size_that_actually_reaches_the_target():
    out = power.required_n(effect=0.3, sd=0.3, target=0.8, n_sims=1200, seed=4)
    assert out["required_n"] is not None
    achieved = power.power_at_n(out["required_n"], 0.3, 0.3, n_sims=3000, seed=21)
    assert achieved["power"] >= 0.72   # simulation noise around the 0.8 target


def test_required_n_never_returns_a_size_below_the_floor():
    out = power.required_n(effect=5.0, sd=0.01, target=0.8, n_sims=400, seed=2)
    assert out["required_n"] >= 6      # n = 5 cannot reach p <= 0.05 at all


def test_an_effect_too_small_to_find_says_so():
    out = power.analyse(effect=0.001, sd=1.0, n_sims=300, seed=6)
    assert out["required_n"] is None
    assert "more runs are not the cheap fix" in out["verdict"]


def test_the_curve_records_every_size_it_tried():
    out = power.required_n(effect=0.4, sd=0.3, target=0.8, n_sims=800, seed=8)
    ns = [row["n"] for row in out["curve"]]
    assert ns == sorted(ns)
    assert ns[0] == 2
    assert out["required_n"] == ns[-1]


# --------------------------------------------------------------------------- #
# The assembled answer                                                         #
# --------------------------------------------------------------------------- #
def test_analyse_reports_both_what_you_have_and_what_you_need():
    out = power.analyse(effect=0.1, sd=0.25, have_n=5, n_sims=500, seed=1)
    assert out["current"]["n"] == 5
    assert out["current"]["attainable"] is False
    assert "cannot reach" in out["verdict"]
    assert out["live_computation"] is True
    assert "sign-flip" in out["test"]


def test_the_method_states_where_enumeration_stops():
    out = power.analyse(effect=0.3, sd=0.2, n_sims=300)
    assert f"n <= {power.EXACT_LIMIT}" in out["method"]
    assert "smoothing it away" in out["method"]


def test_zero_variance_is_handled_explicitly():
    out = power.power_at_n(8, effect=0.5, sd=0.0, n_sims=100)
    assert out["power"] == 1.0
    assert "deterministic" in out["note"]


@pytest.mark.parametrize("kwargs", [
    {"sd": -1.0}, {"alpha": 0.0}, {"alpha": 1.0}, {"target": 0.0}, {"target": 1.5},
])
def test_invalid_inputs_are_rejected(kwargs):
    params = {"effect": 0.1, "sd": 0.2}
    params.update(kwargs)
    with pytest.raises(ValueError):
        power.analyse(n_sims=50, **params)


def test_results_are_reproducible_from_the_seed():
    a = power.power_at_n(10, 0.2, 0.3, n_sims=800, seed=42)
    b = power.power_at_n(10, 0.2, 0.3, n_sims=800, seed=42)
    assert a == b
    c = power.power_at_n(10, 0.2, 0.3, n_sims=800, seed=43)
    assert isinstance(c["power"], float)   # a different seed is allowed to differ
