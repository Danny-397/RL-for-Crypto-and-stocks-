"""Tests for pre-registration.

The claim this feature makes is procedural: the prediction was fixed before the
result existed, and the rule that judges it is the same for everyone. Both are
only true if the parsing stamps a time up front, the band is a constant rather
than a per-experiment choice, and a prediction that cannot be scored says so
instead of being scored against something else.
"""

from __future__ import annotations

import pytest

from server import prereg


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #
def test_no_prediction_is_recorded_as_none():
    """Absence must stay absent — never filled in on the visitor's behalf."""
    assert prereg.parse({}) is None
    assert prereg.parse({"prediction": None}) is None


def test_a_bare_direction_string_is_accepted():
    out = prereg.parse({"prediction": "beats"})
    assert out["direction"] == "beats"
    assert out["statement"] == prereg.DIRECTIONS["beats"]
    assert out["note"] is None


def test_the_registration_is_timestamped_at_parse_time():
    out = prereg.parse({"prediction": {"direction": "loses"}})
    assert out["registered_at_utc"].endswith("Z")
    # The judging rule travels with the registration, so it cannot be restated
    # differently later.
    assert out["rule"] == prereg.RULE
    assert out["match_band"] == prereg.MATCH_BAND


def test_notes_are_kept_verbatim_and_bounded():
    out = prereg.parse({"prediction": {"direction": "matches", "note": "  hunch  "}})
    assert out["note"] == "hunch"
    long = prereg.parse({"prediction": {"direction": "matches", "note": "x" * 900}})
    assert len(long["note"]) == 400


@pytest.mark.parametrize("bad", ["sideways", "", "BEATS_MAYBE"])
def test_unknown_directions_are_rejected(bad):
    with pytest.raises(ValueError, match="unknown prediction direction"):
        prereg.parse({"prediction": bad})


def test_a_malformed_prediction_is_rejected():
    with pytest.raises(ValueError, match="must be an object or a direction string"):
        prereg.parse({"prediction": ["beats"]})


# --------------------------------------------------------------------------- #
# The judging rule                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("excess, label", [
    (0.5, "beats"),
    (0.0201, "beats"),
    (0.02, "matches"),      # the band is inclusive at its edge
    (0.0, "matches"),
    (-0.02, "matches"),
    (-0.0201, "loses"),
    (-0.4, "loses"),
])
def test_classification_uses_one_fixed_band(excess, label):
    assert prereg.classify(excess) == label


def test_the_band_is_a_module_constant_not_a_request_field():
    """A visitor can disagree with the band; they cannot move it after the fact."""
    a = prereg.parse({"prediction": "beats", "match_band": 0.5})
    assert a["match_band"] == prereg.MATCH_BAND == 0.02


# --------------------------------------------------------------------------- #
# Extracting the pre-declared quantity                                         #
# --------------------------------------------------------------------------- #
def test_rollout_excess_is_agent_minus_benchmark():
    result = {"metrics": {"total_return": 0.30}, "bench_metrics": {"total_return": 0.10}}
    assert prereg.observed_excess("rollout", result) == pytest.approx(0.20)


def test_walk_forward_uses_the_mean_across_folds():
    result = {"summary": {"mean_excess_return": -0.07}}
    assert prereg.observed_excess("walk_forward", result) == pytest.approx(-0.07)


def test_distribution_shift_averages_the_regimes():
    result = {"regimes": [{"mean_excess_return": 0.1}, {"mean_excess_return": -0.3}]}
    assert prereg.observed_excess("distribution_shift", result) == pytest.approx(-0.1)


@pytest.mark.parametrize("kind, result", [
    ("counterfactual", {"alternatives": []}),
    ("rollout", {"metrics": {"total_return": 0.1}}),   # benchmark missing
    ("rollout", None),
])
def test_missing_or_inapplicable_results_yield_none(kind, result):
    assert prereg.observed_excess(kind, result) is None


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def test_a_correct_prediction_is_scored_as_matched():
    pred = prereg.parse({"prediction": "beats"})
    result = {"metrics": {"total_return": 0.40}, "bench_metrics": {"total_return": 0.10}}
    out = prereg.evaluate(pred, "rollout", result)
    assert out["matched"] is True
    assert out["observed"] == "beats"
    assert out["observed_excess"] == pytest.approx(0.30)
    # Even a hit is framed as weak evidence — one run is one run.
    assert "weak evidence" in out["verdict"]


def test_a_wrong_prediction_is_recorded_not_softened():
    pred = prereg.parse({"prediction": "beats"})
    result = {"metrics": {"total_return": -0.20}, "bench_metrics": {"total_return": 0.10}}
    out = prereg.evaluate(pred, "rollout", result)
    assert out["matched"] is False
    assert out["observed"] == "loses"
    assert "could have been wrong" in out["verdict"]


def test_an_unscorable_kind_says_so_rather_than_guessing():
    pred = prereg.parse({"prediction": "beats"})
    out = prereg.evaluate(pred, "counterfactual", {"alternatives": []})
    assert out["scorable"] is False
    assert out["matched"] is None
    assert "not scored" in out["reason"]


def test_no_prediction_produces_no_outcome():
    assert prereg.evaluate(None, "rollout", {"metrics": {}, "bench_metrics": {}}) is None


def test_describe_publishes_the_rule_and_every_option():
    d = prereg.describe()
    assert {o["key"] for o in d["directions"]} == {"beats", "matches", "loses"}
    assert d["match_band"] == prereg.MATCH_BAND
    assert "fixed before the run" in d["rule"]
    assert "rollout" in d["scorable_kinds"]
