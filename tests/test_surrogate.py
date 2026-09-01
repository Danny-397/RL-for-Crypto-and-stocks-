"""Tests for serving the surrogate-data falsification test.

This panel makes the project's strongest claim — that the flat performance is the
market's, not the agent's — and it is only entitled to that claim because of the
positive control. So what is pinned here is the *reasoning*: the verdict must
refuse to interpret the real arm when the control fails, and the module must
decline to re-derive statistics it does not have the data for.
"""

from __future__ import annotations

import json

import pytest

from server import surrogate


def _arm(diff: float, p: float, **extra) -> dict:
    row = {
        "market": "stock",
        "edge_structured": 0.2,
        "edge_surrogate": 0.2 - diff,
        "structured_ci": [0.2, 0.1, 0.3],
        "surrogate_ci": [0.0, -0.1, 0.1],
        "diff": diff,
        "p": p,
    }
    row.update(extra)
    return row


def _arms(control_diff=0.4, control_p=0.001, real_diff=-0.1, real_p=0.5):
    return [
        {"arm": "synthetic", "markets": [
            surrogate._row("synthetic", "stock", _arm(control_diff, control_p))]},
        {"arm": "real", "markets": [
            surrogate._row("real", "stock", _arm(real_diff, real_p))]},
    ]


# --------------------------------------------------------------------------- #
# The verdict's logic                                                          #
# --------------------------------------------------------------------------- #
def test_a_passing_control_plus_a_real_null_is_the_headline_claim():
    verdict = surrogate._verdict(_arms())
    assert "has power" in verdict
    assert "no better on real price history" in verdict


def test_a_failed_control_refuses_to_interpret_the_real_arm():
    """The whole argument collapses without the positive control, and must say so."""
    verdict = surrogate._verdict(_arms(control_diff=0.05, control_p=0.6))
    assert "cannot be interpreted" in verdict
    assert "says nothing" in verdict


def test_a_significant_real_result_is_not_overclaimed():
    verdict = surrogate._verdict(_arms(real_diff=0.9, real_p=0.001))
    assert "may be reading genuine ordering" in verdict
    assert "before believing it" in verdict


def test_a_control_that_moves_the_wrong_way_still_fails():
    """A large *negative* control difference is a broken test, not a passing one."""
    verdict = surrogate._verdict(_arms(control_diff=-0.8, control_p=0.001))
    assert "cannot be interpreted" in verdict


def test_a_missing_arm_produces_no_verdict():
    assert surrogate._verdict([{"arm": "real", "markets": []}]) is None


# --------------------------------------------------------------------------- #
# Per-market interpretation                                                    #
# --------------------------------------------------------------------------- #
def test_the_control_row_is_described_as_a_power_check():
    row = surrogate._row("synthetic", "crypto", _arm(0.5, 0.002))
    assert "control passes" in row["interpretation"]
    assert "detect structure when structure is there" in row["interpretation"]


def test_a_real_null_is_stated_as_indistinguishable_not_as_proof():
    row = surrogate._row("real", "stock", _arm(-0.1, 0.5))
    assert "not distinguishable" in row["interpretation"]
    assert row["significant_at_05"] is False


def test_significance_flags_follow_the_p_value():
    assert surrogate._row("real", "stock", _arm(1.0, 0.049))["significant_at_05"] is True
    assert surrogate._row("real", "stock", _arm(1.0, 0.051))["significant_at_05"] is False


# --------------------------------------------------------------------------- #
# What the artifacts can and cannot support                                    #
# --------------------------------------------------------------------------- #
def test_summary_only_artifacts_are_marked_unreanalysable():
    """A p-value cannot be re-derived from a mean, so the module must not pretend."""
    row = surrogate._row("real", "stock", _arm(0.1, 0.4))
    assert row["reanalysable"] is False
    assert row["values_structured"] is None
    assert row["n_pairs"] is None


def test_artifacts_carrying_per_arm_values_become_reanalysable():
    row = surrogate._row("real", "stock", _arm(
        0.1, 0.4, values_structured=[0.1, 0.2, 0.3], values_surrogate=[0.0, 0.1, 0.0]
    ))
    assert row["reanalysable"] is True
    assert row["n_pairs"] == 3


def test_n_pairs_is_preferred_over_inferring_it():
    row = surrogate._row("real", "stock", _arm(
        0.1, 0.4, n_pairs=10, values_structured=[0.1, 0.2], values_surrogate=[0.0, 0.1]
    ))
    assert row["n_pairs"] == 10


# --------------------------------------------------------------------------- #
# The committed artifacts, as they actually are                                #
# --------------------------------------------------------------------------- #
def test_the_committed_results_load_and_carry_their_provenance():
    out = surrogate.results()
    assert out is not None
    arms = {a["arm"]: a for a in out["arms"]}
    assert set(arms) == {"synthetic", "real"}
    for arm in arms.values():
        assert arm["source"].startswith("docs/assets/")
        assert arm["generated_by"].startswith("python tools/surrogate_test.py")
        assert {r["market"] for r in arm["markets"]} == {"stock", "crypto"}


def test_the_committed_control_actually_passes():
    """If this ever fails, the panel's headline claim must stop being made.

    The real arm's null is only readable because the synthetic arm detects a
    planted signal in both markets. That is an empirical fact about the committed
    artifacts, so it is asserted rather than assumed.
    """
    control = next(a for a in surrogate.results()["arms"] if a["arm"] == "synthetic")
    for row in control["markets"]:
        assert row["diff"] > 0
        assert row["p"] < 0.05


def test_the_committed_real_arm_is_a_null():
    real = next(a for a in surrogate.results()["arms"] if a["arm"] == "real")
    assert all(not row["significant_at_05"] for row in real["markets"])


def test_the_served_payload_never_claims_to_be_live():
    out = surrogate.results()
    assert out["live_computation"] is False
    assert out["reanalysable"] is False       # today's artifacts are summary-only
    assert "summary statistics only" in out["reanalysis_note"]
    assert any("committed results, not live" in c for c in out["caveats"])
    assert any("not proof of no structure" in c for c in out["caveats"])


def test_missing_artifacts_return_none_rather_than_a_placeholder(monkeypatch, tmp_path):
    monkeypatch.setattr(surrogate, "ASSETS", str(tmp_path))
    assert surrogate.results() is None


def test_a_corrupt_artifact_is_skipped_not_fatal(monkeypatch, tmp_path):
    (tmp_path / "surrogate_synthetic.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "surrogate_real.json").write_text(
        json.dumps({"stock": _arm(0.1, 0.4)}), encoding="utf-8"
    )
    monkeypatch.setattr(surrogate, "ASSETS", str(tmp_path))
    out = surrogate.results()
    assert [a["arm"] for a in out["arms"]] == ["real"]
    assert out["verdict"] is None   # no control, so no claim


# --------------------------------------------------------------------------- #
# The generator records what the reader needs                                  #
# --------------------------------------------------------------------------- #
def test_the_tool_now_records_per_arm_values():
    """Future regenerations must carry the arrays, or this gap reopens."""
    pytest.importorskip("torch")
    from tools.surrogate_test import _summary

    out = _summary("stock", [0.1, 0.2, 0.3], [0.0, 0.1, 0.0], 0.166, 0.02)
    assert out["n_pairs"] == 3
    assert out["values_structured"] == [0.1, 0.2, 0.3]
    assert out["values_surrogate"] == [0.0, 0.1, 0.0]
