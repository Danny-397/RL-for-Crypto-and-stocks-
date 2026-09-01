"""Tests for the hyper-parameter sensitivity sweep and its serving layer.

The sweep exists to answer "is the conclusion fragile?" — and the easiest way to
ruin it is to let it quietly become "which recipe wins?". Most of what is pinned
here is that refusal: published order preserved, no best-configuration headline,
and a summary that leads with how many recipes produced a positive edge rather
than with the top of a noisy ranking.
"""

from __future__ import annotations

import json

import pytest

from server import hpsweep
from tools.hyperparameter_sweep import KNOBS, summarise, variants


# --------------------------------------------------------------------------- #
# The design                                                                   #
# --------------------------------------------------------------------------- #
def test_the_sweep_is_one_factor_at_a_time():
    """Each variant moves exactly one knob, so effects are attributable."""
    specs = variants()
    assert specs[0]["key"] == "baseline"
    assert specs[0]["knob"] is None
    moved = [s for s in specs[1:]]
    assert len(moved) == sum(len(v) for v in KNOBS.values())
    for spec in moved:
        assert spec["knob"] in KNOBS
        assert spec["value"] in KNOBS[spec["knob"]]


def test_every_knob_is_moved_both_ways():
    """A one-sided nudge could miss a knob the result is sensitive to."""
    from rl_trader.config.training_config import PPOConfig

    defaults = PPOConfig()
    for knob, values in KNOBS.items():
        default = getattr(defaults, knob)
        assert min(values) < default < max(values), knob


def test_the_baseline_is_included_as_a_comparison_point():
    assert sum(1 for s in variants() if s["knob"] is None) == 1


# --------------------------------------------------------------------------- #
# The summary refuses to overclaim                                             #
# --------------------------------------------------------------------------- #
def _rows(edges):
    specs = variants()
    return [
        {"config": s["key"], "knob": s["knob"], "value": s["value"],
         "mean_edge": e, "edge_ci": [e, e - 0.1, e + 0.1], "n_seeds": 3,
         "seed_edges": [e, e, e]}
        for s, e in zip(specs, edges)
    ]


def test_an_all_negative_sweep_makes_the_strong_claim():
    out = summarise("stock", _rows([-0.3] * 9), seeds=3)
    assert out["n_positive"] == 0
    assert "No configuration produced a positive edge" in out["verdict"]
    assert "not an artifact of one unlucky setting" in out["verdict"]


def test_a_positive_configuration_is_reported_as_descriptive_only():
    edges = [-0.3] * 9
    edges[3] = 0.12
    out = summarise("stock", _rows(edges), seeds=3)
    assert out["n_positive"] == 1
    assert "descriptive, not significant" in out["verdict"]
    assert "many more seeds" in out["verdict"]


def test_the_resolution_floor_travels_with_the_summary():
    out = summarise("stock", _rows([-0.1] * 9), seeds=3)
    assert out["sign_test_floor"] == pytest.approx(2 / 2 ** 3)
    assert "cannot produce a p-value below" in out["power_note"]
    assert "Nothing here is a significance claim" in out["power_note"]


def test_the_summary_reports_the_spread_across_recipes():
    edges = [-0.3] * 9
    edges[1], edges[2] = -0.9, -0.1
    out = summarise("stock", _rows(edges), seeds=3)
    assert out["worst_edge"] == pytest.approx(-0.9)
    assert out["best_edge"] == pytest.approx(-0.1)
    assert out["spread"] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Serving                                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture
def artifact(tmp_path, monkeypatch):
    def write(markets):
        payload = {
            "generated": "2026-08-31", "timesteps": 60000, "seeds_per_config": 3,
            "knobs": {k: v for k, v in KNOBS.items()}, "markets": markets,
        }
        path = tmp_path / "hyperparameter_sweep.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(hpsweep, "ASSET", str(path))
        return path
    return write


def _market_block(edges):
    rows = _rows(edges)
    return {"rows": rows, "summary": summarise("stock", rows, seeds=3)}


def test_rows_are_served_in_published_order_not_ranked(artifact):
    """Ranking nine recipes and quoting the winner is the mistake to avoid."""
    edges = [-0.5, -0.9, -0.1, -0.4, -0.3, -0.2, -0.6, -0.7, -0.8]
    artifact({"stock": _market_block(edges)})
    served = hpsweep.results()["markets"][0]["rows"]
    assert [r["config"] for r in served] == [s["key"] for s in variants()]
    assert served[0]["is_baseline"] is True


def test_the_headline_leads_with_the_null_when_nothing_is_positive(artifact):
    artifact({"stock": _market_block([-0.3] * 9), "crypto": _market_block([-0.2] * 9)})
    out = hpsweep.results()
    assert "None of the 18 configurations" in out["headline"]
    assert "not of one unlucky hyper-parameter choice" in out["headline"]


def test_the_headline_stays_cautious_when_something_is_positive(artifact):
    edges = [-0.3] * 9
    edges[5] = 0.2
    artifact({"stock": _market_block(edges)})
    out = hpsweep.results()
    assert "1 of 9 configurations" in out["headline"]
    assert "descriptive rather than significant" in out["headline"]
    assert "not quoting them as a result" in out["headline"]


def test_the_served_payload_never_claims_to_be_live(artifact):
    artifact({"stock": _market_block([-0.3] * 9)})
    out = hpsweep.results()
    assert out["live_computation"] is False
    assert out["generated_by"].startswith("python tools/hyperparameter_sweep.py")
    assert "not a grid search" in out["design"]
    assert any("whether ANY recipe" in c for c in out["caveats"])


def test_a_missing_artifact_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(hpsweep, "ASSET", str(tmp_path / "absent.json"))
    assert hpsweep.results() is None


def test_a_corrupt_artifact_returns_none(monkeypatch, tmp_path):
    path = tmp_path / "hyperparameter_sweep.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(hpsweep, "ASSET", str(path))
    assert hpsweep.results() is None


def test_an_artifact_with_no_markets_returns_none(artifact):
    artifact({})
    assert hpsweep.results() is None
