"""HTTP-level tests for the lab API.

These drive the Flask app through its test client, so they cover routing,
validation, and the async experiment lifecycle end to end.

Network is never touched: ``_fetch_ohlcv`` and ``_market_index`` are monkey-
patched with a locally generated frame, which keeps the suite fast and
deterministic and means CI does not depend on Yahoo being reachable.
"""

from __future__ import annotations

import time

import pytest

flask = pytest.importorskip("flask")

from rl_trader.data.data_loader import generate_synthetic_ohlcv  # noqa: E402
from server import app as server_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    server_app.app.config.update(TESTING=True)
    return server_app.app.test_client()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Serve a deterministic local frame instead of calling Yahoo."""
    df = generate_synthetic_ohlcv(n_steps=420, seed=11)

    monkeypatch.setattr(server_app, "_fetch_ohlcv", lambda ticker, attempts=3: (df.copy(), None))
    monkeypatch.setattr(server_app, "_market_index", lambda market: df.copy())
    return df


def _await(client, exp_id, timeout=25.0):
    """Poll an experiment to completion, as the frontend does."""
    deadline = time.time() + timeout
    body = {}
    while time.time() < deadline:
        body = client.get(f"/api/experiments/{exp_id}").get_json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"experiment {exp_id} did not finish: {body}")


# --------------------------------------------------------------------------- #
# Discovery                                                                    #
# --------------------------------------------------------------------------- #
def test_health_reports_policy_capabilities(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["policies"]
    for name in body["policies"]:
        assert "value_head" in body["capabilities"][name]


def test_meta_declares_that_training_is_not_live(client):
    body = client.get("/api/meta").get_json()
    # The single most important honesty claim on the API.
    assert body["live"]["training"] is False
    assert body["live"]["rollout"] is True
    assert body["live"]["statistics"] is True
    assert "does not train" in body["training_note"]
    assert body["reward_kinds"] and body["markets"]


def test_regimes_are_listed_and_labelled_synthetic(client):
    body = client.get("/api/regimes").get_json()
    assert body["synthetic"] is True
    keys = {r["key"] for r in body["regimes"]}
    assert {"random_walk", "momentum", "mean_reversion", "high_volatility"} <= keys
    assert all(r["description"] for r in body["regimes"])


def test_datasets_carry_provenance(client):
    body = client.get("/api/datasets").get_json()
    assert body["seed_datasets"]
    for row in body["seed_datasets"]:
        assert row["source"] and row["generated_by"]


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #
def test_statistics_reproduces_the_published_interval(client):
    r = client.post("/api/statistics", json={"dataset": "real:stock"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["live_computation"] is True
    pub = body["published"]
    assert body["multi_seed"]["ci_low"] == pytest.approx(pub["ci_low"], abs=5e-4)
    assert body["multi_seed"]["ci_high"] == pytest.approx(pub["ci_high"], abs=5e-4)
    # The seed-axis test must disclose that it cannot reach 0.05 at n=5.
    assert body["benchmark"]["resolution"]["can_reach_05"] is False


def test_statistics_on_the_paper_axis(client):
    r = client.post("/api/statistics", json={"dataset": "assets:stock", "n_perm": 20000})
    body = r.get_json()
    assert body["axis"] == "held_out_ticker"
    assert body["n_pairs"] == 10
    assert body["p_value"] == pytest.approx(0.0021, abs=5e-4)
    assert body["caveat"]


def test_statistics_accepts_caller_values(client):
    r = client.post("/api/statistics", json={"values": [2.75, -0.18, 0.04, -0.21, -0.02]})
    body = r.get_json()
    assert body["single_seed"]["best"] == 2.75
    assert body["distribution"]["counts"]


def test_statistics_rejects_bad_input(client):
    assert client.post("/api/statistics", json={}).status_code == 400
    assert client.post("/api/statistics", json={"dataset": "nope"}).status_code == 404


def test_generalization_returns_real_ablation(client):
    body = client.get("/api/generalization").get_json()
    assert body["live"] is False
    assert body["generated_by"].startswith("python tools/ablation_multiseed.py")
    stock = body["markets"]["stock"]
    # Single-path memorises: a huge in-sample number that does not survive.
    assert stock["single"]["in_sample"]["mean"] > stock["single"]["held_out"]["mean"]
    assert stock["single"]["generalization_gap"] > stock["domain"]["generalization_gap"]


# --------------------------------------------------------------------------- #
# Experiments                                                                  #
# --------------------------------------------------------------------------- #
def test_rollout_experiment_end_to_end(client):
    r = client.post(
        "/api/experiments",
        json={"kind": "rollout", "config": {"market": "stock", "mode": "historical",
                                            "ticker": "TEST", "initial_balance": 50_000}},
    )
    assert r.status_code == 202
    body = _await(client, r.get_json()["id"])
    assert body["status"] == "done", body.get("error")

    result = body["result"]
    assert result["n_steps"] > 50
    assert len(result["feature_names"]) == 28
    assert result["obs_dim"] == result["window_size"] * 28 + 3
    assert result["equity_curve"][0] == pytest.approx(50_000, rel=1e-6)
    assert "inference_note" in result
    # Receipt must identify the code and the data behind the number.
    receipt = body["receipt"]
    assert receipt["experiment_id"] == body["id"]
    assert receipt["provenance"]["dataset_hash"]
    assert receipt["provenance"]["policy"]["sha256"]


def test_rollout_on_a_synthetic_regime_is_labelled(client):
    r = client.post(
        "/api/experiments",
        json={"kind": "rollout", "config": {"market": "stock", "mode": "synthetic",
                                            "regime": "mean_reversion", "seed": 3,
                                            "n_steps": 320}},
    )
    body = _await(client, r.get_json()["id"])
    assert body["status"] == "done", body.get("error")
    meta = body["result"]["meta"]
    assert meta["synthetic"] is True
    assert meta["regime"] == "mean_reversion"
    # The realised statistics of the actual path are reported, not the nominal ones.
    assert meta["realised"]["return_autocorr_lag1"] < 0


def test_transaction_cost_changes_the_trajectory(client):
    """Costs feed back through account state, so they must move the result."""
    def run(cost):
        r = client.post("/api/experiments", json={
            "kind": "rollout",
            "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                       "seed": 5, "n_steps": 320, "transaction_cost": cost},
        })
        return _await(client, r.get_json()["id"])["result"]["metrics"]["final_equity"]

    assert run(0.0) != pytest.approx(run(0.01), rel=1e-6)


def test_xray_returns_the_real_observation(client):
    r = client.post("/api/experiments", json={
        "kind": "rollout",
        "config": {"market": "stock", "mode": "synthetic", "regime": "random_walk",
                   "seed": 1, "n_steps": 320},
    })
    exp_id = r.get_json()["id"]
    _await(client, exp_id)

    body = client.get(f"/api/experiments/{exp_id}/xray?step=25").get_json()
    assert len(body["feature_names"]) == 28
    assert len(body["current"]) == 28
    assert len(body["window_values"]) == body["window"]
    assert body["obs_dim"] == body["window"] * 28 + 3
    assert "standardised" in body["scaling_note"]


def test_counterfactual_experiment(client):
    r = client.post("/api/experiments", json={
        "kind": "counterfactual",
        "step": 30, "actions": [1.0, 0.0, -1.0], "horizon": 3,
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 2, "n_steps": 320},
    })
    body = _await(client, r.get_json()["id"])
    assert body["status"] == "done", body.get("error")
    result = body["result"]
    assert len(result["candidates"]) == 3
    assert result["horizon"] == 3
    assert "does not imply" in result["note"]
    # Long and short from the same state cannot land on the same equity.
    longer = next(c for c in result["candidates"] if c["action"] == 1.0)
    shorter = next(c for c in result["candidates"] if c["action"] == -1.0)
    assert longer["end_equity"] != pytest.approx(shorter["end_equity"], rel=1e-9)


def test_distribution_shift_sweep(client):
    r = client.post("/api/experiments", json={
        "kind": "distribution_shift",
        "regimes": ["random_walk", "mean_reversion"], "seeds": [0, 1],
        "config": {"market": "stock", "mode": "synthetic", "n_steps": 300},
    })
    body = _await(client, r.get_json()["id"], timeout=60)
    assert body["status"] == "done", body.get("error")
    rows = body["result"]["regimes"]
    assert len(rows) == 2
    assert all(len(row["per_seed"]) == 2 for row in rows)
    assert body["result"]["synthetic"] is True


def test_experiment_config_is_reproducible(client):
    r = client.post("/api/experiments", json={
        "kind": "rollout",
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 9, "n_steps": 300},
    })
    exp_id = r.get_json()["id"]
    _await(client, exp_id)
    cfg = client.get(f"/api/experiments/{exp_id}/config").get_json()
    assert cfg["config"]["seed"] == 9
    assert cfg["config"]["regime"] == "momentum"
    # Defaults must be resolved to concrete values, not left as sentinels, or the
    # emitted config would silently rebuild a different environment.
    assert cfg["config"]["transaction_cost"] > 0
    assert cfg["config"]["slippage"] > 0

    # Replaying the returned config must reproduce the result exactly...
    again = client.post("/api/experiments", json={"kind": "rollout", "config": cfg["config"]})
    again_id = again.get_json()["id"]
    first = client.get(f"/api/experiments/{exp_id}").get_json()["result"]["metrics"]
    second = _await(client, again_id)["result"]["metrics"]
    assert second["final_equity"] == pytest.approx(first["final_equity"], rel=1e-9)
    assert second["sharpe"] == pytest.approx(first["sharpe"], rel=1e-9)

    # ...and the config must itself be a fixed point of the round trip.
    replayed = client.get(f"/api/experiments/{again_id}/config").get_json()["config"]
    assert replayed == cfg["config"]


def test_market_defaults_are_applied_when_unspecified(client):
    """An omitted mechanic must fall back to the market preset, never to zero."""
    r = client.post("/api/experiments", json={
        "kind": "rollout",
        "config": {"market": "crypto", "mode": "synthetic", "regime": "momentum",
                   "seed": 4, "n_steps": 300},
    })
    body = _await(client, r.get_json()["id"])
    env = body["receipt"]["provenance"]["env"]
    # crypto_config(): 10 bps cost, 10 bps slippage.
    assert env["transaction_cost"] == pytest.approx(0.0010)
    assert env["slippage"] == pytest.approx(0.0010)


def test_experiment_listing_and_missing_ids(client):
    body = client.get("/api/experiments?limit=5").get_json()
    assert "experiments" in body and "ephemeral" in body["storage"]
    assert client.get("/api/experiments/EXP-NOPE1").status_code == 404
    assert client.get("/api/experiments/EXP-NOPE1/config").status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "rollout", "config": {"market": "forex", "mode": "historical", "ticker": "X"}},
        {"kind": "rollout", "config": {"market": "stock", "mode": "nonsense"}},
        {"kind": "rollout", "config": {"market": "stock", "mode": "historical"}},
        {"kind": "rollout", "config": {"market": "stock", "mode": "synthetic", "regime": "moon"}},
        {"kind": "teleport", "config": {"market": "stock", "mode": "synthetic"}},
    ],
)
def test_invalid_experiment_requests_are_rejected(client, payload):
    assert client.post("/api/experiments", json=payload).status_code == 400


def test_failed_experiment_surfaces_its_error(client, monkeypatch):
    monkeypatch.setattr(
        server_app, "_fetch_ohlcv", lambda ticker, attempts=3: (None, "upstream unavailable")
    )
    r = client.post("/api/experiments", json={
        "kind": "rollout",
        "config": {"market": "stock", "mode": "historical", "ticker": "TEST"},
    })
    body = _await(client, r.get_json()["id"])
    assert body["status"] == "error"
    assert "upstream unavailable" in body["error"]


def test_dashboard_endpoints_still_work(client):
    """The static site depends on these; the lab must not break them."""
    assert client.get("/api/tickers").status_code == 200
    assert client.get("/api/results").status_code == 200
    live = client.get("/api/live?market=stock&ticker=TEST").get_json()
    assert live["equity_agent"] and live["equity_bench"]
    assert live["market"] == "stock"
