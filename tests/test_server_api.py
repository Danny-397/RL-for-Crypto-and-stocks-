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


def test_xray_accounts_for_every_observation_dimension(client):
    """X-Ray must explain all 563 inputs: 20x28 features + 3 account scalars."""
    r = client.post("/api/experiments", json={
        "kind": "rollout",
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 12, "n_steps": 320},
    })
    exp_id = r.get_json()["id"]
    _await(client, exp_id)

    body = client.get(f"/api/experiments/{exp_id}/xray?step=40").get_json()
    n_feat = len(body["feature_names"])
    assert len(body["window_values"]) * n_feat + len(body["account"]) == body["obs_dim"]
    assert set(body["account"]) == set(body["account_names"])

    # Groups arrive as an ordered LIST: a JSON object would get key-sorted in
    # transit and lose the semantic ordering the panel relies on.
    assert isinstance(body["feature_groups"], list)
    assert body["feature_groups"][0]["label"] == "Momentum"
    grouped = [f for g in body["feature_groups"] for f in g["features"]]
    assert sorted(grouped) == sorted(body["feature_names"])


def test_xray_replays_the_policy_to_the_requested_step(client):
    """The account state and action must be the real ones at that bar."""
    cfg = {"market": "stock", "mode": "synthetic", "regime": "momentum",
           "seed": 12, "n_steps": 320}
    r = client.post("/api/experiments", json={"kind": "rollout", "config": cfg})
    exp_id = r.get_json()["id"]
    body = _await(client, exp_id)
    step = 40
    expected = body["result"]["steps"][step]

    xray = client.get(f"/api/experiments/{exp_id}/xray?step={step}").get_json()
    assert xray["step"] == step
    # The action X-Ray reports is the action the trace recorded at that bar.
    assert xray["policy"]["action"] == pytest.approx(expected["action"], abs=1e-6)
    # Position entering the bar matches what the trace recorded.
    assert xray["account"]["position_fraction"] == pytest.approx(
        expected["position_before"], abs=1e-4
    )
    # No critic on the current archives — omitted, never approximated.
    if not xray["policy"]["value_available"]:
        assert xray["policy"]["value"] is None
        assert "omitted rather than approximated" in xray["value_note"]


def test_synthetic_paths_declare_their_inert_features(client):
    """4 of 28 features need a reference index and are zero on synthetic paths.

    Left unstated they read as four suspiciously flat rows in the X-Ray panel,
    which invites the reader to conclude the model ignores them.
    """
    r = client.post("/api/experiments", json={
        "kind": "rollout",
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 3, "n_steps": 320},
    })
    body = _await(client, r.get_json()["id"])
    meta = body["result"]["meta"]
    assert set(meta["inert_features"]) == {
        "rel_return_5", "rel_return_20", "market_trend", "market_ret_20"
    }
    assert "24 of its 28" in meta["inert_features_note"]

    # And they really are zero in the observation.
    xray = client.get(f"/api/experiments/{body['id']}/xray?step=30").get_json()
    by_name = dict(zip(xray["feature_names"], xray["current"]))
    for name in meta["inert_features"]:
        assert by_name[name] == 0.0

    # Real data has a reference index, so nothing is declared inert there.
    r2 = client.post("/api/experiments", json={
        "kind": "rollout",
        "config": {"market": "stock", "mode": "historical", "ticker": "TEST"},
    })
    meta2 = _await(client, r2.get_json()["id"])["result"]["meta"]
    assert "inert_features" not in meta2


def test_statistics_supports_paired_arms(client):
    """Ablation arms share a seed set, so arm-vs-arm must be a paired test."""
    gen = client.get("/api/generalization").get_json()
    stock = gen["markets"]["stock"]
    a = stock["domain"]["held_out"]["per_seed"]
    b = stock["single"]["held_out"]["per_seed"]

    r = client.post("/api/statistics", json={
        "values_a": a, "values_b": b, "axis": "training_seed", "n_perm": 20000,
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["n_pairs"] == len(a)
    assert body["axis"] == "training_seed"
    # Domain randomization holds up out of sample; the single-path agent does not.
    assert body["mean_a"] > body["mean_b"]
    assert body["mean_difference"] > 0
    assert body["a_wins"] == len(a)
    # At n=5 the sign-flip test bottoms out at 0.0625 and must say so.
    assert body["resolution"]["can_reach_05"] is False
    assert body["p_value"] >= body["resolution"]["min_attainable_p"]


def test_statistics_rejects_malformed_paired_arms(client):
    assert client.post("/api/statistics", json={"values_a": [1.0]}).status_code == 400
    assert client.post("/api/statistics",
                       json={"values_a": [1.0, 2.0], "values_b": [1.0]}).status_code == 400


def test_shift_sweep_reports_uncertainty_and_a_reference(client):
    """A mean over a few random paths is not a result unless its spread is shown."""
    r = client.post("/api/experiments", json={
        "kind": "distribution_shift",
        "regimes": ["momentum", "mean_reversion", "high_volatility"],
        "seeds": [0, 1, 2, 3],
        "config": {"market": "stock", "mode": "synthetic", "n_steps": 300},
    })
    body = _await(client, r.get_json()["id"], timeout=90)
    assert body["status"] == "done", body.get("error")
    result = body["result"]

    # The training distribution is momentum-like, so that is the in-distribution
    # baseline the shifted regimes are measured against.
    assert result["reference_regime"] == "momentum"
    refs = [row for row in result["regimes"] if row["is_reference"]]
    assert len(refs) == 1 and refs[0]["regime"] == "momentum"

    for row in result["regimes"]:
        assert row["n_seeds"] == 4
        # Both sides of the excess must be visible, not just the difference.
        assert "mean_agent_return" in row and "mean_benchmark_return" in row
        assert row["mean_excess_return"] == pytest.approx(
            row["mean_agent_return"] - row["mean_benchmark_return"], abs=1e-3
        )
        # Spread is reported, and "distinguishable" is a computed claim.
        assert row["std_excess"] >= 0 and row["sem_excess"] >= 0
        assert isinstance(row["excess_excludes_zero"], bool)
        if row["excess_excludes_zero"]:
            assert abs(row["mean_excess_return"]) > 2 * row["sem_excess"]

    assert "wide" in result["sampling_note"]
    assert "in-distribution" in result["reference_note"]


def test_datasets_expose_the_single_seed_headline(client):
    """The panel contrasts one run against five, so it needs the published one."""
    body = client.get("/api/datasets").get_json()
    head = body["headline_single_seed"]
    assert set(head) >= {"stock", "crypto"}

    # Asserted against docs/results.js rather than a literal. The headline is
    # whatever the current build produced, and pinning last build's number here
    # would turn every legitimate rebuild into a red test while catching nothing
    # a structural check does not. What must hold is that the API and the
    # dashboard are quoting the same run.
    published = client.get("/api/results").get_json()
    for market in ("stock", "crypto"):
        assert head[market]["total_return"] == pytest.approx(
            published["markets"][market]["metrics"]["total_return"], abs=1e-4
        )
    assert head["crypto"]["seed"] == 42
    assert head["crypto"]["source"] == "docs/results.js"

    # And the 5-seed study cannot corroborate it. Where the seeds happen to land
    # is a property of the build -- the previous version of this test pinned a
    # mean near zero and went red the first time the study was legitimately
    # re-run. What is permanent is the design: 5 pairs draw from 2**5 sign
    # assignments, so the test cannot reach p <= 0.05 at all, and no reseeding at
    # this sample size can promote one run into a result.
    stats = client.post("/api/statistics", json={"dataset": "real:crypto"}).get_json()
    assert stats["n_seeds"] == 5
    assert stats["benchmark"]["resolution"]["can_reach_05"] is False
    assert stats["benchmark"]["p_value"] > 0.05
    assert {"mean", "ci_low", "ci_high", "std"} <= set(stats["multi_seed"])


def test_experiment_records_the_callers_question(client):
    """A stated research question is kept verbatim; an absent one stays absent."""
    q = "Does the agent still work when returns mean-revert?"
    r = client.post("/api/experiments", json={
        "kind": "rollout", "question": q,
        "config": {"market": "stock", "mode": "synthetic", "regime": "mean_reversion",
                   "seed": 1, "n_steps": 300},
    })
    body = _await(client, r.get_json()["id"])
    assert body["question"] == q
    assert body["receipt"]["question"] == q
    # It is metadata about the run, not part of the environment description, so
    # it must not leak into the config that gets replayed.
    assert "question" not in body["config"]

    r2 = client.post("/api/experiments", json={
        "kind": "rollout",
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 1, "n_steps": 300},
    })
    body2 = _await(client, r2.get_json()["id"])
    # Never invented on the caller's behalf.
    assert body2["question"] is None


def test_experiment_question_is_bounded_and_trimmed(client):
    r = client.post("/api/experiments", json={
        "kind": "rollout", "question": "  " + ("x" * 900) + "  ",
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 2, "n_steps": 300},
    })
    body = _await(client, r.get_json()["id"])
    assert len(body["question"]) == 400

    blank = client.post("/api/experiments", json={
        "kind": "rollout", "question": "   ",
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 3, "n_steps": 300},
    })
    assert _await(client, blank.get_json()["id"])["question"] is None


def test_experiment_listing_carries_questions(client):
    body = client.get("/api/experiments?limit=50").get_json()
    assert any(row.get("question") for row in body["experiments"])
    assert all("question" in row for row in body["experiments"])


# --------------------------------------------------------------------------- #
# Signal-or-noise: the human pattern-detection test                            #
# --------------------------------------------------------------------------- #
def test_quiz_endpoint_serves_charts_without_the_answers(client):
    body = client.get("/api/perception/quiz?seed=5&n_charts=8").get_json()
    assert len(body["charts"]) == 8
    assert body["params"]["seed"] == 5
    # The key must not reach the browser in any form.
    assert "_key" not in body
    assert "label" not in str(body["charts"])


def test_quiz_endpoint_is_reproducible_from_its_params(client):
    a = client.get("/api/perception/quiz?seed=77&n_charts=8&market=crypto").get_json()
    b = client.get("/api/perception/quiz?seed=77&n_charts=8&market=crypto").get_json()
    assert a["charts"][3]["prices"] == b["charts"][3]["prices"]


def test_scoring_rebuilds_the_same_quiz(client):
    """Scoring must agree with an independent rebuild from the same params.

    A submission of all-ones scores exactly the number of positives, which is
    n/2 by construction — so this pins both the rebuild and the balance.
    """
    quiz = client.get("/api/perception/quiz?seed=21&n_charts=10").get_json()
    out = client.post(
        "/api/perception/score", json={"params": quiz["params"], "answers": [1] * 10}
    ).get_json()
    assert out["correct"] == 5
    assert out["n"] == 10
    assert out["significant_at_05"] is False
    assert out["power"]["min_attainable_p"] == pytest.approx(2 / 2 ** 10, abs=1e-9)
    assert out["reference"]["n"] == 10


def test_scoring_rejects_a_mismatched_answer_sheet(client):
    quiz = client.get("/api/perception/quiz?seed=2&n_charts=8").get_json()
    r = client.post(
        "/api/perception/score", json={"params": quiz["params"], "answers": [1, 0, 1]}
    )
    assert r.status_code == 400
    assert "expected 8 answers" in r.get_json()["error"]


@pytest.mark.parametrize("query", ["n_charts=7", "n_charts=99", "difficulty=magic"])
def test_quiz_endpoint_validates_its_parameters(client, query):
    r = client.get(f"/api/perception/quiz?{query}")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_real_condition_uses_the_price_fetcher(client):
    """The real-data condition must go through the same fetcher as everything else."""
    body = client.get("/api/perception/quiz?difficulty=real&ticker=SPY&n_charts=6").get_json()
    assert body["meta"]["ticker"] == "SPY"
    assert body["meta"]["positive_class"] == "real"
    assert len(body["charts"]) == 6


def test_meta_advertises_the_perception_test(client):
    assert client.get("/api/meta").get_json()["live"]["perception_test"] is True


# --------------------------------------------------------------------------- #
# Feature attribution                                                          #
# --------------------------------------------------------------------------- #
def _rollout(client, **overrides):
    config = {"market": "stock", "mode": "synthetic", "regime": "momentum",
              "seed": 4, "n_steps": 340}
    config.update(overrides)
    r = client.post("/api/experiments", json={"kind": "rollout", "config": config})
    exp_id = r.get_json()["id"]
    _await(client, exp_id)
    return exp_id


def test_attribution_ranks_every_input(client):
    exp_id = _rollout(client)
    body = client.get(f"/api/experiments/{exp_id}/attribution?step=40&bars=25").get_json()

    assert len(body["local"]["market"]) == 28
    assert len(body["local"]["account"]) == 3
    assert len(body["episode"]["features"]) == 28
    assert len(body["groups"]) == 8
    # Ranked, strongest first, on both passes.
    local = [r["abs_delta"] for r in body["local"]["market"]]
    episode = [r["mean_abs_delta"] for r in body["episode"]["features"]]
    assert local == sorted(local, reverse=True)
    assert episode == sorted(episode, reverse=True)


def test_attribution_measures_a_real_effect(client):
    """Occlusion must actually move the deployed policy, or it measures nothing."""
    exp_id = _rollout(client)
    body = client.get(f"/api/experiments/{exp_id}/attribution?bars=25").get_json()
    assert body["episode"]["features"][0]["mean_abs_delta"] > 0.01
    assert -1.0 <= body["base_action"] <= 1.0


def test_attribution_reports_the_structurally_dead_features(client):
    """A synthetic path has no reference index, so cross-asset inputs are inert.

    Naming them is the honest alternative to letting a reader wonder why four
    bars are flat.
    """
    exp_id = _rollout(client)
    body = client.get(f"/api/experiments/{exp_id}/attribution?bars=25").get_json()
    assert set(body["episode"]["dead_features"]) >= {
        "rel_return_5", "rel_return_20", "market_trend", "market_ret_20"
    }
    market_ctx = next(g for g in body["groups"] if g["label"] == "Market context")
    assert market_ctx["share"] == 0.0


def test_attribution_ships_its_own_caveats(client):
    exp_id = _rollout(client)
    body = client.get(f"/api/experiments/{exp_id}/attribution?bars=20").get_json()
    assert "occluded" in body["method"] or "occlude" in body["method"]
    assert any("not causal" in c for c in body["caveats"])
    assert body["live_computation"] is True


def test_attribution_group_shares_sum_to_one(client):
    exp_id = _rollout(client)
    body = client.get(f"/api/experiments/{exp_id}/attribution?bars=20").get_json()
    assert sum(g["share"] for g in body["groups"]) == pytest.approx(1.0, abs=1e-3)


def test_attribution_404s_on_an_unknown_experiment(client):
    r = client.get("/api/experiments/EXP-NOPE/attribution")
    assert r.status_code == 404


def test_meta_advertises_attribution(client):
    assert client.get("/api/meta").get_json()["live"]["attribution"] is True


# --------------------------------------------------------------------------- #
# Walk-forward                                                                 #
# --------------------------------------------------------------------------- #
def test_walk_forward_experiment(client):
    r = client.post("/api/experiments", json={
        "kind": "walk_forward", "n_folds": 4, "scheme": "expanding",
        "compare_leakage": True,
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 2, "n_steps": 1200},
    })
    assert r.status_code == 202
    body = _await(client, r.get_json()["id"], timeout=60)
    assert body["status"] == "done", body.get("error")
    res = body["result"]

    assert len(res["folds"]) == 4
    assert res["summary"]["n_folds"] == 4
    # Disjoint, chronological, and stated as such.
    for fold in res["folds"]:
        assert fold["train_end"] <= fold["test_start"]
    assert "not a retrained walk-forward" in res["fixed_policy_note"]
    assert res["leakage"]["identical"] is False


def test_walk_forward_reports_fold_to_fold_variance(client):
    """The panel's point: one fixed policy swings across chronological blocks."""
    r = client.post("/api/experiments", json={
        "kind": "walk_forward", "n_folds": 4,
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 2, "n_steps": 1200},
    })
    res = _await(client, r.get_json()["id"], timeout=60)["result"]
    s = res["summary"]
    assert s["best_fold_excess"] > s["worst_fold_excess"]
    assert s["sign_test_floor"] == pytest.approx(2 / 2 ** 4)
    assert s["sign_test_can_reach_05"] is False


def test_walk_forward_sliding_scheme(client):
    r = client.post("/api/experiments", json={
        "kind": "walk_forward", "n_folds": 3, "scheme": "sliding",
        "compare_leakage": False,
        "config": {"market": "stock", "mode": "synthetic", "regime": "random_walk",
                   "seed": 8, "n_steps": 1200},
    })
    res = _await(client, r.get_json()["id"], timeout=60)["result"]
    sizes = [f["train_end"] - f["train_start"] for f in res["folds"]]
    assert len(set(sizes)) == 1          # fixed-size window
    assert "leakage" not in res           # not requested, so not invented


@pytest.mark.parametrize("payload", [
    {"scheme": "sideways"}, {"train_min_frac": 0.95},
])
def test_walk_forward_validates_its_options(client, payload):
    body = {"kind": "walk_forward",
            "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                       "seed": 1, "n_steps": 900}}
    body.update(payload)
    r = client.post("/api/experiments", json=body)
    assert r.status_code == 400


def test_meta_describes_the_walk_forward_options(client):
    meta = client.get("/api/meta").get_json()
    assert meta["live"]["walk_forward"] is True
    assert {s["key"] for s in meta["walk_forward"]["schemes"]} == {"expanding", "sliding"}


def test_unknown_experiment_kind_lists_walk_forward(client):
    r = client.post("/api/experiments", json={"kind": "telepathy", "config": {
        "market": "stock", "mode": "synthetic", "regime": "momentum"}})
    assert r.status_code == 400
    assert "walk_forward" in r.get_json()["supported"]


# --------------------------------------------------------------------------- #
# Pre-registration                                                             #
# --------------------------------------------------------------------------- #
def test_a_prediction_is_stamped_before_the_run(client):
    r = client.post("/api/experiments", json={
        "kind": "rollout",
        "question": "Does the agent beat buy-and-hold on a trending path?",
        "prediction": {"direction": "beats", "note": "momentum should suit it"},
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 3, "n_steps": 500},
    })
    assert r.status_code == 202
    created = r.get_json()
    # It is on the record the instant the experiment exists, before any result.
    assert created["prediction"]["direction"] == "beats"
    assert created["prediction"]["registered_at_utc"].endswith("Z")
    assert created["has_result"] is False

    body = _await(client, created["id"])
    outcome = body["prediction_outcome"]
    assert outcome["predicted"] == "beats"
    assert outcome["observed"] in ("beats", "matches", "loses")
    assert outcome["matched"] is (outcome["observed"] == "beats")
    assert outcome["scorable"] is True


def test_the_prediction_is_on_the_reproducibility_receipt(client):
    r = client.post("/api/experiments", json={
        "kind": "rollout", "prediction": "loses",
        "config": {"market": "stock", "mode": "synthetic", "regime": "random_walk",
                   "seed": 6, "n_steps": 400},
    })
    exp_id = r.get_json()["id"]
    _await(client, exp_id)
    receipt = client.get(f"/api/experiments/{exp_id}/config").get_json()["receipt"]
    assert receipt["prediction"]["direction"] == "loses"


def test_an_experiment_without_a_prediction_records_none(client):
    r = client.post("/api/experiments", json={
        "kind": "rollout",
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 7, "n_steps": 400},
    })
    body = _await(client, r.get_json()["id"])
    assert body["prediction"] is None
    assert body["prediction_outcome"] is None


def test_an_unscorable_kind_records_but_does_not_score(client):
    r = client.post("/api/experiments", json={
        "kind": "counterfactual", "step": 30, "actions": [1.0, -1.0],
        "prediction": "beats",
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 4, "n_steps": 400},
    })
    body = _await(client, r.get_json()["id"])
    assert body["prediction"]["direction"] == "beats"
    assert body["prediction_outcome"]["scorable"] is False
    assert body["prediction_outcome"]["matched"] is None


def test_a_bad_prediction_is_rejected_before_anything_runs(client):
    r = client.post("/api/experiments", json={
        "kind": "rollout", "prediction": "sideways",
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum"},
    })
    assert r.status_code == 400
    assert "unknown prediction direction" in r.get_json()["error"]


def test_walk_forward_predictions_score_on_the_fold_mean(client):
    r = client.post("/api/experiments", json={
        "kind": "walk_forward", "n_folds": 3, "prediction": "loses",
        "compare_leakage": False,
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 2, "n_steps": 1200},
    })
    body = _await(client, r.get_json()["id"], timeout=60)
    outcome = body["prediction_outcome"]
    assert outcome["scorable"] is True
    assert outcome["observed_excess"] == pytest.approx(
        body["result"]["summary"]["mean_excess_return"], abs=1e-6
    )


def test_meta_publishes_the_judging_rule(client):
    pre = client.get("/api/meta").get_json()["preregistration"]
    assert {o["key"] for o in pre["directions"]} == {"beats", "matches", "loses"}
    assert pre["match_band"] == 0.02


# --------------------------------------------------------------------------- #
# Human baseline                                                               #
# --------------------------------------------------------------------------- #
def _start_session(client, max_steps=15):
    r = client.post("/api/human/start", json={
        "max_steps": max_steps,
        "config": {"market": "stock", "mode": "synthetic", "regime": "momentum",
                   "seed": 4, "n_steps": 420},
    })
    assert r.status_code == 201
    return r.get_json()


def test_a_session_opens_without_revealing_the_future(client):
    body = _start_session(client)
    assert body["session_id"].startswith("HUM-")
    assert body["step"] == 0
    assert len(body["prices"]) == 20          # the warm-up window, nothing more
    assert "nothing here to read ahead in" in body["lookahead_note"]
    assert "not a like-for-like" in body["information_note"]


def test_each_decision_releases_exactly_one_bar(client):
    body = _start_session(client, max_steps=12)
    sid = body["session_id"]
    for i in range(12):
        out = client.post(f"/api/human/{sid}/step", json={"action": 0.5}).get_json()
        assert out["step"] == i + 1
        assert isinstance(out["price"], float)
        assert "prices" not in out            # one bar, never a series
    assert out["done"] is True


def test_a_finished_session_scores_all_three_on_the_same_bars(client):
    body = _start_session(client, max_steps=12)
    sid = body["session_id"]
    for _ in range(12):
        client.post(f"/api/human/{sid}/step", json={"action": 1.0})
    res = client.post(f"/api/human/{sid}/finish").get_json()

    assert res["bars_traded"] == 12
    n = len(res["you"]["equity_curve"])
    assert len(res["agent"]["equity_curve"]) == n
    assert len(res["benchmark"]["equity_curve"]) == n
    assert res["you_beat_benchmark"] is (
        res["you"]["metrics"]["total_return"] > res["benchmark"]["metrics"]["total_return"]
    )
    assert "single sample" in res["sample_note"]


def test_finishing_twice_returns_the_same_scored_result(client):
    body = _start_session(client, max_steps=10)
    sid = body["session_id"]
    for _ in range(10):
        client.post(f"/api/human/{sid}/step", json={"action": 0.0})
    first = client.post(f"/api/human/{sid}/finish").get_json()
    second = client.post(f"/api/human/{sid}/finish").get_json()
    assert first == second


def test_stepping_after_the_limit_is_a_conflict_not_a_silent_noop(client):
    body = _start_session(client, max_steps=10)
    sid = body["session_id"]
    for _ in range(10):
        client.post(f"/api/human/{sid}/step", json={"action": 0.0})
    r = client.post(f"/api/human/{sid}/step", json={"action": 0.0})
    assert r.status_code == 409
    assert "already finished" in r.get_json()["error"]


def test_an_unknown_session_explains_that_sessions_expire(client):
    r = client.post("/api/human/HUM-NOPE/step", json={"action": 0.0})
    assert r.status_code == 404
    assert "expire" in r.get_json()["error"]


def test_a_bad_config_is_rejected_before_a_session_exists(client):
    r = client.post("/api/human/start", json={"config": {"market": "forex"}})
    assert r.status_code == 400
    assert "unknown market" in r.get_json()["error"]


def test_meta_advertises_the_human_baseline(client):
    meta = client.get("/api/meta").get_json()
    assert meta["live"]["human_baseline"] is True
    assert "ephemeral" in meta["human_sessions"]["storage"]


# --------------------------------------------------------------------------- #
# Surrogate-data test                                                          #
# --------------------------------------------------------------------------- #
def test_surrogate_endpoint_serves_both_arms(client):
    body = client.get("/api/surrogate").get_json()
    arms = {a["arm"]: a for a in body["arms"]}
    assert set(arms) == {"synthetic", "real"}
    # The positive control is served first: it is what licenses reading the rest.
    assert body["arms"][0]["arm"] == "synthetic"
    assert "Theiler" in body["reference"]


def test_surrogate_endpoint_does_not_claim_to_be_live(client):
    body = client.get("/api/surrogate").get_json()
    assert body["live_computation"] is False
    for arm in body["arms"]:
        assert arm["generated_by"].startswith("python tools/surrogate_test.py")


def test_meta_marks_the_surrogate_test_as_not_live(client):
    """Both arms are training runs, so this must not sit under the live flags."""
    assert client.get("/api/meta").get_json()["live"]["surrogate_test"] is False


# --------------------------------------------------------------------------- #
# Baselines on the rollout                                                     #
# --------------------------------------------------------------------------- #
def test_every_rollout_carries_the_baseline_comparison(client):
    exp_id = _rollout(client)
    bl = client.get(f"/api/experiments/{exp_id}").get_json()["result"]["baselines"]
    keys = {r["key"] for r in bl["rows"]}
    assert keys == {"agent", "buy_and_hold", "ma_crossover", "flat", "random"}
    assert bl["live_computation"] is True
    assert bl["agent_rank"] >= 1


def test_the_random_arm_is_many_draws_not_one(client):
    exp_id = _rollout(client)
    bl = client.get(f"/api/experiments/{exp_id}").get_json()["result"]["baselines"]
    random_row = next(r for r in bl["rows"] if r["key"] == "random")
    assert random_row["n_seeds"] > 1
    assert random_row["worst"] < random_row["best"]


def test_both_buy_and_holds_are_reported(client):
    """The charted benchmark is cost-free; the strategy pays costs. Neither hides."""
    exp_id = _rollout(client)
    result = client.get(f"/api/experiments/{exp_id}").get_json()["result"]
    bl = result["baselines"]
    costed = next(r for r in bl["rows"] if r["key"] == "buy_and_hold")["total_return"]
    free = bl["cost_free_benchmark"]["total_return"]
    assert free == pytest.approx(result["bench_metrics"]["total_return"], abs=1e-6)
    assert costed <= free + 1e-9


# --------------------------------------------------------------------------- #
# Hyper-parameter sweep                                                        #
# --------------------------------------------------------------------------- #
def test_hyperparameter_endpoint_passes_the_sweep_through(client, monkeypatch):
    from server import hpsweep

    fixture = {
        "generated": "2026-08-31", "timesteps": 60000, "seeds_per_config": 3,
        "knobs": {"clip_ratio": [0.1, 0.3]}, "markets": [],
        "design": "one factor at a time", "caveats": ["x"],
        "source": "docs/assets/hyperparameter_sweep.json",
        "generated_by": "python tools/hyperparameter_sweep.py",
        "live_computation": False, "headline": "None of the 9 configurations",
    }
    monkeypatch.setattr(hpsweep, "results", lambda: fixture)
    body = client.get("/api/hyperparameters").get_json()
    assert body["headline"].startswith("None of the")
    assert body["live_computation"] is False


def test_a_missing_sweep_artifact_503s_rather_than_faking_one(client, monkeypatch):
    from server import hpsweep

    monkeypatch.setattr(hpsweep, "results", lambda: None)
    r = client.get("/api/hyperparameters")
    assert r.status_code == 503
    assert "unavailable" in r.get_json()["error"]


def test_meta_marks_the_sweep_as_not_live(client):
    assert client.get("/api/meta").get_json()["live"]["hyperparameter_sweep"] is False
