"""Tests for the interactive-lab backend (server/).

These lock in the properties the lab's honesty depends on:

* the served policy is the *same* function the research code trains,
* a rollout trace is internally consistent with the environment that produced it,
* counterfactuals really do replay from an identical state,
* synthetic regimes have the statistical character they claim,
* the statistics layer reproduces the published, committed numbers.

The last one is the important one: if these drift from ``docs/`` the site would
start telling a different story than the paper.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl_trader.config.training_config import stock_config
from rl_trader.data.data_loader import synthetic_market_data
from rl_trader.envs import make_env
from server import precomputed, regimes, stats_api
from server.experiments import ExperimentManager, progress_reporter
from server.policy import load_policies
from server.rollout import counterfactual, observation_detail, run_trace


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def policies():
    pols = load_policies()
    if not pols:
        pytest.skip("no exported policies in server/models")
    return pols


@pytest.fixture(scope="module")
def env_and_policy(policies):
    market = "stock" if "stock" in policies else sorted(policies)[0]
    cfg = stock_config()
    data = synthetic_market_data(market, seed=7, n_steps=320)
    env = make_env(market, data, cfg.env, cfg.reward, random_start=False)
    return env, policies[market], market, cfg


# --------------------------------------------------------------------------- #
# Policy                                                                       #
# --------------------------------------------------------------------------- #
def test_policy_action_is_a_bounded_target_position(env_and_policy):
    env, policy, _market, _cfg = env_and_policy
    obs, _info = env.reset()
    out = policy.evaluate(obs)
    assert -1.0 <= out.action <= 1.0
    # The critic is optional; when absent it must be None, never a stand-in value.
    if not policy.has_value:
        assert out.value is None
    else:
        assert isinstance(out.value, float)


def test_policy_is_deterministic(env_and_policy):
    env, policy, _market, _cfg = env_and_policy
    obs, _info = env.reset()
    assert policy.evaluate(obs).action == policy.evaluate(obs).action


def test_policy_describe_reports_capabilities(policies):
    for policy in policies.values():
        d = policy.describe()
        assert d["has_value_head"] == policy.has_value
        assert len(d["sha256"]) == 16


# --------------------------------------------------------------------------- #
# Rollout traces                                                               #
# --------------------------------------------------------------------------- #
def test_trace_is_internally_consistent(env_and_policy):
    env, policy, market, _cfg = env_and_policy
    trace = run_trace(policy, env, market=market)

    assert len(trace.steps) > 10
    # One equity point per step, plus the starting balance.
    assert len(trace.equity_curve) == len(trace.steps) + 1
    assert len(trace.bench_curve) == len(trace.equity_curve)

    # Steps are contiguous and advance one bar at a time.
    assert [s.step for s in trace.steps] == list(range(len(trace.steps)))
    ts = [s.t for s in trace.steps]
    assert ts == sorted(ts) and np.all(np.diff(ts) == 1)

    for s in trace.steps:
        assert -1.0 <= s.action <= 1.0
        # After rebalancing, exposure equals the requested target.
        assert s.position_after == s.action
        assert s.cost >= 0.0
        assert 0.0 <= s.drawdown <= 1.0
        assert np.isfinite(s.equity) and np.isfinite(s.reward)

    # The recorded equity really is the curve.
    assert trace.equity_curve[1:] == [s.equity for s in trace.steps]
    assert trace.metrics["final_equity"] == pytest.approx(trace.equity_curve[-1], rel=1e-6)


def test_trace_value_flag_matches_policy(env_and_policy):
    env, policy, market, _cfg = env_and_policy
    trace = run_trace(policy, env, market=market)
    assert trace.value_available == policy.has_value
    if not policy.has_value:
        assert all(s.value is None for s in trace.steps)


def test_observation_detail_matches_the_real_input(env_and_policy):
    env, _policy, _market, cfg = env_and_policy
    env.reset()
    detail = observation_detail(
        env, t=60, feature_names=env.data.feature_names, window=cfg.env.window_size
    )
    assert len(detail["feature_names"]) == env.data.features.shape[1]
    assert len(detail["current"]) == env.data.features.shape[1]
    assert len(detail["window_values"]) == cfg.env.window_size
    # The advertised obs_dim must equal what the policy actually consumes.
    assert detail["obs_dim"] == cfg.env.window_size * env.data.features.shape[1] + 3
    # The newest row really is the bar's features.
    assert detail["current"] == [round(float(v), 6) for v in env.data.features[60]]


# --------------------------------------------------------------------------- #
# Counterfactuals                                                              #
# --------------------------------------------------------------------------- #
def test_counterfactual_replays_from_an_identical_state(env_and_policy):
    env, policy, market, _cfg = env_and_policy
    cf = counterfactual(policy, env, target_step=40, actions=[1.0, 0.0, -1.0], market=market)

    assert cf["step"] == 40
    assert len(cf["candidates"]) == 3
    # Every candidate starts from the same equity, so differences are attributable
    # to the action alone.
    assert all(c["steps"] == 1 for c in cf["candidates"])
    equity_before = cf["equity_before"]
    for c in cf["candidates"]:
        assert c["equity_change"] == pytest.approx(c["end_equity"] - equity_before, abs=1e-6)

    # Distinct actions must produce distinct outcomes (the state is real).
    ends = [c["end_equity"] for c in cf["candidates"]]
    assert len(set(round(e, 4) for e in ends)) > 1


def test_counterfactual_agrees_with_the_trace_it_branches_from(env_and_policy):
    """Replaying the agent's *own* action must reproduce the trace exactly."""
    env, policy, market, _cfg = env_and_policy
    trace = run_trace(policy, env, market=market)
    step = 25
    expected = trace.steps[step]

    cf = counterfactual(policy, env, target_step=step, actions=[expected.action], market=market)
    assert cf["agent_action"] == pytest.approx(expected.action, abs=1e-6)
    assert cf["t"] == expected.t
    assert cf["candidates"][0]["end_equity"] == pytest.approx(expected.equity, rel=1e-6)
    assert cf["candidates"][0]["is_agent_action"]


def test_counterfactual_horizon_holds_the_position(env_and_policy):
    env, policy, market, _cfg = env_and_policy
    cf = counterfactual(policy, env, target_step=30, actions=[0.5], market=market, horizon=5)
    assert cf["horizon"] == 5
    assert cf["candidates"][0]["steps"] == 5


# --------------------------------------------------------------------------- #
# Synthetic regimes                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "regime,lo,hi",
    [
        ("momentum", 0.05, 1.0),        # trends persist
        ("random_walk", -0.03, 0.03),   # memoryless
        ("mean_reversion", -1.0, -0.03),  # moves reverse
    ],
)
def test_regimes_have_the_autocorrelation_they_claim(regime, lo, hi):
    """A regime that does not have its advertised character would be a fake label."""
    acs = [
        regimes.build_regime_data(regime, seed=s, n_steps=700)[1]["realised"][
            "return_autocorr_lag1"
        ]
        for s in range(6)
    ]
    mean_ac = float(np.mean(acs))
    assert lo <= mean_ac <= hi, f"{regime}: lag-1 autocorr {mean_ac:+.4f} outside [{lo}, {hi}]"


def test_high_volatility_regime_is_actually_more_volatile():
    _d, calm = regimes.build_regime_data("random_walk", seed=1, n_steps=700)
    _d, wild = regimes.build_regime_data("high_volatility", seed=1, n_steps=700)
    assert wild["realised"]["annualised_vol"] > 2 * calm["realised"]["annualised_vol"]


def test_regime_data_is_labelled_synthetic_and_usable():
    data, meta = regimes.build_regime_data("regime_switch", seed=3, n_steps=400)
    assert meta["synthetic"] is True
    assert np.isfinite(data.features).all()
    assert len(data) > 100
    # Spliced in log-return space, so the price path must stay strictly positive.
    assert (data.prices > 0).all()


def test_every_listed_regime_can_be_built():
    for spec in regimes.list_regimes():
        data, meta = regimes.build_regime_data(spec["key"], seed=0, n_steps=300)
        assert meta["label"] == spec["label"]
        assert len(data) > 50


# --------------------------------------------------------------------------- #
# Statistics — must reproduce the committed, published numbers                 #
# --------------------------------------------------------------------------- #
def test_bootstrap_reproduces_published_confidence_interval():
    ds = precomputed.seed_datasets()
    if "real:stock" not in ds:
        pytest.skip("docs/significance.js not available")
    d = ds["real:stock"]
    out = stats_api.analyze(d["values"], benchmark=d["benchmark"])
    pub = d["published"]
    assert out["multi_seed"]["mean"] == pytest.approx(pub["mean"], abs=5e-4)
    assert out["multi_seed"]["ci_low"] == pytest.approx(pub["ci_low"], abs=5e-4)
    assert out["multi_seed"]["ci_high"] == pytest.approx(pub["ci_high"], abs=5e-4)


def test_per_ticker_test_reproduces_the_published_p_value():
    """The paper pairs across held-out TICKERS; that axis must reproduce."""
    pa = precomputed.paired_asset_datasets()
    if "assets:stock" not in pa:
        pytest.skip("docs/results.js not available")
    d = pa["assets:stock"]
    out = stats_api.compare(d["agent"], d["benchmark"], axis="held_out_ticker")
    # Published: p = 0.0021 (RESULTS.md / docs/significance.js).
    assert out["p_value"] == pytest.approx(0.0021, abs=5e-4)
    assert out["mean_difference"] < 0  # the stock agent trails buy-&-hold
    assert out["n_pairs"] == 10


def test_permutation_floor_is_reported_and_correct():
    """A sign-flip test over n pairs cannot return p below 2 / 2**n."""
    assert stats_api.permutation_floor(5)["min_attainable_p"] == pytest.approx(0.0625)
    assert stats_api.permutation_floor(6)["min_attainable_p"] == pytest.approx(0.03125)
    assert stats_api.permutation_floor(10)["min_attainable_p"] == pytest.approx(0.001953125)
    # n = 5 can never reach significance at 0.05 — the UI must be able to say so.
    assert stats_api.permutation_floor(5)["can_reach_05"] is False
    assert stats_api.permutation_floor(10)["can_reach_05"] is True


def test_analyze_contrasts_single_and_multi_seed():
    out = stats_api.analyze([2.75, -0.18, 0.04, -0.21, -0.02])
    assert out["single_seed"]["best"] == 2.75
    assert out["multi_seed"]["mean"] < 1.0  # the lucky seed does not survive averaging
    assert out["single_seed"]["spread"] == pytest.approx(2.96)


def test_compare_rejects_mismatched_arms():
    with pytest.raises(ValueError):
        stats_api.compare([1.0, 2.0, 3.0], [1.0, 2.0])


def test_seed_datasets_carry_provenance():
    for d in precomputed.seed_datasets().values():
        assert d["source"] and d["generated_by"]
        assert len(d["values"]) == d["seeds"]


# --------------------------------------------------------------------------- #
# Experiment manager                                                           #
# --------------------------------------------------------------------------- #
def _wait(exp, timeout: float = 10.0):
    import time

    deadline = time.time() + timeout
    while exp.status in ("queued", "running") and time.time() < deadline:
        time.sleep(0.01)
    return exp


def test_experiment_runs_and_records_a_receipt():
    mgr = ExperimentManager()
    exp = mgr.create("unit", {"a": 1}, lambda e: {"answer": 42})
    _wait(exp)
    assert exp.status == "done"
    assert exp.result == {"answer": 42}
    assert exp.progress == 1.0
    receipt = exp.receipt()
    assert receipt["experiment_id"] == exp.id
    assert receipt["config"] == {"a": 1}


def test_experiment_records_failures_rather_than_hiding_them():
    mgr = ExperimentManager()

    def boom(_e):
        raise RuntimeError("deliberate")

    exp = _wait(mgr.create("unit", {}, boom))
    assert exp.status == "error"
    assert "deliberate" in exp.error
    assert exp.result is None


def test_progress_reporter_tracks_real_units():
    mgr = ExperimentManager()

    def work(e):
        report = progress_reporter(e, total=4, stage="scoring")
        for i in range(4):
            report(i + 1)
        return {"ok": True}

    exp = _wait(mgr.create("unit", {}, work))
    assert exp.status == "done"
    assert exp.progress == 1.0


def test_registry_lists_and_evicts():
    mgr = ExperimentManager(max_experiments=3)
    for _ in range(5):
        _wait(mgr.create("unit", {}, lambda e: {"ok": True}))
    assert mgr.stats()["total"] <= 3
    listing = mgr.list()
    assert all("id" in row and "status" in row for row in listing)


def test_experiment_ids_are_unique_and_readable():
    mgr = ExperimentManager()
    ids = {mgr.create("unit", {}, lambda e: {}).id for _ in range(25)}
    assert len(ids) == 25
    assert all(i.startswith("EXP-") and len(i) == 9 for i in ids)
