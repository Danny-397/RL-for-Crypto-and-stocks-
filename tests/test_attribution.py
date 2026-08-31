"""Tests for occlusion attribution.

The X-Ray's attribution chart is a ranked bar chart, which is exactly the kind of
output a reader will over-trust. So the index arithmetic underneath it is pinned
against policies whose behaviour is known exactly: if attribution says a feature
moved the action, that feature really is the one the policy read.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl_trader.config.training_config import stock_config
from rl_trader.data.data_loader import FEATURE_GROUPS, synthetic_market_data
from rl_trader.envs import make_env
from server import attribution

WINDOW = 20


class OneInputPolicy:
    """Reads a single observation index and nothing else.

    Gives attribution a ground truth: exactly one feature may show a non-zero
    effect, and it must be the feature that owns that index.
    """

    def __init__(self, index: int, weight: float = 3.0) -> None:
        self.index = index
        self.weight = weight

    def act(self, obs) -> float:
        return float(np.tanh(self.weight * np.asarray(obs, dtype=float)[self.index]))


def _obs(n_features: int, window: int = WINDOW) -> np.ndarray:
    """A deterministic observation with a distinct value in every slot."""
    rng = np.random.default_rng(0)
    return rng.normal(size=window * n_features + 3).astype(np.float32)


# --------------------------------------------------------------------------- #
# Index arithmetic                                                             #
# --------------------------------------------------------------------------- #
def test_feature_indices_stride_the_whole_window():
    """The observation is row-major, so one feature is a column of the window."""
    idx = attribution.feature_indices(window=4, n_features=3, j=1)
    assert list(idx) == [1, 4, 7, 10]


def test_occlusion_isolates_exactly_the_feature_that_was_read():
    names = [f"f{i}" for i in range(6)]
    obs = _obs(len(names))
    # Index 2 in the newest row belongs to feature 2.
    read = (WINDOW - 1) * len(names) + 2
    out = attribution.local_attribution(
        OneInputPolicy(read), obs, np.zeros(len(names)), names, WINDOW
    )
    moved = [r["name"] for r in out["market"] if r["abs_delta"] > 1e-9]
    assert moved == ["f2"]
    assert all(r["abs_delta"] == 0.0 for r in out["account"])


def test_occlusion_covers_the_oldest_bar_too():
    """A feature is occluded across all 20 bars, not just the newest one.

    If the stride were wrong, a policy reading only the oldest bar would look
    like it read nothing at all.
    """
    names = [f"f{i}" for i in range(6)]
    obs = _obs(len(names))
    out = attribution.local_attribution(
        OneInputPolicy(3), obs, np.zeros(len(names)), names, WINDOW  # row 0, feature 3
    )
    moved = [r["name"] for r in out["market"] if r["abs_delta"] > 1e-9]
    assert moved == ["f3"]


def test_account_scalars_are_attributed_separately():
    names = [f"f{i}" for i in range(6)]
    obs = _obs(len(names))
    for k, name in enumerate(attribution.ACCOUNT_NAMES):
        out = attribution.local_attribution(
            OneInputPolicy(len(obs) - 3 + k), obs, np.zeros(len(names)), names, WINDOW
        )
        moved = [r["name"] for r in out["account"] if r["abs_delta"] > 1e-9]
        assert moved == [name]
        assert all(r["abs_delta"] == 0.0 for r in out["market"])


def test_baseline_action_is_the_unoccluded_action():
    names = [f"f{i}" for i in range(6)]
    obs = _obs(len(names))
    policy = OneInputPolicy(7)
    out = attribution.local_attribution(policy, obs, np.zeros(len(names)), names, WINDOW)
    assert out["base_action"] == pytest.approx(policy.act(obs), abs=1e-6)
    # occluded_action and delta must agree with each other.
    for row in out["market"] + out["account"]:
        assert row["occluded_action"] - out["base_action"] == pytest.approx(
            row["delta_action"], abs=2e-6
        )


def test_occluding_to_the_current_value_changes_nothing():
    """Sanity: the effect comes from the substitution, not from the machinery."""
    names = [f"f{i}" for i in range(4)]
    obs = np.zeros(WINDOW * len(names) + 3, dtype=np.float32)
    out = attribution.local_attribution(
        OneInputPolicy(5), obs, np.zeros(len(names)), names, WINDOW
    )
    assert all(r["abs_delta"] == 0.0 for r in out["market"])


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #
def test_group_shares_sum_to_one_and_map_features_correctly():
    rows = [{"name": n, "mean_abs_delta": 1.0}
            for names in FEATURE_GROUPS.values() for n in names]
    groups = attribution.group_shares(rows, FEATURE_GROUPS)
    assert sum(g["share"] for g in groups) == pytest.approx(1.0, abs=1e-3)
    momentum = next(g for g in groups if g["label"] == "Momentum")
    assert momentum["n_features"] == len(FEATURE_GROUPS["Momentum"])
    assert momentum["total_abs_delta"] == pytest.approx(len(FEATURE_GROUPS["Momentum"]))


def test_group_shares_survive_an_all_zero_measurement():
    rows = [{"name": n, "mean_abs_delta": 0.0}
            for names in FEATURE_GROUPS.values() for n in names]
    groups = attribution.group_shares(rows, FEATURE_GROUPS)
    assert all(g["share"] == 0.0 for g in groups)  # no division by zero


def test_dead_features_are_reported():
    rows = [{"name": "a", "mean_abs_delta": 0.5}, {"name": "b", "mean_abs_delta": 0.0}]
    assert attribution.dead_features(rows) == ["b"]


# --------------------------------------------------------------------------- #
# Episode pass, against a real environment                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def env():
    cfg = stock_config()
    data = synthetic_market_data("stock", seed=3, n_steps=400)
    return make_env("stock", data, cfg.env, cfg.reward, random_start=False)


def test_episode_attribution_samples_and_summarises(env):
    names = list(env.data.feature_names)
    policy = OneInputPolicy((WINDOW - 1) * len(names) + 0)  # newest return_1
    out = attribution.episode_attribution(policy, env, names, WINDOW, max_bars=20)

    assert 1 <= out["bars_sampled"] <= 25
    assert out["bars_total"] > out["bars_sampled"]
    assert len(out["features"]) == len(names)
    moved = [r["name"] for r in out["features"] if r["mean_abs_delta"] > 1e-9]
    assert moved == [names[0]]
    # Magnitudes are averages of absolute values, so a peak can never be smaller.
    for row in out["features"]:
        assert row["max_abs_delta"] >= row["mean_abs_delta"] >= 0.0


def test_episode_attribution_is_deterministic(env):
    names = list(env.data.feature_names)
    policy = OneInputPolicy(11)
    a = attribution.episode_attribution(policy, env, names, WINDOW, max_bars=15)
    env.reset()
    b = attribution.episode_attribution(policy, env, names, WINDOW, max_bars=15)
    assert a["features"] == b["features"]


def test_summarise_ranks_and_attaches_its_own_limits():
    names = [f"f{i}" for i in range(4)]
    local = {
        "base_action": 0.1,
        "market": [{"name": "f0", "abs_delta": 0.1, "delta_action": 0.1},
                   {"name": "f1", "abs_delta": 0.9, "delta_action": -0.9}],
        "account": [],
    }
    episode = {
        "bars_sampled": 5, "bars_total": 50, "stride": 10,
        "features": [{"name": "f0", "mean_abs_delta": 0.2, "max_abs_delta": 0.3},
                     {"name": "f1", "mean_abs_delta": 0.7, "max_abs_delta": 0.8}],
        "account": [],
    }
    out = attribution.summarise(local, episode, {"G": names})
    assert [r["name"] for r in out["local"]["market"]] == ["f1", "f0"]
    assert [r["name"] for r in out["episode"]["features"]] == ["f1", "f0"]
    # The caveats travel with the numbers rather than living only in the docs.
    assert len(out["caveats"]) >= 3
    assert any("not causal" in c for c in out["caveats"])
    assert out["live_computation"] is True
