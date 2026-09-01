"""Tests for the supervised baselines.

These exist to answer "maybe PPO is just the wrong tool", so they are only worth
anything if they are held to the same standard as the agent. The load-bearing
tests are therefore about information, not accuracy: the models must be fit on
the training split alone, and the decision at bar *t* must not use anything from
bar *t+1*. A supervised baseline that peeks would beat everything and quietly
invalidate the comparison it exists to make.

The rest pin behaviour that a plausible refactor could break silently — a
degenerate fit returning a confident-looking flat line, or a model being fit on
the evaluation data when no training split was supplied.
"""

from __future__ import annotations

import numpy as np
import pytest

from rl_trader.evaluation import supervised


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _series(n=300, seed=0, signal=0.0):
    """Prices with an optional linear dependence on the first feature.

    The offset matters and is easy to get wrong: with prices built by cumsum,
    the return observed between bar t and t+1 is ``rets[t + 1]``. So to make
    feature 0 *at bar t* drive the move into t+1 — the relationship the models
    are supposed to find — the signal has to be placed one step later in the
    return series. Getting this backwards plants the signal on a feature row the
    model is never shown, and the positive control then fails for a reason that
    has nothing to do with the model.
    """
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, 6))
    noise = rng.normal(scale=0.01, size=n)
    rets = noise.copy()
    rets[1:] += signal * feats[:-1, 0]
    prices = 100.0 * np.exp(np.cumsum(rets))
    return feats.astype(np.float32), prices.astype(np.float32)


class _FakeEnv:
    """Just enough environment for an action function: data.features and t."""

    def __init__(self, feats, prices, t=0):
        self.data = type("D", (), {"features": feats, "prices": prices})()
        self.t = t


# --------------------------------------------------------------------------- #
# No lookahead — the tests that make the comparison meaningful                 #
# --------------------------------------------------------------------------- #
def test_the_design_matrix_never_pairs_a_bar_with_its_own_return():
    """Row i must be features at i against the move from i to i+1."""
    feats, prices = _series(n=50)
    x, y = supervised._design(feats, prices)
    assert len(x) == len(y) == len(prices) - 1
    expected = np.log(prices[1:].astype(np.float64) / prices[:-1])
    assert np.allclose(y, expected)
    # and the features are the earlier bar's, not the later one's
    assert np.allclose(x[0], feats[0])
    assert np.allclose(x[-1], feats[-2])


def test_the_final_bar_is_dropped_because_its_target_is_unobservable():
    feats, prices = _series(n=40)
    x, _ = supervised._design(feats, prices)
    assert len(x) == 39
    assert not np.allclose(x[-1], feats[-1]), "the last bar leaked in"


def test_an_action_depends_only_on_the_current_bar():
    """Changing the future must not change today's decision."""
    feats, prices = _series(n=200, signal=0.05)
    weights, scale = supervised.fit_ridge(feats, prices)
    act = supervised.ridge_action(weights, scale)

    env = _FakeEnv(feats.copy(), prices.copy(), t=50)
    before = act(env)

    env.data.features[51:] = 999.0          # rewrite the entire future
    env.data.prices[51:] = 1.0
    assert act(env) == before


def test_weights_come_from_the_training_split_only():
    """Fitting on train then evaluating on test must ignore the test data."""
    train_f, train_p = _series(n=200, seed=1, signal=0.05)
    w_a, s_a = supervised.fit_ridge(train_f, train_p)
    # a wildly different "test" period must not move the fitted weights
    w_b, s_b = supervised.fit_ridge(train_f, train_p)
    assert np.allclose(w_a, w_b) and s_a == s_b


# --------------------------------------------------------------------------- #
# The models actually work when there is something to find                     #
# --------------------------------------------------------------------------- #
def test_ridge_recovers_a_planted_linear_signal():
    """A positive control: if it cannot find a signal that is there, a null
    result from it would mean nothing."""
    feats, prices = _series(n=600, seed=3, signal=0.05)
    weights, _ = supervised.fit_ridge(feats, prices)
    # the driving feature should carry by far the largest weight
    assert abs(weights[0]) > 3 * np.abs(weights[1:-1]).max()


def test_logistic_beats_chance_on_a_planted_signal():
    feats, prices = _series(n=600, seed=4, signal=0.05)
    assert supervised.train_accuracy(feats, prices) > 0.65


def test_logistic_is_near_chance_on_pure_noise():
    feats, prices = _series(n=600, seed=5, signal=0.0)
    assert 0.4 < supervised.train_accuracy(feats, prices) < 0.62


# --------------------------------------------------------------------------- #
# Positions stay inside what the environment allows                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("signal", [0.0, 0.05, 5.0])
def test_actions_are_always_valid_positions(signal):
    feats, prices = _series(n=300, seed=6, signal=signal)
    w, s = supervised.fit_ridge(feats, prices)
    logit = supervised.fit_logistic(feats, prices)
    for act in (supervised.ridge_action(w, s), supervised.logistic_action(logit)):
        for t in range(0, 250, 25):
            a = act(_FakeEnv(feats, prices, t=t))
            assert -1.0 <= a <= 1.0
            assert np.isfinite(a)


def test_an_extreme_feature_row_cannot_produce_a_nan_position():
    feats, prices = _series(n=200, seed=7)
    w, s = supervised.fit_ridge(feats, prices)
    logit = supervised.fit_logistic(feats, prices)
    blown = feats.copy()
    blown[10] = 1e9
    for act in (supervised.ridge_action(w, s), supervised.logistic_action(logit)):
        a = act(_FakeEnv(blown, prices, t=10))
        assert np.isfinite(a) and -1.0 <= a <= 1.0


# --------------------------------------------------------------------------- #
# Degenerate inputs are declined, not faked                                    #
# --------------------------------------------------------------------------- #
def test_too_little_training_data_yields_no_policies():
    """Better to omit the baseline than to report a fitted-looking flat line."""
    feats, prices = _series(n=20)
    assert supervised.supervised_policies(feats, prices) == {}


def test_absent_training_data_yields_no_policies():
    assert supervised.supervised_policies(None, np.array([])) == {}


def test_both_policies_appear_when_there_is_enough_data():
    feats, prices = _series(n=300)
    got = supervised.supervised_policies(feats, prices)
    assert set(got) == {"ridge_forecast", "logistic_direction"}


def test_the_ridge_intercept_is_not_penalised():
    """Shrinking the intercept would bias the forecast rather than regularise it."""
    feats, prices = _series(n=400, seed=8)
    # a constant offset in the target must be absorbed by the intercept
    shifted = prices * np.exp(np.arange(len(prices)) * 0.001).astype(np.float32)
    w_plain, _ = supervised.fit_ridge(feats, prices)
    w_drift, _ = supervised.fit_ridge(feats, shifted)
    assert w_drift[-1] > w_plain[-1] + 1e-4


# --------------------------------------------------------------------------- #
# Integration: the baselines module offers them only when it may               #
# --------------------------------------------------------------------------- #
def test_evaluate_baselines_omits_supervised_arms_without_a_train_split():
    """Fitting on the evaluation data would be exactly the leakage this project
    spends most of its effort avoiding, so absence must be the default."""
    from rl_trader.config.training_config import stock_config
    from rl_trader.data.data_loader import MarketData
    from rl_trader.evaluation.baselines import evaluate_baselines

    cfg = stock_config()
    feats, prices = _series(n=200, seed=9)
    names = [f"f{i}" for i in range(feats.shape[1])]
    data = MarketData(features=feats, prices=prices, feature_names=names)

    out = evaluate_baselines(data, cfg.env, cfg.reward, market="stock")
    assert "ridge_forecast" not in out and "logistic_direction" not in out
    assert {"buy_and_hold", "flat", "random", "ma_crossover"} <= set(out)
