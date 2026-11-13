"""The hedging environment and its textbook baselines (rl_trader.hedging.core)."""

import math

import numpy as np
import pytest

from rl_trader.hedging import (
    OBS_DIM,
    HedgeSpec,
    VecHedgeEnv,
    action_to_hedge,
    band_hedge,
    bs_call,
    bs_delta,
    delta_hedge,
    hedge_to_action,
    no_hedge,
    norm_cdf,
    run_policy,
    simulate_paths,
    summarize,
)


def _paths(n=4000, vol=0.2, spec=None, seed=0):
    spec = spec or HedgeSpec()
    return simulate_paths(spec, n, np.random.default_rng(seed), vol=np.full(n, vol))


def test_black_scholes_textbook_value_and_cdf():
    assert float(bs_call(100, 100, 1.0, 0.2, 0.05)) == pytest.approx(10.4506, abs=1e-4)
    for x in np.linspace(-6, 6, 241):
        assert float(norm_cdf(x)) == pytest.approx(0.5 * (1 + math.erf(x / math.sqrt(2))), abs=2e-7)


def test_delta_is_the_slope_of_the_price():
    for S in (0.9, 1.0, 1.1):
        h = 1e-5
        fd = (bs_call(S + h, 1.0, 0.1, 0.25) - bs_call(S - h, 1.0, 0.1, 0.25)) / (2 * h)
        assert float(bs_delta(S, 1.0, 0.1, 0.25)) == pytest.approx(float(fd), abs=1e-5)


def test_at_expiry_the_option_is_its_payoff():
    assert float(bs_call(1.2, 1.0, 0.0, 0.3)) == pytest.approx(0.2)
    assert float(bs_call(0.8, 1.0, 0.0, 0.3)) == 0.0
    assert float(bs_delta(1.2, 1.0, 0.0, 0.3)) == 1.0


def test_paths_start_at_one_and_have_the_drawn_volatility():
    spec = HedgeSpec()
    S, sig = simulate_paths(spec, 20000, np.random.default_rng(1), vol=np.full(20000, 0.3))
    assert S.shape == (20000, spec.n_steps + 1) and np.all(S[:, 0] == 1.0)
    realized = np.log(S[:, 1:] / S[:, :-1]).std() * math.sqrt(252)
    assert realized == pytest.approx(0.3, rel=0.02)


def test_the_policy_sees_four_features_and_not_delta():
    S, sig = _paths(8)
    env = VecHedgeEnv(S, sig, HedgeSpec())
    o = env.reset()
    assert o.shape == (8, OBS_DIM) == (8, 4)
    assert np.allclose(o[:, 0], np.log(S[:, 0]))
    assert np.allclose(o[:, 1], 1.0) and np.allclose(o[:, 3], 0.0)


def test_no_lookahead_the_observation_ignores_future_prices():
    spec = HedgeSpec()
    S, sig = _paths(16, spec=spec)
    S2 = S.copy()
    S2[:, 6:] *= 1.3                                         # change only the future
    a, b = VecHedgeEnv(S, sig, spec), VecHedgeEnv(S2, sig, spec)
    a.reset(), b.reset()
    for _ in range(5):
        h = np.full(16, 0.5)
        oa, _, _, _ = a.step(h)
        ob, _, _, _ = b.step(h)
        assert np.allclose(oa, ob)


def test_the_books_balance():
    """Sum of the daily P&L = hedge gains - (payoff - premium) - every cost."""
    spec = HedgeSpec(cost=0.003)
    S, sig = _paths(500, spec=spec)
    env = VecHedgeEnv(S, sig, spec)
    res = run_policy(env, band_hedge(0.05))
    h = res['hedges']
    gains = (h * np.diff(S, axis=1)).sum(axis=1)
    payoff = np.maximum(S[:, -1] - 1.0, 0.0)
    premium = env.V[:, 0]
    traded = np.abs(np.diff(np.concatenate([np.zeros((500, 1)), h], axis=1), axis=1))
    costs = spec.cost * (traded * S[:, :-1]).sum(axis=1) + spec.cost * h[:, -1] * S[:, -1]
    assert np.allclose(res['pnl'], gains - (payoff - premium) - costs)
    assert np.allclose(res['cost'], costs)


def test_delta_hedging_removes_most_of_the_risk_at_zero_cost():
    S, sig = _paths()
    env = VecHedgeEnv(S, sig, HedgeSpec())
    none = summarize(run_policy(env, no_hedge))
    delta = summarize(run_policy(env, delta_hedge))
    assert delta['std'] < none['std'] / 5
    assert abs(delta['mean']) < 0.05                     # no drift, no cost: about break-even
    assert delta['cost'] == 0.0


def test_a_zero_width_band_is_delta_hedging():
    S, sig = _paths(300)
    env = VecHedgeEnv(S, sig, HedgeSpec(cost=0.002))
    a = run_policy(env, delta_hedge)
    b = run_policy(env, band_hedge(0.0))
    assert np.allclose(a['pnl'], b['pnl'])


def test_the_best_band_widens_as_costs_rise():
    """Whalley-Wilmott: with costs, trade less. The baselines must show it before the agent can."""
    S, sig = _paths(6000)
    widths = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
    best = []
    for c in (0.0, 0.0005, 0.002, 0.01):
        env = VecHedgeEnv(S, sig, HedgeSpec(cost=c))
        scores = [summarize(run_policy(env, band_hedge(w)))['objective'] for w in widths]
        best.append(widths[int(np.argmax(scores))])
    assert best[0] == 0.0
    assert best == sorted(best) and best[-1] >= 0.1


def test_action_mapping_round_trips():
    h = np.array([0.0, 0.25, 1.0])
    assert np.allclose(action_to_hedge(hedge_to_action(h)), h)
    assert np.allclose(action_to_hedge(np.array([-3.0, 3.0])), [0.0, 1.0])
