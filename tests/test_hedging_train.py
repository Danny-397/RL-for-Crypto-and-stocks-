"""The PPO hedging trainer (rl_trader.hedging.train): it learns, deterministically, inside the bounds."""

from dataclasses import replace

import numpy as np

from rl_trader.hedging import (
    HedgeSpec,
    VecHedgeEnv,
    no_hedge,
    run_policy,
    simulate_paths,
    summarize,
)
from rl_trader.hedging.train import HedgeTrainConfig, agent_policy, train_hedger


def _short(seed=0, updates=25):
    base = HedgeTrainConfig()
    return replace(base, n_envs=256, updates=updates, seed=seed,
                   ppo=replace(base.ppo, learning_rate=1e-3, minibatch_size=1024, update_epochs=4))


def test_a_short_run_learns_to_hedge():
    spec = HedgeSpec(cost=0.0)
    agent, hist = train_hedger(spec, _short())
    S, sig = simulate_paths(spec, 3000, np.random.default_rng(7))
    env = VecHedgeEnv(S, sig, spec)
    agent_res = summarize(run_policy(env, agent_policy(agent)))
    none = summarize(run_policy(env, no_hedge))
    assert agent_res['std'] < none['std'] / 2            # most of the risk is gone
    assert hist['objective'][-1] > hist['objective'][0]   # and the training objective rose


def test_hedges_stay_in_bounds():
    spec = HedgeSpec(cost=0.002)
    agent, _ = train_hedger(spec, _short(updates=3))
    S, sig = simulate_paths(spec, 500, np.random.default_rng(3))
    res = run_policy(VecHedgeEnv(S, sig, spec), agent_policy(agent))
    assert res['hedges'].min() >= 0.0 and res['hedges'].max() <= 1.0


def test_training_is_reproducible_for_a_seed():
    spec = HedgeSpec(cost=0.002)
    a, ha = train_hedger(spec, _short(seed=3, updates=3))
    b, hb = train_hedger(spec, _short(seed=3, updates=3))
    assert np.allclose(ha['objective'], hb['objective'])
