"""Train and test RL hedging policies against the textbook hedges.

    python tools/hedging_experiment.py train      # 4 cost levels x 10 seeds -> checkpoints/hedging/
    python tools/hedging_experiment.py simulate   # theory checks on simulated paths -> results/hedging_sim.json
    python tools/hedging_experiment.py real       # held-out real SPY history -> results/hedging_real.json
    python tools/hedging_experiment.py site       # docs/hedging.js for the site (+ exported policies)

Every policy is trained only on simulated paths, so all of real history is out of
sample. The no-trade band baseline is tuned per cost on a separate validation set,
so it is the strongest fair version of the textbook answer, and every comparison
is made on a test set neither touched.
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rl_trader.hedging import (  # noqa: E402
    HedgeSpec,
    VecHedgeEnv,
    band_hedge,
    delta_hedge,
    no_hedge,
    run_policy,
    simulate_paths,
    summarize,
)

COSTS = (0.0, 0.0005, 0.002, 0.01)          # 0, 0.05%, 0.2%, 1% of the notional traded
SEEDS = tuple(range(10))
BAND_WIDTHS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4)
CKPT = os.path.join(ROOT, 'checkpoints', 'hedging')
RESULTS = os.path.join(ROOT, 'docs', 'assets')
N_VAL, N_TEST = 20000, 20000
VAL_SEED, TEST_SEED = 1001, 2002


def ckpt_path(cost, seed):
    return os.path.join(CKPT, f'hedge_c{cost:g}_s{seed}.pt')


def _train_one(args):
    cost, seed = args
    import torch
    torch.set_num_threads(2)
    from dataclasses import replace

    from rl_trader.hedging.train import HedgeTrainConfig, train_hedger
    base = HedgeTrainConfig()
    cfg = replace(base, n_envs=1024, updates=300, seed=seed,
                  ppo=replace(base.ppo, learning_rate=1e-3, minibatch_size=2048))
    t0 = time.time()
    agent, hist = train_hedger(HedgeSpec(cost=cost), cfg)
    agent.save(ckpt_path(cost, seed))
    with open(ckpt_path(cost, seed).replace('.pt', '.json'), 'w') as f:
        json.dump(hist, f)
    return cost, seed, time.time() - t0


def train(workers: int = 4):
    os.makedirs(CKPT, exist_ok=True)
    jobs = [(c, s) for c in COSTS for s in SEEDS if not os.path.exists(ckpt_path(c, s))]
    print(f'{len(jobs)} policies to train ({workers} at a time)')
    with Pool(workers) as pool:
        for cost, seed, secs in pool.imap_unordered(_train_one, jobs):
            print(f'  cost {cost:g} seed {seed}: {secs:.0f}s', flush=True)


def load_agent(cost, seed):
    from rl_trader.models.ppo_agent import PPOAgent
    return PPOAgent.from_checkpoint(ckpt_path(cost, seed))


def sim_sets(spec):
    val = simulate_paths(spec, N_VAL, np.random.default_rng(VAL_SEED))
    test = simulate_paths(spec, N_TEST, np.random.default_rng(TEST_SEED))
    return val, test


def tune_band(paths, spec):
    """The band width with the best objective on the validation paths."""
    scores = {w: summarize(run_policy(VecHedgeEnv(*paths, spec), band_hedge(w)))['objective']
              for w in BAND_WIDTHS}
    return max(scores, key=scores.get), scores


def tracking(env, res):
    """Mean |hedge - delta| along the policy's own path, and the share of days it traded."""
    env.reset()
    gaps, trades = [], []
    for t in range(env.spec.n_steps):
        h = res['hedges'][:, t]
        gaps.append(np.abs(h - env.delta))
        trades.append(np.abs(h - env.h) > 0.005)
        env.step(h)
    return float(np.mean(gaps)), float(np.mean(trades))


def simulate():
    from rl_trader.evaluation.statistics import paired_permutation_test
    from rl_trader.hedging.train import agent_policy
    out = {'costs': [], 'widths': list(BAND_WIDTHS)}
    for cost in COSTS:
        spec = HedgeSpec(cost=cost)
        val, test = sim_sets(spec)
        width, val_scores = tune_band(val, spec)
        env = VecHedgeEnv(*test, spec)
        base = {name: summarize(run_policy(env, pol)) for name, pol in
                (('none', no_hedge), ('delta', delta_hedge), ('band', band_hedge(width)))}
        base['delta']['track'], base['delta']['trade_days'] = tracking(env, run_policy(env, delta_hedge))
        seeds = []
        for seed in SEEDS:
            res = run_policy(env, agent_policy(load_agent(cost, seed)))
            s = summarize(res)
            s['track'], s['trade_days'] = tracking(env, res)
            s['seed'] = seed
            seeds.append(s)
        obj = np.array([s['objective'] for s in seeds])
        vs_band = paired_permutation_test(obj, np.full(len(obj), base['band']['objective']))
        vs_delta = paired_permutation_test(obj, np.full(len(obj), base['delta']['objective']))
        row = {'cost': cost, 'band_width': width, 'band_val_scores': val_scores, 'baselines': base, 'seeds': seeds,
               'agent_median': {k: float(np.median([s[k] for s in seeds])) for k in
                                ('objective', 'std', 'cvar95', 'cost', 'turnover', 'track', 'trade_days', 'mean')},
               'vs_band': _test_dict(vs_band, obj - base['band']['objective']),
               'vs_delta': _test_dict(vs_delta, obj - base['delta']['objective'])}
        out['costs'].append(row)
        m = row['agent_median']
        print(f"cost {cost:g}: agent objective {m['objective']:+.3f} (seeds {obj.min():+.3f}..{obj.max():+.3f}) | "
              f"delta {base['delta']['objective']:+.3f} | band({width:g}) {base['band']['objective']:+.3f} | "
              f"none {base['none']['objective']:+.3f} | turnover {m['turnover']:.2f} vs delta "
              f"{base['delta']['turnover']:.2f} | |h-delta| {m['track']:.3f} | p vs band {row['vs_band']['p']:.3f}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'hedging_sim.json'), 'w') as f:
        json.dump(out, f, indent=1)


def _test_dict(test, diffs):
    _, p = test
    return {'p': float(p), 'mean_diff': float(np.mean(diffs)), 'wins': int((diffs > 0).sum()), 'n': len(diffs)}


RAW = os.path.join(ROOT, 'data', 'raw')
VIX_TO_ATM = 0.875        # one-month at-the-money SPY IV / VIX, fitted on Cboe's PutWrite index (AlphaGlyph)
REAL_START = '2007-01-01'


def real_market():
    """Daily SPY closes (dividend-adjusted) and the VIX, cached under data/raw/ (not committed)."""
    import urllib.request

    import pandas as pd
    os.makedirs(RAW, exist_ok=True)
    spy_path, vix_path = os.path.join(RAW, 'hedging_spy.csv'), os.path.join(RAW, 'hedging_vix.csv')
    if not os.path.exists(spy_path):
        import yfinance as yf
        df = yf.download('SPY', start=REAL_START, auto_adjust=True, progress=False)
        close = df['Close']
        close = close.iloc[:, 0] if hasattr(close, 'columns') else close
        close.rename('close').to_csv(spy_path)
    if not os.path.exists(vix_path):
        url = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
        with urllib.request.urlopen(url, timeout=30) as r:
            open(vix_path, 'wb').write(r.read())
    spy = pd.read_csv(spy_path, index_col=0, parse_dates=True)['close'].astype(float).dropna()
    vix = pd.read_csv(vix_path)
    vix = pd.Series(vix['CLOSE'].astype(float).values, index=pd.to_datetime(vix['DATE'], format='%m/%d/%Y'))
    vix = vix.reindex(spy.index).ffill()
    keep = vix.notna()
    return spy[keep], vix[keep]


def real_episodes(n_steps: int = 21):
    """Non-overlapping one-month episodes: prices from 1.0 and the IV each day (0.875 x VIX)."""
    spy, vix = real_market()
    S, sig, starts = [], [], []
    for i in range(0, len(spy) - n_steps, n_steps):
        p = spy.values[i:i + n_steps + 1]
        S.append(p / p[0])
        sig.append(VIX_TO_ATM * vix.values[i:i + n_steps + 1] / 100.0)
        starts.append(str(spy.index[i].date()))
    return np.array(S), np.array(sig), starts


def real():
    from rl_trader.evaluation.statistics import paired_permutation_test
    from rl_trader.hedging.train import agent_policy
    S, sig, starts = real_episodes()
    print(f'{len(S)} real one-month episodes, {starts[0]} to {starts[-1]}')
    out = {'episodes': len(S), 'first': starts[0], 'last': starts[-1], 'costs': [],
           'source': 'SPY daily closes (dividend-adjusted, Yahoo); implied volatility 0.875 x VIX (Cboe)'}
    for cost in COSTS:
        spec = HedgeSpec(cost=cost)
        val, _ = sim_sets(spec)
        width, _ = tune_band(val, spec)                       # tuned on simulation, never on real data
        env = VecHedgeEnv(S, sig, spec)
        base = {name: summarize(run_policy(env, pol)) for name, pol in
                (('none', no_hedge), ('delta', delta_hedge), ('band', band_hedge(width)))}
        band_pnl = run_policy(env, band_hedge(width))['reward']
        seeds = []
        for seed in SEEDS:
            res = run_policy(env, agent_policy(load_agent(cost, seed)))
            s = summarize(res)
            s['seed'] = seed
            # episode-level: does this policy beat the band on the same months?
            _, p_ep = paired_permutation_test(res['reward'], band_pnl, n_perm=20000)
            s['p_vs_band_episodes'] = float(p_ep)
            seeds.append(s)
        obj = np.array([s['objective'] for s in seeds])
        row = {'cost': cost, 'band_width': width, 'baselines': base, 'seeds': seeds,
               'agent_median': {k: float(np.median([s[k] for s in seeds])) for k in
                                ('objective', 'std', 'cvar95', 'cost', 'turnover', 'mean')},
               'vs_band': _test_dict(paired_permutation_test(obj, np.full(len(obj), base['band']['objective'])),
                                     obj - base['band']['objective']),
               'vs_delta': _test_dict(paired_permutation_test(obj, np.full(len(obj), base['delta']['objective'])),
                                      obj - base['delta']['objective'])}
        out['costs'].append(row)
        m = row['agent_median']
        print(f"cost {cost:g}: agent {m['objective']:+.3f} (seeds {obj.min():+.3f}..{obj.max():+.3f}) | "
              f"delta {base['delta']['objective']:+.3f} | band({width:g}) {base['band']['objective']:+.3f} | "
              f"none {base['none']['objective']:+.3f} | std {m['std']:.2f} vs band {base['band']['std']:.2f} | "
              f"cost {m['cost']:.3f} vs band {base['band']['cost']:.3f} | seeds beating band "
              f"{row['vs_band']['wins']}/10, p {row['vs_band']['p']:.3f}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'hedging_real.json'), 'w') as f:
        json.dump(out, f, indent=1)


def _r(x, nd=5):
    """Round floats (recursively) for a compact page file."""
    if isinstance(x, float):
        return float(f'{x:.{nd}g}')
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    return x


def export_policy(agent) -> dict:
    """The deterministic policy as plain weights: the page runs it in the browser (no server)."""
    import torch.nn as nn
    lins = [m for m in agent.ac.trunk if isinstance(m, nn.Linear)]
    w = lambda t: t.detach().cpu().numpy().astype(float).tolist()  # noqa: E731
    out = {'layers': [{'W': w(m.weight), 'b': w(m.bias)} for m in lins],
           'head': {'W': w(agent.ac.policy_mean.weight), 'b': w(agent.ac.policy_mean.bias)}}
    if agent.obs_rms is not None:
        out['obs_mean'] = agent.obs_rms.mean.astype(float).tolist()
        out['obs_std'] = np.sqrt(agent.obs_rms.var + agent.obs_rms.epsilon).astype(float).tolist()
        out['obs_clip'] = float(agent.obs_rms.clip)
    return out


def site():
    with open(os.path.join(RESULTS, 'hedging_sim.json')) as f:
        sim = json.load(f)
    with open(os.path.join(RESULTS, 'hedging_real.json')) as f:
        real_res = json.load(f)
    S, sig, starts = real_episodes()
    policies = {}
    for row in sim['costs']:
        objs = [s['objective'] for s in row['seeds']]
        median_seed = row['seeds'][int(np.argsort(objs)[len(objs) // 2])]['seed']   # typical, not the best
        policies[f"{row['cost']:g}"] = {'seed': median_seed, **export_policy(load_agent(row['cost'], median_seed))}
    data = {'generated': time.strftime('%Y-%m-%d'), 'costs': list(COSTS), 'n_steps': HedgeSpec().n_steps,
            'risk_aversion': HedgeSpec().risk_aversion, 'vol_range': list(HedgeSpec().vol_range),
            'seeds': len(SEEDS), 'sim': sim, 'real': real_res, 'policies': policies,
            'episodes': {'starts': starts, 'S': np.round(S, 5).tolist(), 'sigma': np.round(sig, 4).tolist()}}
    js = ('// Generated by tools/hedging_experiment.py site. Do not edit by hand.\n'
          'window.HEDGING = ' + json.dumps(_r(data), separators=(',', ':')) + ';\n')
    out = os.path.join(ROOT, 'docs', 'hedging.js')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(js)
    print(f'wrote {out} ({len(js) / 1024:.0f} KB)')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'train'
    {'train': train, 'simulate': simulate, 'real': real, 'site': site}[cmd]()
