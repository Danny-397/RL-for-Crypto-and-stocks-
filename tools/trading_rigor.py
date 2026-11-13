"""Is the negative trading result a property of the market, or of this PPO?

Three checks on the real-data study of RESULTS.md §5, all on the same data,
seeds and evaluation as that study's pre-registered replication (seeds 200-209,
the 2026-10-28 data snapshot, 60/40 chronological split, scored per ticker on the
held-out 40% against buy-and-hold):

1. **Other implementations and algorithms.** The same task trained with
   Stable-Baselines3's PPO (same hyper-parameters) and with SB3's SAC, an
   off-policy algorithm from a different family. If they also find no edge, the
   result is not an artifact of this repository's PPO.
2. **Training budget.** One long run per seed, scored at 25k, 50k, 100k, 150k
   (the published budget), 300k, 600k and 1M steps. The PPO has no
   learning-rate schedule, so the 150k checkpoint is the published 150k run;
   later checkpoints show whether more training would have helped.
3. **Time-limit handling.** The published PPO treats the end of a training
   slice as the end of the task. ``ours_tb`` instead bootstraps from the value
   of the state where the data ran out (Pardo et al., 2018) — everything else
   identical.

Arms: ``ours``, ``ours_tb``, ``sb3_ppo``, ``sb3_sac`` (SAC is scored to 300k:
it does a gradient step per environment step and is ~10x slower per step).

    python tools/trading_rigor.py run --market stock --arm ours --seed 200
    python tools/trading_rigor.py run-many --market crypto --arm sb3_sac --seeds 200-203 --jobs 4
    python tools/trading_rigor.py aggregate --runs results/trading_rigor --out docs/assets/trading_rigor.json

``--universe pit`` trains and scores on the hindsight-free stock universe of
data/universe_pit.json instead (RESULTS.md, survivorship). The heavy runs happen
on GitHub Actions (.github/workflows/rigor.yml).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BUDGETS = [25_000, 50_000, 100_000, 150_000, 300_000, 600_000, 1_000_000]
PUBLISHED = 150_000
ARMS = ("ours", "ours_tb", "ours_fixed", "ours_repaired", "sb3_ppo", "sb3_sac")
# ``ours_fixed``: the PPO with every repair from the benchmark diagnosis
# (RESULTS.md §10): separate critic trunk, observations stored as the policy saw
# them, no tanh on the mean, and time-limit bootstrapping.
FIXED = {"bootstrap_truncation": True, "shared_trunk": False,
         "normalize_at_collection": True, "squash_mean": False}
MAX_STEPS = {"sb3_sac": 300_000}
ROLLOUT = 2048


def budgets_for(arm: str, max_steps: int | None = None) -> list[int]:
    cap = min(max_steps or BUDGETS[-1], MAX_STEPS.get(arm, BUDGETS[-1]))
    return [b for b in BUDGETS if b <= cap]


def data_dir_for(universe: str, data_dir: str) -> str:
    return data_dir if universe == "main" else os.path.join(os.path.dirname(data_dir.rstrip("/\\")), "raw_pit")


def data_fingerprint(data_dir: str, market: str) -> str:
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(data_dir, market, "*.csv"))):
        h.update(os.path.basename(p).encode())
        h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


class Scorer:
    """Scores any policy on every ticker's held-out split (as real_significance.py does)."""

    def __init__(self, basket: dict, market: str, cfg) -> None:
        from rl_trader.evaluation.evaluate_agent import ANNUALISATION, compute_metrics

        self.basket, self.market, self.cfg = basket, market, cfg
        periods = ANNUALISATION.get(market, 252)
        w = cfg.env.window_size
        self.bh = []
        for sp in basket.values():
            prices = sp["test"].prices[w - 1:]
            self.bh.append(compute_metrics(cfg.env.initial_balance * (prices / prices[0]), periods)["total_return"])

    def __call__(self, agent) -> list[float]:
        from rl_trader.envs import make_env
        from rl_trader.evaluation.evaluate_agent import backtest

        out = []
        for sp in self.basket.values():
            env = make_env(self.market, sp["test"], self.cfg.env, self.cfg.reward, random_start=False)
            out.append(float(backtest(agent, env, market=self.market).metrics["total_return"]))
        return out


class SB3Adapter:
    """Gives an SB3 model the ``select_action`` interface that ``backtest`` uses."""

    def __init__(self, model, vecnorm=None) -> None:
        self.model, self.vecnorm = model, vecnorm

    def select_action(self, obs, deterministic: bool = True):
        o = np.asarray(obs, dtype=np.float32)[None]
        if self.vecnorm is not None:
            o = self.vecnorm.normalize_obs(o)
        a, _ = self.model.predict(o, deterministic=True)
        return a[0], 0.0, 0.0


def basket_env(train_slices, market, cfg, seed):
    """A gymnasium env that draws a fresh training ticker every episode (domain randomization)."""
    import gymnasium as gym

    from rl_trader.envs import make_env

    class BasketEnv(gym.Env):
        def __init__(self):
            self.rng = random.Random(seed)
            self.inner = make_env(market, self.rng.choice(train_slices), cfg.env, cfg.reward)
            self.observation_space = self.inner.observation_space
            self.action_space = self.inner.action_space
            self.first = True

        def reset(self, *, seed=None, options=None):
            if not self.first:
                self.inner.reload(self.rng.choice(train_slices))
            out = self.inner.reset(seed=seed if self.first else None)
            self.first = False
            return out

        def step(self, action):
            return self.inner.step(action)

    return BasketEnv()


def run_one(market: str, arm: str, seed: int, data_dir: str, out_dir: str,
            universe: str = "main", max_steps: int | None = None, split: str = "ticker") -> str:
    import torch

    from rl_trader.config.training_config import crypto_config, stock_config
    from tools.real_significance import load_real_basket

    torch.set_num_threads(1)
    ddir = data_dir_for(universe, data_dir)
    basket = load_real_basket(ddir, market, split=split)
    if universe == "pit":
        basket.pop("SPY", None)          # the market index, not a member
    if not basket:
        raise SystemExit(f"no data under {os.path.join(ddir, market)}")
    tickers = list(basket)
    train_slices = [basket[t]["train"] for t in tickers]
    cfg = (stock_config if market == "stock" else crypto_config)()
    cfg.market = market
    cfg.train.eval_interval = 0
    cfg.train.seed = seed
    cfg.train.checkpoint_dir = os.path.join("checkpoints", "_rigor")
    score = Scorer(basket, market, cfg)
    budgets = budgets_for(arm, max_steps)
    marks = {b // ROLLOUT if arm != "sb3_sac" else b: b for b in budgets}
    checkpoints: dict = {}
    t0 = time.time()

    if arm in ("ours", "ours_tb", "ours_fixed", "ours_repaired"):
        from rl_trader.training.utils import run_ppo_training

        cfg.ppo.bootstrap_truncation = arm == "ours_tb"
        if arm == "ours_fixed":
            for k, v in FIXED.items():
                setattr(cfg.ppo, k, v)
        if arm == "ours_repaired":
            from rl_trader.config.training_config import repaired_ppo
            repaired_ppo(cfg.ppo)
        cfg.train.total_timesteps = budgets[-1]
        random.seed(seed)

        def factory():
            return random.choice(train_slices)

        def cb(agent, steps):
            u = steps // ROLLOUT
            if u in marks:
                checkpoints[str(marks[u])] = score(agent)

        run_ppo_training(cfg, train_series_factory=factory, callback=cb,
                         logger=_quiet_logger())
    else:
        import torch.nn as nn
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        p = cfg.ppo
        venv = DummyVecEnv([lambda: basket_env(train_slices, market, cfg, seed)])
        if arm == "sb3_ppo":
            venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=p.obs_clip, gamma=p.gamma)
            model = PPO(
                "MlpPolicy", venv, learning_rate=p.learning_rate, n_steps=ROLLOUT,
                batch_size=p.minibatch_size, n_epochs=p.update_epochs, gamma=p.gamma,
                gae_lambda=p.gae_lambda, clip_range=p.clip_ratio, clip_range_vf=p.clip_ratio,
                ent_coef=p.entropy_coef, vf_coef=p.value_coef, max_grad_norm=p.max_grad_norm,
                policy_kwargs=dict(net_arch=dict(pi=list(p.hidden_sizes), vf=list(p.hidden_sizes)),
                                   activation_fn=nn.Tanh, log_std_init=p.init_log_std),
                seed=seed, device="cpu", verbose=0)
            adapter = SB3Adapter(model, venv)
        else:
            # SB3 defaults for SAC, with the same network size. The features are
            # already scaled on the training slice, so no running normaliser.
            model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=300_000,
                        batch_size=256, gamma=p.gamma, policy_kwargs=dict(net_arch=list(p.hidden_sizes)),
                        seed=seed, device="cpu", verbose=0)
            adapter = SB3Adapter(model)

        class Marks(BaseCallback):
            def _on_step(self) -> bool:
                if arm == "sb3_sac" and self.num_timesteps in marks:
                    checkpoints[str(marks[self.num_timesteps])] = score(adapter)
                return True

            def _on_rollout_end(self) -> None:
                # Fires after a rollout is collected and before it is trained on,
                # so the policy has had one update fewer than rollouts collected.
                u = self.num_timesteps // ROLLOUT - 1
                if arm == "sb3_ppo" and u in marks:
                    checkpoints[str(marks[u])] = score(adapter)

        last = budgets[-1] if arm == "sb3_sac" else (budgets[-1] // ROLLOUT) * ROLLOUT
        model.learn(total_timesteps=last, callback=Marks())
        if arm == "sb3_ppo":
            # _on_rollout_end fires before the update that follows the rollout;
            # score the final update's policy too, under the last budget.
            checkpoints[str(budgets[-1])] = score(adapter)

    rec = {"market": market, "arm": arm, "seed": seed, "universe": universe, "split": split,
           "test_bars": {t: len(basket[t]["test"]) for t in tickers}, "tickers": tickers,
           "data": data_fingerprint(ddir, market), "bh": [round(x, 5) for x in score.bh],
           "checkpoints": {k: [round(x, 5) for x in v] for k, v in checkpoints.items()},
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(out_dir, exist_ok=True)
    tag = universe if split == "ticker" else f"{universe}-common"
    path = os.path.join(out_dir, f"{tag}__{market}__{arm}__{seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    pub = rec["checkpoints"].get(str(PUBLISHED))
    msg = f"basket mean at 150k {np.mean(pub):+.1%}" if pub else ""
    print(f"{tag} {market} {arm} seed {seed}: {msg} ({rec['seconds'] / 60:.0f} min)", flush=True)
    return path


def _quiet_logger():
    import logging

    log = logging.getLogger("trading_rigor")
    log.setLevel(logging.WARNING)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())
    return log


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def aggregate(run_dir: str, out: str) -> dict:
    from rl_trader.evaluation.rigor import (
        aggregate_scores,
        probability_of_improvement,
        two_sample_diff,
    )
    from rl_trader.evaluation.statistics import paired_permutation_test

    recs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(run_dir, "*.json")))]
    result: dict = {"budgets": BUDGETS, "published": PUBLISHED, "groups": {}}
    def tag(r):
        return r["universe"] if r.get("split", "ticker") == "ticker" else f"{r['universe']}-common"

    for universe in sorted({tag(r) for r in recs}):
        for market in ("stock", "crypto"):
            rows = [r for r in recs if tag(r) == universe and r["market"] == market]
            if not rows:
                continue
            bh = np.array(rows[0]["bh"])
            g = {"tickers": rows[0]["tickers"], "data": rows[0]["data"], "bh_mean": round(float(bh.mean()), 4),
                 "bh": rows[0]["bh"], "arms": {}}
            for arm in ARMS:
                rs = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["seed"])
                if not rs:
                    continue
                a = {"seeds": [r["seed"] for r in rs], "minutes": round(float(np.mean([r["seconds"] for r in rs])) / 60, 1),
                     "budgets": {}}
                for b in budgets_for(arm):
                    per = [r["checkpoints"].get(str(b)) for r in rs]
                    if any(x is None for x in per):
                        continue
                    per = np.array(per)                       # seeds x tickers
                    seed_means = per.mean(1)
                    by_asset = per.mean(0)
                    diff, p = paired_permutation_test(by_asset, bh)
                    a["budgets"][str(b)] = {
                        "seed_means": [round(float(x), 4) for x in seed_means],
                        **aggregate_scores(seed_means),
                        "win_rate": round(float((per > bh[None, :]).mean()), 3),
                        "diff_vs_bh": round(float(diff), 4), "p_vs_bh": round(float(p), 4),
                        "p_improve_vs_bh": round(probability_of_improvement(per.ravel(), np.repeat(bh, len(rs))), 3),
                    }
                g["arms"][arm] = a
            pub = {arm: g["arms"][arm]["budgets"].get(str(PUBLISHED), {}).get("seed_means")
                   for arm in g["arms"]}
            g["vs_ours_at_150k"] = {arm: two_sample_diff(pub[arm], pub["ours"])
                                    for arm in pub if arm != "ours" and pub.get(arm) and pub.get("ours")}
            result["groups"][f"{universe}/{market}"] = g
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "run-many"):
        r = sub.add_parser(name)
        r.add_argument("--market", choices=["stock", "crypto"], required=True)
        r.add_argument("--arm", choices=ARMS, required=True)
        r.add_argument("--universe", choices=["main", "pit"], default="main")
        r.add_argument("--data-dir", default=os.path.join("data", "raw"))
        r.add_argument("--out", default=os.path.join("results", "trading_rigor"))
        r.add_argument("--max-steps", type=int, default=None, help="cap the budget (smoke tests)")
        r.add_argument("--split", choices=["ticker", "common"], default="ticker",
                       help="ticker: each ticker 60/40 on its own dates (as published); "
                            "common: one calendar cut for every ticker")
        if name == "run":
            r.add_argument("--seed", type=int, default=200)
        else:
            r.add_argument("--seeds", default="200-209")
            r.add_argument("--jobs", type=int, default=4)
    a = sub.add_parser("aggregate")
    a.add_argument("--runs", default=os.path.join("results", "trading_rigor"))
    a.add_argument("--out", default=os.path.join("docs", "assets", "trading_rigor.json"))
    args = ap.parse_args()

    if args.cmd == "run":
        run_one(args.market, args.arm, args.seed, args.data_dir, args.out, args.universe, args.max_steps,
                args.split)
    elif args.cmd == "run-many":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_one, args.market, args.arm, s, args.data_dir, args.out,
                              args.universe, args.max_steps, args.split) for s in parse_seeds(args.seeds)]
            for f in futs:
                f.result()
    else:
        aggregate(args.runs, args.out)


if __name__ == "__main__":
    main()
