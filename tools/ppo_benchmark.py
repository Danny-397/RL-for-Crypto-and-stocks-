"""Does this PPO work? Validate the from-scratch implementation on standard benchmarks.

Every result in this repository comes from a PPO written here from the maths up.
A negative result from it ("PPO finds no edge") is only worth something if the PPO
is known to work, so this script trains it on four standard continuous-control
tasks and compares it, seed for seed, with Stable-Baselines3's PPO, a widely used
reference implementation, under the same hyper-parameters.

Arms
----
ours      the repository's PPOAgent + RolloutBuffer, exactly as the trading
          study uses them (an episode cut off by a time limit is treated as
          ended: its last step's return is not bootstrapped)
ours_tb   the same, but a time-limit cut-off is bootstrapped with the value of
          the state where the episode was cut (what SB3 does)
sb3       stable_baselines3.PPO with the same hyper-parameters

Hyper-parameters are the trading study's defaults (rl_trader.config.PPOConfig),
not values tuned for these tasks: the question is whether the code is right, and
the trading study's settings are the ones it ran with. Remaining differences from
SB3 that hyper-parameters cannot remove: this PPO shares one hidden trunk between
the policy and the value function and squashes the policy mean with tanh; SB3
uses two separate networks and an unsquashed mean.

Tasks (all with actions rescaled to [-1, 1]):

    Pendulum-v1           classic control, 300k steps
    InvertedPendulum-v5   MuJoCo, 300k steps
    HalfCheetah-v5        MuJoCo, 1M steps
    Hopper-v5             MuJoCo, 1M steps

Each run records a learning curve: the mean return of the deterministic policy
over 10 evaluation episodes every 10k steps. The final score is the mean of the
last three evaluations.

    python tools/ppo_benchmark.py run --env Hopper-v5 --arm ours --seed 0
    python tools/ppo_benchmark.py run-many --env Hopper-v5 --arm sb3 --seeds 0-9 --jobs 4
    python tools/ppo_benchmark.py aggregate --runs results/ppo_benchmark --out docs/assets/ppo_benchmark.json

The heavy runs happen on GitHub Actions (.github/workflows/rigor.yml).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TASKS = {
    "Pendulum-v1": 300_000,
    "InvertedPendulum-v5": 300_000,
    "HalfCheetah-v5": 1_000_000,
    "Hopper-v5": 1_000_000,
}
ARMS = ("ours", "ours_tb", "sb3")
EVAL_EVERY = 10_000
EVAL_EPISODES = 10
FINAL_EVALS = 3


@dataclass
class HP:
    """The trading study's PPO defaults (rl_trader.config.PPOConfig / TrainConfig)."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple = (128, 128)
    init_log_std: float = -0.5
    obs_clip: float = 10.0


def make_env(env_id: str, seed: int | None = None):
    import gymnasium as gym
    from gymnasium.wrappers import RescaleAction

    env = RescaleAction(gym.make(env_id), -1.0, 1.0)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def _evaluate(act, env_id: str, seed: int, episodes: int = EVAL_EPISODES) -> float:
    """Mean undiscounted return of a deterministic policy ``act(obs) -> action``."""
    env = make_env(env_id)
    total = []
    for i in range(episodes):
        obs, _ = env.reset(seed=1_000_000 + 1000 * seed + i)  # never a training seed
        done, ret = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(np.clip(act(obs), -1.0, 1.0))
            ret += float(r)
            done = term or trunc
        total.append(ret)
    env.close()
    return float(np.mean(total))


# ── this repository's PPO ─────────────────────────────────────────────────────
def train_ours(env_id: str, seed: int, steps: int, hp: HP, bootstrap_truncation: bool) -> dict:
    import torch

    from rl_trader.config.training_config import PPOConfig
    from rl_trader.models.ppo_agent import PPOAgent
    from rl_trader.training.utils import RolloutBuffer, set_seed

    torch.set_num_threads(1)
    set_seed(seed)
    cfg = PPOConfig(
        learning_rate=hp.learning_rate, gamma=hp.gamma, gae_lambda=hp.gae_lambda,
        clip_ratio=hp.clip_ratio, value_coef=hp.value_coef, entropy_coef=hp.entropy_coef,
        max_grad_norm=hp.max_grad_norm, update_epochs=hp.update_epochs,
        minibatch_size=hp.minibatch_size, hidden_sizes=tuple(hp.hidden_sizes),
        init_log_std=hp.init_log_std, normalize_obs=True, obs_clip=hp.obs_clip,
    )
    env = make_env(env_id, seed)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = PPOAgent(obs_dim, act_dim, cfg, device="cpu")
    buf = RolloutBuffer(hp.n_steps, obs_dim, act_dim)

    def act(o):
        return agent.select_action(o, deterministic=True)[0]

    curve, step = [], 0
    obs, _ = env.reset(seed=seed)
    done = False
    while step < steps:
        buf.reset()
        for _ in range(hp.n_steps):
            agent.observe(obs)
            action, logp, value = agent.select_action(obs)
            nxt, reward, term, trunc, _ = env.step(np.clip(action, -1.0, 1.0))
            reward = float(reward)
            if bootstrap_truncation and trunc and not term:
                reward += hp.gamma * agent.value(nxt)
            done = term or trunc
            buf.add(obs, action, logp, reward, value, done)
            obs = nxt
            step += 1
            if done:
                obs, _ = env.reset()
            if step % EVAL_EVERY == 0:
                curve.append([step, _evaluate(act, env_id, seed)])
        last = 0.0 if done else agent.value(obs)
        buf.compute_gae(last, hp.gamma, hp.gae_lambda)
        agent.update(buf)
    env.close()
    return {"curve": curve}


# ── Stable-Baselines3 reference ──────────────────────────────────────────────
def train_sb3(env_id: str, seed: int, steps: int, hp: HP) -> dict:
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    torch.set_num_threads(1)
    venv = VecNormalize(DummyVecEnv([lambda: make_env(env_id)]), norm_obs=True,
                        norm_reward=False, clip_obs=hp.obs_clip, gamma=hp.gamma)
    venv.seed(seed)
    model = PPO(
        "MlpPolicy", venv, learning_rate=hp.learning_rate, n_steps=hp.n_steps,
        batch_size=hp.minibatch_size, n_epochs=hp.update_epochs, gamma=hp.gamma,
        gae_lambda=hp.gae_lambda, clip_range=hp.clip_ratio, clip_range_vf=hp.clip_ratio,
        ent_coef=hp.entropy_coef, vf_coef=hp.value_coef, max_grad_norm=hp.max_grad_norm,
        policy_kwargs=dict(net_arch=dict(pi=list(hp.hidden_sizes), vf=list(hp.hidden_sizes)),
                           activation_fn=nn.Tanh, log_std_init=hp.init_log_std),
        seed=seed, device="cpu", verbose=0,
    )
    curve: list = []

    def act(o):
        a, _ = model.predict(venv.normalize_obs(o[None].astype(np.float32)), deterministic=True)
        return a[0]

    class Every(BaseCallback):
        def _on_step(self) -> bool:
            if self.num_timesteps % EVAL_EVERY == 0:
                curve.append([self.num_timesteps, _evaluate(act, env_id, seed)])
            return True

    model.learn(total_timesteps=steps, callback=Every())
    venv.close()
    return {"curve": curve}


# ── driver ───────────────────────────────────────────────────────────────────
def run_one(env_id: str, arm: str, seed: int, steps: int | None, out_dir: str) -> str:
    steps = steps or TASKS[env_id]
    hp = HP()
    t0 = time.time()
    if arm == "sb3":
        res = train_sb3(env_id, seed, steps, hp)
    else:
        res = train_ours(env_id, seed, steps, hp, bootstrap_truncation=(arm == "ours_tb"))
    curve = res["curve"]
    final = float(np.mean([r for _, r in curve[-FINAL_EVALS:]])) if curve else float("nan")
    rec = {"env": env_id, "arm": arm, "seed": seed, "steps": steps, "final": final,
           "curve": [[s, round(r, 3)] for s, r in curve], "hp": asdict(hp),
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{env_id}__{arm}__{seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    print(f"{env_id} {arm} seed {seed}: final {final:.1f} ({rec['seconds']:.0f}s)", flush=True)
    return path


def random_policy_score(env_id: str, episodes: int = 20) -> float:
    env = make_env(env_id, 0)
    return _evaluate(lambda o: env.action_space.sample(), env_id, 0, episodes)


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def aggregate(run_dir: str, out: str) -> dict:
    from rl_trader.evaluation.rigor import aggregate_scores, two_sample_diff

    recs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(run_dir, "*.json")))]
    result = {"tasks": {}, "hp": asdict(HP()), "eval_every": EVAL_EVERY,
              "eval_episodes": EVAL_EPISODES, "final_evals": FINAL_EVALS}
    for env_id in TASKS:
        rows = [r for r in recs if r["env"] == env_id]
        if not rows:
            continue
        task = {"steps": rows[0]["steps"], "random": round(random_policy_score(env_id), 2), "arms": {}}
        for arm in ARMS:
            rs = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["seed"])
            if not rs:
                continue
            finals = [r["final"] for r in rs]
            steps_axis = [s for s, _ in rs[0]["curve"]]
            curves = np.array([[v for _, v in r["curve"]] for r in rs if len(r["curve"]) == len(steps_axis)])
            task["arms"][arm] = {
                "seeds": [r["seed"] for r in rs],
                "finals": [round(f, 2) for f in finals],
                **aggregate_scores(finals),
                "curve_steps": steps_axis,
                "curve_mean": [round(float(x), 2) for x in curves.mean(0)],
                "curve_low": [round(float(x), 2) for x in np.percentile(curves, 25, axis=0)],
                "curve_high": [round(float(x), 2) for x in np.percentile(curves, 75, axis=0)],
                "minutes": round(float(np.mean([r["seconds"] for r in rs])) / 60, 1),
            }
        if "ours" in task["arms"] and "sb3" in task["arms"]:
            task["ours_vs_sb3"] = two_sample_diff(task["arms"]["ours"]["finals"], task["arms"]["sb3"]["finals"])
        if "ours_tb" in task["arms"] and "ours" in task["arms"]:
            task["tb_vs_ours"] = two_sample_diff(task["arms"]["ours_tb"]["finals"], task["arms"]["ours"]["finals"])
        result["tasks"][env_id] = task
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--env", choices=list(TASKS), required=True)
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--steps", type=int, default=None, help="override the task's budget (smoke tests)")
    r.add_argument("--out", default=os.path.join("results", "ppo_benchmark"))
    m = sub.add_parser("run-many")
    m.add_argument("--env", choices=list(TASKS), required=True)
    m.add_argument("--arm", choices=ARMS, required=True)
    m.add_argument("--seeds", default="0-9")
    m.add_argument("--jobs", type=int, default=4)
    m.add_argument("--steps", type=int, default=None)
    m.add_argument("--out", default=os.path.join("results", "ppo_benchmark"))
    a = sub.add_parser("aggregate")
    a.add_argument("--runs", default=os.path.join("results", "ppo_benchmark"))
    a.add_argument("--out", default=os.path.join("docs", "assets", "ppo_benchmark.json"))
    args = ap.parse_args()

    if args.cmd == "run":
        run_one(args.env, args.arm, args.seed, args.steps, args.out)
    elif args.cmd == "run-many":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_one, args.env, args.arm, s, args.steps, args.out)
                    for s in parse_seeds(args.seeds)]
            for f in futs:
                f.result()
    else:
        aggregate(args.runs, args.out)


if __name__ == "__main__":
    main()
