"""Multi-seed significance study: is the agent's edge real, or seed luck?

RL results are notoriously seed-sensitive — a single lucky run proves nothing.
This script trains ``--seeds`` independent agents (each with domain-randomized
synthetic training, where a genuine signal exists), evaluates every one of them
on the *same* held-out paths, and then quantifies the result two ways:

1. A **bootstrap 95% confidence interval** on the agent's out-of-sample return
   and Sharpe across seeds — how repeatable is the performance?
2. A **paired permutation test** of the agent vs. buy-&-hold across the held-out
   paths — is the difference statistically distinguishable from noise?

Run from the repo root:

    python tools/significance.py --market stock --seeds 5 --timesteps 40000
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.data.data_loader import synthetic_market_data
from rl_trader.envs import make_env
from rl_trader.evaluation.evaluate_agent import ANNUALISATION, backtest, compute_metrics
from rl_trader.evaluation.statistics import bootstrap_ci, paired_permutation_test
from rl_trader.training.utils import get_logger, run_ppo_training


def _holdout_paths(market: str, n_paths: int, seed: int = 9_999):
    """A fixed bank of held-out synthetic paths, shared across every agent."""
    rng = np.random.default_rng(seed)
    return [synthetic_market_data(market, seed=int(rng.integers(1e9))) for _ in range(n_paths)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=["stock", "crypto"], default="stock")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=40_000)
    parser.add_argument("--eval-paths", type=int, default=20)
    args = parser.parse_args()

    log = get_logger("significance")
    cfg_fn = stock_config if args.market == "stock" else crypto_config
    periods = ANNUALISATION.get(args.market, 252)
    holdout = _holdout_paths(args.market, args.eval_paths)

    # Buy-&-hold on each held-out path (seed-independent reference).
    bh_returns = []
    for data in holdout:
        bh = compute_metrics(
            data.prices / data.prices[0] * cfg_fn().env.initial_balance, periods
        )
        bh_returns.append(bh["total_return"])
    bh_returns = np.array(bh_returns)

    seed_mean_returns, seed_mean_sharpes = [], []
    agent_returns_by_path = np.zeros(args.eval_paths)

    def train_path_factory():
        return synthetic_market_data(args.market)

    for s in range(args.seeds):
        cfg = cfg_fn()
        cfg.train.total_timesteps = args.timesteps
        cfg.train.eval_interval = 0
        cfg.train.seed = 1000 + s
        # Throwaway checkpoints: never clobber the deployment models that
        # build_site_data.py --real writes (which export_policy/baseline_report use).
        cfg.train.checkpoint_dir = os.path.join("checkpoints", "_significance")
        agent, _ = run_ppo_training(cfg, train_series_factory=train_path_factory)

        path_returns = []
        for data in holdout:
            env = make_env(args.market, data, cfg.env, cfg.reward, random_start=False)
            bt = backtest(agent, env, market=args.market)
            path_returns.append(bt.metrics["total_return"])
        path_returns = np.array(path_returns)
        agent_returns_by_path += path_returns

        seed_mean_returns.append(float(path_returns.mean()))
        seed_mean_sharpes.append(
            float(np.mean([
                backtest(agent, make_env(args.market, d, cfg.env, cfg.reward, random_start=False),
                         market=args.market).metrics["sharpe"]
                for d in holdout
            ]))
        )
        log.info("seed %d/%d | OOS mean return %+.2f%%", s + 1, args.seeds,
                 100 * seed_mean_returns[-1])

    agent_returns_by_path /= args.seeds

    ret_ci = bootstrap_ci(seed_mean_returns)
    sharpe_ci = bootstrap_ci(seed_mean_sharpes)
    obs_diff, p_value = paired_permutation_test(agent_returns_by_path, bh_returns)

    print("\n" + "=" * 64)
    print(f"MULTI-SEED SIGNIFICANCE - {args.market.upper()} "
          f"({args.seeds} seeds, {args.eval_paths} held-out paths)")
    print("=" * 64)
    print(f"Agent OOS return : mean {ret_ci.mean:+.2%}  "
          f"95% CI [{ret_ci.low:+.2%}, {ret_ci.high:+.2%}]")
    print(f"Agent OOS Sharpe : mean {sharpe_ci.mean:+.2f}  "
          f"95% CI [{sharpe_ci.low:+.2f}, {sharpe_ci.high:+.2f}]")
    print(f"Buy & hold return: mean {bh_returns.mean():+.2%}")
    print(f"Agent vs B&H     : {obs_diff:+.2%}  (paired permutation p = {p_value:.4f})")
    verdict = "DISTINGUISHABLE from B&H" if p_value < 0.05 else "NOT distinguishable from B&H"
    print(f"Verdict          : agent is {verdict} at alpha=0.05")
    print("=" * 64)


if __name__ == "__main__":
    main()
