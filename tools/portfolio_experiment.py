"""Cross-sectional portfolio experiment: train + evaluate vs. quant baselines.

Trains a PPO agent that allocates across a whole basket at once, then backtests
it on the held-out test split against the benchmarks a quant would reach for:
equal-weight (passive), cross-sectional momentum (the classic factor), and random
weights. With ``--seeds > 1`` it repeats training across seeds and reports a
bootstrap confidence interval on the agent's return — the same honesty discipline
the single-asset study uses, so a lucky run can't masquerade as an edge.

Run from the repo root (uses cached real OHLCV; pass --synthetic for no data):

    python tools/portfolio_experiment.py --market stock --timesteps 150000
    python tools/portfolio_experiment.py --market stock --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.data.portfolio_data import load_portfolio, synthetic_portfolio
from rl_trader.envs.portfolio_env import PortfolioTradingEnv
from rl_trader.evaluation.evaluate_agent import ANNUALISATION
from rl_trader.evaluation.portfolio_eval import evaluate_portfolio_baselines, portfolio_backtest
from rl_trader.evaluation.statistics import bootstrap_ci
from rl_trader.training.portfolio import train_portfolio


def _splits(args):
    if args.synthetic:
        return synthetic_portfolio(n_assets=args.n_assets, market=args.market, seed=args.seed)
    return load_portfolio(args.data_dir, args.market)


def _train_one(args, splits, seed: int):
    cfg = stock_config() if args.market == "stock" else crypto_config()
    cfg.market = args.market
    cfg.train.total_timesteps = args.timesteps
    cfg.train.seed = seed
    cfg.train.eval_interval = 0
    cfg.train.checkpoint_dir = os.path.join("checkpoints", "_portfolio")
    agent, _ = train_portfolio(cfg, splits)
    return agent, cfg



def _record(args, tickers, rows: dict) -> None:
    """Merge this market's table into docs/assets/portfolio.json.

    Until now this experiment only printed, so RESULTS.md carried the table
    typed by hand -- and it drifted: the run that produced the published
    numbers put the allocator at -43.4% where the pinned rebuild puts it at
    +38.7%. Written per market and merged, because the two markets are run as
    separate commands and neither should clobber the other's result.

    tools/sync_docs.py regenerates the table in RESULTS.md from this file.
    """
    import datetime
    out = os.path.join("docs", "assets", "portfolio.json")
    payload = {"markets": {}}
    if os.path.exists(out):
        try:
            with open(out, encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError:
            pass
    payload.setdefault("markets", {})[args.market] = {
        "generated": datetime.date.today().isoformat(),
        "seed": args.seed,
        "timesteps": args.timesteps,
        "n_assets": len(tickers),
        "tickers": list(tickers),
        "strategies": {
            name: {k: float(m[k]) for k in ("total_return", "sharpe", "max_drawdown")}
            for name, m in rows.items()
        },
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=["stock", "crypto"], default="stock")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1, help=">1 runs a multi-seed CI study")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-assets", type=int, default=5, help="synthetic basket size")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    args = parser.parse_args()

    splits = _splits(args)
    periods = ANNUALISATION.get(args.market, 252)
    n_assets = splits["train"].n_assets
    tickers = splits["train"].tickers

    # Baselines are deterministic given the data — compute once.
    cfg0 = stock_config() if args.market == "stock" else crypto_config()
    baselines = evaluate_portfolio_baselines(splits["test"], cfg0.env, cfg0.reward, periods)

    print("\n" + "=" * 70)
    print(f"CROSS-SECTIONAL PORTFOLIO — {args.market.upper()}  "
          f"({n_assets} assets: {', '.join(tickers)})")
    print("=" * 70)

    if args.seeds <= 1:
        agent, cfg = _train_one(args, splits, args.seed)
        env = PortfolioTradingEnv(splits["test"], cfg.env, cfg.reward, random_start=False)
        bt = portfolio_backtest(agent, env, periods)
        rows = {"PPO portfolio agent": bt["metrics"]}
        for name, r in baselines.items():
            rows[name] = r["metrics"]
        print(f"\n{'Strategy':<28}{'Return':>10}{'Sharpe':>9}{'Max DD':>9}")
        print("-" * 56)
        for name, m in rows.items():
            print(f"{name:<28}{m['total_return']:>+9.1%}{m['sharpe']:>9.2f}{m['max_drawdown']:>9.1%}")
        _record(args, tickers, rows)
    else:
        seed_returns = []
        for s in range(args.seeds):
            agent, cfg = _train_one(args, splits, 100 + s)
            env = PortfolioTradingEnv(splits["test"], cfg.env, cfg.reward, random_start=False)
            r = portfolio_backtest(agent, env, periods)["metrics"]["total_return"]
            seed_returns.append(r)
            print(f"  seed {s + 1}/{args.seeds}: agent return {r:+.2%}")
        ci = bootstrap_ci(seed_returns)
        eq = baselines["equal_weight"]["metrics"]["total_return"]
        print(f"\nAgent return : mean {ci.mean:+.2%}  95% CI [{ci.low:+.2%}, {ci.high:+.2%}]  (n={args.seeds})")
        print(f"Equal-weight : {eq:+.2%}")
        beats = "ABOVE" if ci.low > eq else ("BELOW" if ci.high < eq else "OVERLAPS")
        print(f"Verdict      : agent CI {beats} equal-weight")
    print("=" * 70)


if __name__ == "__main__":
    main()
