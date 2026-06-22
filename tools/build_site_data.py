"""Generate REAL backtest results for the web prototype.

Trains a PPO agent on each market, backtests it on the held-out test split,
computes a buy-&-hold baseline on the same window, and writes
``docs/results.js`` — a global ``window.RL_RESULTS`` object the static site
loads directly. Emitting a JS global (rather than JSON) means the page renders
real model output with **no web server**, working under ``file://`` and GitHub
Pages alike.

Run from the repo root:

    python tools/build_site_data.py --timesteps 60000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import random

import numpy as np

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.data.data_loader import (
    add_technical_indicators,
    load_ohlcv_csv,
    prepare_market_data,
    synthetic_market_data,
)
from rl_trader.envs import make_env
from rl_trader.evaluation.evaluate_agent import ANNUALISATION, backtest, compute_metrics
from rl_trader.training.utils import run_ppo_training


def _downsample(arr, n: int = 160) -> list:
    """Reduce a long series to ~n points so the embedded file stays small."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= n:
        return [round(float(v), 4) for v in arr]
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return [round(float(v), 4) for v in arr[idx]]


def _pick(seq, n: int = 160) -> list:
    """Downsample a non-numeric sequence (e.g. date strings) on the same index
    grid as :func:`_downsample`, so labels stay aligned with the plotted points."""
    seq = list(seq)
    if len(seq) <= n:
        return seq
    idx = np.linspace(0, len(seq) - 1, n).astype(int)
    return [seq[i] for i in idx]


def _drawdown(equity) -> list:
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    return ((peak - equity) / peak).tolist()


def _sortino(returns: np.ndarray, periods: int) -> float:
    downside = returns[returns < 0]
    dd = downside.std() if len(downside) else 0.0
    if dd < 1e-12:
        return 0.0
    return float(np.sqrt(periods) * returns.mean() / dd)


def _backtest_on_path(agent, market, cfg, periods, seed) -> dict:
    """Backtest the agent on one held-out path; return its result + baseline.

    Evaluation paths are deliberately shorter than training paths (~2.5 years of
    daily bars) so compounding stays in a realistic range.
    """
    data = synthetic_market_data(market, seed=seed, n_steps=650)
    env = make_env(market, data, cfg.env, cfg.reward, random_start=False)
    res = backtest(agent, env, market=market)

    w = cfg.env.window_size
    prices = data.prices[w - 1 :]
    bench_equity = cfg.env.initial_balance * (prices / prices[0])

    agent_m = dict(res.metrics)
    agent_m["sortino"] = _sortino(res.returns, periods)
    return {
        "res": res,
        "bench_equity": bench_equity,
        "agent_m": agent_m,
        "bench_m": compute_metrics(bench_equity, periods),
    }


def run_market(market: str, cfg, timesteps: int, seed: int, n_eval: int = 40) -> dict:
    """Train with domain randomization, then evaluate over many unseen paths.

    Reporting the **mean** over ``n_eval`` independent held-out paths (plus a
    win-rate vs. buy-&-hold) is far more honest than a single backtest, which is
    hostage to one path's luck. The displayed curves use the *median-return*
    path so the visual matches the typical case.
    """
    cfg.market = market
    cfg.train.total_timesteps = timesteps
    cfg.train.seed = seed
    cfg.train.eval_interval = 0  # skip in-loop validation for speed

    # Domain-randomized training: a fresh synthetic path every episode forces a
    # generalizable policy. None of the evaluation paths below are ever trained on.
    factory = lambda: synthetic_market_data(market)  # noqa: E731
    agent, history = run_ppo_training(cfg, train_series_factory=factory)

    periods = ANNUALISATION.get(market, 252)
    runs = [
        _backtest_on_path(agent, market, cfg, periods, seed=10_000 + k)
        for k in range(n_eval)
    ]
    return _aggregate(runs, history, cfg)


def _aggregate(runs: list, history: dict, cfg) -> dict:
    """Average a list of backtest runs into the payload the website consumes.

    Reporting the mean over many held-out backtests (plus a win-rate vs.
    buy-&-hold) is the honest way to summarise performance; the displayed curves
    use the median-return run so the visual reflects the typical case.
    """
    def _mean(which: str, key: str) -> float:
        return float(np.mean([r[which][key] for r in runs]))

    agent_metrics = {
        k: round(_mean("agent_m", k), 4)
        for k in ("total_return", "sharpe", "sortino", "max_drawdown", "final_equity")
    }
    agent_metrics["n_steps"] = int(np.mean([len(r["res"].actions) for r in runs]))
    bench_metrics = {
        k: round(_mean("bench_m", k), 4)
        for k in ("total_return", "sharpe", "max_drawdown", "final_equity")
    }
    win_rate = float(np.mean([r["agent_m"]["total_return"] > r["bench_m"]["total_return"] for r in runs]))

    order = np.argsort([r["agent_m"]["total_return"] for r in runs])
    rep = runs[int(order[len(order) // 2])]
    rep_dates = rep.get("dates", [])

    all_actions = np.concatenate([r["res"].actions for r in runs])
    if len(all_actions) > 600:
        idx = np.linspace(0, len(all_actions) - 1, 600).astype(int)
        all_actions = all_actions[idx]

    return {
        "equity_agent": _downsample(rep["res"].equity_curve),
        "equity_bench": _downsample(rep["bench_equity"]),
        "drawdown": _downsample(_drawdown(rep["res"].equity_curve)),
        "actions": [round(float(a), 3) for a in all_actions],
        "start_date": rep_dates[0] if rep_dates else None,
        "end_date": rep_dates[-1] if rep_dates else None,
        "dates": _pick(rep_dates[1:]) if rep_dates else None,
        "metrics": agent_metrics,
        "bench_metrics": bench_metrics,
        "win_rate": round(win_rate, 3),
        "n_eval": len(runs),
        "training": {
            "update": history["update"],
            "reward": [round(float(r), 4) for r in history["mean_episode_return"]],
        },
        "initial_balance": cfg.env.initial_balance,
    }


def _per_ticker_record(run: dict) -> dict:
    """Compact per-ticker payload for the interactive explorer.

    Keeps each named asset's own agent-vs-buy&hold curves, plus the price series
    and the agent's per-step position (``actions_t``) so the frontend can render a
    "watch the agent act" scrubber. ``actions`` is one shorter than the equity
    curve (an action per transition), so the price line is aligned to it by
    dropping the first bar — that way price[i] and actions_t[i] share an index.
    """
    res = run["res"]
    prices_aligned = np.asarray(run["prices"], dtype=float)[1:]  # match actions length
    dates = run.get("dates", [])
    dates_aligned = dates[1:] if dates else []                   # match actions/price
    agent_m, bench_m = run["agent_m"], run["bench_m"]
    return {
        "ticker": run["ticker"],
        "equity_agent": _downsample(res.equity_curve),
        "equity_bench": _downsample(run["bench_equity"]),
        "price": _downsample(prices_aligned),
        "actions_t": _downsample(res.actions),
        "dates": _pick(dates_aligned),
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
        "drawdown": _downsample(_drawdown(res.equity_curve)),
        "metrics": {
            k: round(float(agent_m[k]), 4)
            for k in ("total_return", "sharpe", "sortino", "max_drawdown", "final_equity")
        },
        "bench_metrics": {
            k: round(float(bench_m[k]), 4)
            for k in ("total_return", "sharpe", "max_drawdown", "final_equity")
        },
        "latest_action": round(float(res.actions[-1]), 3),
    }


def load_real_basket(data_dir: str, market: str, train_frac: float = 0.6) -> dict:
    """Load every ticker CSV under ``data_dir/market`` into train/test splits.

    Each ticker is split **chronologically** (older ``train_frac`` for training,
    the recent remainder held out for testing) with the feature scaler fit on
    that ticker's training slice only — a clean walk-forward setup with no
    look-ahead leakage.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, market, "*.csv")))
    basket = {}
    for path in paths:
        ticker = os.path.splitext(os.path.basename(path))[0]
        df = load_ohlcv_csv(path)
        splits = prepare_market_data(df, market=market, train_frac=train_frac, val_frac=0.0)
        if len(splits["train"]) > 60 and len(splits["test"]) > 60:
            # Recover the held-out test-window dates (prepare_market_data drops the
            # date column). The chronological split matches prepare_market_data's:
            # test = rows[int(n*train_frac):], on the indicator-featured frame.
            featured = add_technical_indicators(df)
            train_end = int(len(featured) * train_frac)
            splits["dates"] = featured["date"].astype(str).str.slice(0, 10).tolist()[train_end:]
            basket[ticker] = splits
    return basket


def run_market_real(market: str, cfg, timesteps: int, seed: int, data_dir: str) -> dict:
    """Train on a basket of real tickers, then backtest on each one's held-out
    recent period — a multi-asset walk-forward evaluation on real market data."""
    cfg.market = market
    cfg.train.total_timesteps = timesteps
    cfg.train.seed = seed
    cfg.train.eval_interval = 0

    basket = load_real_basket(data_dir, market)
    if not basket:
        raise SystemExit(
            f"No CSVs found in {os.path.join(data_dir, market)} — run tools/fetch_data.py first."
        )
    tickers = list(basket)
    train_slices = [basket[t]["train"] for t in tickers]

    # Domain randomization across real assets: each episode trains on a random
    # ticker's history, so the policy must generalize rather than fit one name.
    random.seed(seed)
    factory = lambda: random.choice(train_slices)  # noqa: E731
    agent, history = run_ppo_training(cfg, train_series_factory=factory)

    periods = ANNUALISATION.get(market, 252)
    runs = []
    for t in tickers:
        test = basket[t]["test"]
        env = make_env(market, test, cfg.env, cfg.reward, random_start=False)
        res = backtest(agent, env, market=market)
        w = cfg.env.window_size
        prices = test.prices[w - 1 :]
        dates = basket[t]["dates"][w - 1 :]  # aligned to prices / equity curve
        bench_equity = cfg.env.initial_balance * (prices / prices[0])
        agent_m = dict(res.metrics)
        agent_m["sortino"] = _sortino(res.returns, periods)
        runs.append({
            "res": res, "bench_equity": bench_equity, "ticker": t, "prices": prices,
            "dates": dates,
            "agent_m": agent_m, "bench_m": compute_metrics(bench_equity, periods),
        })

    payload = _aggregate(runs, history, cfg)
    payload["assets"] = tickers
    payload["per_ticker"] = [_per_ticker_record(r) for r in runs]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build docs/results.js from real runs.")
    parser.add_argument("--timesteps", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real", action="store_true",
                        help="Train/evaluate on real OHLCV under --data-dir (run fetch_data.py first).")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default=os.path.join("docs", "results.js"))
    args = parser.parse_args()

    if args.real:
        markets = {
            "stock": run_market_real("stock", stock_config(), args.timesteps, args.seed, args.data_dir),
            "crypto": run_market_real("crypto", crypto_config(), args.timesteps, args.seed, args.data_dir),
        }
        n_stock = len(markets["stock"].get("assets", []))
        n_crypto = len(markets["crypto"].get("assets", []))
        data_source = (
            f"real daily OHLCV via Yahoo Finance · {n_stock} stocks + {n_crypto} crypto pairs · "
            f"walk-forward held-out test period"
        )
    else:
        markets = {
            "stock": run_market("stock", stock_config(), args.timesteps, args.seed),
            "crypto": run_market("crypto", crypto_config(), args.timesteps, args.seed),
        }
        data_source = "synthetic trending series (momentum); swap in real CSVs via --real"

    payload = {
        "generated": _dt.date.today().isoformat(),
        "data_source": data_source,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "markets": markets,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("/* Auto-generated by tools/build_site_data.py — real backtest output. */\n")
        fh.write("window.RL_RESULTS = ")
        json.dump(payload, fh, indent=0)
        fh.write(";\n")

    print(f"\nWrote {args.out}  (mean over {payload['markets']['stock']['n_eval']} held-out paths)")
    for mkt in ("stock", "crypto"):
        m = payload["markets"][mkt]
        a, b = m["metrics"], m["bench_metrics"]
        print(
            f"  {mkt:<6} agent: ret {a['total_return']:+.2%}  sharpe {a['sharpe']:+.2f}  "
            f"maxDD {a['max_drawdown']:.2%}   |   buy&hold: ret {b['total_return']:+.2%}  "
            f"sharpe {b['sharpe']:+.2f}  maxDD {b['max_drawdown']:.2%}   |   win-rate {m['win_rate']:.0%}"
        )


if __name__ == "__main__":
    main()
