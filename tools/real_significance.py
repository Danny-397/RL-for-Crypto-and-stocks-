"""Real-data significance: is the out-of-sample crypto/stock edge statistically real?

The headline dashboard number comes from one training seed on one walk-forward
split. This script closes that gap with the same rigor the synthetic study uses,
but on **real markets**:

1. Train ``--seeds`` independent agents on the real basket (domain-randomized
   across tickers), each on a throwaway checkpoint so deployment models are safe.
2. For every seed, backtest on each ticker's held-out test period and record the
   agent's return vs. buy-&-hold's.
3. Report (a) a bootstrap 95% CI on the basket-mean agent return *across seeds*
   — how repeatable is it? — and (b) a paired permutation test of the agent vs.
   buy-&-hold *across the held-out tickers* — is the cross-sectional edge
   distinguishable from noise?

Run from the repo root (after tools/fetch_data.py):

    python tools/real_significance.py --seeds 5 --timesteps 150000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import numpy as np

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.data.data_loader import attach_market_index, load_ohlcv_csv, prepare_market_data
from rl_trader.envs import make_env
from rl_trader.evaluation.evaluate_agent import ANNUALISATION, backtest, compute_metrics
from rl_trader.evaluation.statistics import bootstrap_ci, paired_permutation_test
from rl_trader.training.utils import get_logger, run_ppo_training


def common_split_date(frames: dict, train_frac: float = 0.6):
    """The date that splits the window *every* ticker trades in at ``train_frac``.

    Used by ``split="common"`` so that no ticker's training period overlaps any
    ticker's test period. See RESULTS.md §10: with per-ticker splits, tickers
    whose history starts later are trained on dates that are already test dates
    for tickers whose history starts earlier.
    """
    from rl_trader.data.data_loader import add_technical_indicators

    dates = {t: add_technical_indicators(df)["date"] for t, df in frames.items()}
    start = max(d.iloc[0] for d in dates.values())
    end = min(d.iloc[-1] for d in dates.values())
    shortest = min((d[(d >= start) & (d <= end)] for d in dates.values()), key=len).reset_index(drop=True)
    return shortest.iloc[int(len(shortest) * train_frac)]


def load_real_basket(data_dir: str, market: str, train_frac: float = 0.6, split: str | None = None) -> dict:
    """Load every ticker CSV under ``data_dir/market`` into train/test splits.

    Chronological split with the scaler fit on the training slice only.

    ``split="ticker"`` (as published) cuts each ticker at ``train_frac`` of *its
    own* history. Every ticker's test period follows its own training period, but
    when histories start on different dates the test periods do not line up, so
    one ticker can be trained on dates that are another's test period. For the
    stock basket every series covers the same dates and the two modes coincide;
    for crypto they do not. ``split="common"`` cuts every ticker on one calendar
    date (:func:`common_split_date`), so no training date is a test date anywhere.

    The default is ``"common"`` from v2.3.0, and ``"ticker"`` under
    ``RL_TRADER_PUBLISHED=1``.
    """
    from rl_trader.config.training_config import published_mode

    if split is None:
        split = "ticker" if published_mode() else "common"
    from rl_trader.data.data_loader import add_technical_indicators

    frames = {}
    for path in sorted(glob.glob(os.path.join(data_dir, market, "*.csv"))):
        ticker = os.path.splitext(os.path.basename(path))[0]
        frames[ticker] = attach_market_index(load_ohlcv_csv(path), data_dir, market)
    cut = common_split_date(frames, train_frac) if split == "common" and frames else None
    basket = {}
    for ticker, df in frames.items():
        frac = train_frac
        if cut is not None:
            d = add_technical_indicators(df)["date"]
            k = int((d < cut).sum())
            frac = (k + 0.5) / len(d)          # so int(n * frac) == k inside the splitter
        splits = prepare_market_data(df, market=market, train_frac=frac, val_frac=0.0)
        if len(splits["train"]) > 60 and len(splits["test"]) > 60:
            basket[ticker] = splits
    return basket


def _alpha_beta(agent_steps: np.ndarray, asset_steps: np.ndarray, periods: int):
    """Annualised alpha and beta of the agent's daily returns on the asset's own.

    Total return against buy-and-hold rewards whoever held the most of a rising
    asset. Regressing on the asset separates the exposure the agent carried
    (beta) from what it added beyond that exposure (alpha), which is the
    comparison a quant would make. Zero risk-free rate, as everywhere else here.
    """
    n = min(len(agent_steps), len(asset_steps))
    a, b = agent_steps[:n], asset_steps[:n]
    var = float(np.var(b))
    beta = float(np.cov(a, b, bias=True)[0, 1] / var) if var > 1e-18 else 0.0
    alpha = float((a.mean() - beta * b.mean()) * periods)
    return alpha, beta


def _evaluate_basket(agent, basket, market, cfg, periods):
    """Per-ticker results on the held-out test split.

    Returns the agent's and buy-&-hold's total returns (the published study) and,
    alongside them, risk-adjusted measures: each side's annualised Sharpe ratio,
    and the agent's annualised alpha and beta on the asset it traded.
    """
    agent_rets, bh_rets = [], []
    agent_sh, bh_sh, alphas, betas = [], [], [], []
    w = cfg.env.window_size
    for sp in basket.values():
        test = sp["test"]
        env = make_env(market, test, cfg.env, cfg.reward, random_start=False)
        res = backtest(agent, env, market=market)
        agent_rets.append(res.metrics["total_return"])
        prices = test.prices[w - 1 :]
        bh = compute_metrics(cfg.env.initial_balance * (prices / prices[0]), periods)
        bh_rets.append(bh["total_return"])
        agent_sh.append(res.metrics["sharpe"])
        bh_sh.append(bh["sharpe"])
        alpha, beta = _alpha_beta(res.returns, np.diff(prices) / prices[:-1], periods)
        alphas.append(alpha)
        betas.append(beta)
    risk = {"agent_sharpe": np.array(agent_sh), "bh_sharpe": np.array(bh_sh),
            "alpha": np.array(alphas), "beta": np.array(betas)}
    return np.array(agent_rets), np.array(bh_rets), risk


def run_market(market: str, cfg_fn, seeds: int, timesteps: int, data_dir: str, log,
               seed_base: int = 100):
    basket = load_real_basket(data_dir, market)
    if not basket:
        raise SystemExit(f"No CSVs in {os.path.join(data_dir, market)} — run tools/fetch_data.py.")
    tickers = list(basket)
    train_slices = [basket[t]["train"] for t in tickers]
    periods = ANNUALISATION.get(market, 252)

    seed_mean_returns, seed_win_rates = [], []
    agent_by_asset = np.zeros(len(tickers))
    bh_by_asset = None
    risk_sum = {k: np.zeros(len(tickers)) for k in ("agent_sharpe", "alpha", "beta")}
    bh_sharpe = None
    seed_alphas = []

    for s in range(seeds):
        cfg = cfg_fn()
        cfg.market = market
        cfg.train.total_timesteps = timesteps
        cfg.train.eval_interval = 0
        cfg.train.seed = seed_base + s
        cfg.train.checkpoint_dir = os.path.join("checkpoints", "_realsig")
        random.seed(seed_base + s)

        def factory():
            return random.choice(train_slices)

        agent, _ = run_ppo_training(cfg, train_series_factory=factory)
        a, b, risk = _evaluate_basket(agent, basket, market, cfg, periods)
        bh_by_asset = b
        agent_by_asset += a
        for k in risk_sum:
            risk_sum[k] += risk[k]
        bh_sharpe = risk["bh_sharpe"]
        seed_alphas.append(float(risk["alpha"].mean()))
        seed_mean_returns.append(float(a.mean()))
        seed_win_rates.append(float((a > b).mean()))
        log.info("[%s] seed %d/%d | basket-mean agent %+.2f%% | win-rate %.0f%%",
                 market, s + 1, seeds, 100 * seed_mean_returns[-1], 100 * seed_win_rates[-1])

    agent_by_asset /= seeds
    ret_ci = bootstrap_ci(seed_mean_returns)
    obs_diff, p_value = paired_permutation_test(agent_by_asset, bh_by_asset)

    # Risk-adjusted: the same paired test across tickers, on Sharpe ratios, and a
    # sign-flip test of the agent's alpha on the traded asset against zero.
    per_asset = {k: v / seeds for k, v in risk_sum.items()}
    sharpe_diff, sharpe_p = paired_permutation_test(per_asset["agent_sharpe"], bh_sharpe)
    alpha_mean, alpha_p = paired_permutation_test(per_asset["alpha"], np.zeros(len(tickers)))
    alpha_ci = bootstrap_ci(seed_alphas)
    risk_summary = {
        "agent_sharpe": round(float(per_asset["agent_sharpe"].mean()), 3),
        "bh_sharpe": round(float(bh_sharpe.mean()), 3),
        "sharpe_diff": round(float(sharpe_diff), 3),
        "sharpe_p": round(float(sharpe_p), 4),
        "alpha": round(float(alpha_mean), 4),
        "alpha_ci_low": round(float(alpha_ci.low), 4),
        "alpha_ci_high": round(float(alpha_ci.high), 4),
        "alpha_p": round(float(alpha_p), 4),
        "beta": round(float(per_asset["beta"].mean()), 3),
        "per_ticker": {t: {"agent_sharpe": round(float(per_asset["agent_sharpe"][i]), 3),
                           "bh_sharpe": round(float(bh_sharpe[i]), 3),
                           "alpha": round(float(per_asset["alpha"][i]), 4),
                           "beta": round(float(per_asset["beta"][i]), 3)}
                       for i, t in enumerate(tickers)},
    }
    return {
        "tickers": tickers,
        "ret_ci": ret_ci,
        "seed_returns": [round(x, 4) for x in seed_mean_returns],
        "mean_win_rate": float(np.mean(seed_win_rates)),
        "bh_mean": float(bh_by_asset.mean()),
        "diff": obs_diff,
        "p": p_value,
        "risk": risk_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--market", choices=["stock", "crypto", "both"], default="both")
    # The published study used seeds 100.. ; a replication picks a base it has never run.
    parser.add_argument("--seed-base", type=int, default=100)
    parser.add_argument("--out", default=os.path.join("docs", "significance.js"),
                        help="where to write the per-seed results (default: the site's data file)")
    args = parser.parse_args()

    log = get_logger("real_significance")
    markets = ["stock", "crypto"] if args.market == "both" else [args.market]
    cfg_fns = {"stock": stock_config, "crypto": crypto_config}

    print("\n" + "=" * 70)
    print(f"REAL-DATA SIGNIFICANCE  ({args.seeds} seeds, {args.timesteps:,} timesteps)")
    print("=" * 70)
    site = {}
    for market in markets:
        r = run_market(market, cfg_fns[market], args.seeds, args.timesteps, args.data_dir, log,
                       seed_base=args.seed_base)
        ci = r["ret_ci"]
        verdict = "DISTINGUISHABLE" if r["p"] < 0.05 else "not distinguishable"
        print(f"\n{market.upper()}  ({len(r['tickers'])} held-out tickers)")
        print(f"  Agent basket return : mean {ci.mean:+.2%}  95% CI [{ci.low:+.2%}, {ci.high:+.2%}]")
        print(f"  Buy & hold (mean)   : {r['bh_mean']:+.2%}")
        print(f"  Mean win-rate       : {r['mean_win_rate']:.0%} of tickers beat buy-&-hold")
        print(f"  Agent vs B&H        : {r['diff']:+.2%}  (paired permutation p = {r['p']:.4f})")
        print(f"  Verdict             : cross-sectional edge is {verdict} at alpha=0.05")
        site[market] = {
            "seed_returns": r["seed_returns"], "mean": round(ci.mean, 4),
            "ci_low": round(ci.low, 4), "ci_high": round(ci.high, 4),
            "bh": round(r["bh_mean"], 4), "p": round(r["p"], 4), "seeds": args.seeds,
            "seed_base": args.seed_base, "diff": round(r["diff"], 4),
            "risk": r["risk"],
        }
    print("=" * 70)

    # Emit a small JS global the dashboard plots as a "multi-seed distribution" chart.
    out = args.out
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("/* Auto-generated by tools/real_significance.py — per-seed real-data returns. */\n")
        fh.write("window.RL_SIGNIFICANCE = ")
        json.dump(site, fh, indent=0)
        fh.write(";\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
