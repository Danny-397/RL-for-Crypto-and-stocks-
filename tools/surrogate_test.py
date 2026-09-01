"""Surrogate-data falsification test — is the agent exploiting *structure*, or noise?

An original probe, borrowed from nonlinear time-series analysis (surrogate-data
testing; Theiler et al., 1992) and applied here to a trading RL agent. The other
experiments in this repo show the agent *doesn't* beat buy-&-hold on real markets.
This one asks the sharper question: **is that because the agent is weak, or because
there is no exploitable temporal structure to find?**

Method
------
Build a **return-shuffled surrogate** of a price series: randomly permute its daily
log-returns and re-integrate. Because a sum is permutation-invariant, the surrogate
ends at the **exact same price** — so buy-&-hold is *identical* — but every bit of
temporal structure a timing agent could exploit (momentum, autocorrelation,
volatility clustering) is destroyed. We then train and evaluate the same PPO recipe
on the real series and on its surrogate, and compare the agent's edge over B&H.

Two arms
--------
* ``--mode synthetic`` (positive control): on synthetic data with a *known* AR(1)
  momentum signal, a working test must find edge_structured >> edge_surrogate — the
  agent exploits the signal, and loses that edge once shuffling removes it. This
  proves the falsification test has statistical power.
* ``--mode real``: apply the identical lens to the real basket. If
  edge_real ≈ edge_surrogate (indistinguishable), the agent finds no more
  exploitable structure in real prices than in pure noise — i.e. the underperformance
  is *signal absence*, not a broken agent. That is the novel, falsifiable claim.

Run from the repo root:

    python tools/surrogate_test.py --mode synthetic --seeds 3 --timesteps 40000
    python tools/surrogate_test.py --mode real      --seeds 3 --timesteps 120000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd

# The report tables use Unicode (− Δ …); make stdout tolerate them on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - older interpreters / redirected streams
    pass

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.data.data_loader import (
    attach_market_index,
    generate_synthetic_ohlcv,
    load_ohlcv_csv,
    market_data_from_df,
    market_regime,
    prepare_market_data,
)
from rl_trader.envs import make_env
from rl_trader.evaluation.evaluate_agent import ANNUALISATION, backtest, compute_metrics
from rl_trader.evaluation.statistics import bootstrap_ci, paired_permutation_test
from rl_trader.training.utils import get_logger, run_ppo_training

CFG = {"stock": stock_config, "crypto": crypto_config}


# --------------------------------------------------------------------------- #
# Surrogate construction                                                       #
# --------------------------------------------------------------------------- #
def surrogate_df(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Return a return-shuffled surrogate of an OHLCV frame.

    The daily log-return *multiset* is preserved (so the final price — and thus
    buy-&-hold — is identical), but the temporal ordering is randomised, erasing
    every exploitable pattern. OHLC bars and the cross-asset index are rebuilt
    consistently so the whole feature pipeline still runs unchanged.
    """
    out = df.reset_index(drop=True)
    close = out["close"].to_numpy(float)
    logret = np.diff(np.log(np.clip(close, 1e-9, None)))
    rng.shuffle(logret)
    new_close = np.empty_like(close)
    new_close[0] = close[0]
    new_close[1:] = close[0] * np.exp(np.cumsum(logret))  # same endpoint => same B&H

    o = np.empty_like(new_close)
    o[0] = new_close[0]
    o[1:] = new_close[:-1]
    hl = np.abs(((out["high"] - out["low"]) / out["close"]).to_numpy(float))
    rng.shuffle(hl)  # shuffle the intrabar ranges too — no structure survives
    half = 0.5 * hl * new_close
    hi = np.maximum(o, new_close) + half
    lo = np.minimum(o, new_close) - half
    vol = out["volume"].to_numpy(float).copy()
    rng.shuffle(vol)

    res = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": new_close, "volume": vol})
    if "date" in out.columns:
        res["date"] = out["date"].to_numpy()
    if "_mkt_close" in out.columns:  # destroy structure in the reference index as well
        mc = out["_mkt_close"].to_numpy(float)
        mlr = np.diff(np.log(np.clip(mc, 1e-9, None)))
        rng.shuffle(mlr)
        nmc = np.empty_like(mc)
        nmc[0] = mc[0]
        nmc[1:] = mc[0] * np.exp(np.cumsum(mlr))
        res["_mkt_close"] = nmc
    return res


def _bh_return(md, cfg, periods) -> float:
    return compute_metrics(cfg.env.initial_balance * (md.prices / md.prices[0]), periods)["total_return"]


def _train(market, factory, seeds, timesteps, seed_base):
    """Train `seeds` domain-randomized agents on `factory`; yield each agent."""
    for s in range(seeds):
        cfg = CFG[market]()
        cfg.market = market
        cfg.train.total_timesteps = timesteps
        cfg.train.eval_interval = 0
        cfg.train.seed = seed_base + s
        cfg.train.checkpoint_dir = os.path.join("checkpoints", "_surrogate")
        agent, _ = run_ppo_training(cfg, train_series_factory=factory)
        yield agent, cfg


# --------------------------------------------------------------------------- #
# Synthetic arm — positive control (a signal provably exists)                  #
# --------------------------------------------------------------------------- #
def _synth_df(market: str, seed):
    vol, drift, mom = market_regime(market)
    return generate_synthetic_ohlcv(
        n_steps=1_400, annual_vol=vol, annual_drift=drift, momentum=mom, seed=seed
    )


def run_synthetic(market, seeds, timesteps, eval_paths, log) -> dict:
    periods = ANNUALISATION.get(market, 252)
    hrng = np.random.default_rng(7_777)
    base = [_synth_df(market, int(hrng.integers(1e9))) for _ in range(eval_paths)]
    struct_hold = [market_data_from_df(df) for df in base]
    srng = np.random.default_rng(1_234)
    surr_hold = [market_data_from_df(surrogate_df(df, srng)) for df in base]

    def edge(holdout, factory, seed_base):
        cfg0 = CFG[market]()
        bh = np.array([_bh_return(md, cfg0, periods) for md in holdout])
        agent_by_path = np.zeros(len(holdout))
        for i, (agent, cfg) in enumerate(_train(market, factory, seeds, timesteps, seed_base)):
            rets = np.array([
                backtest(agent, make_env(market, md, cfg.env, cfg.reward, random_start=False),
                         market=market).metrics["total_return"]
                for md in holdout
            ])
            agent_by_path += rets
            log.info("[%s/%s] seed %d/%d | agent %+.1f%% vs B&H %+.1f%%",
                     market, factory.__name__, i + 1, seeds, 100 * rets.mean(), 100 * bh.mean())
        return agent_by_path / seeds - bh  # per-path edge over buy-&-hold

    frng = np.random.default_rng(99)

    def structured():  # noqa: D401 - factory
        return market_data_from_df(_synth_df(market, None))

    def surrogate():  # noqa: D401 - factory
        return market_data_from_df(surrogate_df(_synth_df(market, None), frng))

    edge_struct = edge(struct_hold, structured, 500)
    edge_surr = edge(surr_hold, surrogate, 800)
    diff, p = paired_permutation_test(edge_struct, edge_surr)
    return _summary(market, edge_struct, edge_surr, diff, p)


# --------------------------------------------------------------------------- #
# Real arm — the falsification claim on actual markets                         #
# --------------------------------------------------------------------------- #
def load_basket(data_dir, market, surrogate=False, rng=None, train_frac=0.6):
    basket = {}
    for path in sorted(glob.glob(os.path.join(data_dir, market, "*.csv"))):
        ticker = os.path.splitext(os.path.basename(path))[0]
        df = attach_market_index(load_ohlcv_csv(path), data_dir, market)
        if surrogate:
            df = surrogate_df(df, rng)
        splits = prepare_market_data(df, market=market, train_frac=train_frac, val_frac=0.0)
        if len(splits["train"]) > 60 and len(splits["test"]) > 60:
            basket[ticker] = splits
    return basket


def _basket_edge(market, basket, seeds, timesteps, seed_base, log, tag):
    periods = ANNUALISATION.get(market, 252)
    tickers = list(basket)
    train_slices = [basket[t]["train"] for t in tickers]
    agent_by_asset = np.zeros(len(tickers))
    bh_by_asset = None
    for s in range(seeds):
        random.seed(seed_base + s)

        def factory():
            return random.choice(train_slices)

        cfg = None
        for agent, cfg in _train(market, factory, 1, timesteps, seed_base + s):
            pass  # single seed per loop so the RNG seeding above lines up
        a, b = [], []
        w = cfg.env.window_size
        for sp in basket.values():
            test = sp["test"]
            env = make_env(market, test, cfg.env, cfg.reward, random_start=False)
            a.append(backtest(agent, env, market=market).metrics["total_return"])
            prices = test.prices[w - 1:]
            b.append(compute_metrics(cfg.env.initial_balance * (prices / prices[0]), periods)["total_return"])
        a, b = np.array(a), np.array(b)
        bh_by_asset = b
        agent_by_asset += a
        log.info("[%s/%s] seed %d/%d | basket agent %+.1f%% vs B&H %+.1f%%",
                 market, tag, s + 1, seeds, 100 * a.mean(), 100 * b.mean())
    return agent_by_asset / seeds - bh_by_asset  # per-ticker edge over B&H


def run_real(market, seeds, timesteps, data_dir, log) -> dict:
    real = load_basket(data_dir, market, surrogate=False)
    if not real:
        raise SystemExit(f"No CSVs in {os.path.join(data_dir, market)} — run tools/fetch_data.py.")
    surr = load_basket(data_dir, market, surrogate=True, rng=np.random.default_rng(2_024))
    # Pair ticker-by-ticker: restrict both baskets to the same tickers, same order,
    # so the surrogate is a like-for-like counterfactual of each real series.
    common = sorted(set(real) & set(surr))
    real = {t: real[t] for t in common}
    surr = {t: surr[t] for t in common}
    edge_real = _basket_edge(market, real, seeds, timesteps, 300, log, "real")
    edge_surr = _basket_edge(market, surr, seeds, timesteps, 600, log, "surrogate")
    diff, p = paired_permutation_test(edge_real, edge_surr)
    return _summary(market, edge_real, edge_surr, diff, p)


def _summary(market, edge_struct, edge_surr, diff, p) -> dict:
    """Summarise one market's two arms.

    The per-arm values are recorded alongside the summary statistics, not just
    the means. Without them a published p-value can only be quoted, never
    recomputed — anyone re-reading the artifact (the interactive lab included)
    would have to take the number on trust or reconstruct it from a mean, which
    is not possible. ``n_pairs`` is stored explicitly for the same reason: it
    fixes the permutation test's resolution floor at ``2 / 2**n``.
    """
    cs, cu = bootstrap_ci(edge_struct), bootstrap_ci(edge_surr)
    return {
        "market": market,
        "edge_structured": float(np.mean(edge_struct)),
        "edge_surrogate": float(np.mean(edge_surr)),
        "structured_ci": [round(cs.mean, 4), round(cs.low, 4), round(cs.high, 4)],
        "surrogate_ci": [round(cu.mean, 4), round(cu.low, 4), round(cu.high, 4)],
        "diff": float(diff),
        "p": float(p),
        "n_pairs": int(len(edge_struct)),
        "values_structured": [float(v) for v in edge_struct],
        "values_surrogate": [float(v) for v in edge_surr],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--market", choices=["stock", "crypto", "both"], default="both")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=40_000)
    parser.add_argument("--eval-paths", type=int, default=12, help="held-out paths (synthetic mode)")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    args = parser.parse_args()

    log = get_logger("surrogate_test")
    markets = ["stock", "crypto"] if args.market == "both" else [args.market]

    print("\n" + "=" * 74)
    print(f"SURROGATE-DATA FALSIFICATION TEST  —  {args.mode.upper()} MODE  "
          f"({args.seeds} seeds, {args.timesteps:,} steps)")
    print("=" * 74)
    print("edge = agent return − buy-&-hold return  (buy-&-hold is IDENTICAL on a surrogate)")
    print("-" * 74)
    print(f"{'Market':<9}{'Edge (structured)':>20}{'Edge (surrogate)':>20}{'Δ':>10}{'p':>8}")
    print("-" * 74)

    results = {}
    for market in markets:
        r = run_synthetic(market, args.seeds, args.timesteps, args.eval_paths, log) \
            if args.mode == "synthetic" \
            else run_real(market, args.seeds, args.timesteps, args.data_dir, log)
        results[market] = r
        print(f"{market:<9}{r['edge_structured']:>+19.1%}{r['edge_surrogate']:>+19.1%}"
              f"{r['diff']:>+9.1%}{r['p']:>8.3f}")
    print("-" * 74)
    if args.mode == "synthetic":
        print("Expected (positive control): structured edge >> surrogate edge, small p —\n"
              "the agent exploits the known signal and loses it once shuffling removes it.")
    else:
        print("Reading it: if Δ is small and p is large, real markets offer the agent no\n"
              "more exploitable structure than pure noise — the null this project argues for.")
    print("=" * 74)

    out = os.path.join("docs", "assets", f"surrogate_{args.mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
