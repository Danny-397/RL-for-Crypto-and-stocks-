"""Render the figures embedded in README.md / RESULTS.md from real results.

Produces three PNGs in ``docs/assets/`` using a dark theme that matches the web
dashboard:

    fig_ablation.png   out-of-sample return, single-path vs domain-randomized
    fig_baselines.png   agent vs every baseline on the real test data
    fig_equity.png      representative held-out equity curve (agent vs buy&hold)

Run from the repo root (after a ``--real`` build + an ablation run):

    python tools/make_figures.py
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rl_trader.config.training_config import crypto_config, stock_config  # noqa: E402
from rl_trader.data.data_loader import load_ohlcv_csv, prepare_market_data  # noqa: E402
from rl_trader.envs import make_env  # noqa: E402
from rl_trader.evaluation.baselines import evaluate_baselines  # noqa: E402
from rl_trader.evaluation.evaluate_agent import backtest  # noqa: E402
from rl_trader.models.ppo_agent import PPOAgent  # noqa: E402

ASSETS = os.path.join("docs", "assets")
BG = "#0e141d"
FG = "#e7edf5"
VOLT = "#d4ff3f"
CYAN = "#36e0ff"
GREY = "#8c98a8"
RED = "#ff6b6b"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": "#2a323e", "grid.color": "#222a35",
    "font.size": 11, "axes.titleweight": "bold",
})


def _cfg(market):
    return stock_config() if market == "stock" else crypto_config()


def _basket(market, data_dir="data/raw"):
    out = {}
    for path in sorted(glob.glob(os.path.join(data_dir, market, "*.csv"))):
        ticker = os.path.splitext(os.path.basename(path))[0]
        splits = prepare_market_data(load_ohlcv_csv(path), market=market,
                                     train_frac=0.6, val_frac=0.0)
        if len(splits["test"]) > 60:
            out[ticker] = splits
    return out


# --------------------------------------------------------------------------- #
def fig_ablation():
    """Out-of-sample return: single-path vs domain-randomized (the money chart)."""
    with open(os.path.join(ASSETS, "ablation.json"), encoding="utf-8") as fh:
        data = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, market in zip(axes, ("stock", "crypto")):
        d = data[market]
        means = [d["single"]["oos_mean"] * 100, d["domain"]["oos_mean"] * 100]
        errs = [d["single"]["oos_std"] * 100, d["domain"]["oos_std"] * 100]
        ax.bar(["single-path", "domain-random"], means, yerr=errs,
               color=[RED, VOLT], capsize=6, width=0.6, edgecolor="none")
        ax.axhline(0, color="#3a434f", linewidth=1)
        ax.set_title(market.capitalize())
        ax.set_ylabel("Out-of-sample return (%)")
        ax.grid(axis="y", alpha=0.3)
        # Annotate the absurd in-sample numbers above the single-path bar.
        ax.annotate(f"in-sample:\n+{d['single']['in'] * 100:,.0f}%",
                    xy=(0, means[0]), xytext=(0, max(means) + 60),
                    ha="center", fontsize=9, color=GREY)
    fig.suptitle("Domain randomization fixes overfitting (out-of-sample return)",
                 fontsize=13, color=FG)
    fig.tight_layout()
    out = os.path.join(ASSETS, "fig_ablation.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def _agent_and_baselines(market):
    cfg = _cfg(market)
    agent = PPOAgent.from_checkpoint(os.path.join("checkpoints", f"ppo_{market}.pt"))
    basket = _basket(market)
    agent_runs, baseline_acc, equities = [], {}, []
    for ticker, splits in basket.items():
        test = splits["test"]
        env = make_env(market, test, cfg.env, cfg.reward, random_start=False)
        res = backtest(agent, env, market=market)
        agent_runs.append(res.metrics["total_return"])
        w = cfg.env.window_size
        prices = test.prices[w - 1:]
        bench = cfg.env.initial_balance * (prices / prices[0])
        equities.append((res.metrics["total_return"], res.equity_curve, bench, ticker))
        for name, m in evaluate_baselines(test, cfg.env, cfg.reward, market=market).items():
            baseline_acc.setdefault(name, []).append(m["total_return"])
    means = {"PPO agent": float(np.mean(agent_runs))}
    for name, vals in baseline_acc.items():
        means[name] = float(np.mean(vals))
    # Representative equity = ticker with median agent return.
    equities.sort(key=lambda e: e[0])
    rep = equities[len(equities) // 2]
    return means, rep


def fig_baselines_and_equity():
    """Agent-vs-baselines bars + a representative held-out equity curve."""
    order = ["PPO agent", "buy_and_hold", "ma_crossover", "random", "flat"]
    labels = {"PPO agent": "PPO agent", "buy_and_hold": "Buy & hold",
              "ma_crossover": "MA cross", "random": "Random", "flat": "Flat"}
    colors = {"PPO agent": CYAN, "buy_and_hold": VOLT, "ma_crossover": "#9be15d",
              "random": RED, "flat": GREY}

    bfig, baxes = plt.subplots(1, 2, figsize=(10, 4.2))
    efig, eaxes = plt.subplots(1, 2, figsize=(10, 4.0))

    for bax, eax, market in zip(baxes, eaxes, ("stock", "crypto")):
        means, rep = _agent_and_baselines(market)
        names = [n for n in order if n in means]
        vals = [means[n] * 100 for n in names]
        bax.bar([labels[n] for n in names], vals,
                color=[colors[n] for n in names], width=0.66, edgecolor="none")
        bax.axhline(0, color="#3a434f", linewidth=1)
        bax.set_title(market.capitalize())
        bax.set_ylabel("Mean return (%)")
        bax.grid(axis="y", alpha=0.3)
        bax.tick_params(axis="x", labelrotation=20)

        _, agent_eq, bench_eq, ticker = rep
        eax.plot(bench_eq / bench_eq[0], color=GREY, linewidth=1.8, label="Buy & hold")
        eax.plot(np.asarray(agent_eq) / agent_eq[0], color=VOLT, linewidth=2.0, label="PPO agent")
        eax.set_title(f"{market.capitalize()} — {ticker} (held-out)")
        eax.set_xlabel("Trading day")
        eax.set_ylabel("Growth of $1")
        eax.grid(alpha=0.3)
        eax.legend(facecolor=BG, edgecolor="#2a323e", labelcolor=FG, fontsize=9)

    bfig.suptitle("Real out-of-sample: agent vs. baselines (mean over basket)",
                  fontsize=13, color=FG)
    bfig.tight_layout()
    bout = os.path.join(ASSETS, "fig_baselines.png")
    bfig.savefig(bout, dpi=130)
    plt.close(bfig)
    print(f"wrote {bout}")

    efig.suptitle("Representative held-out equity curve", fontsize=13, color=FG)
    efig.tight_layout()
    eout = os.path.join(ASSETS, "fig_equity.png")
    efig.savefig(eout, dpi=130)
    plt.close(efig)
    print(f"wrote {eout}")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    fig_ablation()
    fig_baselines_and_equity()


if __name__ == "__main__":
    main()
