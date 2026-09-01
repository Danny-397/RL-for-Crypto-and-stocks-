"""Does a plain supervised model find what the RL agent could not?

The rule-based baselines (buy-and-hold, moving-average crossover, random, flat)
establish that the agent is not useless. They cannot answer the objection a
skeptical reader actually has: perhaps the market does contain exploitable
structure and PPO is simply the wrong tool for extracting it.

This fits two ordinary supervised models — ridge regression on the next bar's
return, and logistic regression on its direction — over the same training split,
using the same 28 features, and trades them through the same environment with
the same costs and slippage. Then it puts all of it beside the trained agent on
the same held-out tickers.

Both possible outcomes are worth publishing:

* Supervised also fails -> two unrelated method classes agree, and the "no
  exploitable structure" reading is much harder to dispute.
* Supervised wins -> the negative result was about the algorithm, not the
  market, which is a more interesting finding and changes the conclusion.

In-sample directional accuracy is reported alongside, because it separates a
model that cannot fit its own training data from one that fits it well and still
earns nothing out of sample. Only the second says anything about markets.

Run from the repo root::

    python tools/supervised_report.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_trader.config.training_config import crypto_config, stock_config  # noqa: E402
from rl_trader.envs import make_env  # noqa: E402
from rl_trader.evaluation.baselines import evaluate_baselines  # noqa: E402
from rl_trader.evaluation.evaluate_agent import ANNUALISATION, backtest  # noqa: E402
from rl_trader.evaluation.supervised import fit_logistic, train_accuracy  # noqa: E402
from rl_trader.models.ppo_agent import PPOAgent  # noqa: E402
from tools.real_significance import load_real_basket  # noqa: E402

OUT = os.path.join("docs", "assets", "supervised.json")
CFG = {"stock": stock_config, "crypto": crypto_config}


def run_market(market: str, data_dir: str) -> dict:
    cfg = CFG[market]()
    periods = ANNUALISATION.get(market, 252)
    basket = load_real_basket(data_dir, market)

    agent = PPOAgent.from_checkpoint(
        os.path.join("checkpoints", f"ppo_{market}.pt"))

    per_strategy: dict = {}
    agent_returns, accuracies = [], []

    for ticker, splits in sorted(basket.items()):
        train, test = splits["train"], splits["test"]

        env = make_env(market, test, cfg.env, cfg.reward, random_start=False)
        agent_returns.append(backtest(agent, env, market=market)
                             .metrics["total_return"])

        results = evaluate_baselines(test, cfg.env, cfg.reward, market=market,
                                     train_data=train)
        for name, metrics in results.items():
            per_strategy.setdefault(name, []).append(metrics)

        weights = fit_logistic(train.features, train.prices)
        accuracies.append(train_accuracy(train.features, train.prices, weights))

    keys = ("total_return", "sharpe", "max_drawdown")
    strategies = {
        name: {k: float(np.mean([m[k] for m in rows])) for k in keys}
        for name, rows in per_strategy.items()
    }
    strategies["PPO agent"] = {
        "total_return": float(np.mean(agent_returns)),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
    }

    bh = strategies["buy_and_hold"]["total_return"]
    beats_agent = [
        name for name, m in strategies.items()
        if name not in ("PPO agent",)
        and m["total_return"] > strategies["PPO agent"]["total_return"]
    ]
    return {
        "n_tickers": len(basket),
        "buy_and_hold": bh,
        "strategies": strategies,
        "supervised_beats_agent": sorted(
            n for n in beats_agent
            if n in ("ridge_forecast", "logistic_direction")),
        "any_beats_buy_and_hold": sorted(
            name for name, m in strategies.items()
            if m["total_return"] > bh and name != "buy_and_hold"),
        "logistic_train_accuracy": float(np.mean(accuracies)),
        "periods_per_year": periods,
    }


def verdict(markets: dict) -> str:
    """State what the comparison licenses, and nothing more."""
    sup = {"ridge_forecast", "logistic_direction"}
    beat_bh = {m for m, b in markets.items()
               if sup & set(b["any_beats_buy_and_hold"])}
    beat_agent = {m for m, b in markets.items() if b["supervised_beats_agent"]}

    if not beat_bh:
        base = ("Neither supervised model beat buy-and-hold in either market. "
                "Two unrelated method classes — a policy-gradient RL agent and "
                "ordinary linear regression on the same inputs — reach the same "
                "place, which is what one would expect if the features carry no "
                "exploitable structure rather than if PPO were simply the wrong "
                "tool.")
    else:
        base = ("A supervised model beat buy-and-hold in "
                f"{', '.join(sorted(beat_bh))}. That is evidence the features do "
                "carry extractable structure and the RL agent failed to use it — "
                "a result about the algorithm, not about the market.")

    if beat_agent:
        names = ", ".join(sorted(beat_agent))
        base += (f" On {names} it also out-performed the trained agent, so the "
                 "agent is not even the best use of its own inputs there.")

    accuracy = float(np.mean([b["logistic_train_accuracy"]
                              for b in markets.values()]))
    base += f" In-sample directional accuracy averaged {accuracy:.1%}."
    if accuracy < 0.52:
        # Nothing was learned at all, so nothing about the market follows.
        base += (" That is close enough to a coin flip that the models did not "
                 "fit even their training data, so their out-of-sample failure "
                 "says nothing about the market on its own.")
    else:
        # The informative case: it fitted, and the fit did not travel.
        base += (" So the models did find in-sample structure — better than a "
                 "coin flip on the direction — and none of it survived into the "
                 "held-out period. That is the same pattern as the "
                 "domain-randomization ablation, reached by a completely "
                 "different method: what is learnable here is the training "
                 "sample, not the market.")
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markets", nargs="+", default=["stock", "crypto"])
    ap.add_argument("--data-dir", default="data/raw")
    args = ap.parse_args()

    markets = {}
    for market in args.markets:
        markets[market] = run_market(market, args.data_dir)
        block = markets[market]
        print(f"\n=== {market}  ({block['n_tickers']} held-out tickers) ===")
        for name, m in sorted(block["strategies"].items(),
                              key=lambda kv: -kv[1]["total_return"]):
            print(f"  {name:22} {m['total_return']:+8.2%}")
        print(f"  logistic in-sample accuracy: "
              f"{block['logistic_train_accuracy']:.1%}")

    payload = {
        "generated": datetime.date.today().isoformat(),
        "question": ("Would an ordinary supervised model have found structure "
                     "the RL agent missed?"),
        "method": ("Ridge regression on the next bar's log return and logistic "
                   "regression on its direction, both fit on the training split "
                   "only, using the same 28 features the agent observes, traded "
                   "through the same environment with the same costs."),
        "markets": markets,
        "verdict": verdict(markets),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n{payload['verdict']}\n\nwrote {OUT}")


if __name__ == "__main__":
    main()
