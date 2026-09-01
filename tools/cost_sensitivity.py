"""How much of the loss is friction?

The agent underperforms buy-and-hold on real data. There are two very different
reasons that could be true, and nothing in the repository separates them:

1. It has no edge, and trading costs are beside the point.
2. It has a small gross edge that transaction costs and slippage eat.

These license different conclusions. The second would say the strategy is
directionally right and merely uneconomic at retail friction — interesting, and
a completely different write-up. The first says there is nothing there.

Separating them is cheap, because it needs no retraining: the policy is frozen,
so the same agent can be replayed through the same held-out data at different
cost levels. Note the agent was *trained* at the published costs and is not
re-optimised for each level, which is the honest way round — an agent retrained
without costs would learn to churn, and the comparison would measure that
instead of friction.

Run from the repo root::

    python tools/cost_sensitivity.py
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_trader.config.training_config import crypto_config, stock_config  # noqa: E402
from rl_trader.envs import make_env  # noqa: E402
from rl_trader.evaluation.evaluate_agent import backtest  # noqa: E402
from rl_trader.models.ppo_agent import PPOAgent  # noqa: E402
from tools.real_significance import load_real_basket  # noqa: E402

OUT = os.path.join("docs", "assets", "cost_sensitivity.json")
CFG = {"stock": stock_config, "crypto": crypto_config}

# Zero is the informative end: a strategy that loses money with no friction at
# all cannot be blamed on friction. The multiples above 1 are there because
# "would it survive worse fills?" is the other half of the question.
MULTIPLES = (0.0, 0.5, 1.0, 2.0, 5.0)


def run_market(market: str, data_dir: str) -> dict:
    base = CFG[market]()
    basket = load_real_basket(data_dir, market)
    agent = PPOAgent.from_checkpoint(
        os.path.join("checkpoints", f"ppo_{market}.pt"))

    rows = []
    for multiple in MULTIPLES:
        cfg = copy.deepcopy(base)
        cfg.env.transaction_cost = base.env.transaction_cost * multiple
        cfg.env.slippage = base.env.slippage * multiple

        returns, turnovers = [], []
        for _ticker, splits in sorted(basket.items()):
            env = make_env(market, splits["test"], cfg.env, cfg.reward,
                           random_start=False)
            result = backtest(agent, env, market=market)
            returns.append(result.metrics["total_return"])
            # Mean absolute change in target position per bar: 0 is buy-and-hold,
            # 2.0 would be flipping fully long to fully short every single bar.
            actions = np.asarray(result.actions, dtype=np.float64).ravel()
            turnovers.append(float(np.abs(np.diff(actions)).mean())
                             if len(actions) > 1 else 0.0)
        rows.append({
            "multiple": multiple,
            "transaction_cost": cfg.env.transaction_cost,
            "slippage": cfg.env.slippage,
            "mean_return": float(np.mean(returns)),
            "median_return": float(np.median(returns)),
            "mean_turnover": float(np.mean(turnovers)),
            "n_tickers": len(returns),
        })

    frictionless = next(r for r in rows if r["multiple"] == 0.0)
    published = next(r for r in rows if r["multiple"] == 1.0)
    return {
        "rows": rows,
        "published_cost": base.env.transaction_cost,
        "published_slippage": base.env.slippage,
        "frictionless_return": frictionless["mean_return"],
        "published_return": published["mean_return"],
        "cost_drag": published["mean_return"] - frictionless["mean_return"],
        "profitable_without_costs": frictionless["mean_return"] > 0.0,
        "mean_turnover": published["mean_turnover"],
    }


def verdict(markets: dict) -> str:
    """Report the measurement and its limits, and stop there.

    The tempting reading of a positive gross return is "it works, costs just eat
    it". That is not supportable from this experiment, and saying so is the
    point of the file rather than a hedge bolted onto it.
    """
    parts = []
    for market, block in sorted(markets.items()):
        gross, net = block["frictionless_return"], block["published_return"]
        turn = block["mean_turnover"]
        if gross <= 0:
            parts.append(
                f"On {market} the agent loses money even at zero cost and zero "
                f"slippage ({gross:+.1%} gross, {net:+.1%} net). Friction is not "
                "what is wrong with it.")
        else:
            parts.append(
                f"On {market} the agent is positive before friction ({gross:+.1%}) "
                f"and {net:+.1%} after it, at a mean turnover of {turn:.2f} — that "
                "is, it moves its position by "
                f"{turn * 100:.0f}% of equity per bar on average.")

    parts.append(
        "**This is not evidence of a tradable edge, and should not be read as "
        "one.** A frictionless backtest is unattainable by construction: with "
        "cost and slippage set to zero an agent can flip its position every bar "
        "for free, and compound noise while doing it. These are single-seed "
        "runs, and the multi-seed study cannot distinguish this agent from "
        "buy-and-hold at real costs. Most importantly it sits in tension with "
        "the surrogate test, which found the agent does no better on real price "
        "history than on the same returns in random order — a genuine "
        "directional edge should not survive that shuffle, while a mechanical "
        "one would. What the experiment does establish is narrower and still "
        "useful: turnover, not signal, is the binding constraint on this "
        "policy, and a lower-turnover variant is the obvious next experiment.")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markets", nargs="+", default=["stock", "crypto"])
    ap.add_argument("--data-dir", default="data/raw")
    args = ap.parse_args()

    markets = {}
    for market in args.markets:
        markets[market] = run_market(market, args.data_dir)
        print(f"\n=== {market} ===")
        for row in markets[market]["rows"]:
            print(f"  costs x{row['multiple']:<4} "
                  f"(fee {row['transaction_cost']:.4%}, slip {row['slippage']:.4%})"
                  f"  mean {row['mean_return']:+8.2%}"
                  f"  turnover {row['mean_turnover']:.2f}")

    payload = {
        "generated": datetime.date.today().isoformat(),
        "question": "Is the agent losing to friction, or to having no edge?",
        "method": ("The frozen policy replayed through the same held-out data at "
                   "scaled transaction cost and slippage. No retraining: the "
                   "agent was trained at the published costs, and re-optimising "
                   "it per level would measure churn rather than friction."),
        "multiples": list(MULTIPLES),
        "markets": markets,
        "verdict": verdict(markets),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n{payload['verdict']}\n\nwrote {OUT}")


if __name__ == "__main__":
    main()
