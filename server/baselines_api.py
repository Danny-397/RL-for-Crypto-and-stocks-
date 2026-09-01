"""Score the trained agent against the naive strategies, live.

Beating buy-and-hold is the bar this project talks about most, but it is not the
first bar a skeptical reader has in mind. The first one is: **does it beat a coin
flip?** Until now the answer existed only in ``RESULTS.md``. This computes it on
whatever series the visitor just ran, through the same environment.

The strategies come from :func:`rl_trader.evaluation.baselines.evaluate_baselines`
— imported, not reimplemented — so the site cannot drift from the research code.

Two details that decide whether the comparison means anything
-------------------------------------------------------------
**A single random run is not a baseline.** One draw from a random policy is one
sample of a very wide distribution; quoting it would be exactly the single-run
mistake this project exists to criticise. So the random arm is run over many
seeds and reported as a mean with its full observed range, and the agent is only
described as beating it when it clears that whole range.

**There are two buy-and-holds and they are not the same number.** The equity
benchmark drawn on every chart here is cost-free — it is the price series scaled
to the starting balance. The buy-and-hold *strategy* below goes through the
environment and pays the entry cost like everything else. Both are reported,
labelled, because silently swapping one for the other would flatter or penalise
the agent by a few basis points for no stated reason.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from rl_trader.evaluation.baselines import evaluate_baselines

# How many independent draws stand behind the random arm. Cheap — each is one
# NumPy pass over a few hundred bars — and enough to show a range rather than a
# point.
RANDOM_SEEDS = 24

LABELS = {
    "agent": "The trained agent",
    "buy_and_hold": "Buy & hold (through the cost model)",
    "ma_crossover": "10/30 moving-average crossover",
    "flat": "Never trade (hold cash)",
    "random": "Random positions",
}

DESCRIPTIONS = {
    "buy_and_hold": "Target position 1.0 at every bar, paying the entry cost once.",
    "ma_crossover": "Fully long while the 10-bar average is above the 30-bar, else flat.",
    "flat": "The do-nothing control: it cannot lose money, and it cannot make any.",
    "random": f"A uniformly random target position every bar, over {RANDOM_SEEDS} seeds.",
}


def _row(key: str, metrics: Dict[str, float], **extra) -> dict:
    row = {
        "key": key,
        "label": LABELS.get(key, key),
        "description": DESCRIPTIONS.get(key),
        "total_return": round(float(metrics["total_return"]), 6),
        "sharpe": round(float(metrics["sharpe"]), 6),
        "max_drawdown": round(float(metrics["max_drawdown"]), 6),
    }
    row.update(extra)
    return row


def compare(
    data,
    env_config,
    reward_config,
    market: str,
    agent_metrics: Dict[str, float],
    cost_free_benchmark: Optional[Dict[str, float]] = None,
    n_random: int = RANDOM_SEEDS,
) -> dict:
    """Run every baseline on the agent's own series and rank them together."""
    deterministic = evaluate_baselines(data, env_config, reward_config, market, seed=0)

    # The random arm again, many times, so it is a distribution and not an anecdote.
    draws: List[Dict[str, float]] = []
    for seed in range(max(1, int(n_random))):
        draws.append(
            evaluate_baselines(data, env_config, reward_config, market, seed=seed)["random"]
        )
    returns = np.array([d["total_return"] for d in draws], dtype=float)
    random_row = _row(
        "random",
        {
            "total_return": float(returns.mean()),
            "sharpe": float(np.mean([d["sharpe"] for d in draws])),
            "max_drawdown": float(np.mean([d["max_drawdown"] for d in draws])),
        },
        n_seeds=len(draws),
        worst=round(float(returns.min()), 6),
        best=round(float(returns.max()), 6),
    )

    rows = [_row("agent", agent_metrics)]
    for key in ("buy_and_hold", "ma_crossover", "flat"):
        if key in deterministic:
            rows.append(_row(key, deterministic[key]))
    rows.append(random_row)

    agent_return = rows[0]["total_return"]
    ranked = sorted(rows, key=lambda r: -r["total_return"])
    beaten = [r["label"] for r in rows[1:] if agent_return > r["total_return"]]
    # "Beat random" has to mean beating the whole observed range, not its mean:
    # clearing the average of a wide distribution is not evidence of skill.
    clears_random_range = bool(agent_return > random_row["best"])

    out: Dict[str, Any] = {
        "rows": rows,
        "ranked": [r["key"] for r in ranked],
        "agent_rank": ranked.index(rows[0]) + 1,
        "n_strategies": len(rows),
        "beaten": beaten,
        "beats_random_mean": bool(agent_return > random_row["total_return"]),
        "beats_every_random_draw": clears_random_range,
        "random_range": [random_row["worst"], random_row["best"]],
        "live_computation": True,
        "note": (
            "Every strategy here runs through the same environment on the same "
            "bars, paying the same transaction cost and slippage."
        ),
        "random_note": (
            f"The random arm is {random_row['n_seeds']} independent draws, reported "
            f"as a mean with its full observed range "
            f"({random_row['worst']:+.1%} to {random_row['best']:+.1%}). One random "
            "run would be a single sample of a very wide distribution — the same "
            "mistake this project's seed testing exists to catch."
        ),
        "verdict": _verdict(agent_return, rows, random_row, clears_random_range),
    }
    if cost_free_benchmark is not None:
        out["cost_free_benchmark"] = {
            "label": "Buy & hold (cost-free reference)",
            "total_return": round(float(cost_free_benchmark["total_return"]), 6),
            "note": (
                "This is the benchmark line drawn on the charts: the price series "
                "scaled to the starting balance, paying no costs. The strategy in "
                "the table above pays the entry cost, so the two differ slightly."
            ),
        }
    return out


def _verdict(agent_return: float, rows: List[dict], random_row: dict, clears: bool) -> str:
    """Rank the agent honestly, leading with the least flattering true statement."""
    losses = [r["label"] for r in rows[1:] if agent_return <= r["total_return"]]
    parts = [f"The agent returned {agent_return:+.1%} on this series."]
    if not clears:
        parts.append(
            f"It did not clear every random draw (the best of "
            f"{random_row['n_seeds']} random runs made {random_row['best']:+.1%}), so "
            "on this one path it is not distinguishable from luck."
        )
    else:
        parts.append(
            f"It beat all {random_row['n_seeds']} random draws, the best of which "
            f"made {random_row['best']:+.1%}."
        )
    if losses:
        parts.append("It was beaten by: " + ", ".join(losses) + ".")
    else:
        parts.append("It came top of every baseline here — on one path.")
    return " ".join(parts)
