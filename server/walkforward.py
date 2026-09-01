"""Rolling walk-forward evaluation, made visible.

The project's leakage controls — chronological splits, scalers fit on training
rows only — are the least glamorous and most important thing in it, and until now
they were invisible: a claim in the README and a few lines in
:mod:`rl_trader.evaluation.walk_forward`. This runs the *real* splitter
(:func:`~rl_trader.evaluation.walk_forward.generate_folds`, imported, not
reimplemented) over a real price series and evaluates each out-of-sample block.

What this is, precisely
-----------------------
A textbook walk-forward **retrains** on each fold's training window. This backend
cannot train — no PyTorch, a fraction of a CPU — so it does the honest subset:

* the **fold geometry** is real, straight from the research code's splitter;
* the **scaling** is real, and is the point: each fold's feature scaler is fit on
  that fold's training rows only, exactly as training does it;
* the **policy is the single deployed one, unchanged across folds**.

So this measures *how much one fixed policy's out-of-sample result swings across
chronological blocks* — fold-to-fold variance, which is a real and under-reported
source of the "my backtest worked" illusion. It is **not** a retrained
walk-forward, and every response says so.

The leakage comparison
----------------------
Running the same folds a second time with the scaler fit on the *whole* series —
the standard mistake — gives a directly measured answer to "how much does that
shortcut change the number?" Both arms are real runs of the same policy over the
same prices; only the scaling differs. Nothing here is asserted; the difference
is whatever it comes out as, including when it is small.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from rl_trader.data.data_loader import (
    FEATURE_COLUMNS,
    MarketData,
    add_technical_indicators,
)
from rl_trader.envs import make_env
from rl_trader.evaluation.evaluate_agent import ANNUALISATION, compute_metrics
from rl_trader.evaluation.walk_forward import generate_folds

SCHEMES = ("expanding", "sliding")
SCALINGS = ("train_only", "full_sample")

SCHEME_NOTES = {
    "expanding": (
        "Anchored: each fold trains on everything before its test block, so the "
        "training window grows. More data per fold, but stale history is never "
        "dropped."
    ),
    "sliding": (
        "Fixed-size rolling window: each fold sees only the most recent stretch "
        "before its test block. Adapts to regime change, at the cost of data."
    ),
}

FIXED_POLICY_NOTE = (
    "This is not a retrained walk-forward. The fold geometry and the per-fold "
    "scaling are real, but the same deployed policy is evaluated on every fold — "
    "this backend cannot train. What it measures is how far one fixed policy's "
    "out-of-sample result swings from block to block. Retraining per fold is "
    "rl_trader.evaluation.walk_forward.rolling_walk_forward, run offline."
)

LEAKAGE_NOTE = (
    "Both arms are real runs of the same policy over the same prices. They differ "
    "only in how the feature scaler was fit: on each fold's training rows (correct) "
    "or on the whole series including the test block (the standard mistake). The "
    "gap between them is the measured cost of that shortcut on this series."
)


def _fold_rows(fold, n_rows: int) -> dict:
    tr, te = fold.train, fold.test
    return {
        "fold": int(fold.index),
        "train_start": int(tr.start or 0),
        "train_end": int(tr.stop),
        "test_start": int(te.start),
        "test_end": int(te.stop if te.stop is not None else n_rows),
    }


def fold_plan(
    n_rows: int, n_folds: int = 4, scheme: str = "expanding",
    train_min_frac: float = 0.4, dates: Optional[Sequence[str]] = None,
) -> List[dict]:
    """The fold geometry alone, with no evaluation — the splitting story.

    Uses the research code's own splitter, so the picture the site draws cannot
    drift from the folds the offline tooling actually produces.
    """
    plan = []
    for fold in generate_folds(n_rows, n_folds, train_min_frac, scheme):
        row = _fold_rows(fold, n_rows)
        row["train_bars"] = row["train_end"] - row["train_start"]
        row["test_bars"] = row["test_end"] - row["test_start"]
        if dates is not None and len(dates) >= n_rows:
            row["train_from"] = dates[row["train_start"]]
            row["train_to"] = dates[max(row["train_start"], row["train_end"] - 1)]
            row["test_from"] = dates[row["test_start"]]
            row["test_to"] = dates[min(len(dates) - 1, row["test_end"] - 1)]
        # The property that makes the whole thing an out-of-sample test at all.
        row["disjoint"] = row["train_end"] <= row["test_start"]
        plan.append(row)
    return plan


def _run_policy(policy, env, market: str, initial_balance: float) -> dict:
    """One deterministic pass of the fixed policy over a test block."""
    obs, info = env.reset()
    equity = [info["equity"]]
    actions: List[float] = []
    done = False
    while not done:
        a = float(policy.act(obs))
        obs, _r, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        equity.append(info["equity"])
        actions.append(a)
        done = term or trunc
    periods = ANNUALISATION.get(market, 252)
    metrics = compute_metrics(np.asarray(equity, dtype=float), periods)
    return {
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "mean_position": round(float(np.mean(actions)) if actions else 0.0, 6),
        "bars": len(actions),
    }


def evaluate_folds(
    policy,
    df: pd.DataFrame,
    market: str,
    cfg_obj,
    n_folds: int = 4,
    scheme: str = "expanding",
    train_min_frac: float = 0.4,
    scaling: str = "train_only",
    dates: Optional[Sequence[str]] = None,
    progress=None,
) -> List[dict]:
    """Score the fixed policy on each fold's out-of-sample block.

    ``scaling`` selects where the feature scaler is fit:
    ``"train_only"`` uses that fold's training rows (correct); ``"full_sample"``
    uses the entire series, test block included — reproducing the leakage this
    project's pipeline is built to avoid, so its cost can be measured rather than
    asserted.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    if scaling not in SCALINGS:
        raise ValueError(f"unknown scaling {scaling!r}; expected one of {SCALINGS}")

    featured = add_technical_indicators(df)
    feat_all = featured[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    prices_all = featured["close"].to_numpy(dtype=np.float32)
    n_rows = len(prices_all)
    periods = ANNUALISATION.get(market, 252)

    if scaling == "full_sample":
        full_mean = feat_all.mean(axis=0)
        full_std = feat_all.std(axis=0)
        full_std[full_std < 1e-8] = 1.0

    rows: List[dict] = []
    for i, fold in enumerate(generate_folds(n_rows, n_folds, train_min_frac, scheme)):
        tr, te = fold.train, fold.test
        if scaling == "train_only":
            mean = feat_all[tr].mean(axis=0)
            std = feat_all[tr].std(axis=0)
            std[std < 1e-8] = 1.0
        else:
            mean, std = full_mean, full_std

        test_feat = ((feat_all[te] - mean) / std).astype(np.float32)
        test_prices = prices_all[te]
        if len(test_prices) <= cfg_obj.env.window_size + 1:
            continue  # too short to step through; skipped rather than faked

        data = MarketData(test_feat, test_prices, FEATURE_COLUMNS)
        env = make_env(market, data, cfg_obj.env, cfg_obj.reward, random_start=False)
        out = _run_policy(policy, env, market, cfg_obj.env.initial_balance)

        w = cfg_obj.env.window_size
        bench_prices = test_prices[w - 1:]
        bench = cfg_obj.env.initial_balance * (bench_prices / bench_prices[0])
        bench_metrics = compute_metrics(np.asarray(bench, dtype=float), periods)

        row = _fold_rows(fold, n_rows)
        row.update({
            "agent_return": out["metrics"]["total_return"],
            "agent_sharpe": out["metrics"]["sharpe"],
            "agent_max_drawdown": out["metrics"]["max_drawdown"],
            "benchmark_return": round(float(bench_metrics["total_return"]), 6),
            "benchmark_sharpe": round(float(bench_metrics["sharpe"]), 6),
            "excess_return": round(
                out["metrics"]["total_return"] - float(bench_metrics["total_return"]), 6
            ),
            "mean_position": out["mean_position"],
            "test_bars": out["bars"],
            "train_bars": row["train_end"] - row["train_start"],
        })
        if dates is not None and len(dates) >= n_rows:
            row["test_from"] = dates[row["test_start"]]
            row["test_to"] = dates[min(len(dates) - 1, row["test_end"] - 1)]
        rows.append(row)
        if progress:
            progress(i + 1)
    return rows


def summarise(rows: Sequence[dict]) -> dict:
    """Aggregate across folds, with the resolution limit stated up front.

    A handful of folds is a handful of observations. The sign test over ``n``
    folds cannot go below ``2 / 2**n``, the same floor the seed-level tests hit,
    so it is reported beside the spread rather than left for a reader to assume
    that four folds settle anything.
    """
    if not rows:
        return {"n_folds": 0}
    excess = np.array([r["excess_return"] for r in rows], dtype=float)
    agent = np.array([r["agent_return"] for r in rows], dtype=float)
    bench = np.array([r["benchmark_return"] for r in rows], dtype=float)
    n = len(excess)
    sd = float(excess.std(ddof=1)) if n > 1 else 0.0
    floor = (2.0 / (2 ** n)) if 0 < n <= 30 else 0.0
    return {
        "n_folds": n,
        "mean_agent_return": round(float(agent.mean()), 6),
        "mean_benchmark_return": round(float(bench.mean()), 6),
        "mean_excess_return": round(float(excess.mean()), 6),
        "std_excess_return": round(sd, 6),
        "worst_fold_excess": round(float(excess.min()), 6),
        "best_fold_excess": round(float(excess.max()), 6),
        "folds_beaten": int((excess > 0).sum()),
        "sign_test_floor": round(floor, 12),
        "sign_test_can_reach_05": bool(floor < 0.05),
        "spread_note": (
            f"The agent beat buy-and-hold on {int((excess > 0).sum())} of {n} folds, "
            f"with per-fold excess ranging from {excess.min():+.1%} to "
            f"{excess.max():+.1%}. A sign test over {n} folds cannot produce a "
            f"p-value below {floor:.4f}, so this design "
            + ("can" if floor < 0.05 else "cannot")
            + " reach significance at 0.05 whatever the effect size."
        ),
    }


def leakage_delta(correct: Sequence[dict], leaked: Sequence[dict]) -> dict:
    """Fold-by-fold difference between train-only and full-sample scaling."""
    pairs = []
    by_fold = {r["fold"]: r for r in leaked}
    for row in correct:
        other = by_fold.get(row["fold"])
        if other is None:
            continue
        pairs.append({
            "fold": row["fold"],
            "train_only_return": row["agent_return"],
            "full_sample_return": other["agent_return"],
            "delta": round(other["agent_return"] - row["agent_return"], 6),
        })
    deltas = np.array([p["delta"] for p in pairs], dtype=float)
    return {
        "per_fold": pairs,
        "mean_delta": round(float(deltas.mean()), 6) if len(deltas) else 0.0,
        "max_abs_delta": round(float(np.abs(deltas).max()), 6) if len(deltas) else 0.0,
        "identical": bool(len(deltas) and np.all(np.abs(deltas) < 1e-9)),
        "note": LEAKAGE_NOTE,
    }


def describe() -> Dict[str, Any]:
    """Static description of the available knobs, for the frontend's selector."""
    return {
        "schemes": [{"key": k, "description": v} for k, v in SCHEME_NOTES.items()],
        "scalings": [
            {"key": "train_only",
             "label": "Fit the scaler on training rows only (correct)"},
            {"key": "full_sample",
             "label": "Fit on the whole series, test block included (leaks)"},
        ],
        "fixed_policy_note": FIXED_POLICY_NOTE,
    }
