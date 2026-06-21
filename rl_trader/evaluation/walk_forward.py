"""Rolling (multi-fold) walk-forward evaluation.

A single 60/40 train/test split answers one question on one slice of history —
and that slice might just have been kind (or cruel) to the strategy. Rolling
walk-forward re-asks the question across several chronological folds: train on
everything up to a cut-off, test on the block that follows, advance, repeat.
Aggregating across folds turns a point estimate into a *distribution* you can
put a confidence interval around.

:func:`generate_folds` is a pure function (no training) so the splitting logic
is unit-testable in isolation; :func:`rolling_walk_forward` trains + scores an
agent on each fold and is exercised by the offline tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Fold:
    """One walk-forward fold: contiguous train then (disjoint, later) test rows."""

    index: int
    train: slice
    test: slice


def generate_folds(
    n_rows: int,
    n_folds: int = 4,
    train_min_frac: float = 0.4,
    scheme: str = "expanding",
) -> List[Fold]:
    """Split ``n_rows`` chronological rows into ``n_folds`` walk-forward folds.

    The first ``train_min_frac`` of the data seeds the initial training window;
    the remainder is divided into ``n_folds`` equal, *out-of-sample* test blocks.
    For each fold the test block is everything after the current cut-off, up to
    the next one.

    Parameters
    ----------
    scheme:
        ``"expanding"`` — train on *all* rows before the test block (anchored
        walk-forward; more data each fold). ``"sliding"`` — train only on the
        most recent ``train_min_frac`` window (fixed-size rolling window; adapts
        to regime change, ignores stale history).
    """
    if not 0.0 < train_min_frac < 1.0:
        raise ValueError("train_min_frac must be in (0, 1)")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    first_train_end = int(n_rows * train_min_frac)
    test_total = n_rows - first_train_end
    if test_total < n_folds:
        raise ValueError("not enough rows for the requested number of folds")

    block = test_total // n_folds
    train_window = first_train_end  # fixed window size for the sliding scheme
    folds: List[Fold] = []
    for i in range(n_folds):
        test_start = first_train_end + i * block
        test_end = n_rows if i == n_folds - 1 else test_start + block
        train_start = 0 if scheme == "expanding" else max(0, test_start - train_window)
        folds.append(Fold(i, slice(train_start, test_start), slice(test_start, test_end)))
    return folds


def rolling_walk_forward(
    df: pd.DataFrame,
    config,
    n_folds: int = 4,
    scheme: str = "expanding",
    timesteps: int = 60_000,
) -> List[Dict[str, float]]:
    """Train + score an agent on each walk-forward fold of a real OHLCV frame.

    Returns one dict per fold with the agent's and buy-&-hold's test-set total
    return and Sharpe — ready to feed into :mod:`rl_trader.evaluation.statistics`
    for a confidence interval across folds.
    """
    # Imported lazily: this pulls in torch, which the pure splitter above must
    # not require.
    from ..data.data_loader import add_technical_indicators, FEATURE_COLUMNS, MarketData
    from ..envs import make_env
    from ..evaluation.evaluate_agent import ANNUALISATION, backtest, compute_metrics
    from ..training.utils import run_ppo_training

    featured = add_technical_indicators(df)
    feat_all = featured[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    prices_all = featured["close"].to_numpy(dtype=np.float32)
    periods = ANNUALISATION.get(config.market, 252)

    results: List[Dict[str, float]] = []
    for fold in generate_folds(len(prices_all), n_folds, scheme=scheme):
        tr, te = fold.train, fold.test
        # Scale on the fold's training rows only — no leakage from the test block.
        mean = feat_all[tr].mean(axis=0)
        std = feat_all[tr].std(axis=0)
        std[std < 1e-8] = 1.0

        def scale(a: np.ndarray) -> np.ndarray:
            return ((a - mean) / std).astype(np.float32)

        train_df = featured.iloc[tr].reset_index(drop=True)
        agent, _ = run_ppo_training(
            _with_timesteps(config, timesteps), df=_ohlcv(train_df)
        )
        test_data = MarketData(scale(feat_all[te]), prices_all[te], FEATURE_COLUMNS)
        env = make_env(config.market, test_data, config.env, config.reward, random_start=False)
        bt = backtest(agent, env, market=config.market)
        bh = compute_metrics(prices_all[te] / prices_all[te][0] * config.env.initial_balance, periods)
        results.append(
            {
                "fold": fold.index,
                "agent_return": bt.metrics["total_return"],
                "agent_sharpe": bt.metrics["sharpe"],
                "bh_return": bh["total_return"],
                "bh_sharpe": bh["sharpe"],
            }
        )
    return results


def _ohlcv(featured_df: pd.DataFrame) -> pd.DataFrame:
    """Strip engineered columns back to raw OHLCV for the training pipeline."""
    return featured_df[["open", "high", "low", "close", "volume"]].copy()


def _with_timesteps(config, timesteps: int):
    """Shallow-copy a config with an overridden training budget."""
    import copy

    cfg = copy.deepcopy(config)
    cfg.train.total_timesteps = timesteps
    cfg.train.eval_interval = 0
    return cfg
