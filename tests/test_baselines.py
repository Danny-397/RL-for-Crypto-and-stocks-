"""Tests for the non-learned baseline strategies."""

import numpy as np

from rl_trader.config.training_config import stock_config
from rl_trader.data.data_loader import prepare_market_data
from rl_trader.evaluation.baselines import evaluate_baselines


def test_baselines_run_and_report_expected_keys():
    cfg = stock_config()
    splits = prepare_market_data(None, market="stock", synthetic_steps=900, seed=0)
    results = evaluate_baselines(splits["test"], cfg.env, cfg.reward, market="stock")

    assert set(results) == {"buy_and_hold", "flat", "random", "ma_crossover"}
    for metrics in results.values():
        for key in ("total_return", "sharpe", "max_drawdown", "final_equity"):
            assert key in metrics
            assert np.isfinite(metrics[key])


def test_flat_baseline_preserves_capital():
    """Never trading should leave equity exactly at the starting balance."""
    cfg = stock_config()
    splits = prepare_market_data(None, market="stock", synthetic_steps=900, seed=1)
    results = evaluate_baselines(splits["test"], cfg.env, cfg.reward, market="stock")

    # Flat = all cash, no trades, no costs -> ~0% return and ~0 drawdown.
    assert abs(results["flat"]["total_return"]) < 1e-6
    assert results["flat"]["max_drawdown"] < 1e-6
