"""Backtesting metrics and plotting."""

from .evaluate_agent import BacktestResult, backtest, compute_metrics

__all__ = ["BacktestResult", "backtest", "compute_metrics"]
