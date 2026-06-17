"""Plotting helpers for backtests and training diagnostics.

All functions accept an optional ``save_path``; when provided the figure is
written to disk (handy for headless servers and for embedding results in the
README). Matplotlib is imported lazily so the core training/eval code has no
hard plotting dependency at import time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def plot_equity_curve(equity_curve: np.ndarray, title: str = "Equity Curve",
                      save_path: Optional[str] = None):
    """Plot portfolio value over time."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_curve, color="#1f77b4", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def plot_drawdown(equity_curve: np.ndarray, title: str = "Drawdown",
                 save_path: Optional[str] = None):
    """Plot the drawdown curve (depth below the running high-water mark)."""
    import matplotlib.pyplot as plt

    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = (running_peak - equity_curve) / running_peak

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(drawdown)), -drawdown * 100, 0, color="#d62728", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def plot_action_distribution(actions: np.ndarray, title: str = "Action Distribution",
                            save_path: Optional[str] = None):
    """Histogram of target position fractions chosen by the agent."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(actions, bins=40, color="#2ca02c", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Target position (fraction of equity)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig


def plot_training_curve(history: dict, key: str = "mean_episode_return",
                       save_path: Optional[str] = None):
    """Plot a metric from the training ``history`` dict against update index."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["update"], history[key], color="#9467bd", linewidth=1.5)
    ax.set_title(f"Training: {key}")
    ax.set_xlabel("PPO update")
    ax.set_ylabel(key)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    return fig
