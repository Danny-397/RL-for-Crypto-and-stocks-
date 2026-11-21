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


