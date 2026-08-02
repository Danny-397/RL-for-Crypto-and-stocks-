"""Print-ready figures for paper/rl_trader.tex.

The figures in docs/assets/ are dark-themed to match the website; dropping them
into a white two-column paper looks wrong and prints badly. These are light,
serif, and sized to the column measure (3.4in) so LaTeX does not rescale them.

Renders from docs/assets/ablation_multiseed.json (see tools/ablation_multiseed.py),
so the figure shows the per-seed spread rather than one run -- which is the
point Table 1 is being rewritten to make.

Run from the repo root:
    python tools/paper_figures.py      # writes paper/figures/*.pdf and *.png
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = os.path.join("docs", "assets", "ablation_multiseed.json")
OUT = os.path.join("paper", "figures")

# Muted, print-safe, distinguishable in greyscale.
RED = "#b2182b"
BLUE = "#2166ac"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 400,
})


def _compact(frac: float) -> str:
    """Format a return fraction compactly enough to fit the column measure.

    +145.8 (i.e. +14,580%) -> '+15,000%' is still too wide, so switch to
    scientific-ish shorthand above 1000%.
    """
    pct = frac * 100
    if abs(pct) >= 1_000_000:
        return f"{pct/1_000_000:+.1f}M%"
    if abs(pct) >= 1_000:
        return f"{pct/1_000:+.0f}k%"
    return f"{pct:+.0f}%"


def fig_ablation(data: dict) -> None:
    markets = list(data["summary"]["markets"].keys())
    fig, axes = plt.subplots(1, len(markets), figsize=(3.4, 1.95), sharey=False)
    if len(markets) == 1:
        axes = [axes]

    for ax, market in zip(axes, markets):
        m = data["summary"]["markets"][market]
        xs = [0, 1]
        means = [m["single"]["oos_mean"] * 100, m["domain"]["oos_mean"] * 100]
        cis = [m["single"]["oos_ci"], m["domain"]["oos_ci"]]
        colors = [RED, BLUE]

        for x, mu, ci, c in zip(xs, means, cis, colors):
            lo, hi = ci[0] * 100, ci[1] * 100
            ax.bar(x, mu, width=0.55, color=c, alpha=0.32, edgecolor=c, linewidth=0.9)
            ax.plot([x, x], [lo, hi], color=c, linewidth=1.1, solid_capstyle="butt")

        # Individual seeds, so the reader sees n and the spread, not just a bar.
        for x, key, c in zip(xs, ("single", "domain"), colors):
            pts = np.array(m[key]["oos_per_seed"]) * 100
            jit = np.linspace(-0.13, 0.13, len(pts))
            ax.plot(x + jit, pts, "o", ms=2.0, color=c, mew=0, alpha=0.85, zorder=3)

        ax.axhline(0, color="#666666", linewidth=0.6, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(["single-\npath", "domain-\nrandom"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(market.capitalize())
        if ax is axes[0]:
            ax.set_ylabel("Out-of-sample return (%)")

        # The in-sample number is the memorization artifact; give its range, not a
        # point. Headroom first so the label never lands on a data point.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.26 * (hi - lo))
        s = m["single"]
        ax.text(
            0.5, 0.99, f"in-sample {_compact(s['in_min'])} to {_compact(s['in_max'])}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=5.4, color="#444444",
        )

    fig.tight_layout(pad=0.35, w_pad=0.9)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig_ablation.{ext}")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print("wrote", path)
    plt.close(fig)


def main() -> None:
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run ablation_multiseed.py first")
    with open(src, encoding="utf-8") as fh:
        data = json.load(fh)
    fig_ablation(data)


if __name__ == "__main__":
    main()
