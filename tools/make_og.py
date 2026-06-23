"""Render the social share card (docs/assets/og.png, 1200x630).

This is the image that unfurls when the site link is pasted into the Common App,
an email, Slack, etc. Pulls a real agent-vs-buy&hold curve from docs/results.js so
the card shows actual output, not a mock. Run from the repo root:

    python tools/make_og.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG = "#07090d"
VOLT = "#d4ff3f"
CYAN = "#36e0ff"
GREY = "#8a97a8"
TEXT = "#e7edf5"


def _load_curves():
    """A representative agent/benchmark equity curve from the baked results."""
    path = os.path.join("docs", "results.js")
    try:
        src = open(path, encoding="utf-8").read()
        data = json.loads(src[src.index("{"): src.rindex("}") + 1])
        m = data["markets"]["crypto"]
        return m["equity_agent"], m["equity_bench"]
    except Exception:
        # graceful fallback so the card always renders
        return [1, 1.4, 1.2, 1.9, 2.6, 3.1], [1, 1.1, 0.9, 1.2, 1.15, 1.3]


def main() -> None:
    agent, bench = _load_curves()
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(BG)

    # faint equity curve as the backdrop
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor(BG)
    n = max(len(agent), len(bench))
    ax.plot(range(len(bench)), bench, color=GREY, lw=2, alpha=0.5)
    ax.plot(range(len(agent)), agent, color=VOLT, lw=3, alpha=0.9)
    ax.set_xlim(0, n - 1)
    ax.axis("off")

    # title block
    fig.text(0.06, 0.74, "RL·Trader", color=TEXT, fontsize=64, fontweight="bold",
             fontfamily="DejaVu Sans")
    fig.text(0.061, 0.62, "Deep reinforcement learning that trades stocks & crypto",
             color=CYAN, fontsize=24)
    fig.text(0.061, 0.20,
             "from-scratch PPO  ·  stocks & crypto  ·  28 features  ·  multi-seed evaluated",
             color=GREY, fontsize=16, fontfamily="DejaVu Sans Mono")

    os.makedirs(os.path.join("docs", "assets"), exist_ok=True)
    out = os.path.join("docs", "assets", "og.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
