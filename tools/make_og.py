"""Render the social share card (docs/assets/og.png, 1200x630).

This is the image that unfurls when the site link is pasted into an email, Slack,
an application form, etc. It shows the same figure as the site's hero: one dot per
complete training run on held-out data, against buy-and-hold, read from
docs/significance.js so the card cannot drift from the published result. The
palette matches docs/styles.css. Run from the repo root:

    python tools/make_og.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# docs/styles.css tokens
BG = "#0b0b0c"
TEXT = "#f5f5f7"
MUTED = "#a1a1a6"
FAINT = "#86868b"
LINE = "#2a2a2e"
ACCENT = "#ff9f0a"   # the agent's series
BENCH = "#8e8e93"    # buy & hold

ROWS = [("crypto", "Crypto"), ("stock", "Stocks")]


def _font() -> str:
    """The closest installed face to the site's system stack."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Inter", "SF Pro Display", "Helvetica Neue", "Segoe UI", "Arial"):
        if name in installed:
            return name
    return "DejaVu Sans"


def _load_significance() -> dict:
    path = os.path.join("docs", "significance.js")
    src = open(path, encoding="utf-8").read()
    return json.loads(src[src.index("{"): src.rindex("}") + 1])


def _pct(v: float) -> str:
    return f"{v * 100:+.1f}%".replace("-", "−")


def _p(v: float) -> str:
    return f"p = {v:.3f}" if v < 0.01 else f"p = {v:.2f}"


def main() -> None:
    sig = _load_significance()
    plt.rcParams["font.family"] = _font()

    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(BG)

    # Title block
    fig.text(0.06, 0.865, "RL Trader", color=MUTED, fontsize=20, fontweight="semibold")
    fig.text(0.06, 0.70, "Is the agent learning, or just lucky?", color=TEXT,
             fontsize=46, fontweight="bold")
    fig.text(0.06, 0.615,
             "A from-scratch deep-RL study of memorisation and seed luck, on crypto and stocks.",
             color=MUTED, fontsize=18)

    # The figure: one dot per training run, held-out return, shared axis
    ax = fig.add_axes([0.06, 0.15, 0.88, 0.36])
    ax.set_facecolor(BG)
    vals = []
    for key, _ in ROWS:
        d = sig[key]
        vals += list(d["seed_returns"]) + [d["bh"], d["ci_low"], d["ci_high"], 0.0]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.75, len(ROWS) - 0.4)
    ax.axis("off")

    ax.axvline(0, color=LINE, lw=1.2, ls=(0, (3, 3)), zorder=0)
    ax.text(0, -0.73, "0%", color=FAINT, fontsize=12, ha="center", va="bottom")

    for i, (key, label) in enumerate(ROWS):
        d = sig[key]
        y = len(ROWS) - 1 - i
        ax.axhline(y, color=LINE, lw=1, zorder=0)
        ax.fill_betweenx([y - 0.22, y + 0.22], d["ci_low"], d["ci_high"],
                         color=ACCENT, alpha=0.14, lw=0, zorder=1)
        n = len(d["seed_returns"])
        offsets = [((j % 5) - 2) * 0.07 for j in range(n)]
        ax.scatter(d["seed_returns"], [y + o for o in offsets], s=70, color=ACCENT,
                   edgecolors=BG, linewidths=1.2, zorder=3)
        ax.plot([d["bh"], d["bh"]], [y - 0.28, y + 0.28], color=BENCH, lw=3,
                solid_capstyle="round", zorder=2)
        ax.text(d["bh"], y - 0.34, f"buy & hold {_pct(d['bh'])}", color=MUTED,
                fontsize=12, va="top",
                ha="right" if d["bh"] > hi - (hi - lo) * 0.2 else "left")
        verdict = "indistinguishable" if d["p"] >= 0.05 else (
            "worse" if d["mean"] < d["bh"] else "better")
        ax.text(lo - pad, y + 0.34,
                f"{label}   mean {_pct(d['mean'])}  ·  vs buy & hold: {verdict} ({_p(d['p'])})",
                color=TEXT, fontsize=14, fontweight="semibold", ha="left")

    n_seeds = len(sig["crypto"]["seed_returns"])
    fig.text(0.06, 0.06,
             f"Each dot is one complete training run ({n_seeds} seeds per market), "
             "scored on data it never saw.",
             color=FAINT, fontsize=13)
    fig.text(0.94, 0.06, "rl-for-crypto-and-stocks.vercel.app", color=FAINT,
             fontsize=13, ha="right")

    os.makedirs(os.path.join("docs", "assets"), exist_ok=True)
    out = os.path.join("docs", "assets", "og.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
