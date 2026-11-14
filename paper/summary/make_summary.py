"""Build the one-page plain-English summary, docs/assets/rl-trader-summary.pdf.

Written for readers outside the field. Every figure is read from the committed
artifact docs/assets/depth.json, so the page cannot drift from the study. Needs
matplotlib and Playwright (Chromium):

    python paper/summary/make_summary.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"
INK, MUTED, GRID, RUN = "#1d1d1f", "#6e6e73", "#e5e5ea", "#c2410c"


def chart(depth: dict) -> str:
    """Every training run as a dot, buy-and-hold as a labelled tick, per market.

    One colour for every run: whether a run made or lost money is read from its
    position against the zero line, not from a red/green pair that colour-blind
    readers cannot separate.
    """
    fig, ax = plt.subplots(figsize=(6.6, 2.0), dpi=300)
    for key, y in (("stock", 1.0), ("crypto", 0.0)):
        r = depth["seeds"][key]
        xs = [v * 100 for v in r["seed_returns"]]
        offs = [((i % 5) - 2) * 0.07 for i in range(len(xs))]
        ax.scatter(xs, [y + o for o in offs], s=14, color=RUN, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=3)
        bh = r["bh"] * 100
        ax.plot([bh, bh], [y - 0.28, y + 0.28], color=INK, lw=2, zorder=4)
        ax.annotate(f"buy & hold {bh:+.0f}%", (bh, y + 0.3), ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_yticks([1, 0], ["Stocks", "Crypto"], fontsize=8, color=INK)
    ax.set_ylim(-0.55, 1.7)
    ax.set_xlabel("Return on years of prices the agent never saw (%)", fontsize=7.5, color=MUTED)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


CSS = """
@page { size: Letter; margin: 0.55in 0.6in; }
body { font-family: Inter, "Segoe UI", Helvetica, Arial, sans-serif; color: INK;
  font-size: 9.6pt; line-height: 1.42; margin: 0; }
h1 { font-size: 18pt; margin: 0 0 2pt; letter-spacing: -0.01em; }
.sub { color: MUTED; margin: 0 0 8pt; font-size: 9pt; }
h2 { font-size: 9.5pt; margin: 9pt 0 3pt; color: RUN; text-transform: uppercase;
  letter-spacing: 0.05em; }
p { margin: 0 0 5pt; }
ul { margin: 0 0 5pt 14pt; padding: 0; } li { margin: 0 0 3pt; }
.fig { margin: 4pt 0 2pt; } .fig img { width: 100%; }
.cap { color: MUTED; font-size: 8pt; margin: 0 0 5pt; }
.links { border-top: 1px solid GRID; margin-top: 9pt; padding-top: 6pt;
  color: MUTED; font-size: 8.3pt; }
b { font-weight: 650; }
""".replace("INK", INK).replace("MUTED", MUTED).replace("GRID", GRID).replace("RUN", RUN)


def signed(v: float) -> str:
    """A whole-number percentage with a real minus sign."""
    return ("−" if v < 0 else "+") + f"{abs(v) * 100:.0f}%"


def page(depth: dict) -> str:
    s, c = depth["seeds"]["stock"], depth["seeds"]["crypto"]
    folds = depth["walk_forward"]
    nf = len(folds["stock"]["folds"])
    stock_behind = folds["stock"]["pooled"]["folds_behind"]
    crypto_behind = folds["crypto"]["pooled"]["folds_behind"]
    hit = c["risk"]["behaviour"]["hit_rate"]
    losing = sum(v < 0 for v in c["seed_returns"])
    runs = s["seeds"] * len(s["tickers"])
    img = chart(depth)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<h1>Is the AI learning, or just lucky?</h1>
<p class="sub">RL Trader &middot; Daniel Lichtenberger &middot; independent research in deep
reinforcement learning, 2026</p>

<h2>The question</h2>
<p>Reinforcement learning is how an AI teaches itself by trial and error, the method behind
game-playing programs. It is also easy to fool yourself with: an agent can memorise its
practice data, or one lucky training run can look like a breakthrough. I built a trading
agent from scratch (its learning algorithm written by hand in PyTorch) to study both traps, using stock and
crypto markets because they are noisy enough to make both easy to fall into.</p>

<h2>What happened</h2>
<p>An early version looked spectacular: one run nearly quadrupled its money on crypto, while
simply buying and holding gained 19%. Instead of reporting it, I tried to break it.</p>
<div class="fig"><img src="data:image/png;base64,{img}" alt="Each dot is one complete
training run. Most lose money, and on stocks buy-and-hold is far to the right."></div>
<p class="cap">Each dot is one complete training run ({s['seeds']} per market: same code and
data, a different random starting point), scored on years of prices it never saw. Dots left
of the line lost money.</p>
<ul>
<li><b>On stocks it is clearly worse than just holding:</b> {signed(s['mean'])} on average
against buy-and-hold's {signed(s['bh'])}, and behind in all {runs} runs.</li>
<li><b>On crypto, {losing} of {c['seeds']} runs lose money.</b> Retrained in {nf} different
periods of history, it trails buy-and-hold in {stock_behind} of {nf} periods on stocks and
{crypto_behind} of {nf} on crypto; the exception was a crash, which it lost less in.</li>
<li><b>Why:</b> each day it bets on up or down, and its bets are right {hit * 100:.1f}% of the
time, a coin flip. It also pays trading costs to change its bet almost every day.</li>
</ul>

<h2>How I tested it</h2>
<p>Each claim had to survive the checks professional researchers use. I wrote down my
predictions <i>before</i> running the final experiments, so the results could not shape them
(one of six turned out wrong, and the write-up says so). I repeated the key experiments over
dozens of independent training runs and used statistical tests to separate skill from luck.
I checked my algorithm against a standard reference library, found two real flaws in my own
study, fixed them, and showed the conclusion did not change.</p>

<h2>Where the AI does work</h2>
<p>Given a decision problem instead of a prediction problem (hedging an option contract when
every trade costs money), the same code learns what finance theory predicts and, on 19 years
of real market data, beats a standard textbook strategy when trading is expensive.</p>

<h2>What I learned</h2>
<p>The most valuable result was a negative one: a convincing number is not evidence until it
survives a serious attempt to disprove it. The tools I built to catch my own false positive
are the real contribution.</p>

<div class="links">Live site and interactive lab: rl-for-crypto-and-stocks.vercel.app
&nbsp;&middot;&nbsp; Code, data and full results: github.com/Danny-397/RL-for-Crypto-and-stocks-
&nbsp;&middot;&nbsp; Paper: doi.org/10.5281/zenodo.23843503</div>
</body></html>"""


def main() -> None:
    depth = json.loads((ASSETS / "depth.json").read_text(encoding="utf-8"))
    out_html = ROOT / "paper" / "summary" / "_summary.html"
    out_html.write_text(page(depth), encoding="utf-8")
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            pg = browser.new_page()
            pg.goto(out_html.as_uri())
            pg.pdf(path=str(ASSETS / "rl-trader-summary.pdf"), format="Letter",
                   print_background=True, prefer_css_page_size=True)
            browser.close()
    finally:
        os.remove(out_html)
    print("wrote", ASSETS / "rl-trader-summary.pdf")


if __name__ == "__main__":
    main()
