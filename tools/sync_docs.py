"""Regenerate the result tables in README.md and RESULTS.md from the artifacts.

The website used to carry these same numbers as hand-typed HTML, and a rebuild
moved them without anything noticing: the ablation table went on claiming a +36%
held-out return for two hours after the artifact said +47%. That was fixed by
generating the page from ``docs/assets/*.json`` instead of transcribing it.

The prose documents have exactly the same problem and are read far more often.
At the time this was written README.md and RESULTS.md still quoted a +275% crypto
run, a -2.7% multi-seed mean and ``p = 0.97``, all superseded, in a repository
whose stated thesis is that single-run numbers should not be trusted.

So the tables are generated here and pasted into marked regions:

    <!-- BEGIN GENERATED: significance-full -->
    ...anything in here is overwritten...
    <!-- END GENERATED: significance-full -->

Markers are HTML comments, which GitHub renders as nothing. Everything outside
them is hand-written and left alone -- this replaces the numbers, not the
argument about them.

Usage::

    python tools/sync_docs.py            # rewrite the marked regions
    python tools/sync_docs.py --check    # exit 1 if anything is out of date

``--check`` is what the test suite runs, so a rebuild that moves a number turns
the suite red instead of quietly making the documentation wrong.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from typing import Callable, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
DOCS = os.path.join(REPO, "docs")

# Markdown hides its markers in an HTML comment; LaTeX in a % comment. Both are
# invisible in the rendered output, so a reader never sees the machinery.
MARKERS = {
    ".md": ("<!-- BEGIN GENERATED: {} -->", "<!-- END GENERATED: {} -->"),
    ".tex": ("% BEGIN GENERATED: {}", "% END GENERATED: {}"),
}
BEGIN, END = MARKERS[".md"]


def markers_for(name: str):
    """Marker pair for a target, chosen by extension."""
    return MARKERS.get(os.path.splitext(name)[1], MARKERS[".md"])


# --------------------------------------------------------------------------- #
# Reading the artifacts                                                        #
# --------------------------------------------------------------------------- #
def _js_global(filename: str) -> Optional[dict]:
    """Parse one of the ``window.RL_* = {...};`` shims the site loads."""
    path = os.path.join(DOCS, filename)
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        return json.loads(text[text.index("=") + 1: text.rindex(";")])
    except (ValueError, json.JSONDecodeError):
        return None


def _asset(name: str) -> Optional[dict]:
    path = os.path.join(DOCS, "assets", name)
    if not os.path.exists(path):
        return None
    try:
        with io.open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Formatting                                                                   #
# --------------------------------------------------------------------------- #
def pct(v: float, dp: int = 1) -> str:
    """A signed percentage using a real minus sign, as the documents do."""
    s = f"{abs(v) * 100:,.{dp}f}"
    return ("−" if v < 0 else "+") + s + "%"


def kpct(v: float) -> str:
    """Compact form for the in-sample figures, which span six decades."""
    p, a = v * 100, abs(v * 100)
    sign = "−" if p < 0 else "+"
    if a >= 1e6:
        return f"{sign}{a / 1e6:.1f}M%"
    if a >= 1e3:
        return f"{sign}{round(a / 1e3):,.0f}k%"
    return f"{sign}{a:.0f}%"


def num(v: float, dp: int = 2) -> str:
    """A signed number using the same minus sign as the percentages beside it."""
    return ("−" if v < 0 else "+") + f"{abs(v):.{dp}f}"


def pval(p: float) -> str:
    """Below 0.01 the leading digits carry the information; above it they do not."""
    return f"{p:.4f}" if p < 0.01 else f"{p:.2f}"


def _verdict(mean: float, bh: float, p: float) -> str:
    if p >= 0.05:
        return "**indistinguishable** from B&H"
    return ("significantly **worse** than B&H" if mean < bh
            else "significantly **better** than B&H")


# --------------------------------------------------------------------------- #
# The generated blocks                                                         #
# --------------------------------------------------------------------------- #
def significance_full() -> Optional[str]:
    """The full multi-seed table, for RESULTS.md."""
    sig = _js_global("significance.js")
    if not sig:
        return None
    rows = [
        "| Market | Agent return (95% CI across seeds) | Buy & hold | "
        "Agent − B&H | p-value | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, label in (("stock", "Stock"), ("crypto", "Crypto")):
        r = sig.get(key)
        if not r:
            continue
        emph = f"**{pval(r['p'])}**" if r["p"] < 0.05 else pval(r["p"])
        rows.append(
            f"| {label} | **{pct(r['mean'])}** `[{pct(r['ci_low'])}, "
            f"{pct(r['ci_high'])}]` | {pct(r['bh'])} | "
            f"{pct(r['mean'] - r['bh'])} | {emph} | "
            f"{_verdict(r['mean'], r['bh'], r['p'])} |")
    n = sig.get("stock", {}).get("seeds") or sig.get("crypto", {}).get("seeds")
    rows.append("")
    rows.append(
        f"*{n} independent training seeds per market. The p-value is a paired "
        "permutation test of agent vs. buy-and-hold across the held-out tickers "
        "(10 equities, 6 crypto pairs), not across seeds — the two axes answer "
        "different questions and their p-values are not comparable.*")
    return "\n".join(rows)


def significance_brief() -> Optional[str]:
    """The same study, condensed, for the README."""
    sig = _js_global("significance.js")
    if not sig:
        return None
    rows = ["| Market | Agent return (95% CI, seeds) | Buy & hold | Verdict |",
            "|---|---:|---:|---|"]
    for key, label in (("crypto", "Crypto"), ("stock", "Stock")):
        r = sig.get(key)
        if not r:
            continue
        verdict = (f"indistinguishable (p = {pval(r['p'])})" if r["p"] >= 0.05
                   else f"significantly **worse** (p = {pval(r['p'])})"
                   if r["mean"] < r["bh"]
                   else f"significantly **better** (p = {pval(r['p'])})")
        rows.append(
            f"| {label} | **{pct(r['mean'])}** `[{pct(r['ci_low'], 0)}, "
            f"{pct(r['ci_high'], 0)}]` | {pct(r['bh'], 0)} | {verdict} |")
    return "\n".join(rows)


def seed_spread() -> Optional[str]:
    """What the individual seeds did -- the point of the whole exercise."""
    sig = _js_global("significance.js")
    if not sig:
        return None
    out = []
    for key, label in (("stock", "Stock"), ("crypto", "Crypto")):
        r = sig.get(key)
        if not r or not r.get("seed_returns"):
            continue
        vals = ", ".join(pct(v) for v in r["seed_returns"])
        out.append(f"- **{label}** — {vals}")
    return "\n".join(out) if out else None


def headline_run() -> Optional[str]:
    """The published single-seed run the multi-seed study is contrasted against."""
    res = _js_global("results.js")
    if not res:
        return None
    lines = []
    for key, label in (("crypto", "Crypto"), ("stock", "Stock")):
        m = (res.get("markets") or {}).get(key)
        if not m:
            continue
        n = m.get("n_eval") or 0
        wins = round((m.get("win_rate") or 0.0) * n)
        lines.append(
            f"- **{label}** — agent {pct(m['metrics']['total_return'])} vs. "
            f"buy-and-hold {pct(m['bench_metrics']['total_return'])}, "
            f"beating buy-and-hold on {wins} of {n} tickers "
            f"(Sharpe {num(m['metrics']['sharpe'])}, "
            f"max drawdown {m['metrics']['max_drawdown'] * 100:.1f}%).")
    if not lines:
        return None
    head = (f"Seed {res.get('seed')}, {res.get('timesteps', 0):,} timesteps, "
            f"test window ending {(res.get('markets', {}).get('stock', {}) or {}).get('end_date', '?')}:")
    return head + "\n\n" + "\n".join(lines)


def ablation_table() -> Optional[str]:
    """Single-asset vs cross-asset training, across seeds."""
    art = _asset("ablation_multiseed.json")
    if not art:
        return None
    s = art["summary"]
    n = len(s.get("seeds", []))
    rows = ["| Training setup | In-sample (range over seeds) | "
            "Held-out return (mean, 95% CI) |", "|---|---:|---:|"]
    for market, label in (("stock", "Stocks"), ("crypto", "Crypto")):
        m = s["markets"].get(market)
        if not m:
            continue
        for arm, how in (("single", "single asset"), ("domain", "across tickers")):
            d = m[arm]
            rows.append(
                f"| {label} · {how} | {kpct(d['in_min'])} to {kpct(d['in_max'])} "
                f"| **{kpct(d['oos_mean'])}** `[{kpct(d['oos_ci'][0])}, "
                f"{kpct(d['oos_ci'][1])}]` |")
    rows.append("")
    rows.append(
        f"*{n} seeds × {s.get('timesteps', 0) // 1000}k steps, bootstrap 95% CI, "
        "held-out paths held identical across arms. In-sample is a range rather "
        "than a mean because it varies by an order of magnitude across seeds; "
        "quoting one run of it would imply a precision that is not there.*")
    return "\n".join(rows)


def _surrogate_arm(name: str) -> Optional[str]:
    """One arm of the falsification test.

    Kept as two blocks rather than one table because RESULTS.md argues at length
    between them -- the control has to be read, and believed, before the real
    arm's null means anything.
    """
    art = _asset(f"surrogate_{name}.json")
    if not art:
        return None
    rows = ["| Market | Edge vs. B&H (structured) | Edge vs. B&H (surrogate) | "
            "Δ | permutation p | pairs |", "|---|---:|---:|---:|---:|---:|"]
    for market in sorted(art):
        r = art[market]
        p_ = r["p"]
        mark = f"**{p_:.4f}**" if p_ < 0.05 else f"{p_:.4f}"
        rows.append(
            f"| {market.capitalize()} | **{pct(r['edge_structured'], 1)}** | "
            f"{pct(r['edge_surrogate'], 1)} | {pct(r['diff'], 1)} | {mark} | "
            f"{r.get('n_pairs', '?')} |")
    return "\n".join(rows)


def surrogate_control() -> Optional[str]:
    return _surrogate_arm("synthetic")


def surrogate_real() -> Optional[str]:
    return _surrogate_arm("real")


def baselines_table() -> Optional[str]:
    """Agent against every baseline, mean over the held-out basket."""
    art = _asset("baselines.json")
    if not art:
        return None
    order = ["PPO agent", "buy_and_hold", "ma_crossover", "flat", "random"]
    pretty = {"PPO agent": "**PPO agent**", "buy_and_hold": "Buy & hold",
              "ma_crossover": "MA crossover", "flat": "Flat (cash)",
              "random": "Random"}
    out: List[str] = []
    for market, label in (("stock", "Stocks"), ("crypto", "Crypto")):
        block = art.get("markets", {}).get(market)
        if not block:
            continue
        wins, n = block.get("agent_beats_bh"), block.get("n_tickers")
        out.append(f"**{label}** — mean over {n} held-out tickers "
                   f"(the agent beats buy-and-hold on {wins} of {n}):")
        out.append("")
        out.append("| Strategy | Return | Sharpe | Max DD |")
        out.append("|---|---:|---:|---:|")
        for key in order:
            r = block.get("strategies", {}).get(key)
            if not r:
                continue
            out.append(f"| {pretty.get(key, key)} | {pct(r['total_return'])} | "
                       f"{num(r['sharpe'])} | {r['max_drawdown'] * 100:.1f}% |")
        out.append("")
    return "\n".join(out).rstrip() if out else None


def portfolio_table() -> Optional[str]:
    """Cross-sectional allocation, agent against the benchmarks a quant uses."""
    art = _asset("portfolio.json")
    if not art:
        return None
    pretty = {"PPO portfolio agent": "PPO portfolio agent",
              "equal_weight": "**Equal-weight (1/N)**",
              "cross_sectional_momentum": "Cross-sectional momentum",
              "random": "Random weights"}
    order = ["PPO portfolio agent", "equal_weight", "cross_sectional_momentum", "random"]
    out: List[str] = []
    for market, label in (("stock", "Stock"), ("crypto", "Crypto")):
        b = art.get("markets", {}).get(market)
        if not b:
            continue
        out.append(f"**{label}** — {b['n_assets']}-name basket, held-out test "
                   f"(seed {b['seed']}, {b['timesteps']:,} steps):")
        out.append("")
        out.append("| Strategy | Return | Sharpe | Max DD |")
        out.append("|---|---:|---:|---:|")
        for key in order:
            m = b.get("strategies", {}).get(key)
            if not m:
                continue
            out.append(f"| {pretty.get(key, key)} | {pct(m['total_return'])} | "
                       f"{num(m['sharpe'])} | {m['max_drawdown'] * 100:.1f}% |")
        out.append("")
    return "\n".join(out).rstrip() if out else None


def significance_synth() -> Optional[str]:
    """The synthetic-data multi-seed study, where a signal provably exists."""
    art = _asset("significance_synth.json")
    if not art:
        return None
    rows = ["| Market | Agent OOS return (95% CI) | Agent OOS Sharpe (95% CI) | "
            "Buy & hold | Agent − B&H | p-value |",
            "|---|---:|---:|---:|---:|---:|"]
    seen = False
    for market, label in (("stock", "Stock"), ("crypto", "Crypto")):
        r = art.get("markets", {}).get(market)
        if not r:
            continue
        seen = True
        rows.append(
            f"| {label} | {pct(r['return_mean'])} `[{pct(r['return_ci'][0])}, "
            f"{pct(r['return_ci'][1])}]` | {num(r['sharpe_mean'])} "
            f"`[{num(r['sharpe_ci'][0])}, {num(r['sharpe_ci'][1])}]` | "
            f"{pct(r['bh_mean'])} | {pct(r['diff'])} | {pval(r['p'])} |")
    return "\n".join(rows) if seen else None


def attribution_table() -> Optional[str]:
    """Which inputs the deployed policies read, by occlusion.

    Quoted by hand until the archives were rebuilt and the top three features
    changed identity in both markets while the magnitudes barely moved. The
    instability is the point, so it travels with the table.
    """
    art = _asset("attribution.json")
    if not art:
        return None
    rows = ["| | Strongest inputs | Largest account-state effect |", "|---|---|---:|"]
    for market, label in (("stock", "Stock"), ("crypto", "Crypto")):
        m = art.get("markets", {}).get(market)
        if not m:
            continue
        top = " · ".join(f"`{f['name']}` {f['mean_abs_delta']:.2f}"
                         for f in m["features"][:3])
        rows.append(f"| **{label}** | {top} | {m['account_max']:.3f} |")
    rows.append("")
    rows.append(f"*{art['episode_bars']}-bar {art['source']} paths, seed "
                f"{art['seed']}, {art['sampled_bars']} sampled bars. "
                f"{art['units'].capitalize()}.*")
    return "\n".join(rows)


def sweep_table() -> Optional[str]:
    """One knob at a time, either side of each published default.

    Served and rendered in published order, never ranked: the question is
    whether the conclusion is fragile, not which recipe wins. Ranking nine noisy
    configurations and quoting the top one is exactly the error the rest of this
    document is about.
    """
    art = _asset("hyperparameter_sweep.json")
    if not art:
        return None
    out: List[str] = []
    for market, label in (("stock", "Stocks"), ("crypto", "Crypto")):
        block = art.get("markets", {}).get(market)
        if not block:
            continue
        rows = block.get("rows", [])
        n_pos = sum(1 for r in rows if r["mean_edge"] > 0)
        out.append(f"**{label}** — {len(rows)} configurations, "
                   f"{n_pos} with a positive edge over buy-and-hold:")
        out.append("")
        out.append("| Configuration | Edge vs. buy & hold | 95% CI |")
        out.append("|---|---:|---:|")
        for r in rows:
            name = r["config"] + (" *(published default)*" if r.get("knob") is None else "")
            ci = r["edge_ci"]
            out.append(f"| {name} | {pct(r['mean_edge'])} | "
                       f"`[{pct(ci[1])}, {pct(ci[2])}]` |")
        out.append("")
    if not out:
        return None
    out.append(f"*{art.get('seeds_per_config', '?')} seeds per configuration, "
               f"{art.get('timesteps', 0) // 1000}k steps each, one knob moved at a "
               "time with everything else held at its default. Not a grid search: "
               "the question is whether the negative result is fragile, not which "
               "recipe wins. With this many seeds no single row is a significance "
               "claim — the sign test cannot reach p ≤ 0.05 at that sample size.*")
    return "\n".join(out)


def supervised_table() -> Optional[str]:
    """The agent against learned baselines on the same inputs and costs."""
    art = _asset("supervised.json")
    if not art:
        return None
    pretty = {
        "PPO agent": "**PPO agent**",
        "ridge_forecast": "Ridge regression *(learned)*",
        "logistic_direction": "Logistic direction *(learned)*",
        "buy_and_hold": "Buy & hold",
        "ma_crossover": "MA crossover",
        "flat": "Flat (cash)",
        "random": "Random",
    }
    out: List[str] = []
    for market, label in (("stock", "Stocks"), ("crypto", "Crypto")):
        block = art.get("markets", {}).get(market)
        if not block:
            continue
        out.append(f"**{label}** \u2014 mean over {block['n_tickers']} held-out "
                   "tickers, ranked:")
        out.append("")
        out.append("| Strategy | Return |")
        out.append("|---|---:|")
        rows = sorted(block["strategies"].items(),
                      key=lambda kv: -kv[1]["total_return"])
        for name, m in rows:
            out.append(f"| {pretty.get(name, name)} | {pct(m['total_return'])} |")
        out.append("")
        out.append(f"*Logistic in-sample directional accuracy: "
                   f"{block['logistic_train_accuracy'] * 100:.1f}%.*")
        out.append("")
    return "\n".join(out).rstrip() if out else None


def supervised_brief() -> Optional[str]:
    """Condensed for the README: the learned arms against the passive benchmark."""
    art = _asset("supervised.json")
    if not art:
        return None
    rows = ["| Market | Ridge | Logistic | PPO agent | Buy & hold |",
            "|---|---:|---:|---:|---:|"]
    seen = False
    for market, label in (("stock", "Stocks"), ("crypto", "Crypto")):
        b = art.get("markets", {}).get(market)
        if not b:
            continue
        seen = True
        st = b["strategies"]
        rows.append(
            f"| {label} | {pct(st['ridge_forecast']['total_return'])} | "
            f"{pct(st['logistic_direction']['total_return'])} | "
            f"{pct(st['PPO agent']['total_return'])} | "
            f"**{pct(st['buy_and_hold']['total_return'])}** |")
    return "\n".join(rows) if seen else None


def cost_table() -> Optional[str]:
    """Held-out return against the friction it pays."""
    art = _asset("cost_sensitivity.json")
    if not art:
        return None
    out: List[str] = []
    for market, label in (("stock", "Stocks"), ("crypto", "Crypto")):
        block = art.get("markets", {}).get(market)
        if not block:
            continue
        out.append(f"**{label}**:")
        out.append("")
        out.append("| Costs | Fee | Slippage | Mean held-out return | Turnover |")
        out.append("|---|---:|---:|---:|---:|")
        for r in block["rows"]:
            marker = " *(published)*" if r["multiple"] == 1.0 else ""
            out.append(
                f"| {r['multiple']:g}\u00d7{marker} | {r['transaction_cost'] * 100:.3f}% "
                f"| {r['slippage'] * 100:.3f}% | {pct(r['mean_return'])} "
                f"| {r['mean_turnover']:.2f} |")
        out.append("")
    if not out:
        return None
    out.append("*Turnover is the mean absolute change in target position per bar: "
               "0 would be buy-and-hold, 2.0 would be flipping fully long to fully "
               "short every bar. The policy is frozen and replayed \u2014 it is not "
               "retrained per cost level, which would measure churn rather than "
               "friction.*")
    return "\n".join(out)


def surrogate_robust() -> Optional[str]:
    """Mean-based against median-based, so a fat tail cannot hide in either.

    Computed live by server/surrogate.py from the per-pair values, not read from
    a stored field -- so this table cannot drift from the artifacts even if
    someone regenerates them.
    """
    try:
        from server import surrogate as _surrogate
    except ImportError:            # pragma: no cover - server extras absent
        return None
    served = _surrogate.results()
    if not served:
        return None

    rows = ["| Arm | Market | Mean diff | p (mean) | Median diff | p (median) | Floor |",
            "|---|---|---:|---:|---:|---:|---:|"]
    labels = {"synthetic": "Control", "real": "Real"}
    any_row = False
    for arm in served["arms"]:
        for m in arm["markets"]:
            rb = m.get("robust")
            if not rb:
                continue
            any_row = True
            mark = lambda v: f"**{v:.4f}**" if v < 0.05 else f"{v:.4f}"  # noqa: E731
            rows.append(
                f"| {labels.get(arm['arm'], arm['arm'])} | {m['market']} | "
                f"{m['diff']:+.3f} | {mark(m['p'])} | {rb['median_diff']:+.3f} | "
                f"{mark(rb['p'])} | {rb['floor']:.4f} |")
    if not any_row:
        return None
    rows.append("")
    rows.append("*Same paired sign-flip null, enumerated exactly; only the "
                "statistic differs. `Floor` is the smallest p-value the design "
                "can produce at that many pairs.*")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# The same results, as LaTeX                                                   #
# --------------------------------------------------------------------------- #
# The paper has drifted from the artifacts before: tools/ablation_multiseed.py
# records that its Table 1 once read +5821%/+18709% while ablation.json said
# +14581%/+1956390% -- two different runs of a quantity that is not stable across
# seeds. A paper gets a DOI, so a wrong number in it is permanent. These render
# the same artifacts into the paper's table bodies, between "% BEGIN GENERATED"
# markers, leaving every caption and \toprule where the author put them.


def _tex_pc(v: float, dp: int = 1) -> str:
    r"""A percentage the way the paper writes them, e.g. ``$+79.7\pc$``."""
    sign = "-" if v < 0 else "+"
    return "$" + sign + f"{abs(v) * 100:,.{dp}f}" + r"\pc$"


def _tex_k(v: float) -> str:
    """An in-sample figure, unitless and compact, as Table 1 has them."""
    a, sign = abs(v * 100), ("-" if v < 0 else "+")
    if a >= 1e6:
        return f"{sign}{a / 1e6:.1f}M"
    if a >= 1e3:
        return f"{sign}{round(a / 1e3):,.0f}k"
    return f"{sign}{a:.0f}"


def tex_ablation() -> Optional[str]:
    """Body rows for tab:ablation."""
    art = _asset("ablation_multiseed.json")
    if not art:
        return None
    rows = []
    for market, label in (("stock", "Stock "), ("crypto", "Crypto")):
        m = art["summary"]["markets"].get(market)
        if not m:
            continue
        for arm in ("single", "domain"):
            d = m[arm]
            lo, hi = d["oos_ci"]
            mean = f"{d['oos_mean'] * 100:+.0f}"
            # the paper bolds the arm that generalizes
            cell = r"\mathbf{" + mean + "}" if arm == "domain" else mean
            rows.append(
                f"{label} & {arm} & ${_tex_k(d['in_min'])}$--${_tex_k(d['in_max'])}$"
                f" & ${cell}$ "
                + r"{\scriptsize$[" + f"{lo * 100:+.0f},{hi * 100:+.0f}" + r"]$}\\")
    return "\n".join(rows)


def tex_significance() -> Optional[str]:
    """Body rows for tab:real."""
    sig = _js_global("significance.js")
    if not sig:
        return None
    rows = []
    for key, label in (("crypto", "Crypto"), ("stock", "Stock ")):
        r = sig.get(key)
        if not r:
            continue
        verdict = ("indist." if r["p"] >= 0.05
                   else ("worse" if r["mean"] < r["bh"] else "better"))
        rows.append(
            f"{label} & {_tex_pc(r['mean'])} "
            f"$[{r['ci_low'] * 100:+.0f},{r['ci_high'] * 100:+.0f}]$ & "
            f"{_tex_pc(r['mean'] - r['bh'], 0)} & "
            f"{verdict} ($p{{=}}{r['p']:.4f}$)" + r"\\")
    return "\n".join(rows)


def _tex_surrogate(name: str) -> Optional[str]:
    art = _asset(f"surrogate_{name}.json")
    if not art:
        return None
    rows = []
    for market in sorted(art):
        r = art[market]
        rows.append(
            f"{market.capitalize():<6} & {_tex_pc(r['edge_structured'])} & "
            f"{_tex_pc(r['edge_surrogate'])} & {_tex_pc(r['diff'])} & "
            f"${r['p']:.4f}$" + r"\\")
    return "\n".join(rows)


def tex_surrogate_control() -> Optional[str]:
    return _tex_surrogate("synthetic")


def tex_surrogate_real() -> Optional[str]:
    return _tex_surrogate("real")


BLOCKS: Dict[str, Callable[[], Optional[str]]] = {
    "significance-full": significance_full,
    "significance-brief": significance_brief,
    "seed-spread": seed_spread,
    "headline-run": headline_run,
    "ablation-table": ablation_table,
    "surrogate-control": surrogate_control,
    "surrogate-real": surrogate_real,
    "baselines-table": baselines_table,
    "portfolio-table": portfolio_table,
    "significance-synth": significance_synth,
    "attribution-table": attribution_table,
    "sweep-table": sweep_table,
    "supervised-table": supervised_table,
    "supervised-brief": supervised_brief,
    "cost-table": cost_table,
    "surrogate-robust": surrogate_robust,
    "tex-ablation": tex_ablation,
    "tex-significance": tex_significance,
    "tex-surrogate-control": tex_surrogate_control,
    "tex-surrogate-real": tex_surrogate_real,
}

TARGETS = ("README.md", "RESULTS.md", os.path.join("paper", "rl_trader.tex"))


# --------------------------------------------------------------------------- #
# Splicing                                                                     #
# --------------------------------------------------------------------------- #
def apply_blocks(text: str, target: str = "README.md") -> tuple:
    """Return ``(new_text, [keys touched], [keys whose data is missing])``."""
    begin_t, end_t = markers_for(target)
    touched, missing = [], []
    for key, build in BLOCKS.items():
        begin, end = begin_t.format(key), end_t.format(key)
        if begin not in text:
            continue
        if end not in text:
            raise SystemExit(f"'{begin}' has no matching '{end}'")
        body = build()
        if body is None:
            # An absent artifact must not silently blank a table that is
            # currently correct; leave it and say so.
            missing.append(key)
            continue
        pattern = re.compile(
            re.escape(begin) + r".*?" + re.escape(end), re.S)
        text = pattern.sub(lambda _m: f"{begin}\n{body}\n{end}", text, count=1)
        touched.append(key)
    return text, touched, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="Report drift and exit 1 instead of rewriting.")
    args = ap.parse_args()

    stale, wrote, missing_any = [], [], set()
    for name in TARGETS:
        path = os.path.join(REPO, name)
        if not os.path.exists(path):
            continue
        with io.open(path, encoding="utf-8", newline="") as fh:
            original = fh.read()
        updated, touched, missing = apply_blocks(
            original.replace("\r\n", "\n"), name)
        missing_any |= set(missing)
        if updated != original.replace("\r\n", "\n"):
            stale.append(name)
            if not args.check:
                with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(updated)
                wrote.append(f"{name} ({', '.join(touched)})")
        elif touched:
            print(f"  up to date: {name} ({', '.join(touched)})")

    for key in sorted(missing_any):
        print(f"  SKIPPED {key}: no artifact yet, left as-is")

    if args.check:
        if stale:
            print("\nOut of date: " + ", ".join(stale))
            print("Run: python tools/sync_docs.py")
            sys.exit(1)
        print("\nDocumentation matches the artifacts.")
        return
    for line in wrote:
        print(f"  rewrote {line}")
    if not wrote:
        print("\nNothing to do.")


if __name__ == "__main__":
    main()
