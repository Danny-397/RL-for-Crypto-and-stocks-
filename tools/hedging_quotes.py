"""Part 2 on real option quotes: does the hedging result survive real prices?

Part 2 (RESULTS.md §9) tests its trained hedging agents on real SPY history, but
the option itself is *modelled*: marked at Black-Scholes with an implied
volatility of 0.875 x VIX. This replaces the model with the market, using the
daily SPY option chains Tradeski archives (Danny-397/Tradeski,
``data/options-archive/``: bid, ask, volume and open interest per strike,
captured after each close since 2026-11-06).

Three parts, each as far as the archive allows:

A. **The volatility assumption.** Each archived day, the at-the-money implied
   volatility of the SPY expiry nearest one month, backed out of real mid quotes,
   against 0.875 x that day's VIX close.
B. **One-day hedging errors on real quotes.** For every near-the-money call quoted
   on two consecutive trading days, the error of each hedge over that day,
   ``h * dS - dV`` with ``dV`` the change in the real mid: no hedge, Black-Scholes
   delta at the option's own implied volatility, and the ten zero-cost agents
   (each starting from a delta hedge). Available as soon as two days exist.
C. **Full one-month episodes on real quotes.** A contract followed through its
   last 21 trading days, marked at its mid each day and at its payoff at expiry,
   hedged by every Part 2 policy at every cost level, scored exactly as §9 scores
   them. This needs a contract quoted on all 22 days, and the archive keeps each
   week's last expiry that far out only from Danny-397/Tradeski#55 on, so the
   first episode completes 21 trading days after that change takes effect, and
   one more each week after.

Implied volatilities use r = 0 and no dividends, as Part 2's model does, so they
are the volatilities that model would need to reproduce the quotes.

    python tools/hedging_quotes.py export-policies   # 40 agents -> data/hedging_policies.npz
    python tools/hedging_quotes.py run               # -> docs/assets/hedging_quotes.json

.github/workflows/hedging-quotes.yml re-runs it every week.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_trader.hedging.core import (  # noqa: E402
    HedgeSpec,
    VecHedgeEnv,
    band_hedge,
    bs_call,
    bs_delta,
    delta_hedge,
    no_hedge,
    run_policy,
    summarize,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_URL = "https://raw.githubusercontent.com/Danny-397/Tradeski/main/data/options-archive"
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
CACHE = os.path.join(ROOT, "data", "raw", "options-archive")
POLICIES = os.path.join(ROOT, "data", "hedging_policies.npz")
OUT = os.path.join(ROOT, "docs", "assets", "hedging_quotes.json")
COSTS = (0.0, 0.0005, 0.002, 0.01)
SEEDS = tuple(range(10))
VIX_TO_ATM = 0.875
LIFE = 21                       # trading days, as Part 2's options
TRADING_DAYS = 252
MONEYNESS_BAND = 0.03           # part B: strikes within 3% of the spot
MAX_SPREAD = 0.25               # a quote whose spread exceeds 25% of its mid is not used

# NYSE full-day closures (Good Friday included), for counting trading days.
NYSE_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


def trading_days_between(a: str, b: str) -> int:
    """Trading days after ``a`` up to and including ``b``."""
    d, end, n = dt.date.fromisoformat(a), dt.date.fromisoformat(b), 0
    while d < end:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d.isoformat() not in NYSE_HOLIDAYS:
            n += 1
    return n


def next_trading_day(a: str) -> str:
    d = dt.date.fromisoformat(a)
    while True:
        d += dt.timedelta(days=1)
        if d.weekday() < 5 and d.isoformat() not in NYSE_HOLIDAYS:
            return d.isoformat()


# ── data ─────────────────────────────────────────────────────────────────────
def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rl-trader-research"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def load_archive(source: str = ARCHIVE_URL) -> dict:
    """date -> that day's SPY snapshot (spot and expiries), from a directory or URL."""
    days = {}
    if os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "20*.json")))
        raw = {os.path.basename(p)[:-5]: json.load(open(p, encoding="utf-8")) for p in paths}
    else:
        os.makedirs(CACHE, exist_ok=True)
        index = json.loads(_get(f"{source}/index.json"))["days"]
        raw = {}
        for d in index:
            path = os.path.join(CACHE, f"{d}.json")
            if not os.path.exists(path):          # archived days never change
                open(path, "wb").write(_get(f"{source}/{d}.json"))
            raw[d] = json.load(open(path, encoding="utf-8"))
    for d, snap in raw.items():
        spy = snap.get("symbols", {}).get("SPY")
        if spy and spy.get("spot"):
            days[d] = spy
    return dict(sorted(days.items()))


def load_vix() -> dict:
    text = _get(VIX_URL).decode()
    out = {}
    for line in text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 5:
            m, d, y = parts[0].split("/")
            out[f"{y}-{int(m):02d}-{int(d):02d}"] = float(parts[4])
    return out


def mid(row) -> float | None:
    """Mid of [strike, bid, ask, volume, oi], or None if the quote is not usable."""
    _, bid, ask = row[0], row[1], row[2]
    if bid is None or ask is None or bid <= 0 or ask < bid:
        return None
    m = 0.5 * (bid + ask)
    return m if (ask - bid) <= MAX_SPREAD * m else None


def calls(snap: dict, expiry: str) -> dict:
    """strike -> mid for the calls of one expiry with usable quotes."""
    out = {}
    for row in snap["expiries"].get(expiry, {}).get("calls", []):
        m = mid(row)
        if m is not None:
            out[float(row[0])] = m
    return out


def implied_vol(price: float, s: float, k: float, tau: float) -> float | None:
    """Black-Scholes volatility (r = 0) that reproduces ``price``, by bisection."""
    intrinsic = max(s - k, 0.0)
    if tau <= 0 or price <= intrinsic + 1e-6 or price >= s:
        return None
    lo, hi = 1e-4, 5.0
    for _ in range(100):
        m = 0.5 * (lo + hi)
        if float(bs_call(np.array(s), k, tau, m)) > price:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


# ── policies ─────────────────────────────────────────────────────────────────
def export_policies(out: str = POLICIES) -> None:
    """The 40 trained agents (4 costs x 10 seeds) as plain arrays, for runners without checkpoints."""
    import torch.nn as nn

    from tools.hedging_experiment import load_agent

    arrays = {}
    for cost in COSTS:
        for seed in SEEDS:
            a = load_agent(cost, seed)
            key = f"c{cost:g}_s{seed}"
            lins = [m for m in a.ac.trunk if isinstance(m, nn.Linear)]
            for i, lin in enumerate(lins):
                arrays[f"{key}_w{i}"] = lin.weight.detach().numpy().astype(np.float32)
                arrays[f"{key}_b{i}"] = lin.bias.detach().numpy().astype(np.float32)
            arrays[f"{key}_wm"] = a.ac.policy_mean.weight.detach().numpy().astype(np.float32)
            arrays[f"{key}_bm"] = a.ac.policy_mean.bias.detach().numpy().astype(np.float32)
            arrays[f"{key}_mean"] = a.obs_rms.mean.astype(np.float32)
            arrays[f"{key}_std"] = np.sqrt(a.obs_rms.var + a.obs_rms.epsilon).astype(np.float32)
            arrays[f"{key}_clip"] = np.array(a.obs_rms.clip, dtype=np.float32)
            arrays[f"{key}_n"] = np.array(len(lins))
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({len(COSTS) * len(SEEDS)} policies)")


class NumpyHedger:
    """A trained hedging agent's deterministic policy, from data/hedging_policies.npz."""

    def __init__(self, arrays, cost: float, seed: int):
        k = f"c{cost:g}_s{seed}"
        self.p = {name[len(k) + 1:]: arrays[name] for name in arrays.files if name.startswith(k + "_")}
        self.n = int(self.p["n"])

    def hedge(self, obs: np.ndarray) -> np.ndarray:
        p = self.p
        h = np.clip((obs - p["mean"]) / p["std"], -float(p["clip"]), float(p["clip"]))
        for i in range(self.n):
            h = h @ p[f"w{i}"].T + p[f"b{i}"]
            if i < self.n - 1:                  # no tanh after the last trunk layer
                h = np.tanh(h)
        a = h @ p["wm"].T + p["bm"]
        return np.clip((np.tanh(a[:, 0]) + 1.0) / 2.0, 0.0, 1.0)

    def policy(self):
        return lambda env: self.hedge(env.obs())


def load_hedgers(path: str = POLICIES) -> dict:
    arrays = np.load(path)
    return {(c, s): NumpyHedger(arrays, c, s) for c in COSTS for s in SEEDS}


# ── part A ───────────────────────────────────────────────────────────────────
def part_a(days: dict, vix: dict) -> list:
    rows = []
    for d, snap in days.items():
        S = float(snap["spot"])
        exps = [(abs(trading_days_between(d, e) - LIFE), e) for e in snap["expiries"]
                if trading_days_between(d, e) >= 10]
        if not exps:
            continue
        e = min(exps)[1]
        n = trading_days_between(d, e)
        q = calls(snap, e)
        below = [k for k in q if k <= S]
        above = [k for k in q if k >= S]
        if not below or not above:
            continue
        k0, k1 = max(below), min(above)
        tau = n / TRADING_DAYS
        v0, v1 = implied_vol(q[k0], S, k0, tau), implied_vol(q[k1], S, k1, tau)
        if v0 is None or v1 is None:
            continue
        w = 0.0 if k1 == k0 else (S - k0) / (k1 - k0)
        atm = (1 - w) * v0 + w * v1
        row = {"date": d, "expiry": e, "trading_days": n, "spot": round(S, 2), "atm_iv": round(atm, 4)}
        if d in vix:
            model = VIX_TO_ATM * vix[d] / 100.0
            near = [k for k in q if abs(k / S - 1) <= MONEYNESS_BAND]
            errs = [float(bs_call(np.array(S), k, tau, model)) / q[k] - 1 for k in near]
            row.update({"vix": vix[d], "model_iv": round(model, 4), "ratio_to_vix": round(atm / (vix[d] / 100), 4),
                        "model_price_error_median": round(float(np.median(errs)), 4) if errs else None})
        rows.append(row)
    return rows


# ── part B ───────────────────────────────────────────────────────────────────
def part_b(days: dict, hedgers: dict) -> dict:
    dates = list(days)
    obs_rows, moves = [], []
    for d0 in dates:
        d1 = next_trading_day(d0)
        if d1 not in days:
            continue
        s0, s1 = days[d0], days[d1]
        S0, S1 = float(s0["spot"]), float(s1["spot"])
        for e in s0["expiries"]:
            n0 = trading_days_between(d0, e)
            if not 2 <= n0 <= LIFE:
                continue
            q0 = calls(s0, e)
            q1 = calls(s1, e) if e != d1 else {}
            for k, m0 in q0.items():
                if abs(k / S0 - 1) > MONEYNESS_BAND:
                    continue
                m1 = max(S1 - k, 0.0) if e == d1 else q1.get(k)
                if m1 is None:
                    continue
                iv0 = implied_vol(m0, S0, k, n0 / TRADING_DAYS)
                if iv0 is None:
                    continue
                delta = float(bs_delta(np.array(S0 / k), 1.0, n0 / TRADING_DAYS, iv0))
                obs_rows.append([math.log(S0 / k), n0 / LIFE, iv0, delta])
                moves.append((d0, S0 / k, S1 / k, m0 / k, m1 / k, delta))
    if not moves:
        return {"n": 0}
    obs = np.array(obs_rows, dtype=np.float32)
    ds = np.array([m[2] - m[1] for m in moves])
    dv = np.array([m[4] - m[3] for m in moves])
    pairs = sorted({m[0] for m in moves})
    hedges = {"none": np.zeros(len(moves)), "delta": np.array([m[5] for m in moves])}
    for s in SEEDS:
        hedges[f"agent_s{s}"] = hedgers[(0.0, s)].hedge(obs)
    err = {name: 100.0 * (h * ds - dv) for name, h in hedges.items()}
    rms = {name: float(np.sqrt(np.mean(e ** 2))) for name, e in err.items()}
    agent = [rms[f"agent_s{s}"] for s in SEEDS]
    return {
        "n": len(moves), "day_pairs": pairs, "rms": {k: round(v, 4) for k, v in rms.items()},
        "agent_rms_median": round(float(np.median(agent)), 4),
        "agent_vs_delta": round(float(np.median(agent)) / rms["delta"], 4),
        "agent_hedge_gap_to_delta": round(float(np.median([np.mean(np.abs(hedges[f'agent_s{s}'] - hedges['delta']))
                                                            for s in SEEDS])), 4),
    }


# ── part C ───────────────────────────────────────────────────────────────────
def episodes(days: dict) -> list:
    """Contracts quoted on every one of their last LIFE + 1 trading days."""
    out = []
    expiries = sorted({e for s in days.values() for e in s["expiries"]})
    for e in expiries:
        if e not in days:                   # need the spot on expiry day for the payoff
            continue
        path = [d for d in days if d < e and trading_days_between(d, e) <= LIFE]
        if not path or trading_days_between(path[0], e) != LIFE or len(path) != LIFE:
            continue
        S0 = float(days[path[0]]["spot"])
        common = set(calls(days[path[0]], e))
        for d in path[1:]:
            common &= set(calls(days[d], e))
        if not common:
            continue
        k = min(common, key=lambda x: abs(x / S0 - 1))
        S = [float(days[d]["spot"]) for d in path] + [float(days[e]["spot"])]
        V = [calls(days[d], e)[k] for d in path] + [max(S[-1] - k, 0.0)]
        sig = [implied_vol(v, s, k, trading_days_between(d, e) / TRADING_DAYS) or np.nan
               for v, s, d in zip(V[:-1], S[:-1], path)]
        sig = list(np.nan_to_num(np.array(sig), nan=float(np.nanmean(sig)))) + [sig[-1] if sig else 0.2]
        out.append({"expiry": e, "start": path[0], "strike": k,
                    "S": [x / k for x in S], "V": [x / k for x in V], "sigma": sig})
    return out


def part_c(days: dict, hedgers: dict, band_widths: dict) -> dict:
    eps = episodes(days)
    if not eps:
        return {"n": 0, "episodes": []}
    S = np.array([e["S"] for e in eps])
    V = np.array([e["V"] for e in eps])
    sig = np.array([e["sigma"] for e in eps])
    rows = []
    for cost in COSTS:
        spec = HedgeSpec(cost=cost)
        env = VecHedgeEnv(S, sig, spec, V=V)
        base = {name: summarize(run_policy(env, pol)) for name, pol in
                (("none", no_hedge), ("delta", delta_hedge), ("band", band_hedge(band_widths[cost])))}
        agents = [summarize(run_policy(env, hedgers[(cost, s)].policy()))["objective"] for s in SEEDS]
        rows.append({"cost": cost, "band_width": band_widths[cost],
                     "baselines": {k: round(v["objective"], 4) for k, v in base.items()},
                     "agent_median": round(float(np.median(agents)), 4),
                     "seeds_beating_band": int(sum(a > base["band"]["objective"] for a in agents))})
    return {"n": len(eps), "episodes": [{k: e[k] for k in ("expiry", "start", "strike")} for e in eps],
            "costs": rows}


def run(source: str = ARCHIVE_URL) -> dict:
    days = load_archive(source)
    vix = load_vix()
    hedgers = load_hedgers()
    real = json.load(open(os.path.join(ROOT, "docs", "assets", "hedging_real.json"), encoding="utf-8"))
    widths = {r["cost"]: r["band_width"] for r in real["costs"]}
    a = part_a(days, vix)
    out = {
        "generated": dt.date.today().isoformat(),
        "archive": {"first": next(iter(days)), "last": list(days)[-1], "days": len(days),
                    "source": "Danny-397/Tradeski data/options-archive (SPY)"},
        "vix_to_atm_assumed": VIX_TO_ATM,
        "part_a": {"rows": a, "ratio_median": round(float(np.median([r["ratio_to_vix"] for r in a if "ratio_to_vix" in r])), 4)
                   if any("ratio_to_vix" in r for r in a) else None},
        "part_b": part_b(days, hedgers),
        "part_c": part_c(days, hedgers, widths),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    pa, pb, pc = out["part_a"], out["part_b"], out["part_c"]
    print(f"archive {out['archive']['first']}..{out['archive']['last']} ({len(days)} days)")
    print(f"A: ATM IV / VIX median {pa['ratio_median']} (assumed {VIX_TO_ATM}) over {len(a)} days")
    if pb.get("n"):
        print(f"B: {pb['n']} one-day moves; RMS error none {pb['rms']['none']:.3f}, delta {pb['rms']['delta']:.3f}, "
              f"agent median {pb['agent_rms_median']:.3f} (agent/delta {pb['agent_vs_delta']:.3f})")
    print(f"C: {pc['n']} complete one-month episodes")
    print(f"wrote {OUT}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export-policies")
    r = sub.add_parser("run")
    r.add_argument("--archive", default=ARCHIVE_URL, help="archive directory or base URL")
    args = ap.parse_args()
    if args.cmd == "export-policies":
        export_policies()
    else:
        run(args.archive)


if __name__ == "__main__":
    main()
