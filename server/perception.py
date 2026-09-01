"""Signal-or-noise: a controlled test of *human* pattern detection.

The project's central claim is that structure which looks obvious is usually not
there, and that telling the difference needs statistics rather than eyes. This
module lets a visitor test that on themselves, under a design strict enough to
mean something.

Two conditions
--------------
``synthetic``
    Half the charts are drawn with the AR(1) return autocorrelation the agents
    were actually **trained against** (:func:`rl_trader.data.data_loader.
    market_regime`); half come from the same generator with ``momentum = 0`` —
    pure geometric Brownian motion. So the question is precisely "can you see the
    signal the agent was built to find?"

``real``
    Half the charts are contiguous, disjoint slices of a real ticker's history;
    the other half are those *same* slices with their daily log returns randomly
    permuted. A permutation preserves the entire marginal distribution — mean,
    variance, skew, fat tails — and destroys only the temporal ordering. Any
    ability to tell them apart is therefore ability to see **time structure**
    (volatility clustering, autocorrelation), not distributional shape.

Confound control
----------------
A test whose classes differ in drift or volatility measures nothing about pattern
detection, because those are visible for free. So every chart's log returns are
standardised to an identical volatility and given a drift drawn from one common
distribution regardless of class. Standardisation is affine, so it leaves
autocorrelation — the only class-carrying property — exactly intact. Classes are
exactly balanced, then shuffled.

Statelessness
-------------
A quiz is a deterministic function of ``(difficulty, market/ticker, seed,
n_charts)``. The answer key is never sent to the client; scoring rebuilds the
identical quiz from those fields and compares. No server-side session, and
nothing on the wire to tamper with.

Honesty
-------
Scoring reports an exact two-sided binomial test, the smallest p-value the design
can attain at all (``2^(1-n)``), and its **power** against plausible true
accuracies. A handful of guesses cannot separate skill from luck — the same
resolution limit that governs this project's seed-level tests, met here by the
visitor personally.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from rl_trader.data.data_loader import generate_synthetic_ohlcv, market_regime

DIFFICULTIES = ("synthetic", "real")

CHART_BARS = 180            # bars per synthetic chart
MIN_CHARTS, MAX_CHARTS = 4, 12
TARGET_DAILY_SIGMA = 0.015  # every chart gets exactly this return volatility
DRIFT_SD = 0.0004           # per-chart drift, drawn identically for both classes
MIN_REAL_BARS = 40          # below this a slice is too short to judge


# --------------------------------------------------------------------------- #
# Exact binomial inference                                                     #
# --------------------------------------------------------------------------- #
def _pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def binomial_p_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value.

    The total probability of every outcome no more likely than the observed one.
    Computed exactly rather than by a normal approximation, because ``n`` here is
    tiny — precisely where an approximation would mislead.
    """
    if n <= 0:
        return 1.0
    observed = _pmf(k, n, p)
    total = sum(
        _pmf(j, n, p) for j in range(n + 1) if _pmf(j, n, p) <= observed * (1 + 1e-9)
    )
    return float(min(1.0, total))


def _rejection_set(n: int, alpha: float) -> List[int]:
    """Which scores out of ``n`` would actually be called significant."""
    return [k for k in range(n + 1) if binomial_p_two_sided(k, n) <= alpha]


def power_analysis(
    n: int, alpha: float = 0.05, accuracies: Sequence[float] = (0.6, 0.7, 0.8)
) -> dict:
    """Probability this design would detect a genuinely skilled guesser.

    Enumerated exactly over the binomial, not simulated. The point it makes is
    deliberately uncomfortable: at these sample sizes a real but moderate ability
    is very likely to go undetected, so failing to reach significance is not
    evidence of no skill.
    """
    reject = _rejection_set(n, alpha)
    return {
        "alpha": alpha,
        "n_trials": n,
        "min_attainable_p": round(binomial_p_two_sided(n, n), 12),
        "significant_scores": reject,
        "power": [
            {"true_accuracy": acc, "power": round(sum(_pmf(k, n, acc) for k in reject), 4)}
            for acc in accuracies
        ],
        "explanation": (
            f"With {n} charts, only scores of {reject or 'none'} out of {n} reach "
            f"p <= {alpha}. Every other score is indistinguishable from guessing, "
            "however confident it felt."
        ),
    }


# --------------------------------------------------------------------------- #
# Chart construction                                                           #
# --------------------------------------------------------------------------- #
def _lag1(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a return series (nan when undefined)."""
    if len(x) < 3 or x.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _standardise(log_returns: np.ndarray, drift: float) -> np.ndarray:
    """Force a common volatility and drift onto one chart's log returns.

    Affine, so the autocorrelation survives exactly while everything a viewer
    could otherwise read off the axis is erased.
    """
    lr = np.asarray(log_returns, dtype=float)
    lr = lr - lr.mean()
    scale = lr.std()
    if scale > 1e-12:
        lr = lr / scale
    return lr * TARGET_DAILY_SIGMA + drift


def _prices(log_returns: np.ndarray) -> np.ndarray:
    return 100.0 * np.exp(np.cumsum(np.insert(log_returns, 0, 0.0)))


def _balanced_labels(rng: np.random.Generator, n: int) -> np.ndarray:
    """Exactly ``n/2`` of each class, in random order."""
    labels = np.array([1] * (n // 2) + [0] * (n - n // 2), dtype=int)
    rng.shuffle(labels)
    return labels


def _synthetic_charts(
    rng: np.random.Generator, labels: np.ndarray, market: str
) -> Tuple[List[dict], dict]:
    vol, annual_drift, phi = market_regime(market)
    charts: List[dict] = []
    for i, label in enumerate(labels):
        sub_seed = int(rng.integers(0, 2 ** 31 - 1))
        chart_drift = float(rng.normal(0.0, DRIFT_SD))
        df = generate_synthetic_ohlcv(
            n_steps=CHART_BARS + 1,
            annual_vol=vol,
            annual_drift=annual_drift,
            momentum=float(phi) if label else 0.0,
            seed=sub_seed,
        )
        lr = np.diff(np.log(df["close"].to_numpy(dtype=float)))
        charts.append({"index": i, "label": int(label),
                       "log_returns": _standardise(lr, chart_drift)})

    meta = {
        "market": market,
        "signal_phi": round(float(phi), 4),
        "control_phi": 0.0,
        "bars_per_chart": CHART_BARS,
        "positive_class": "trending",
        "prompt": "Which of these have positively autocorrelated (trending) returns?",
        "design": (
            f"Signal charts use the AR(1) coefficient the {market} agents were "
            f"trained against (phi = {float(phi):.2f}); control charts use the same "
            "generator with phi = 0, a pure random walk. Every chart is then "
            "standardised to the same volatility and given a drift drawn from one "
            "common distribution, so neither scale nor volatility can leak the answer."
        ),
    }
    return charts, meta


def _real_charts(
    rng: np.random.Generator, labels: np.ndarray, ticker: str, fetch_ohlcv: Callable
) -> Tuple[List[dict], dict]:
    df, err = fetch_ohlcv(ticker)
    if df is None:
        raise ValueError(err or f"no price history available for {ticker}")
    all_lr = np.diff(np.log(np.asarray(df["close"], dtype=float)))
    n = len(labels)
    seg = len(all_lr) // n
    if seg < MIN_REAL_BARS:
        raise ValueError(
            f"{ticker} has only {len(all_lr)} returns — not enough for {n} disjoint "
            f"slices of at least {MIN_REAL_BARS} bars"
        )

    charts: List[dict] = []
    for i, label in enumerate(labels):
        chunk = all_lr[i * seg: (i + 1) * seg].copy()
        if not label:
            chunk = rng.permutation(chunk)  # same returns, ordering destroyed
        charts.append({
            "index": i,
            "label": int(label),
            "log_returns": _standardise(chunk, float(rng.normal(0.0, DRIFT_SD))),
        })

    meta = {
        "ticker": ticker,
        "bars_per_chart": int(seg),
        "positive_class": "real",
        "prompt": (
            "Which of these are real market history, and which are the same "
            "returns reshuffled?"
        ),
        "design": (
            f"Every chart is a disjoint slice of {ticker}'s own daily history. Half "
            "are left in their real order; half have their daily returns randomly "
            "permuted. A permutation keeps the exact same set of returns — same "
            "mean, variance, skew and fat tails — and destroys only the ordering, so "
            "the only thing separating the classes is time structure."
        ),
    }
    return charts, meta


# --------------------------------------------------------------------------- #
# Quiz assembly                                                                #
# --------------------------------------------------------------------------- #
def build_quiz(
    difficulty: str = "synthetic",
    seed: int = 0,
    n_charts: int = 8,
    market: str = "stock",
    ticker: Optional[str] = None,
    fetch_ohlcv: Optional[Callable] = None,
) -> dict:
    """Build one quiz, including its answer key under ``_key``.

    Anything serving this to a browser must strip ``_key`` — :func:`public` does
    exactly that.
    """
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty {difficulty!r}; expected {DIFFICULTIES}")
    n_charts = int(n_charts)
    if not MIN_CHARTS <= n_charts <= MAX_CHARTS:
        raise ValueError(f"n_charts must be between {MIN_CHARTS} and {MAX_CHARTS}")
    if n_charts % 2:
        raise ValueError("n_charts must be even so the two classes are exactly balanced")

    rng = np.random.default_rng(int(seed))
    labels = _balanced_labels(rng, n_charts)

    if difficulty == "synthetic":
        charts, meta = _synthetic_charts(rng, labels, market)
    else:
        if fetch_ohlcv is None:
            raise ValueError("the real-data condition needs a price fetcher")
        charts, meta = _real_charts(rng, labels, (ticker or "SPY").upper(), fetch_ohlcv)

    for c in charts:
        lr = c.pop("log_returns")
        c["autocorr_lag1"] = round(_lag1(lr), 4)
        c["prices"] = [round(float(v), 4) for v in _prices(lr)]

    signal = [c["autocorr_lag1"] for c in charts if c["label"] == 1]
    control = [c["autocorr_lag1"] for c in charts if c["label"] == 0]
    meta.update({
        "difficulty": difficulty,
        "seed": int(seed),
        "n_charts": n_charts,
        "synthetic": difficulty == "synthetic",
        # Measured on the charts actually generated, never the nominal parameter.
        "realised": {
            "mean_autocorr_signal": round(float(np.nanmean(signal)), 4),
            "mean_autocorr_control": round(float(np.nanmean(control)), 4),
            # At these sample sizes the two classes routinely overlap. Saying so is
            # part of the lesson, not a defect to hide.
            "classes_overlap": bool(np.nanmin(signal) < np.nanmax(control)),
        },
        "normalisation": (
            f"All charts standardised to a {TARGET_DAILY_SIGMA:.1%} daily return "
            "volatility, with a drift drawn from one shared distribution — so scale, "
            "volatility and overall trendiness cannot give the answer away."
        ),
    })
    return {"meta": meta, "charts": charts, "_key": [int(x) for x in labels]}


def public(quiz: dict) -> dict:
    """The quiz as served: charts with no labels and no answer key."""
    return {
        "meta": quiz["meta"],
        "charts": [{"index": c["index"], "prices": c["prices"]} for c in quiz["charts"]],
        "n_charts": len(quiz["charts"]),
        "note": (
            "The answer key is not sent to the browser. Answers are scored against "
            "a rebuild of this exact quiz from its seed."
        ),
    }


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def _autocorr_rule(quiz: dict) -> dict:
    """How a one-line statistic does on the same charts the visitor just saw.

    The classes are balanced by construction, so ranking charts by their sample
    lag-1 autocorrelation and calling the top half "signal" is a fully specified,
    fair rule — no fitting, no tuned threshold. It is the honest reference point:
    a quantity the eye cannot see, computed in one line.
    """
    charts = quiz["charts"]
    n = len(charts)
    acs = np.array([c["autocorr_lag1"] for c in charts], dtype=float)
    order = np.argsort(np.nan_to_num(acs, nan=-np.inf))[::-1]
    predicted = np.zeros(n, dtype=int)
    predicted[order[: n // 2]] = 1
    truth = np.array(quiz["_key"], dtype=int)
    correct = int((predicted == truth).sum())
    return {
        "name": "lag-1 autocorrelation ranking",
        "correct": correct,
        "n": n,
        "accuracy": round(correct / n, 4),
        "p_value": round(binomial_p_two_sided(correct, n), 6),
        "description": (
            "Rank every chart by its sample lag-1 return autocorrelation and call "
            "the top half signal. No parameters, no fitting."
        ),
        "caveat": (
            "This rule reads the same finite sample you did, so it is not an oracle "
            "— on a short slice it can rank a random walk above a trending path."
        ),
    }


def score_quiz(quiz: dict, answers: Sequence[int]) -> dict:
    """Score a submission against a rebuilt quiz and report honest inference."""
    key = quiz["_key"]
    n = len(key)
    if len(answers) != n:
        raise ValueError(f"expected {n} answers, got {len(answers)}")
    picks = [1 if int(a) else 0 for a in answers]
    per_chart = [
        {
            "index": i,
            "your_answer": picks[i],
            "truth": key[i],
            "correct": picks[i] == key[i],
            "autocorr_lag1": quiz["charts"][i]["autocorr_lag1"],
        }
        for i in range(n)
    ]
    correct = sum(1 for r in per_chart if r["correct"])
    p = binomial_p_two_sided(correct, n)

    return {
        "meta": quiz["meta"],
        "per_chart": per_chart,
        "correct": correct,
        "n": n,
        "accuracy": round(correct / n, 4),
        "expected_by_chance": n / 2.0,
        "p_value": round(p, 6),
        "significant_at_05": bool(p <= 0.05),
        "test": "exact two-sided binomial test against p = 0.5",
        "power": power_analysis(n),
        "reference": _autocorr_rule(quiz),
        "live_computation": True,
        "verdict": _verdict(correct, n, p),
    }


def _verdict(correct: int, n: int, p: float) -> str:
    """A plain reading that never claims more than ``n`` trials can support."""
    if p > 0.05 and correct > n / 2:
        return (
            f"{correct} of {n} is above chance, but not distinguishably so "
            f"(p = {p:.3f}). At this sample size that is the expected outcome even "
            "for someone genuinely skilled — see the power analysis."
        )
    if p > 0.05:
        return (
            f"{correct} of {n} is indistinguishable from guessing (p = {p:.3f}). "
            "That is the same problem the project's seed-level tests have: too few "
            "trials to resolve anything."
        )
    if correct > n / 2:
        return (
            f"{correct} of {n}, p = {p:.3f} — better than chance at this sample "
            "size. Worth repeating with a fresh seed before believing it: one "
            "significant result out of several attempts is a different claim."
        )
    return (
        f"{correct} of {n}, p = {p:.3f} — reliably *worse* than chance, which takes "
        "a consistently wrong rule rather than bad luck."
    )
