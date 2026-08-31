"""RL-Trader backend API — the experiment engine behind the interactive lab.

Serves the trained PPO policies for live inference without PyTorch: the actor is
a small MLP run with a few NumPy matmuls (weights in ``server/models/*.npz``,
exported by ``tools/export_policy.py``). That keeps the container tiny and
cold-starts fast on Render's free tier.

Endpoints
---------
Dashboard (unchanged, the static site depends on these)
    GET  /health                          liveness + loaded policy capabilities
    GET  /api/results                     the precomputed dashboard results
    GET  /api/live?market=&ticker=        quick agent-vs-benchmark equity curve
    GET  /api/tickers                     suggested tickers per market

Lab
    GET  /api/meta                        capabilities, provenance, what is live
    GET  /api/regimes                     synthetic distribution-shift regimes
    GET  /api/datasets                    real committed per-seed datasets
    GET  /api/generalization              real single-path vs domain-random results
    POST /api/statistics                  live bootstrap / permutation inference
    POST /api/experiments                 create an experiment (async)
    GET  /api/experiments                 list recent experiments
    GET  /api/experiments/<id>            status, progress, result, receipt
    GET  /api/experiments/<id>/config     the exact config needed to reproduce
    GET  /api/experiments/<id>/xray?step= the full observation at one bar

What is and is not live
-----------------------
Rollouts, counterfactuals, distribution-shift sweeps and all statistical
inference run **live** on request. Training does not: this container has no
PyTorch and a fraction of a CPU, and every seed in a multi-seed study is a full
PPO run. Training-derived results are therefore served from the repository's real
committed experiments, labelled with their source and the command that
regenerates them. ``/api/meta`` states this explicitly so the frontend never has
to guess.

The frontend works fully without this API (it ships a baked ``results.js``); the
API is what turns the page into a laboratory.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

# Make the repo's `rl_trader` package importable without installing it (and
# without pulling in PyTorch — only the numpy/pandas/gymnasium layers are used).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rl_trader.config.training_config import crypto_config, stock_config  # noqa: E402
from rl_trader.data.data_loader import market_data_from_df  # noqa: E402
from rl_trader.envs import make_env  # noqa: E402
from rl_trader.evaluation.evaluate_agent import ANNUALISATION, compute_metrics  # noqa: E402
from server import lab, precomputed, regimes, stats_api  # noqa: E402
from server.experiments import ExperimentManager, code_version  # noqa: E402
from server.policy import load_policies  # noqa: E402

RESULTS_PATH = os.path.join(_REPO_ROOT, "docs", "results.js")
TICKERS = {
    "stock": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD"],
}
API_VERSION = "0.2.0"

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
CORS(app)  # public, read-only API — allow any origin

_POLICIES = load_policies()
MANAGER = ExperimentManager()


def policy_action(market: str, obs: np.ndarray) -> float:
    """Deterministic target position in [-1, 1] for a single observation."""
    return _POLICIES[market].act(obs)


# ── Caching helpers ─────────────────────────────────────────────────────────
_CACHE: dict[str, tuple[float, dict]] = {}
_TTL = 1800  # 30 min — recent daily bars don't change intraday


def _cached(key: str):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    return None


def _store(key: str, value: dict) -> dict:
    _CACHE[key] = (time.time(), value)
    return value


def _downsample(arr, n: int = 160) -> list:
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= n:
        return [round(float(v), 4) for v in arr]
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return [round(float(v), 4) for v in arr[idx]]


# ── Price data ──────────────────────────────────────────────────────────────
try:
    from curl_cffi import requests as _cffi_requests
    _YF_SESSION = _cffi_requests.Session(impersonate="chrome")  # dodge Yahoo bot rate-limits
except Exception:  # pragma: no cover - curl_cffi is optional
    _YF_SESSION = None


def _fetch_ohlcv(ticker: str, attempts: int = 3):
    """Download ~2y of daily OHLCV, retrying transient yfinance failures.

    Returns ``(dataframe, None)`` on success or ``(None, message)`` on failure. A
    browser-impersonating curl_cffi session (when available) plus a few retries
    smooth over the rate-limiting / empty-response flakiness that otherwise makes
    arbitrary-ticker lookups fail intermittently on the free tier.
    """
    import yfinance as yf
    cols = ["open", "high", "low", "close", "volume"]
    last = "no data returned"
    for i in range(attempts):
        try:
            kw = dict(period="2y", interval="1d", auto_adjust=True, progress=False, threads=False)
            try:
                raw = yf.download(ticker, session=_YF_SESSION, **kw) if _YF_SESSION else yf.download(ticker, **kw)
            except TypeError:  # this yfinance build doesn't accept a session kwarg
                raw = yf.download(ticker, **kw)
            if getattr(raw.columns, "nlevels", 1) > 1:  # yfinance MultiIndex
                raw.columns = raw.columns.get_level_values(0)
            df = raw.rename(columns=str.lower)
            if all(c in df.columns for c in cols):
                df = df[cols].dropna()
                # need enough rows to survive the ~120-bar feature warm-up + a window
                if len(df) >= 200:
                    return df, None
                last = f"only {len(df)} rows of history for {ticker}"
            else:
                last = f"no price data returned for {ticker}"
        except Exception as exc:  # pragma: no cover - network
            last = f"fetch error: {str(exc)[:100]}"
        if i < attempts - 1:
            time.sleep(0.7 * (i + 1))
    return None, last


_INDEX_CACHE: dict = {}


def _market_index(market: str):
    """Fetch (and briefly cache) the reference index OHLCV for cross-asset features."""
    tk = "SPY" if market == "stock" else "BTC-USD"
    hit = _INDEX_CACHE.get(tk)
    if hit and (time.time() - hit[0]) < 1800:  # 30-min TTL
        return hit[1]
    df, _err = _fetch_ohlcv(tk)
    if df is not None:
        _INDEX_CACHE[tk] = (time.time(), df)
    return df


# ── Dashboard routes (unchanged contract) ───────────────────────────────────
@app.get("/health")
def health():
    return jsonify(
        status="ok",
        policies=sorted(_POLICIES),
        version=API_VERSION,
        code_version=code_version(),
        capabilities={
            name: {"value_head": p.has_value, "obs_dim": p.obs_dim}
            for name, p in _POLICIES.items()
        },
    )


@app.get("/api/results")
def api_results():
    """Return the precomputed dashboard results (parsed from docs/results.js)."""
    try:
        src = open(RESULTS_PATH, encoding="utf-8").read()
        obj = src[src.index("{"): src.rstrip().rstrip(";").rindex("}") + 1]
        return app.response_class(obj, mimetype="application/json")
    except Exception as exc:
        return jsonify(error=f"results unavailable: {exc}"), 503


@app.get("/api/tickers")
def api_tickers():
    return jsonify(TICKERS)


@app.get("/api/live")
def api_live():
    """Run the trained agent live on recent real prices for one ticker."""
    market = request.args.get("market", "stock").lower()
    ticker = request.args.get("ticker", TICKERS.get(market, ["AAPL"])[0]).upper()
    if market not in _POLICIES:
        return jsonify(error=f"unknown/unloaded market '{market}'"), 400

    cache_key = f"{market}:{ticker}"
    cached = _cached(cache_key)
    if cached:
        return jsonify(cached)

    df, err = _fetch_ohlcv(ticker)
    if df is None:
        code = 422 if "rows of history" in err else 502
        return jsonify(error=err), code

    idx = _market_index(market)
    if idx is not None:
        df = df.copy()
        df["_mkt_close"] = idx["close"].reindex(df.index).ffill().bfill()

    cfg = crypto_config() if market == "crypto" else stock_config()
    data = market_data_from_df(df.reset_index(drop=True))
    env = make_env(market, data, cfg.env, cfg.reward, random_start=False)

    obs, info = env.reset()
    equity = [info["equity"]]
    actions = []
    done = False
    while not done:
        a = policy_action(market, obs)
        obs, _, term, trunc, info = env.step(np.array([a], dtype=np.float32))
        equity.append(info["equity"])
        actions.append(round(a, 3))
        done = term or trunc

    equity = np.asarray(equity, dtype=float)
    w = cfg.env.window_size
    prices = data.prices[w - 1:]
    bench = cfg.env.initial_balance * (prices / prices[0])
    periods = ANNUALISATION.get(market, 252)

    result = {
        "market": market, "ticker": ticker,
        "latest_action": actions[-1] if actions else 0.0,
        "equity_agent": _downsample(equity),
        "equity_bench": _downsample(bench),
        "metrics": {k: round(float(v), 4) for k, v in compute_metrics(equity, periods).items()},
        "bench_metrics": {k: round(float(v), 4) for k, v in compute_metrics(bench, periods).items()},
        "n_days": len(actions),
    }
    return jsonify(_store(cache_key, result))


# ── Lab: capability discovery ───────────────────────────────────────────────
@app.get("/api/meta")
def api_meta():
    """What this backend can actually do — so the UI never has to guess."""
    return jsonify(
        version=API_VERSION,
        code_version=code_version(),
        policies={name: p.describe() for name, p in _POLICIES.items()},
        markets=sorted(_POLICIES),
        reward_kinds=list(lab.REWARD_KINDS),
        evaluation_modes=list(lab.EVALUATION_MODES),
        tickers=TICKERS,
        live={
            "rollout": True,
            "counterfactual": True,
            "distribution_shift": True,
            "statistics": True,
            "training": False,
        },
        training_note=(
            "This backend does not train. It has no PyTorch and a fraction of a "
            "CPU, and each seed of a multi-seed study is a full PPO run. "
            "Training-derived results are served from the repository's real "
            "committed experiments, each labelled with the command that "
            "regenerates it."
        ),
        inference_note=lab.INFERENCE_NOTE,
        results_provenance=precomputed.results_provenance(),
        experiments=MANAGER.stats(),
    )


@app.get("/api/regimes")
def api_regimes():
    return jsonify(regimes=regimes.list_regimes(), synthetic=True)


@app.get("/api/datasets")
def api_datasets():
    """Real, committed per-seed and per-ticker datasets available for analysis."""
    return jsonify(
        seed_datasets=precomputed.dataset_catalog(),
        paired_asset_datasets=[
            {
                "key": d["key"],
                "label": d["label"],
                "market": d["market"],
                "n_pairs": d["n_pairs"],
                "axis": d["axis"],
                "source": d["source"],
                "caveat": d["caveat"],
            }
            for d in precomputed.paired_asset_datasets().values()
        ],
        headline_single_seed=precomputed.headline_single_seed(),
        note=(
            "These are real results committed to the repository, not live runs. "
            "Statistical inference over them runs live."
        ),
    )


@app.get("/api/generalization")
def api_generalization():
    """The real single-path vs domain-randomized ablation (Agent A / Agent B)."""
    out = precomputed.generalization_results()
    if out is None:
        return jsonify(error="ablation results unavailable"), 503
    return jsonify(out)


# ── Lab: live statistics ────────────────────────────────────────────────────
@app.post("/api/statistics")
def api_statistics():
    """Recompute real statistics on demand over a real dataset.

    Accepts either a ``dataset`` key (from ``/api/datasets``) or explicit
    ``values``. Bootstrap and permutation parameters are caller-controlled, so a
    visitor can watch a genuine p-value move as the design changes.
    """
    payload = request.get_json(silent=True) or {}
    n_boot = payload.get("n_boot", 10_000)
    n_perm = payload.get("n_perm", 20_000)
    confidence = payload.get("confidence", 0.95)
    seed = int(payload.get("seed", 0))

    try:
        # Two caller-supplied arms, paired element-wise. Used for arm-vs-arm
        # comparisons where both were trained on the *same* seed set (e.g. the
        # ablation), which is a genuinely paired design — testing one arm against
        # the other's scalar mean would be a weaker and different question.
        if payload.get("values_a") is not None or payload.get("values_b") is not None:
            a, b = payload.get("values_a"), payload.get("values_b")
            if not (isinstance(a, list) and isinstance(b, list) and a and b):
                return jsonify(error="'values_a' and 'values_b' must be non-empty lists"), 400
            if len(a) > 1000 or len(b) > 1000:
                return jsonify(error="at most 1000 values per arm"), 400
            out = stats_api.compare(
                a, b, n_perm=n_perm, confidence=confidence, n_boot=n_boot, seed=seed,
                axis=str(payload.get("axis", "paired"))[:40],
            )
            out["live_computation"] = True
            return jsonify(out)

        key = payload.get("dataset")
        if key and key.startswith("assets:"):
            pa = precomputed.paired_asset_datasets().get(key)
            if pa is None:
                return jsonify(error=f"unknown dataset {key!r}"), 404
            out = stats_api.compare(
                pa["agent"], pa["benchmark"],
                n_perm=n_perm, confidence=confidence, n_boot=n_boot, seed=seed,
                axis=pa["axis"],
            )
            out.update({
                "dataset": key, "labels": pa["labels"], "source": pa["source"],
                "generated_by": pa["generated_by"], "caveat": pa["caveat"],
                "live_computation": True,
            })
            return jsonify(out)

        if key:
            ds = precomputed.seed_datasets().get(key)
            if ds is None:
                return jsonify(error=f"unknown dataset {key!r}"), 404
            values, benchmark = ds["values"], ds.get("benchmark")
            meta = {
                "dataset": key, "label": ds["label"], "source": ds["source"],
                "generated_by": ds["generated_by"], "published": ds.get("published"),
            }
        else:
            values = payload.get("values")
            if not isinstance(values, list) or not values:
                return jsonify(error="provide a 'dataset' key or a non-empty 'values' list"), 400
            if len(values) > 1000:
                return jsonify(error="at most 1000 values"), 400
            benchmark = payload.get("benchmark")
            meta = {"dataset": None, "label": "caller-supplied values"}

        out = stats_api.analyze(
            values, benchmark=benchmark, confidence=confidence,
            n_boot=n_boot, n_perm=n_perm, seed=seed,
        )
        out["distribution"] = stats_api.bootstrap_distribution(
            values, n_boot=min(int(n_boot), 5000), seed=seed
        )
        out.update(meta)
        out["live_computation"] = True
        return jsonify(out)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400


# ── Lab: experiments ────────────────────────────────────────────────────────
@app.post("/api/experiments")
def api_create_experiment():
    """Create and start an experiment. Returns its id immediately."""
    payload = request.get_json(silent=True) or {}
    kind = str(payload.get("kind", "rollout")).lower()

    try:
        config = lab.parse_config(payload.get("config", payload), sorted(_POLICIES))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    policy = _POLICIES.get(config["market"])
    if policy is None:
        return jsonify(error=f"no policy loaded for market {config['market']!r}"), 400

    if kind == "rollout":
        runner = lab.make_rollout_runner(policy, config, _fetch_ohlcv, _market_index)
    elif kind == "counterfactual":
        step = int(payload.get("step", 0))
        actions = payload.get("actions") or [1.0, 0.0, -1.0]
        if not isinstance(actions, list) or not 1 <= len(actions) <= 11:
            return jsonify(error="'actions' must be a list of 1-11 values"), 400
        try:
            actions = [max(-1.0, min(1.0, float(a))) for a in actions]
        except (TypeError, ValueError):
            return jsonify(error="'actions' must be numeric"), 400
        horizon = max(1, min(60, int(payload.get("horizon", 1))))
        runner = lab.make_counterfactual_runner(
            policy, config, step, actions, horizon, _fetch_ohlcv, _market_index
        )
    elif kind == "distribution_shift":
        keys = payload.get("regimes") or [r["key"] for r in regimes.list_regimes()]
        known = {r["key"] for r in regimes.list_regimes()}
        bad = [k for k in keys if k not in known]
        if bad:
            return jsonify(error=f"unknown regimes: {bad}"), 400
        seeds = payload.get("seeds") or [0, 1, 2]
        if not isinstance(seeds, list) or not 1 <= len(seeds) <= 10:
            return jsonify(error="'seeds' must be a list of 1-10 integers"), 400
        seeds = [int(s) for s in seeds]
        runner = lab.make_shift_runner(policy, config, keys, seeds, _fetch_ohlcv, _market_index)
    else:
        return jsonify(
            error=f"unknown experiment kind {kind!r}",
            supported=["rollout", "counterfactual", "distribution_shift"],
        ), 400

    exp = MANAGER.create(kind, config, runner, provenance={"api_version": API_VERSION})
    return jsonify(exp.summary()), 202


@app.get("/api/experiments")
def api_list_experiments():
    limit = request.args.get("limit", default=50, type=int) or 50
    return jsonify(
        experiments=MANAGER.list(limit=limit, kind=request.args.get("kind")),
        stats=MANAGER.stats(),
        storage="ephemeral — experiments live in memory and are lost on restart",
    )


@app.get("/api/experiments/<experiment_id>")
def api_get_experiment(experiment_id: str):
    exp = MANAGER.get(experiment_id.upper())
    if exp is None:
        return jsonify(error=f"no experiment {experiment_id!r}"), 404
    return jsonify(exp.full() if exp.status in ("done", "error") else exp.summary())


@app.get("/api/experiments/<experiment_id>/config")
def api_experiment_config(experiment_id: str):
    """Everything needed to reproduce this experiment."""
    exp = MANAGER.get(experiment_id.upper())
    if exp is None:
        return jsonify(error=f"no experiment {experiment_id!r}"), 404
    return jsonify(
        experiment_id=exp.id, kind=exp.kind, config=exp.config, receipt=exp.receipt()
    )


@app.get("/api/experiments/<experiment_id>/xray")
def api_experiment_xray(experiment_id: str):
    """The full observation the agent consumed at one bar of this experiment."""
    exp = MANAGER.get(experiment_id.upper())
    if exp is None:
        return jsonify(error=f"no experiment {experiment_id!r}"), 404
    step = request.args.get("step", default=0, type=int) or 0
    try:
        return jsonify(
            lab.xray_at(
                exp.config, step, _fetch_ohlcv, _market_index,
                policy=_POLICIES.get(exp.config.get("market")),
            )
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
