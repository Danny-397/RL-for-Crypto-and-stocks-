"""RL-Trader backend API (Render-ready, featherweight).

Serves the trained PPO policy for *live* inference without PyTorch or ONNX — the
actor is a small MLP run with a few NumPy matmuls (weights in
``server/models/*.npz``, exported by ``tools/export_policy.py``). That keeps the
container tiny and cold-starts fast on Render's free tier.

Endpoints
---------
GET /health                         liveness + which policies loaded
GET /api/results                    the precomputed dashboard results
GET /api/live?market=&ticker=       fetch recent real prices, run the agent live,
                                    return its equity curve vs. buy-&-hold

The frontend works fully without this API (it ships a baked ``results.js``); the
API is an *optional* live-inference enhancement.
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

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_PATH = os.path.join(_REPO_ROOT, "docs", "results.js")
TICKERS = {
    "stock": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD"],
}

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
CORS(app)  # public, read-only API — allow any origin


# ── Policy (numpy forward pass) ─────────────────────────────────────────────
def _load_policy(market: str) -> dict:
    path = os.path.join(MODELS_DIR, f"ppo_{market}.npz")
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


_POLICIES: dict[str, dict] = {}
for _m in ("stock", "crypto"):
    try:
        _POLICIES[_m] = _load_policy(_m)
    except Exception as exc:  # pragma: no cover - missing model is non-fatal
        app.logger.warning("could not load %s policy: %s", _m, exc)


def policy_action(market: str, obs: np.ndarray) -> float:
    """Deterministic target position in [-1, 1] for a single observation."""
    p = _POLICIES[market]
    x = obs.reshape(1, -1).astype(np.float32)
    # Apply the exported observation normaliser (if the policy was trained with one).
    if "obs_mean" in p:
        clip = float(p["obs_clip"]) if "obs_clip" in p else 10.0
        x = np.clip((x - p["obs_mean"]) / p["obs_std"], -clip, clip).astype(np.float32)
    n = int(p["n_trunk"])
    for i in range(n):
        x = x @ p[f"w{i}"].T + p[f"b{i}"]
        if i < n - 1:
            x = np.tanh(x)
    action = np.tanh(x @ p["wm"].T + p["bm"])
    return float(action.reshape(-1)[0])


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


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify(status="ok", policies=sorted(_POLICIES), version="0.1.0")


@app.get("/api/results")
def api_results():
    """Return the precomputed dashboard results (parsed from docs/results.js)."""
    try:
        src = open(RESULTS_PATH, encoding="utf-8").read()
        obj = src[src.index("{"): src.rstrip().rstrip(";").rindex("}") + 1]
        return app.response_class(obj, mimetype="application/json")
    except Exception as exc:
        return jsonify(error=f"results unavailable: {exc}"), 503


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

    try:
        import yfinance as yf
        raw = yf.download(ticker, period="2y", interval="1d",
                          auto_adjust=True, progress=False)
        if getattr(raw.columns, "nlevels", 1) > 1:  # yfinance returns a MultiIndex
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
        if len(df) < 120:
            return jsonify(error=f"not enough data for {ticker}"), 422
    except Exception as exc:
        return jsonify(error=f"data fetch failed for {ticker}: {exc}"), 502

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


@app.get("/api/tickers")
def api_tickers():
    return jsonify(TICKERS)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
