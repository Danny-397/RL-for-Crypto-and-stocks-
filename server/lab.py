"""Experiment construction and runners for the lab endpoints.

This module turns a JSON experiment config into a real environment + policy and
runs it, keeping :mod:`server.app` a thin router.

One behaviour is called out explicitly everywhere it applies, because it is the
easiest thing on the whole site to accidentally misrepresent:

    **At inference the reward function is an observer, not a driver.**

The deployed policy is a fixed function of the observation. Changing the reward
formulation changes the reward *reported* at each bar, but it cannot change the
agent's actions — the agent is not learning here. Transaction cost and slippage
*do* change the trajectory, because they change the account state that feeds
back into the next observation. Every rollout response carries a note saying so,
so a visitor who flips the reward selector and sees an unchanged equity curve
learns why instead of assuming the control is broken.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rl_trader.config.training_config import crypto_config, stock_config
from rl_trader.data.data_loader import market_data_from_df
from rl_trader.envs import make_env

from . import regimes
from .experiments import Experiment, progress_reporter
from .rollout import counterfactual, observation_detail, run_trace

REWARD_KINDS = ("return", "dsr")
EVALUATION_MODES = ("historical", "synthetic")

INFERENCE_NOTE = (
    "The policy is fixed and not learning here. Reward formulation affects the "
    "reward reported per bar, not the actions taken. Transaction cost and "
    "slippage do change the trajectory, because they alter the account state fed "
    "back into the next observation."
)


# --------------------------------------------------------------------------- #
# Config parsing                                                               #
# --------------------------------------------------------------------------- #
def _num(value, default: float, lo: float, hi: float) -> float:
    try:
        return float(min(hi, max(lo, float(value))))
    except (TypeError, ValueError):
        return default


def parse_config(payload: Dict[str, Any], available_markets: List[str]) -> Dict[str, Any]:
    """Validate and normalise an experiment config from untrusted JSON."""
    market = str(payload.get("market", "stock")).lower()
    if market not in available_markets:
        raise ValueError(
            f"unknown market {market!r}; available: {sorted(available_markets)}"
        )

    mode = str(payload.get("mode", "historical")).lower()
    if mode not in EVALUATION_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {EVALUATION_MODES}")

    reward_kind = str(payload.get("reward", "return")).lower()
    if reward_kind not in REWARD_KINDS:
        raise ValueError(f"unknown reward {reward_kind!r}; expected one of {REWARD_KINDS}")

    # Resolve every mechanic to a concrete value against the market preset, rather
    # than carrying an "unspecified" sentinel. The config this returns is what the
    # receipt publishes and what /config hands back, so it must round-trip
    # exactly: re-submitting it has to rebuild an identical environment.
    preset = _base_config(market)
    cfg: Dict[str, Any] = {
        "market": market,
        "mode": mode,
        "reward": reward_kind,
        "initial_balance": _num(
            payload.get("initial_balance", preset.env.initial_balance),
            preset.env.initial_balance, 100.0, 1e9,
        ),
        "transaction_cost": _num(
            payload.get("transaction_cost", preset.env.transaction_cost),
            preset.env.transaction_cost, 0.0, 0.01,
        ),
        "slippage": _num(
            payload.get("slippage", preset.env.slippage),
            preset.env.slippage, 0.0, 0.01,
        ),
        "allow_short": bool(payload.get("allow_short", preset.env.allow_short)),
        "max_position": _num(
            payload.get("max_position", preset.env.max_position),
            preset.env.max_position, 0.0, 1.0,
        ),
    }
    if mode == "historical":
        cfg["ticker"] = str(payload.get("ticker", "")).upper().strip()
        if not cfg["ticker"]:
            raise ValueError("historical mode requires a ticker")
    else:
        regime = str(payload.get("regime", "random_walk")).lower()
        known = {r["key"] for r in regimes.list_regimes()}
        if regime not in known:
            raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(known)}")
        cfg["regime"] = regime
        cfg["seed"] = int(_num(payload.get("seed", 0), 0, 0, 2**31 - 1))
        cfg["n_steps"] = int(_num(payload.get("n_steps", 650), 650, 200, 2000))
    return cfg


def _base_config(market: str):
    return crypto_config() if market == "crypto" else stock_config()


def apply_config(cfg_obj, config: Dict[str, Any]):
    """Overlay the request's mechanics onto a market preset.

    Every value in ``config`` is concrete by construction (see
    :func:`parse_config`), so this is a straight assignment — no sentinels, and
    no branch that could make a replayed config diverge from the original.
    """
    cfg_obj.env.initial_balance = config["initial_balance"]
    cfg_obj.env.transaction_cost = config["transaction_cost"]
    cfg_obj.env.slippage = config["slippage"]
    cfg_obj.env.allow_short = config["allow_short"]
    cfg_obj.env.max_position = config["max_position"]
    cfg_obj.reward.kind = config["reward"]
    return cfg_obj


def _frame_hash(df: pd.DataFrame) -> str:
    """Stable content hash of the price series backing an experiment."""
    closes = np.asarray(df["close"], dtype=np.float64)
    return hashlib.sha256(closes.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Environment construction                                                     #
# --------------------------------------------------------------------------- #
def build_environment(
    config: Dict[str, Any], fetch_ohlcv, market_index
) -> Tuple[Any, Any, Optional[List[str]], Dict[str, Any]]:
    """Build the environment described by ``config``.

    ``fetch_ohlcv`` / ``market_index`` are injected so this module stays free of
    network concerns (and so tests can drive it with local frames).

    Returns ``(env, cfg_obj, dates, meta)``.
    """
    market = config["market"]
    cfg_obj = apply_config(_base_config(market), config)

    if config["mode"] == "historical":
        df, err = fetch_ohlcv(config["ticker"])
        if df is None:
            raise ValueError(err or f"no price data for {config['ticker']}")
        dates = [str(d)[:10] for d in df.index] if df.index is not None else None
        # Cross-asset context must match training, or four features silently read 0.
        idx = market_index(market)
        frame = df.copy()
        if idx is not None:
            frame["_mkt_close"] = idx["close"].reindex(frame.index).ffill().bfill()
        data = market_data_from_df(frame.reset_index(drop=True))
        # add_technical_indicators drops warm-up rows; realign the date labels.
        if dates is not None:
            dates = dates[len(dates) - len(data.prices):]
        meta = {
            "source": "real",
            "synthetic": False,
            "ticker": config["ticker"],
            "bars": int(len(data.prices)),
            "dataset_hash": _frame_hash(df),
            "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": "Yahoo Finance (yfinance), ~2y daily bars",
        }
    else:
        data, regime_meta = regimes.build_regime_data(
            config["regime"], seed=config["seed"], n_steps=config["n_steps"]
        )
        dates = None
        meta = {
            "source": "synthetic",
            "synthetic": True,
            "bars": int(len(data.prices)),
            "dataset_hash": hashlib.sha256(
                np.asarray(data.prices, dtype=np.float64).tobytes()
            ).hexdigest()[:16],
            **regime_meta,
        }

    env = make_env(market, data, cfg_obj.env, cfg_obj.reward, random_start=False)
    return env, cfg_obj, dates, meta


# --------------------------------------------------------------------------- #
# Runners                                                                      #
# --------------------------------------------------------------------------- #
def make_rollout_runner(policy, config, fetch_ohlcv, market_index):
    """Runner for a single full-trace episode (Agent Playground + X-Ray)."""

    def run(exp: Experiment) -> Dict[str, Any]:
        exp.stage = "building environment"
        env, cfg_obj, dates, meta = build_environment(config, fetch_ohlcv, market_index)
        exp.provenance.update(meta)
        exp.provenance["policy"] = policy.describe()
        exp.provenance["env"] = {
            "window_size": cfg_obj.env.window_size,
            "transaction_cost": cfg_obj.env.transaction_cost,
            "slippage": cfg_obj.env.slippage,
            "initial_balance": cfg_obj.env.initial_balance,
            "allow_short": cfg_obj.env.allow_short,
            "max_position": cfg_obj.env.max_position,
            "reward_kind": cfg_obj.reward.kind,
        }

        exp.stage = "running episode"
        report = progress_reporter(exp, total=max(1, meta["bars"]), stage="stepping")
        trace = run_trace(
            policy,
            env,
            market=config["market"],
            dates=dates,
            initial_balance=cfg_obj.env.initial_balance,
        )
        report(meta["bars"])

        out = trace.to_dict()
        out["meta"] = meta
        out["feature_names"] = list(env.data.feature_names)
        out["obs_dim"] = cfg_obj.env.window_size * env.data.features.shape[1] + 3
        out["window_size"] = cfg_obj.env.window_size
        out["inference_note"] = INFERENCE_NOTE
        if not trace.value_available:
            out["value_note"] = (
                "This policy archive contains no critic head, so no value estimate "
                "is shown. It is omitted rather than approximated."
            )
        return out

    return run


def make_counterfactual_runner(policy, config, step, actions, horizon, fetch_ohlcv, market_index):
    """Runner for the 'what if?' panel at one bar of an episode."""

    def run(exp: Experiment) -> Dict[str, Any]:
        exp.stage = "building environment"
        env, cfg_obj, dates, meta = build_environment(config, fetch_ohlcv, market_index)
        exp.provenance.update(meta)
        exp.provenance["policy"] = policy.describe()

        exp.stage = "replaying alternatives"
        out = counterfactual(
            policy,
            env,
            target_step=step,
            actions=actions,
            market=config["market"],
            horizon=horizon,
            dates=dates,
        )
        out["meta"] = meta
        out["inference_note"] = INFERENCE_NOTE
        return out

    return run


def make_shift_runner(policy, config, regime_keys, seeds, fetch_ohlcv, market_index):
    """Runner for a distribution-shift sweep across synthetic regimes.

    The same fixed policy meets several controlled distributions, several seeds
    each. Progress is the real completed-run count.
    """

    def run(exp: Experiment) -> Dict[str, Any]:
        total = max(1, len(regime_keys) * len(seeds))
        report = progress_reporter(exp, total=total, stage="evaluating regimes")
        exp.provenance["policy"] = policy.describe()

        rows: List[dict] = []
        done = 0
        for key in regime_keys:
            per_seed = []
            for seed in seeds:
                local = dict(config)
                local.update({"mode": "synthetic", "regime": key, "seed": int(seed)})
                env, cfg_obj, _dates, meta = build_environment(
                    local, fetch_ohlcv, market_index
                )
                trace = run_trace(
                    policy,
                    env,
                    market=config["market"],
                    initial_balance=cfg_obj.env.initial_balance,
                )
                per_seed.append(
                    {
                        "seed": int(seed),
                        "agent_return": trace.metrics["total_return"],
                        "benchmark_return": trace.bench_metrics["total_return"],
                        "sharpe": trace.metrics["sharpe"],
                        "max_drawdown": trace.metrics["max_drawdown"],
                        "excess": round(
                            trace.metrics["total_return"]
                            - trace.bench_metrics["total_return"],
                            6,
                        ),
                        "realised_autocorr": meta["realised"]["return_autocorr_lag1"],
                    }
                )
                done += 1
                report(done)

            agent = np.array([r["agent_return"] for r in per_seed], dtype=float)
            excess = np.array([r["excess"] for r in per_seed], dtype=float)
            rows.append(
                {
                    "regime": key,
                    "label": next(
                        (r["label"] for r in regimes.list_regimes() if r["key"] == key), key
                    ),
                    "n_seeds": len(per_seed),
                    "mean_agent_return": round(float(agent.mean()), 6),
                    "mean_excess_return": round(float(excess.mean()), 6),
                    "std_excess": round(float(excess.std(ddof=1)), 6) if len(excess) > 1 else 0.0,
                    "per_seed": per_seed,
                }
            )

        return {
            "market": config["market"],
            "regimes": rows,
            "synthetic": True,
            "note": (
                "Controlled synthetic distributions, not realistic market simulators. "
                "They exist to measure how a fixed policy behaves as the distribution "
                "moves away from what it was trained on."
            ),
            "inference_note": INFERENCE_NOTE,
        }

    return run


def xray_at(config, step: int, fetch_ohlcv, market_index) -> Dict[str, Any]:
    """The full observation the agent consumed at one step of an episode."""
    env, cfg_obj, dates, meta = build_environment(config, fetch_ohlcv, market_index)
    env.reset()
    first_t = env.t
    t = min(max(first_t, first_t + int(step)), len(env.data.prices) - 1)
    detail = observation_detail(
        env, t=t, feature_names=env.data.feature_names, window=cfg_obj.env.window_size
    )
    detail["step"] = int(step)
    detail["date"] = dates[t] if dates and t < len(dates) else None
    detail["price"] = round(float(env.data.prices[t]), 6)
    detail["meta"] = meta
    detail["scaling_note"] = (
        "Feature values are standardised (z-scored) exactly as the agent receives "
        "them, not raw indicator levels."
    )
    return detail
