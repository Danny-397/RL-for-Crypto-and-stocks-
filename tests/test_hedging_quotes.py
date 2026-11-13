"""The real-quote hedging test (tools/hedging_quotes.py), on synthetic archives."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from rl_trader.hedging.core import bs_call
from tools import hedging_quotes as hq


def test_trading_days_skip_weekends_and_holidays():
    assert hq.trading_days_between("2026-11-13", "2026-11-16") == 1           # Fri -> Mon
    assert hq.trading_days_between("2026-11-25", "2026-11-27") == 1           # Thanksgiving
    assert hq.next_trading_day("2026-12-24") == "2026-12-28"                  # Christmas Friday
    assert hq.trading_days_between("2026-11-17", "2026-12-18") == 22


def test_implied_vol_round_trips():
    for vol in (0.08, 0.2, 0.6):
        p = float(bs_call(np.array(100.0), 103.0, 20 / 252, vol))
        assert hq.implied_vol(p, 100.0, 103.0, 20 / 252) == pytest.approx(vol, abs=1e-6)
    assert hq.implied_vol(1.0, 100.0, 90.0, 0.1) is None                      # below intrinsic


def test_unusable_quotes_are_dropped():
    assert hq.mid([100, 1.0, 1.2, 0, 0]) == pytest.approx(1.1)
    assert hq.mid([100, 0.0, 0.2, 0, 0]) is None                               # no bid
    assert hq.mid([100, 1.0, 2.0, 0, 0]) is None                               # spread > 25% of mid


def test_numpy_hedger_matches_the_torch_agent(tmp_path, monkeypatch):
    import torch

    from rl_trader.hedging.core import OBS_DIM
    from rl_trader.hedging.train import HedgeTrainConfig
    from rl_trader.models.ppo_agent import PPOAgent

    torch.manual_seed(0)
    agent = PPOAgent(OBS_DIM, 1, HedgeTrainConfig().ppo)
    rng = np.random.default_rng(0)
    for _ in range(50):
        agent.observe(np.array([rng.normal(0, .05), rng.uniform(), rng.uniform(.1, .45), rng.uniform()]))
    monkeypatch.setattr("tools.hedging_experiment.load_agent", lambda c, s: agent)
    path = tmp_path / "p.npz"
    hq.export_policies(str(path))
    h = hq.load_hedgers(str(path))[(0.002, 3)]
    obs = np.array([[rng.normal(0, .05), rng.uniform(), rng.uniform(.1, .45), rng.uniform()] for _ in range(30)],
                   dtype=np.float32)
    torch_h = [float(np.clip((agent.select_action(o, deterministic=True)[0][0] + 1) / 2, 0, 1)) for o in obs]
    assert np.max(np.abs(h.hedge(obs) - np.array(torch_h))) < 1e-5


def _archive(start: str, n_days: int, expiry: str, vol: float = 0.2, spot0: float = 100.0):
    """A synthetic SPY archive: one expiry, strikes 90..110, priced at ``vol``."""
    days, d, S = {}, start, spot0
    rng = np.random.default_rng(0)
    for _ in range(n_days):
        tau = hq.trading_days_between(d, expiry) / hq.TRADING_DAYS
        rows = []
        for k in range(90, 111):
            p = float(bs_call(np.array(S), float(k), tau, vol)) if tau > 0 else max(S - k, 0)
            rows.append([float(k), round(p * 0.99, 4), round(p * 1.01, 4), 10, 100])
        days[d] = {"spot": S, "expiries": {expiry: {"calls": rows, "puts": []}} if d < expiry else {}}
        S *= float(np.exp(rng.normal(0, 0.01)))
        d = hq.next_trading_day(d)
    return days


def test_a_contract_seen_every_day_makes_one_episode():
    expiry = "2026-12-18"
    days = _archive("2026-11-17", 23, expiry)   # one spare day before the episode
    eps = hq.episodes(days)
    assert len(eps) == 1
    e = eps[0]
    assert hq.trading_days_between(e["start"], expiry) == hq.LIFE
    assert e["start"] == "2026-11-18" and len(e["S"]) == hq.LIFE + 1
    assert e["V"][-1] == pytest.approx(max(e["S"][-1] - 1.0, 0.0))            # payoff, strike units
    assert np.allclose(e["sigma"][:-1], 0.2, atol=0.01)                        # recovered from the mids


def test_a_missing_day_means_no_episode():
    days = _archive("2026-11-17", 23, "2026-12-18")
    days.pop("2026-12-01")
    assert hq.episodes(days) == []


def test_one_day_errors_favour_delta_over_no_hedge():
    days = _archive("2026-11-17", 6, "2026-12-18")

    class Fixed:
        def hedge(self, obs):
            return np.full(len(obs), 0.5)

    hedgers = {(c, s): Fixed() for c in hq.COSTS for s in hq.SEEDS}
    b = hq.part_b(days, hedgers)
    assert b["n"] > 0
    assert b["rms"]["delta"] < b["rms"]["none"]
    # Quotes from the model the delta comes from: daily rebalancing leaves only the
    # gamma error of 1% moves, a fraction of the unhedged error.
    assert b["rms"]["delta"] < 0.25 * b["rms"]["none"]


def test_part_c_scores_every_cost(monkeypatch):
    days = _archive("2026-11-17", 23, "2026-12-18")

    class Fixed:
        def policy(self):
            return lambda env: env.delta

    hedgers = {(c, s): Fixed() for c in hq.COSTS for s in hq.SEEDS}
    c = hq.part_c(days, hedgers, {cost: 0.0 for cost in hq.COSTS})
    assert c["n"] == 1 and [r["cost"] for r in c["costs"]] == list(hq.COSTS)
    # A policy that is delta scores exactly as the delta baseline.
    assert all(r["agent_median"] == pytest.approx(r["baselines"]["delta"]) for r in c["costs"])
    assert dt.date.fromisoformat(c["episodes"][0]["expiry"])
