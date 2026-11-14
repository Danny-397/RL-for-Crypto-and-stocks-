"""The pure parts of the 2026-11-14 depth study (docs/preregistration/2026-11-14-depth.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.basket_walk_forward import fold_dates
from tools.real_significance import _behaviour, _robust


def _days(start, n):
    return list(pd.date_range(start, periods=n, freq="D"))


def test_folds_sit_on_the_window_every_ticker_trades_in():
    early, late = _days("2020-01-01", 1000), _days("2021-01-01", 634)   # both end 2022-09-26
    bounds = fold_dates([early, late], 4)
    assert len(bounds) == 4
    # First test date: 40% into the common window (late's 634 days), not into early's.
    assert bounds[0][0] == late[int(634 * 0.4)]
    assert bounds[-1][1] is None
    starts = [b[0] for b in bounds]
    assert starts == sorted(starts)
    for (lo, hi), (nlo, _) in zip(bounds, bounds[1:]):
        assert hi == nlo                      # blocks are contiguous and disjoint


def test_folds_refuse_a_window_too_short_to_split():
    with pytest.raises(ValueError):
        fold_dates([_days("2020-01-01", 5)], 4)


def test_behaviour_of_a_policy_that_never_trades():
    b = _behaviour(np.zeros(50), np.full(50, 0.01))
    assert b["frac_flat"] == 1.0 and b["pos_abs"] == 0.0 and b["turnover"] == 0.0
    assert np.isnan(b["hit_rate"])


def test_behaviour_of_a_policy_that_flips_every_day():
    pos = np.tile([1.0, -1.0], 25)
    move = np.tile([0.01, 0.01], 25)                       # the asset only rises
    b = _behaviour(pos, move)
    assert b["pos_mean"] == pytest.approx(0.0)
    assert b["pos_abs"] == pytest.approx(1.0)
    assert b["frac_long"] == b["frac_short"] == 0.5
    assert b["turnover"] == pytest.approx(2.0)
    assert b["hit_rate"] == pytest.approx(0.5)


def test_behaviour_clips_actions_like_the_environment():
    b = _behaviour(np.full(10, 3.0), np.full(10, -0.01))
    assert b["pos_abs"] == 1.0 and b["hit_rate"] == 0.0


def test_robust_counts_runs_that_beat_the_benchmark():
    runs = np.array([[0.1, -0.2], [0.3, -0.1], [0.2, 0.5]])  # 3 seeds x 2 tickers
    bh = np.array([0.0, 0.0])
    r = _robust(runs, bh)
    assert r["p_improve"] == pytest.approx(4 / 6, abs=1e-3)
    assert r["tickers_behind"] == 0                          # ticker 2 averages +0.067
    assert r["p_improve_low"] <= r["p_improve"] <= r["p_improve_high"]
    assert r["seed_means"]["n"] == 3
