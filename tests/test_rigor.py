"""Tests for the RL-reporting statistics and the rigor tools' plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from rl_trader.evaluation.rigor import (
    aggregate_scores,
    holm,
    iqm,
    performance_profile,
    probability_of_improvement,
    two_sample_diff,
)


def test_iqm_matches_trimmed_mean_when_n_divisible_by_4():
    x = [1, 2, 3, 4, 5, 6, 7, 100]
    assert iqm(x) == pytest.approx(np.mean([3, 4, 5, 6]))


def test_iqm_weights_boundaries_for_ten_runs():
    x = list(range(10))                       # middle 50%: 2.5 .. 7.5
    expected = (0.5 * 2 + 3 + 4 + 5 + 6 + 0.5 * 7) / 5
    assert iqm(x) == pytest.approx(expected)


def test_iqm_ignores_one_outlier_seed():
    base = [1.0] * 9
    assert iqm(base + [1e6]) == pytest.approx(1.0)


def test_aggregate_scores_interval_contains_point():
    s = aggregate_scores(np.random.default_rng(0).normal(size=10))
    assert s["n"] == 10 and s["iqm_low"] <= s["iqm"] <= s["iqm_high"]


def test_probability_of_improvement_counts_ties_as_half():
    assert probability_of_improvement([1, 2], [1, 2]) == pytest.approx(0.5)
    assert probability_of_improvement([3, 4], [1, 2]) == 1.0


def test_performance_profile_is_fraction_above():
    assert performance_profile([1, 2, 3, 4], [0, 2.5, 10]) == [1.0, 0.5, 0.0]


def test_two_sample_diff_detects_a_clear_gap_and_not_noise():
    rng = np.random.default_rng(1)
    far = two_sample_diff(rng.normal(5, 1, 10), rng.normal(0, 1, 10))
    assert far["p"] < 0.01 and far["low"] > 0
    same = two_sample_diff(rng.normal(0, 1, 10), rng.normal(0, 1, 10))
    assert same["p"] > 0.05 and same["low"] < 0 < same["high"]


def test_two_sample_diff_exact_p_for_small_groups():
    # 3 vs 3: C(6,3) = 20 relabellings; the observed split is the most extreme
    # of two (it and its mirror), so p = 2/20.
    r = two_sample_diff([10, 11, 12], [0, 1, 2])
    assert r["p"] == pytest.approx(0.1)


def test_holm_step_down():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"]["p_holm"] == pytest.approx(0.03)
    assert out["c"]["p_holm"] == pytest.approx(0.06)
    assert out["b"]["p_holm"] == pytest.approx(0.06)          # monotone
    assert out["a"]["reject"] and not out["b"]["reject"]


def test_trading_rigor_budgets_and_seeds():
    from tools.trading_rigor import BUDGETS, PUBLISHED, budgets_for, parse_seeds

    assert PUBLISHED in BUDGETS
    assert budgets_for("sb3_sac")[-1] == 300_000
    assert budgets_for("ours")[-1] == 1_000_000
    assert budgets_for("ours", 150_000)[-1] == PUBLISHED
    assert parse_seeds("200-203,209") == [200, 201, 202, 203, 209]


def test_published_budget_checkpoint_is_the_published_run():
    # run_ppo_training does total // rollout updates; a 150k run is 73 updates,
    # and the checkpoint the long run scores at 150k must be that same update.
    from tools.trading_rigor import ROLLOUT

    assert 150_000 // ROLLOUT == 73


def test_defaults_are_the_repaired_ppo():
    from rl_trader.config.training_config import REPAIRED_PPO, PPOConfig

    cfg = PPOConfig()
    assert {k: getattr(cfg, k) for k in REPAIRED_PPO} == REPAIRED_PPO


def test_published_switch_restores_the_v221_ppo(monkeypatch):
    from rl_trader.config.training_config import PUBLISHED_PPO, PPOConfig

    monkeypatch.setenv("RL_TRADER_PUBLISHED", "1")
    cfg = PPOConfig()
    assert {k: getattr(cfg, k) for k in PUBLISHED_PPO} == PUBLISHED_PPO


def test_published_switch_restores_the_per_ticker_split(monkeypatch, tmp_path):
    # Two series on different dates: the default cuts both on one date, the
    # published switch cuts each on its own history.
    from tools.real_significance import load_real_basket

    d = tmp_path / "crypto"
    d.mkdir()
    # Same last date, different first dates (BBB starts 600 days later).
    for name, start, n, seed in (("AAA", "2015-01-01", 1500, 1), ("BBB", "2016-08-23", 900, 2)):
        _frame(start, n, seed).to_csv(d / f"{name}.csv", index=False)
    common = load_real_basket(str(tmp_path), "crypto")
    assert len(common["AAA"]["test"]) == len(common["BBB"]["test"])
    monkeypatch.setenv("RL_TRADER_PUBLISHED", "1")
    per = load_real_basket(str(tmp_path), "crypto")
    assert len(per["AAA"]["test"]) != len(per["BBB"]["test"])


def test_hedging_keeps_its_published_ppo():
    from rl_trader.config.training_config import PUBLISHED_PPO
    from rl_trader.hedging.train import HedgeTrainConfig

    ppo = HedgeTrainConfig().ppo
    assert {k: getattr(ppo, k) for k in PUBLISHED_PPO} == PUBLISHED_PPO


def test_training_callback_sees_every_update():
    from rl_trader.config.training_config import stock_config
    from rl_trader.training.utils import run_ppo_training

    cfg = stock_config()
    cfg.train.total_timesteps = 2 * 256
    cfg.train.rollout_length = 256
    cfg.train.eval_interval = 0
    cfg.ppo.minibatch_size = 64
    cfg.ppo.update_epochs = 1
    seen = []
    run_ppo_training(cfg, callback=lambda agent, steps: seen.append(steps))
    assert seen == [256, 512]


def _frame(start: str, n: int, seed: int):
    import pandas as pd

    from rl_trader.data.data_loader import generate_synthetic_ohlcv

    df = generate_synthetic_ohlcv(n_steps=n, seed=seed)
    df["date"] = pd.date_range(start, periods=len(df), freq="D")
    return df


def test_common_split_keeps_every_training_date_before_every_test_date():
    from rl_trader.data.data_loader import add_technical_indicators
    from tools.real_significance import common_split_date

    frames = {"early": _frame("2015-01-01", 3000, 1), "late": _frame("2019-01-01", 1500, 2)}
    cut = common_split_date(frames, 0.6)
    late = add_technical_indicators(frames["late"])["date"]
    # 60% of the window both trade in, which is the late series' own window.
    assert late.iloc[0] < cut < late.iloc[-1]
    assert abs((late < cut).mean() - 0.6) < 0.01


def test_per_ticker_split_overlaps_when_histories_start_apart():
    # The defect the common split removes: cut each series at 60% of its own rows
    # and the late series is still training after the early one starts testing.
    from rl_trader.data.data_loader import add_technical_indicators

    early = add_technical_indicators(_frame("2015-01-01", 3000, 1))["date"]
    late = add_technical_indicators(_frame("2019-01-01", 1500, 2))["date"]
    early_test_start = early.iloc[int(len(early) * 0.6)]
    late_train_end = late.iloc[int(len(late) * 0.6) - 1]
    assert late_train_end > early_test_start
