"""Tests for the signal-vs-noise human pattern-detection test.

This panel invites a visitor to conclude something about themselves, so the
experimental design has to hold up: the two classes must be exactly balanced,
they must differ *only* in temporal structure, the answer key must never leave
the server, and the inference reported afterwards must be exact rather than
approximate. Each of those is pinned here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from server import perception


# --------------------------------------------------------------------------- #
# Exact binomial inference                                                     #
# --------------------------------------------------------------------------- #
def test_binomial_matches_hand_computed_values():
    # A perfect score: both tails, so 2 * 0.5**n.
    assert perception.binomial_p_two_sided(8, 8) == pytest.approx(2 / 2 ** 8)
    assert perception.binomial_p_two_sided(0, 8) == pytest.approx(2 / 2 ** 8)
    # Dead-centre: nothing is less likely than the mode, so everything is summed.
    assert perception.binomial_p_two_sided(4, 8) == pytest.approx(1.0)
    # 7 of 8 = both 7s and both 8s.
    expected = 2 * (math.comb(8, 7) + math.comb(8, 8)) / 2 ** 8
    assert perception.binomial_p_two_sided(7, 8) == pytest.approx(expected)


def test_binomial_is_symmetric_and_bounded():
    for n in (4, 8, 12):
        for k in range(n + 1):
            p = perception.binomial_p_two_sided(k, n)
            assert 0.0 < p <= 1.0
            assert p == pytest.approx(perception.binomial_p_two_sided(n - k, n))


def test_power_analysis_is_exact_and_monotone():
    """Power must rise with true skill, and the floor must match 2^(1-n)."""
    out = perception.power_analysis(12)
    assert out["min_attainable_p"] == pytest.approx(2 / 2 ** 12)
    powers = [row["power"] for row in out["power"]]
    assert powers == sorted(powers)
    # Every "significant" score really is significant, and no other score is.
    for k in range(13):
        significant = perception.binomial_p_two_sided(k, 12) <= 0.05
        assert (k in out["significant_scores"]) is significant


def test_power_is_low_enough_to_matter():
    """The uncomfortable claim the UI makes must actually be true.

    At n = 8 a genuinely 70%-accurate guesser is detected under 10% of the time.
    If this ever stopped holding, the panel's headline would be wrong.
    """
    row = next(r for r in perception.power_analysis(8)["power"] if r["true_accuracy"] == 0.7)
    assert row["power"] < 0.10


# --------------------------------------------------------------------------- #
# Design: balance, determinism, and what leaves the server                     #
# --------------------------------------------------------------------------- #
def test_classes_are_exactly_balanced():
    for seed in range(6):
        quiz = perception.build_quiz("synthetic", seed=seed, n_charts=8)
        assert sum(quiz["_key"]) == 4


def test_quiz_is_deterministic_in_its_parameters():
    """Scoring rebuilds the quiz from its seed, so this is load-bearing."""
    a = perception.build_quiz("synthetic", seed=11, n_charts=8, market="crypto")
    b = perception.build_quiz("synthetic", seed=11, n_charts=8, market="crypto")
    assert a["_key"] == b["_key"]
    assert a["charts"][0]["prices"] == b["charts"][0]["prices"]


def test_a_different_seed_is_a_different_quiz():
    a = perception.build_quiz("synthetic", seed=1, n_charts=8)
    b = perception.build_quiz("synthetic", seed=2, n_charts=8)
    assert a["charts"][0]["prices"] != b["charts"][0]["prices"]


def test_public_view_carries_no_answers():
    quiz = perception.build_quiz("synthetic", seed=3, n_charts=8)
    pub = perception.public(quiz)
    blob = repr(pub)
    assert "_key" not in pub
    assert all("label" not in c for c in pub["charts"])
    # Nothing anywhere in the served payload should expose the truth.
    assert "label" not in blob
    assert set(pub["charts"][0]) == {"index", "prices"}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"difficulty": "nope"}, "unknown difficulty"),
        ({"n_charts": 7}, "even"),
        ({"n_charts": 2}, "between"),
        ({"n_charts": 40}, "between"),
        ({"difficulty": "real"}, "price fetcher"),
    ],
)
def test_invalid_requests_are_rejected(kwargs, message):
    params = {"difficulty": "synthetic", "seed": 0, "n_charts": 8}
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        perception.build_quiz(**params)


# --------------------------------------------------------------------------- #
# Confound control                                                             #
# --------------------------------------------------------------------------- #
def test_standardisation_preserves_autocorrelation():
    """The whole design rests on this: erase scale, keep the only real signal."""
    rng = np.random.default_rng(0)
    raw = rng.normal(0.0, 0.03, size=400)
    raw[1:] += 0.5 * raw[:-1]  # inject autocorrelation
    scaled = perception._standardise(raw, drift=0.001)
    assert perception._lag1(scaled) == pytest.approx(perception._lag1(raw), abs=1e-9)


def test_every_chart_has_the_same_volatility():
    """Volatility must carry no information about which class a chart is in."""
    quiz = perception.build_quiz("synthetic", seed=5, n_charts=12)
    stds = []
    for c in quiz["charts"]:
        lr = np.diff(np.log(np.asarray(c["prices"], dtype=float)))
        stds.append(float(lr.std()))
    # Prices are rounded to 4dp for the wire, so allow that much slack.
    assert np.allclose(stds, perception.TARGET_DAILY_SIGMA, rtol=2e-3)


def test_total_return_does_not_separate_the_classes():
    """If drift leaked the answer, the test would measure nothing interesting."""
    signal, control = [], []
    for seed in range(30):
        quiz = perception.build_quiz("synthetic", seed=seed, n_charts=8)
        for c in quiz["charts"]:
            total = c["prices"][-1] / c["prices"][0] - 1.0
            (signal if c["label"] else control).append(total)
    # Same drift distribution for both classes: the means should be close
    # relative to how much they individually vary.
    spread = np.std(signal + control)
    assert abs(np.mean(signal) - np.mean(control)) < 0.35 * spread


def test_the_signal_class_really_is_more_autocorrelated():
    """The one property that *should* differ, verified as measured — not assumed.

    This is the regression guard that would have caught the earlier bug where a
    'mean reversion' regime was byte-identical to a random walk.
    """
    signal, control = [], []
    for seed in range(25):
        meta = perception.build_quiz("synthetic", seed=seed, n_charts=8)["meta"]
        signal.append(meta["realised"]["mean_autocorr_signal"])
        control.append(meta["realised"]["mean_autocorr_control"])
    assert np.mean(signal) > np.mean(control) + 0.05
    assert abs(np.mean(control)) < 0.05  # the control really is memoryless


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def test_a_perfect_score_is_scored_as_perfect():
    quiz = perception.build_quiz("synthetic", seed=9, n_charts=12)
    out = perception.score_quiz(quiz, quiz["_key"])
    assert out["correct"] == 12
    assert out["accuracy"] == 1.0
    # score_quiz rounds the p-value for the wire; compare at that resolution.
    assert out["p_value"] == pytest.approx(2 / 2 ** 12, abs=1e-6)
    assert out["significant_at_05"] is True
    assert all(r["correct"] for r in out["per_chart"])


def test_an_inverted_answer_sheet_is_significantly_wrong():
    quiz = perception.build_quiz("synthetic", seed=9, n_charts=12)
    out = perception.score_quiz(quiz, [1 - k for k in quiz["_key"]])
    assert out["correct"] == 0
    assert out["significant_at_05"] is True
    assert "worse* than chance" in out["verdict"]


def test_a_chance_score_is_reported_as_indistinguishable():
    quiz = perception.build_quiz("synthetic", seed=4, n_charts=8)
    answers = list(quiz["_key"])
    for i in range(0, 8, 2):  # flip half of them
        answers[i] = 1 - answers[i]
    out = perception.score_quiz(quiz, answers)
    assert out["correct"] == 4
    assert out["significant_at_05"] is False
    assert "indistinguishable" in out["verdict"]


def test_wrong_number_of_answers_is_an_error():
    quiz = perception.build_quiz("synthetic", seed=0, n_charts=8)
    with pytest.raises(ValueError, match="expected 8 answers"):
        perception.score_quiz(quiz, [1, 0, 1])


def test_the_statistical_reference_beats_the_eye_on_average():
    """The panel claims a one-line statistic outperforms visual inspection.

    It is only allowed to claim that because it is measured here: the rule is
    scored on the same charts, over many quizzes, and must land clearly above
    the chance rate of n/2.
    """
    scores = [
        perception._autocorr_rule(perception.build_quiz("synthetic", seed=s, n_charts=8))[
            "correct"
        ]
        for s in range(30)
    ]
    assert np.mean(scores) > 5.5  # chance is 4.0
    assert min(scores) >= 2       # and it is not an oracle either


# --------------------------------------------------------------------------- #
# The real-data condition                                                      #
# --------------------------------------------------------------------------- #
def _fake_fetcher(n_rows: int = 600):
    """A deterministic stand-in for the network price fetcher."""
    rng = np.random.default_rng(42)
    lr = rng.normal(0.0004, 0.02, size=n_rows)
    close = 100.0 * np.exp(np.cumsum(lr))
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1e6,
    })
    return lambda ticker: (df, None)


def test_real_condition_uses_disjoint_slices_of_one_series():
    fetch = _fake_fetcher(600)
    quiz = perception.build_quiz("real", seed=1, n_charts=6, ticker="SPY", fetch_ohlcv=fetch)
    assert quiz["meta"]["ticker"] == "SPY"
    # 599 returns over 6 charts -> 99 bars each, plus the anchor price.
    assert quiz["meta"]["bars_per_chart"] == 99
    assert all(len(c["prices"]) == 100 for c in quiz["charts"])


def test_shuffled_charts_keep_the_exact_same_return_distribution():
    """The surrogate property that makes this a fair test.

    A permuted chart must be distributionally identical to the slice it came
    from — only the ordering may differ. If the shuffle changed the marginal, a
    visitor could win on fat tails alone and the test would measure the wrong
    thing.
    """
    fetch = _fake_fetcher(600)
    close = np.asarray(fetch("SPY")[0]["close"], dtype=float)
    source = np.diff(np.log(close))
    quiz = perception.build_quiz("real", seed=2, n_charts=6, ticker="SPY", fetch_ohlcv=fetch)
    seg = quiz["meta"]["bars_per_chart"]

    for i, chart in enumerate(quiz["charts"]):
        got = np.diff(np.log(np.asarray(chart["prices"], dtype=float)))
        want = perception._standardise(source[i * seg:(i + 1) * seg], drift=0.0)
        # Compare centred, sorted returns: identical multiset either way.
        assert np.allclose(np.sort(got - got.mean()), np.sort(want), atol=1e-5)
        # A real chart preserves the order; a shuffled one must not.
        in_order = np.allclose(got - got.mean(), want, atol=1e-5)
        assert in_order is bool(chart["label"])


def test_real_condition_fails_loudly_without_data():
    def broken(_ticker):
        return None, "upstream refused the request"

    with pytest.raises(ValueError, match="upstream refused"):
        perception.build_quiz("real", seed=0, n_charts=6, ticker="X", fetch_ohlcv=broken)


def test_real_condition_refuses_slices_too_short_to_judge():
    with pytest.raises(ValueError, match="not enough"):
        perception.build_quiz(
            "real", seed=0, n_charts=12, ticker="X", fetch_ohlcv=_fake_fetcher(200)
        )
