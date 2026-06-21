"""Tests for the statistics helpers and walk-forward fold generation."""

import numpy as np
import pytest

from rl_trader.evaluation.statistics import (
    bootstrap_ci,
    paired_permutation_test,
    summarize,
)
from rl_trader.evaluation.walk_forward import generate_folds


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    samples = rng.normal(0.05, 0.1, size=200)
    est = bootstrap_ci(samples, seed=0)
    assert est.low < est.mean < est.high
    assert est.n == 200
    # Interval should contain the true mean of a well-behaved sample.
    assert est.low < 0.05 < est.high


def test_bootstrap_ci_handles_degenerate_inputs():
    assert bootstrap_ci([]).n == 0
    one = bootstrap_ci([0.3])
    assert one.mean == one.low == one.high == pytest.approx(0.3)


def test_permutation_detects_real_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.2, size=15)
    b = rng.normal(0.0, 0.2, size=15)
    _diff, p = paired_permutation_test(a, b, seed=0)
    assert p < 0.01


def test_permutation_pvalue_large_when_no_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 0.2, size=15)
    b = rng.normal(0.0, 0.2, size=15)
    _diff, p = paired_permutation_test(a, b, seed=0)
    assert p > 0.1


def test_permutation_requires_equal_length():
    with pytest.raises(ValueError):
        paired_permutation_test([1, 2, 3], [1, 2])


def test_summarize_keys():
    out = summarize([0.1, 0.2, 0.3, 0.4])
    assert set(out) >= {"mean", "ci_low", "ci_high", "std", "min", "max", "n"}
    assert out["n"] == 4


def test_generate_folds_are_chronological_and_disjoint():
    folds = generate_folds(1000, n_folds=4, train_min_frac=0.4, scheme="expanding")
    assert len(folds) == 4
    for f in folds:
        # Train strictly precedes test; test is non-empty.
        assert f.train.stop == f.test.start
        assert f.test.stop > f.test.start
    assert folds[-1].test.stop == 1000
    # Expanding scheme: train always starts at 0 and grows.
    assert all(f.train.start == 0 for f in folds)
    assert folds[0].train.stop < folds[-1].train.stop


def test_sliding_folds_use_fixed_window():
    folds = generate_folds(1000, n_folds=4, train_min_frac=0.4, scheme="sliding")
    widths = {f.train.stop - f.train.start for f in folds}
    assert len(widths) == 1  # constant training-window size


def test_generate_folds_validates_inputs():
    with pytest.raises(ValueError):
        generate_folds(1000, train_min_frac=1.5)
    with pytest.raises(ValueError):
        generate_folds(10, n_folds=50)
