"""Tests for the surrogate-data falsification test (tools/surrogate_test.py).

The scientific validity of the surrogate test rests on one invariant: shuffling
the daily returns must **destroy temporal order while leaving buy-&-hold
unchanged**. If the surrogate ended at a different price, the comparison would be
confounded. These tests pin that invariant down.
"""

import numpy as np

from rl_trader.data.data_loader import generate_synthetic_ohlcv
from tools.surrogate_test import surrogate_df


def _sample_df():
    return generate_synthetic_ohlcv(n_steps=500, annual_vol=0.3, momentum=0.6, seed=1)


def test_surrogate_preserves_final_price():
    """Same endpoint => buy-&-hold return is identical on the surrogate."""
    df = _sample_df()
    surr = surrogate_df(df, np.random.default_rng(0))
    assert np.isclose(surr["close"].iloc[-1], df["close"].iloc[-1], rtol=1e-6)
    assert np.isclose(surr["close"].iloc[0], df["close"].iloc[0], rtol=1e-6)


def test_surrogate_preserves_return_distribution():
    """The daily log-return *multiset* is preserved — only its order changes."""
    df = _sample_df()
    surr = surrogate_df(df, np.random.default_rng(0))
    orig = np.diff(np.log(df["close"].to_numpy(float)))
    new = np.diff(np.log(surr["close"].to_numpy(float)))
    assert np.allclose(np.sort(orig), np.sort(new), atol=1e-8)


def test_surrogate_actually_reorders():
    """Sanity: the surrogate is not a no-op — temporal order really is destroyed."""
    df = _sample_df()
    surr = surrogate_df(df, np.random.default_rng(0))
    orig = np.diff(np.log(df["close"].to_numpy(float)))
    new = np.diff(np.log(surr["close"].to_numpy(float)))
    assert not np.allclose(orig, new)
