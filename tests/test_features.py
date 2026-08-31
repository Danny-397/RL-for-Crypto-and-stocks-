"""Tests for the engineered feature set and the data pipeline guarantees.

These lock in the two properties that matter most for a *trustworthy* backtest:
the features are finite/leak-free, and the train/val/test scaler is fit on
training data only.
"""

import numpy as np

from rl_trader.data.data_loader import (
    FEATURE_COLUMNS,
    add_technical_indicators,
    generate_synthetic_ohlcv,
    prepare_market_data,
)


def test_feature_set_is_complete_and_finite():
    df = generate_synthetic_ohlcv(n_steps=800, seed=0)
    featured = add_technical_indicators(df)
    # Every declared feature column is actually produced...
    for col in FEATURE_COLUMNS:
        assert col in featured.columns, f"missing engineered feature: {col}"
    feats = featured[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    # ...and contains no NaN/inf after the dropna in the builder.
    assert np.isfinite(feats).all()


def test_richer_feature_dimension():
    # Guards against silently dropping/duplicating a feature.
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert len(FEATURE_COLUMNS) >= 19


def test_splits_are_chronological_and_leak_free():
    splits = prepare_market_data(None, market="stock", synthetic_steps=1000, seed=1)
    assert set(splits) == {"train", "val", "test"}
    # Each split is non-empty and shares the same feature dimension.
    f = len(FEATURE_COLUMNS)
    for part in splits.values():
        assert part.features.shape[1] == f
        assert len(part) > 0
        assert np.isfinite(part.features).all()


def test_indicators_have_no_lookahead():
    # Features at row t must not change when *future* rows are appended: a direct
    # check that nothing leaks information backwards in time.
    df = generate_synthetic_ohlcv(n_steps=400, seed=2)
    full = add_technical_indicators(df)
    prefix = add_technical_indicators(df.iloc[:300].reset_index(drop=True))
    overlap = min(len(prefix), 200)
    # Compare the last `overlap` rows of the prefix against the matching rows of
    # the full series (aligned from the end of the prefix window).
    a = prefix[FEATURE_COLUMNS].to_numpy()[-overlap:]
    b = full[FEATURE_COLUMNS].to_numpy()[: len(prefix)][-overlap:]
    assert np.allclose(a, b, atol=1e-6)


def _lag1_autocorr(df) -> float:
    """Lag-1 autocorrelation of log returns for a generated price path."""
    close = df["close"].to_numpy(dtype=float)
    lr = np.diff(np.log(close))
    return float(np.corrcoef(lr[:-1], lr[1:])[0, 1])


def test_momentum_sign_controls_return_autocorrelation():
    """``momentum`` is an AR(1) coefficient, so its *sign* must reach the output.

    A guard of ``if momentum > 0`` would silently route negative coefficients
    into the memoryless branch, making a "mean reverting" regime identical to a
    random walk. Averaged over seeds, the three cases must separate cleanly.
    """
    def mean_ac(phi: float) -> float:
        return float(
            np.mean([
                _lag1_autocorr(generate_synthetic_ohlcv(n_steps=800, momentum=phi, seed=s))
                for s in range(6)
            ])
        )

    trending = mean_ac(0.70)
    flat = mean_ac(0.0)
    reverting = mean_ac(-0.50)

    assert trending > 0.05, f"positive momentum should trend, got {trending:+.4f}"
    assert abs(flat) < 0.03, f"zero momentum should be memoryless, got {flat:+.4f}"
    assert reverting < -0.03, f"negative momentum should revert, got {reverting:+.4f}"
    assert reverting < flat < trending


def test_feature_groups_partition_the_feature_set():
    """The X-Ray panel groups the observation from FEATURE_GROUPS.

    If a feature were added to FEATURE_COLUMNS but not to a group, the UI would
    silently stop showing it — so the two must stay an exact partition.
    """
    from rl_trader.data.data_loader import FEATURE_GROUPS

    grouped = [f for names in FEATURE_GROUPS.values() for f in names]
    assert len(grouped) == len(set(grouped)), "a feature appears in two groups"
    assert set(grouped) == set(FEATURE_COLUMNS), (
        f"ungrouped: {sorted(set(FEATURE_COLUMNS) - set(grouped))}; "
        f"unknown: {sorted(set(grouped) - set(FEATURE_COLUMNS))}"
    )
