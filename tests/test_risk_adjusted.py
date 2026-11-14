"""The risk-adjusted comparison in tools/real_significance.py.

Alpha and beta are only meaningful if the agent's daily returns line up with the
asset's. These pin the regression on cases whose answer is known exactly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from real_significance import _alpha_beta  # noqa: E402

RNG = np.random.default_rng(0)
ASSET = RNG.normal(0.0008, 0.015, 750)


def test_holding_the_asset_is_beta_one_and_no_alpha():
    alpha, beta = _alpha_beta(ASSET.copy(), ASSET, 252)
    assert abs(beta - 1.0) < 1e-9
    assert abs(alpha) < 1e-9


def test_half_exposure_is_beta_one_half():
    alpha, beta = _alpha_beta(0.5 * ASSET, ASSET, 252)
    assert abs(beta - 0.5) < 1e-9
    assert abs(alpha) < 1e-9


def test_a_constant_daily_edge_is_annualised_alpha():
    alpha, beta = _alpha_beta(ASSET + 0.0001, ASSET, 252)
    assert abs(beta - 1.0) < 1e-9
    assert abs(alpha - 0.0252) < 1e-9


def test_being_flat_has_no_exposure():
    alpha, beta = _alpha_beta(np.zeros_like(ASSET), ASSET, 252)
    assert beta == 0.0 and alpha == 0.0
