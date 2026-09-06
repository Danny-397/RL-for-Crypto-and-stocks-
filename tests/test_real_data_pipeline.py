"""A tool that scores the agent must build the observation it was trained on.

``tools/make_figures.py`` loaded the real CSVs without merging the market index.
Four of the agent's 28 inputs are cross-asset -- ``rel_return_5``,
``rel_return_20``, ``market_trend``, ``market_ret_20`` -- and with no reference
series they do not degrade gracefully: they come back at exactly zero, standard
deviation zero, for every bar. So the figures and ``docs/assets/baselines.json``
scored the deployed policy on an observation it had never seen in training,
while every other real-data tool merged the index and scored it correctly.

Nothing failed. The numbers simply disagreed: RESULTS.md carried a held-out
stock return of -14.1% in its baselines table and -4.7% in the headline run and
supervised table, for what the prose called the same run. Two tables, one seed,
one checkpoint, different answers -- in a document whose argument is that a
number which has drifted from the run behind it is a false claim.

These tests pin the rule rather than the digits: if a tool reads a real OHLCV
CSV, it merges the index first.
"""

from __future__ import annotations

import ast
import io
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
DATA = os.path.join(REPO, "data", "raw")

# The four inputs that have no value without a reference series.
CROSS_ASSET = ("rel_return_5", "rel_return_20", "market_trend", "market_ret_20")


def _tool_sources():
    for name in sorted(os.listdir(TOOLS)):
        if not name.endswith(".py") or name.startswith("_"):
            continue
        with io.open(os.path.join(TOOLS, name), encoding="utf-8") as fh:
            yield name, fh.read()


def _calls_named(tree: ast.AST, wanted: str) -> bool:
    """Is ``wanted`` called anywhere in this module?"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.id if isinstance(fn, ast.Name)
                else fn.attr if isinstance(fn, ast.Attribute) else None)
        if name == wanted:
            return True
    return False


REAL_DATA_TOOLS = [
    (name, src) for name, src in _tool_sources()
    if _calls_named(ast.parse(src), "load_ohlcv_csv")
]


def test_the_scan_finds_the_real_data_tools():
    """Guard the guard: an empty list would make every case below vacuous."""
    assert len(REAL_DATA_TOOLS) >= 5, [n for n, _ in REAL_DATA_TOOLS]


@pytest.mark.parametrize("name,src", REAL_DATA_TOOLS,
                         ids=[n for n, _ in REAL_DATA_TOOLS])
def test_a_tool_reading_real_csvs_attaches_the_market_index(name, src):
    """Otherwise it feeds the policy four dead inputs and reports the result."""
    tree = ast.parse(src)
    if not _calls_named(tree, "prepare_market_data"):
        pytest.skip(f"{name} loads CSVs but does not build features from them")
    assert _calls_named(tree, "attach_market_index"), (
        f"{name} builds features from a real CSV without merging the market "
        f"index; the cross-asset features {CROSS_ASSET} will be identically "
        f"zero and the agent will be scored on an observation it never trained "
        f"on. Wrap the frame: attach_market_index(load_ohlcv_csv(path), "
        f"data_dir, market)."
    )


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA, "stock")),
                    reason="real basket not fetched; run tools/fetch_data.py")
def test_omitting_the_index_flatlines_exactly_the_cross_asset_features():
    """The reason the rule above exists, measured rather than asserted.

    If this ever fails because the columns are merely *different* rather than
    dead, the static check is still right but its rationale has changed and the
    docstring above needs rewriting.
    """
    from rl_trader.data.data_loader import attach_market_index, load_ohlcv_csv, prepare_market_data

    path = os.path.join(DATA, "stock", "AAPL.csv")
    if not os.path.exists(path):
        pytest.skip("AAPL.csv not in the fetched basket")

    bare = prepare_market_data(load_ohlcv_csv(path), market="stock",
                               train_frac=0.6, val_frac=0.0)["test"]
    merged = prepare_market_data(
        attach_market_index(load_ohlcv_csv(path), DATA, "stock"),
        market="stock", train_frac=0.6, val_frac=0.0)["test"]

    names = list(getattr(merged, "feature_names", None) or [])
    assert names, "feature names are needed to identify the cross-asset columns"
    assert bare.features.shape == merged.features.shape, (
        "the observation must keep its width either way -- that is precisely "
        "why the omission was silent rather than a shape error")

    moved = {names[i] for i in range(len(names))
             if np.abs(bare.features[:, i] - merged.features[:, i]).max() > 1e-9}
    assert moved == set(CROSS_ASSET), (
        f"expected exactly the cross-asset inputs to change, got {sorted(moved)}")

    for feature in CROSS_ASSET:
        col = bare.features[:, names.index(feature)]
        assert col.std() == 0.0 and np.all(col == 0.0), (
            f"{feature} was expected to be identically zero without an index")
