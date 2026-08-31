"""Tests for dataset pinning in ``tools/fetch_data.py``.

The published figures were produced from a data window that has since moved,
because ``PERIOD`` is relative. These lock in the machinery that stops that
happening again: a fixed calendar clip, a content-hashed snapshot, and a verify
step that fails loudly on drift instead of silently rebuilding on new data.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tools.fetch_data import _clip, verify_snapshot, write_snapshot


def _frame(dates, close=1.0):
    return pd.DataFrame(
        {
            "date": dates,
            "open": close, "high": close, "low": close,
            "close": close, "volume": 1000.0,
        }
    )


@pytest.fixture
def dataset(tmp_path):
    """A miniature data/raw tree: two markets, two tickers."""
    root = tmp_path / "raw"
    (root / "stock").mkdir(parents=True)
    (root / "crypto").mkdir(parents=True)
    _frame(["2024-01-01", "2024-01-02", "2024-01-03"]).to_csv(
        root / "stock" / "AAPL.csv", index=False
    )
    _frame(["2024-01-01", "2024-01-02"]).to_csv(
        root / "crypto" / "BTC-USD.csv", index=False
    )
    return root


# --------------------------------------------------------------------------- #
# Clipping                                                                     #
# --------------------------------------------------------------------------- #
def test_clip_pins_both_ends():
    df = _frame(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    out = _clip(df, "2024-01-02", "2024-01-03")
    assert list(out["date"]) == ["2024-01-02", "2024-01-03"]


def test_clip_is_a_noop_without_bounds():
    df = _frame(["2024-01-01", "2024-01-02"])
    assert list(_clip(df, None, None)["date"]) == ["2024-01-01", "2024-01-02"]


def test_clip_makes_a_longer_fetch_identical_to_a_shorter_one():
    """The point of pinning: extra history must not change the pinned window."""
    short = _clip(_frame(["2024-01-01", "2024-01-02"]), None, "2024-01-02")
    long = _clip(
        _frame(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]), None, "2024-01-02"
    )
    pd.testing.assert_frame_equal(short, long)


# --------------------------------------------------------------------------- #
# Snapshot + verify                                                            #
# --------------------------------------------------------------------------- #
def test_snapshot_records_every_file(dataset, tmp_path):
    path = tmp_path / "SNAPSHOT.json"
    snap = write_snapshot(str(dataset), str(path), "2024-01-01", "2024-01-03")

    assert snap["pinned_start"] == "2024-01-01"
    assert snap["pinned_end"] == "2024-01-03"
    assert set(snap["files"]) == {"stock/AAPL", "crypto/BTC-USD"}
    entry = snap["files"]["stock/AAPL"]
    assert entry["rows"] == 3
    assert entry["first_date"] == "2024-01-01" and entry["last_date"] == "2024-01-03"
    assert len(entry["sha256"]) == 64
    # Written to disk and valid JSON.
    assert json.loads(path.read_text(encoding="utf-8"))["files"] == snap["files"]


def test_verify_passes_on_an_unchanged_dataset(dataset, tmp_path):
    path = tmp_path / "SNAPSHOT.json"
    write_snapshot(str(dataset), str(path), None, "2024-01-03")
    assert verify_snapshot(str(dataset), str(path)) == []


def test_verify_detects_changed_content(dataset, tmp_path):
    """The exact failure that made the published numbers unreproducible."""
    path = tmp_path / "SNAPSHOT.json"
    write_snapshot(str(dataset), str(path), None, "2024-01-03")

    # Simulate a later re-fetch that picked up extra bars.
    _frame(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]).to_csv(
        dataset / "stock" / "AAPL.csv", index=False
    )
    problems = verify_snapshot(str(dataset), str(path))
    assert len(problems) == 1
    assert "stock/AAPL" in problems[0]
    assert "content differs" in problems[0]
    assert "found 4 rows" in problems[0]


def test_verify_detects_missing_and_extra_files(dataset, tmp_path):
    path = tmp_path / "SNAPSHOT.json"
    write_snapshot(str(dataset), str(path), None, None)

    (dataset / "crypto" / "BTC-USD.csv").unlink()
    _frame(["2024-01-01"]).to_csv(dataset / "stock" / "MSFT.csv", index=False)

    problems = verify_snapshot(str(dataset), str(path))
    joined = " | ".join(problems)
    assert "crypto/BTC-USD: missing" in joined
    assert "stock/MSFT: present but not in the snapshot" in joined


def test_snapshot_hash_is_content_sensitive(dataset, tmp_path):
    path = tmp_path / "SNAPSHOT.json"
    before = write_snapshot(str(dataset), str(path), None, None)["files"]["stock/AAPL"]["sha256"]
    _frame(["2024-01-01", "2024-01-02", "2024-01-03"], close=2.0).to_csv(
        dataset / "stock" / "AAPL.csv", index=False
    )
    after = write_snapshot(str(dataset), str(path), None, None)["files"]["stock/AAPL"]["sha256"]
    # Same dates and row count, different prices — the hash must still move.
    assert before != after
