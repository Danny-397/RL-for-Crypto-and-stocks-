"""README.md and RESULTS.md must agree with the experiments they describe.

The site's tables were hand-typed until a rebuild moved them and nothing noticed;
they are generated now, and ``tests/test_site_numbers.py`` keeps them that way.
The prose documents had the same defect and are read far more often — at one
point RESULTS.md carried a +275.5% crypto agent, a −2.7% multi-seed mean and
``p ≈ 0.97``, every one of them superseded, in a repository whose argument is
that single-run numbers should not be believed.

``tools/sync_docs.py`` regenerates the tables from ``docs/`` artifacts. These
tests make a rebuild that moves a number turn the suite red rather than quietly
make the documentation wrong.
"""

from __future__ import annotations

import io
import os
import re

import pytest

from tools import sync_docs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name: str) -> str:
    with io.open(os.path.join(REPO, name), encoding="utf-8") as fh:
        return fh.read().replace("\r\n", "\n")


@pytest.fixture(scope="module")
def readme() -> str:
    return _read("README.md")


@pytest.fixture(scope="module")
def results() -> str:
    return _read("RESULTS.md")


# --------------------------------------------------------------------------- #
# The documents match the artifacts                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sync_docs.TARGETS)
def test_the_generated_tables_are_up_to_date(name):
    """The failure this exists to catch is a rebuild landing without a re-sync."""
    original = _read(name)
    updated, touched, _missing = sync_docs.apply_blocks(original, name)
    assert touched, f"{name} has no generated regions any more"
    assert updated == original, (
        f"{name} is out of date with docs/. Run: python tools/sync_docs.py")


def test_every_marker_in_the_documents_has_a_generator():
    """A marker with no generator would silently never update."""
    known = set(sync_docs.BLOCKS)
    for name in sync_docs.TARGETS:
        begin, _end = sync_docs.markers_for(name)
        pattern = re.escape(begin).replace(re.escape("{}"), "([a-z-]+)")
        for key in re.findall(pattern, _read(name)):
            assert key in known, f"{name}: no generator for '{key}'"


def test_every_marker_is_closed():
    for name in sync_docs.TARGETS:
        text = _read(name)
        begin, end = sync_docs.markers_for(name)

        def esc(m):
            return re.escape(m).replace(re.escape("{}"), "([a-z-]+)")

        opened = re.findall(esc(begin), text)
        closed = re.findall(esc(end), text)
        assert opened == closed, f"{name}: markers are unbalanced"
        assert len(opened) == len(set(opened)), f"{name}: a key is marked twice"


def test_the_generators_that_have_artifacts_produce_tables():
    built = {k: fn() for k, fn in sync_docs.BLOCKS.items()}
    assert any(v for v in built.values()), "no artifact is readable at all"
    for key, body in built.items():
        if body is None:
            continue      # artifact not generated yet; sync_docs leaves it alone
        assert body.strip(), key
        # a table (markdown pipes or LaTeX ampersands) or a list -- never a shell
        looks_like_a_table = "|" in body or "&" in body
        looks_like_a_list = "\n- " in body or body.lstrip().startswith("-")
        assert looks_like_a_table or looks_like_a_list, key


# --------------------------------------------------------------------------- #
# What is quoted by hand must be labelled as superseded                        #
# --------------------------------------------------------------------------- #
SUPERSEDED_COMMIT = "d4c0ef9"


@pytest.mark.parametrize("name", sync_docs.TARGETS)
def test_the_old_headline_is_only_quoted_as_history(name):
    """+275% was real, and is no longer current. Saying so is the whole point.

    Every mention must read as history in its own sentence, and the document must
    cite the commit the run came from so a reader can check it against git rather
    than against the description here. The label is checked locally because that
    is what someone skimming actually sees; the citation is checked per document
    because it sensibly appears once per section, not beside every mention.
    """
    text = _read(name)
    mentions = list(re.finditer(r"\+275", text))
    if not mentions:
        return
    assert SUPERSEDED_COMMIT in text, (
        f"{name} quotes a +275% figure but never cites the commit it came from")
    for match in mentions:
        window = text[max(0, match.start() - 300): match.start() + 300]
        assert re.search(
            r"earlier|superseded|Superseded|moved that crypto headline",
            window), (
            f"{name}: the +275% figure at offset {match.start()} is not labelled "
            "as an earlier result")


@pytest.mark.parametrize("name", sync_docs.TARGETS)
def test_the_superseded_multi_seed_numbers_are_gone(name):
    """These were the v1 study, and every one of them has since moved."""
    text = _read(name)
    for stale in ("p ≈ 0.97", "−2.7%, 95% CI", "straddles zero"):
        assert stale not in text, f"{name}: still carries the superseded '{stale}'"


def test_the_crypto_interval_is_not_described_as_straddling_zero(results: str):
    """It did in v1 and does not now, so the argument had to change with it.

    Pinned because this is the sentence most likely to be copied forward by
    reflex: the conclusion (not distinguishable from buy-and-hold) survived the
    rebuild, but the reason for it did not.
    """
    sig = sync_docs._js_global("significance.js")
    if not sig or "crypto" not in sig:
        pytest.skip("no significance artifact")
    excludes_zero = sig["crypto"]["ci_low"] > 0 or sig["crypto"]["ci_high"] < 0
    if excludes_zero:
        assert "straddles zero" not in results
