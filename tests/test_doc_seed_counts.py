"""Prose must not quote a seed count the experiments no longer use.

The tables in README.md, RESULTS.md and the paper are generated from artifacts,
so they stay current. The sentences around them are not: when the ablation and
the real-market study moved from 5 seeds to 10, the tables changed and the prose
kept describing the 5-seed run ("one of five seeds returned +8%", "with 5 seeds
the test cannot reach p <= 0.05"), contradicting the table directly above it in
three documents at once.

So every mention of five seeds/pairs/runs has to be one of the studies that
really does use five, or history that says it is history. A new one fails here
until it is either corrected or added below with its reason.
"""
from __future__ import annotations

import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCS = [
    "README.md", "RESULTS.md", "paper/rl_trader.tex", "paper/README.md",
    "docs/index.html", "docs/lab.js", "docs/app.js",
]

FIVE = re.compile(r"\b(?:5|five)[- ](?:[\w-]+ ){0,2}(?:seeds?|pairs|runs)\b", re.I)

# Substring of the line -> why five is right there.
ALLOWED = {
    '"run 5 seeds"': "rhetorical: why training is not a live button",
    "five-seed count was fixed": "history of the ablation's seed count",
    "For each of 5 seeds it": "the multi-fold walk-forward (RESULTS 5c) is 5 seeds",
    "95% CI over 5 seeds": "walk-forward table header",
    "the §5 seed-luck story": "a section reference, not a seed count",
    "the budget to 5 seeds / 60k steps": "the surrogate positive control is 5 seeds",
    "4 rolling folds x 5 seeds": "walk-forward summary",
    "repeatably profitable across 5 seeds": "the synthetic significance study (RESULTS 2) is 5 seeds",
    "5 seeds, 60k steps; $p$ from a": "surrogate table caption",
    "folds $\\times$ 5 seeds": "walk-forward summary in the paper",
    "Table 4 was regenerated at 5 seeds": "surrogate table history",
    "at five seeds it": "history: why the study was rerun at ten",
    "with 5 seeds a ` +": "history: why the study was rerun at ten",
    'so "run 5 seeds" cannot be': "rhetorical: why training is not a live button",
}


def _artifact_seed_counts() -> tuple[int, int]:
    with open(os.path.join(REPO, "docs", "assets", "ablation_multiseed.json"),
              encoding="utf-8") as fh:
        ablation = len(json.load(fh)["summary"]["seeds"])
    with open(os.path.join(REPO, "docs", "significance.js"), encoding="utf-8") as fh:
        shim = fh.read()
    real = json.loads(shim[shim.index("=") + 1: shim.rindex(";")])["crypto"]["seeds"]
    return ablation, real


def test_the_headline_studies_are_not_five_seeds():
    """If either study goes back to five, the allow-list above needs revisiting."""
    assert _artifact_seed_counts() == (10, 10)


def test_every_mention_of_five_seeds_is_accounted_for():
    stray = []
    for rel in DOCS:
        with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if FIVE.search(line) and not any(a in line for a in ALLOWED):
                    stray.append(f"{rel}:{n}: {line.strip()[:120]}")
    assert not stray, (
        "prose quotes five seeds where the experiments now use ten -- correct it, "
        "or add the line to ALLOWED with the reason five is right there:\n"
        + "\n".join(stray)
    )


def test_no_document_still_calls_the_domain_arm_unseeded():
    """The DR stream has been seeded since v2.1 (seeded_path_factory)."""
    stale = re.compile(r"fresh \\emph\{unseeded\}|draws a fresh \*\*unseeded\*\*|"
                       r"cannot be reproduced from a single run", re.I)
    hits = []
    for rel in DOCS:
        with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
            text = fh.read()
        hits += [rel for _ in stale.finditer(text)]
    assert not hits, f"stale 'unseeded domain arm' claim in {sorted(set(hits))}"
