"""Every p-value the paper reports, corrected for multiple testing.

The study reports fourteen hypothesis tests across five sections. Each was
reported at alpha = 0.05 on its own; with that many tests, a false positive
somewhere is likely (1 - 0.95**14 = 51% if every null were true). This collects them from the result files the tables are
generated from (no number is typed in here) and applies the Holm-Bonferroni
step-down correction twice:

* **per question**: the family is the tests answering one research question
  (e.g. the four hedging costs). This is the conventional unit.
* **whole paper**: one family of all fourteen. The strictest reading.

Correction only matters for tests whose claim is a difference; for the tests
whose claim is "no difference", adjusting p upward makes that claim stronger.

A test whose claim survives only the first is reported as such.

    python tools/multiple_testing.py      # writes docs/assets/multiple_testing.json
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_trader.evaluation.rigor import holm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _json(rel: str):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


def _js_global(rel: str):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        text = fh.read()
    return json.loads(re.search(r"=\s*(\{.*\})\s*;?\s*$", text, re.S).group(1))


def collect() -> list[dict]:
    """(family, test id, description, p, claimed) for every reported test."""
    tests = []
    synth = _json("docs/assets/significance_synth.json")["markets"]
    for m in ("stock", "crypto"):
        tests.append({"family": "§2 synthetic multi-seed", "id": f"synth-{m}",
                      "test": f"{m}: agent vs buy-and-hold, synthetic paths", "p": synth[m]["p"],
                      "claimed": "no difference"})
    sig = _js_global("docs/significance.js")
    rep = _js_global("docs/preregistration/2026-10-28-fresh-seeds.results.js")
    for m in ("stock", "crypto"):
        tests.append({"family": "§5 real-data significance", "id": f"real-{m}",
                      "test": f"{m}: agent vs buy-and-hold, held-out tickers", "p": sig[m]["p"],
                      "claimed": "worse than buy-and-hold" if m == "stock" else "no difference"})
        tests.append({"family": "§5 pre-registered replication", "id": f"rep-{m}",
                      "test": f"{m}: replication, seeds 200-209", "p": rep[m]["p"],
                      "claimed": "worse than buy-and-hold" if m == "stock" else "no difference"})
    sur_s = _json("docs/assets/surrogate_synthetic.json")
    sur_r = _json("docs/assets/surrogate_real.json")
    for m in ("stock", "crypto"):
        tests.append({"family": "§6 surrogate test", "id": f"surr-control-{m}",
                      "test": f"{m}: positive control, structured vs shuffled", "p": sur_s[m]["p"],
                      "claimed": "test has power (difference)"})
        tests.append({"family": "§6 surrogate test", "id": f"surr-real-{m}",
                      "test": f"{m}: real vs shuffled prices", "p": sur_r[m]["p"],
                      "claimed": "no difference"})
    hedge = _json("docs/assets/hedging_real.json")
    for r in hedge["costs"]:
        c = "none" if r["cost"] == 0 else f"{r['cost'] * 100:g}%"
        tests.append({"family": "§9 hedging vs tuned band", "id": f"hedge-{c}",
                      "test": f"cost {c}: agent vs tuned band, across seeds", "p": r["vs_band"]["p"],
                      "claimed": "agent better" if r["vs_band"]["p"] < 0.05 else "no difference"})
    return tests


def correct(tests: list[dict]) -> dict:
    whole = holm({t["id"]: t["p"] for t in tests})
    fams: dict = {}
    for t in tests:
        fams.setdefault(t["family"], {})[t["id"]] = t["p"]
    per = {}
    for f in fams.values():
        per.update(holm(f))
    rows = []
    for t in tests:
        sig = t["p"] <= 0.05
        rows.append({**t, "p": round(t["p"], 4), "p_family": per[t["id"]]["p_holm"],
                     "p_paper": whole[t["id"]]["p_holm"], "significant_raw": sig,
                     "survives_family": per[t["id"]]["reject"], "survives_paper": whole[t["id"]]["reject"]})
    return {"n_tests": len(rows), "rows": rows,
            "raw_significant": sum(r["significant_raw"] for r in rows),
            "survive_family": sum(r["survives_family"] for r in rows),
            "survive_paper": sum(r["survives_paper"] for r in rows)}


def main() -> None:
    out = correct(collect())
    path = os.path.join(ROOT, "docs", "assets", "multiple_testing.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for r in out["rows"]:
        flag = "yes" if r["survives_paper"] else ("family only" if r["survives_family"] else "-")
        print(f"{r['id']:<20} p={r['p']:<7} family={r['p_family']:<7} paper={r['p_paper']:<7} "
              f"{'*' if r['significant_raw'] else ' '} {flag}")
    print(f"{out['raw_significant']} of {out['n_tests']} significant at 0.05; "
          f"{out['survive_family']} after Holm per question; {out['survive_paper']} after Holm across the paper")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
