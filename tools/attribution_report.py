"""Record which inputs the deployed policies actually read.

The lab computes occlusion attribution live, but the README quotes a ranking, and
that quote was measured against policy archives that have since been replaced.
When it was re-measured against the rebuilt archives the top three features
changed identity in both markets — the magnitudes barely moved, but the names did.

That instability is worth publishing rather than hiding: a ranked bar chart is
exactly the kind of output a reader takes for a discovered mechanism, and two runs
of the same recipe disagreeing about which feature comes first is the strongest
available argument for reading it as local sensitivity instead.

Writes ``docs/assets/attribution.json``; ``tools/sync_docs.py`` renders it into
the README.

Run from the repo root (no server needed — it drives the app in-process)::

    python tools/attribution_report.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "assets", "attribution.json")

TERMINAL = ("done", "complete", "finished", "error", "failed")


def measure(market: str, bars: int, sampled: int, seed: int, source: str) -> dict:
    sys.path.insert(0, REPO)
    from server.app import app

    client = app.test_client()
    config = {"market": market, "mode": "synthetic", "source": source,
              "bars": bars, "seed": seed}
    created = client.post("/api/experiments", json=config).get_json()
    experiment_id = created.get("id") or created.get("experiment_id")
    if not experiment_id:
        raise SystemExit(f"{market}: could not create experiment: {created}")

    for _ in range(600):
        status = client.get(f"/api/experiments/{experiment_id}").get_json().get("status")
        if status in TERMINAL:
            break
        time.sleep(0.5)

    payload = client.get(
        f"/api/experiments/{experiment_id}/attribution?bars={sampled}").get_json()
    if "error" in payload:
        raise SystemExit(f"{market}: {payload['error']}")

    episode = payload["episode"]
    return {
        "features": [
            {"name": row["name"],
             "mean_abs_delta": round(float(row["mean_abs_delta"]), 4)}
            for row in episode["features"][:5]
        ],
        "account_max": round(
            max(float(a["mean_abs_delta"]) for a in episode["account"]), 4),
        "account_max_name": max(
            episode["account"], key=lambda a: a["mean_abs_delta"])["name"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", type=int, default=600, help="episode length")
    ap.add_argument("--sampled", type=int, default=80,
                    help="bars sampled along the episode")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--source", default="momentum")
    ap.add_argument("--markets", nargs="+", default=["stock", "crypto"])
    args = ap.parse_args()

    markets = {}
    for market in args.markets:
        markets[market] = measure(market, args.bars, args.sampled,
                                  args.seed, args.source)
        top = markets[market]["features"][0]
        print(f"{market:7} top: {top['name']} {top['mean_abs_delta']:.2f}  "
              f"account max {markets[market]['account_max']:.3f}")

    payload = {
        "generated": datetime.date.today().isoformat(),
        "episode_bars": args.bars,
        "sampled_bars": args.sampled,
        "seed": args.seed,
        "source": args.source,
        "units": ("fraction of equity: 0.22 means occluding the input moves the "
                  "requested position by 22 percentage points of exposure"),
        "caveat": ("Local sensitivity of the deployed policy, not a causal claim "
                   "about markets. The ranking is not stable across retraining: "
                   "re-measuring against rebuilt archives changed which feature "
                   "came first in both markets."),
        "markets": markets,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
