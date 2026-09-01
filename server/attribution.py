"""Occlusion attribution: which inputs is the policy actually reading?

The X-Ray shows all 563 numbers the agent consumes. It never said which of them
*mattered*. This answers that directly, and cheaply enough to run live: hold the
observation fixed, replace one input with an uninformative baseline, and measure
how far the action moves.

Method
------
The observation is a ``window x n_features`` block flattened row-major, followed
by three account scalars. Occluding feature ``j`` therefore means overwriting
``j, j + F, j + 2F, ...`` — the whole 20-bar history of that one feature, not a
single cell — with its mean over the series. Features are standardised, so that
baseline is roughly zero and means "this indicator sat at its average". The
account scalars are occluded to their episode-start values (flat position, full
cash, equity at par): the state of an agent that has done nothing yet.

The reported quantity is the change in the deterministic target position, in the
action's own units. A feature worth 0.30 moves the agent's requested exposure by
30% of equity when it is taken away.

What this is and is not
-----------------------
This is **local sensitivity**, not causal importance, and the difference is worth
being blunt about:

* The 28 features are heavily correlated. Occluding one leaves much of its
  information present in the others, so shared contributions are systematically
  understated. A feature reading zero here is not proven irrelevant.
* Replacing an input with its mean is itself an intervention. The resulting
  vector may be one no real market would produce, and the network's response
  there is not guaranteed to be meaningful.
* A single bar is one point in input space. The episode-level pass averages the
  magnitude over many bars, which is more stable but still not global.

Those limits are returned in the payload, not left in this docstring, because a
ranked bar chart is exactly the kind of output a reader will over-read.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

ACCOUNT_NAMES = ("position_fraction", "cash_fraction", "equity_normalised")
# An agent that has just reset: flat, fully in cash, equity at par. This is the
# uninformative state for the account block, in the same spirit as "this feature
# sat at its average" for the market block.
ACCOUNT_BASELINE = (0.0, 1.0, 1.0)


def feature_indices(window: int, n_features: int, j: int) -> np.ndarray:
    """Every position feature ``j`` occupies in the flattened observation."""
    return np.arange(j, window * n_features, n_features)


def _occluded(obs: np.ndarray, idx: Sequence[int], value: float) -> np.ndarray:
    out = np.array(obs, dtype=np.float32, copy=True)
    out[np.asarray(idx, dtype=int)] = value
    return out


def local_attribution(
    policy,
    obs: np.ndarray,
    feature_means: np.ndarray,
    feature_names: Sequence[str],
    window: int,
) -> Dict[str, List[dict]]:
    """Occlude each feature (and each account scalar) once, at this observation.

    Returns market-feature and account rows separately: they are different kinds
    of input and averaging them into one ranking would invite a comparison that
    is not quite apples to apples.
    """
    n_features = len(feature_names)
    base = float(policy.act(obs))

    market: List[dict] = []
    for j, name in enumerate(feature_names):
        idx = feature_indices(window, n_features, j)
        alt = float(policy.act(_occluded(obs, idx, float(feature_means[j]))))
        market.append({
            "name": name,
            "delta_action": round(alt - base, 6),
            "abs_delta": round(abs(alt - base), 6),
            "occluded_action": round(alt, 6),
        })

    account: List[dict] = []
    tail = len(obs) - 3
    for k, name in enumerate(ACCOUNT_NAMES):
        alt = float(policy.act(_occluded(obs, [tail + k], ACCOUNT_BASELINE[k])))
        account.append({
            "name": name,
            "delta_action": round(alt - base, 6),
            "abs_delta": round(abs(alt - base), 6),
            "occluded_action": round(alt, 6),
        })

    return {"base_action": round(base, 6), "market": market, "account": account}


def episode_attribution(
    policy,
    env,
    feature_names: Sequence[str],
    window: int,
    max_bars: int = 60,
) -> dict:
    """Average occlusion magnitude over bars the policy actually visits.

    The episode is replayed under the policy's own decisions, and attribution is
    measured at evenly spaced bars along it. Averaging the *magnitude* is
    deliberate: signed effects cancel across bars, which would make a feature the
    agent leans on hard in both directions look like one it ignores.
    """
    feature_means = np.asarray(env.data.features, dtype=np.float64).mean(axis=0)
    n_features = len(feature_names)

    obs, _info = env.reset()
    total_bars = max(1, len(env.data.prices) - window)
    stride = max(1, total_bars // max(1, int(max_bars)))

    sums = np.zeros(n_features)
    peaks = np.zeros(n_features)
    acct_sums = np.zeros(3)
    sampled = 0
    step = 0

    while True:
        if step % stride == 0:
            out = local_attribution(policy, obs, feature_means, feature_names, window)
            mags = np.array([r["abs_delta"] for r in out["market"]])
            sums += mags
            peaks = np.maximum(peaks, mags)
            acct_sums += np.array([r["abs_delta"] for r in out["account"]])
            sampled += 1
        action = policy.act(obs)
        obs, _r, term, trunc, _i = env.step(np.array([action], dtype=np.float32))
        step += 1
        if term or trunc:
            break

    denom = max(1, sampled)
    features = [
        {
            "name": feature_names[j],
            "mean_abs_delta": round(float(sums[j] / denom), 6),
            "max_abs_delta": round(float(peaks[j]), 6),
        }
        for j in range(n_features)
    ]
    account = [
        {"name": ACCOUNT_NAMES[k], "mean_abs_delta": round(float(acct_sums[k] / denom), 6)}
        for k in range(3)
    ]
    return {
        "bars_sampled": sampled,
        "bars_total": int(step),
        "stride": stride,
        "features": features,
        "account": account,
    }


def group_shares(
    rows: Sequence[dict], groups: Dict[str, Sequence[str]], key: str = "mean_abs_delta"
) -> List[dict]:
    """Roll per-feature magnitudes up into the pipeline's own feature groups.

    Shares are of the summed magnitude, which is a description of this
    measurement — not a decomposition of the policy. Correlated features double-
    count, so these do not partition anything; they rank.
    """
    by_name = {r["name"]: float(r.get(key, 0.0)) for r in rows}
    out = []
    for label, names in groups.items():
        present = [n for n in names if n in by_name]
        total = sum(by_name[n] for n in present)
        out.append({
            "label": label,
            "n_features": len(present),
            "total_abs_delta": round(total, 6),
            "mean_abs_delta": round(total / len(present), 6) if present else 0.0,
        })
    grand = sum(g["total_abs_delta"] for g in out)
    for g in out:
        g["share"] = round(g["total_abs_delta"] / grand, 4) if grand > 1e-12 else 0.0
    return out


def dead_features(rows: Sequence[dict], key: str = "mean_abs_delta",
                  tol: float = 1e-6) -> List[str]:
    """Features that moved the action by nothing at all, anywhere sampled.

    On synthetic paths the four cross-asset features are structurally zero, so
    this list is expected to be non-empty there — and saying which ones is more
    useful than letting a reader puzzle over four flat bars.
    """
    return [r["name"] for r in rows if abs(float(r.get(key, 0.0))) <= tol]


CAVEATS = [
    "Occlusion measures local sensitivity, not causal importance.",
    "The 28 features are correlated, so a feature's information survives its own "
    "removal through the others. Contributions are understated, and a low bar is "
    "not proof of irrelevance.",
    "Replacing an input with its mean can produce a vector no real market would "
    "generate; the network's response there is not guaranteed to be meaningful.",
    "Magnitudes are in units of the target position: 0.30 means the requested "
    "exposure moves by 30% of equity when that input is removed.",
]

METHOD_NOTE = (
    "Each feature is occluded across its entire window (all 20 bars of it) and "
    "replaced by its mean over this series; the account scalars are replaced by "
    "their episode-start values. The policy is re-evaluated and the change in the "
    "deterministic target position is recorded."
)


def summarise(
    local: dict,
    episode: Optional[dict],
    groups: Dict[str, Sequence[str]],
) -> dict:
    """Assemble the response, ranked, with its own limits attached."""
    local_market = sorted(local["market"], key=lambda r: -r["abs_delta"])
    ranked = sorted(episode["features"], key=lambda r: -r["mean_abs_delta"]) if episode else []
    return {
        "base_action": local["base_action"],
        "local": {"market": local_market, "account": local["account"]},
        "episode": (
            {
                **{k: v for k, v in episode.items() if k != "features"},
                "features": ranked,
                "dead_features": dead_features(ranked),
            }
            if episode
            else None
        ),
        "groups": group_shares(ranked or local_market, groups,
                               key="mean_abs_delta" if ranked else "abs_delta"),
        "method": METHOD_NOTE,
        "caveats": CAVEATS,
        "live_computation": True,
    }
