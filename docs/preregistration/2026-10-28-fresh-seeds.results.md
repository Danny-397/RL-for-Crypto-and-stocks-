# Results: fresh-seed replication (registered 2026-10-28)

Registration: [`2026-10-28-fresh-seeds.md`](2026-10-28-fresh-seeds.md), merged to
`main` as commit `d344b4e` (2026-10-27 21:22 ET) before the run started.
Run: 2026-10-27 21:23–22:06 ET, exactly the registered command; data verified
against the registered snapshot first. Raw output:
[`2026-10-28-fresh-seeds.results.js`](2026-10-28-fresh-seeds.results.js).

**Deviations: none.** No crash, no rerun, no change to code, seeds, budget or data.

## Outcome

| Market | Agent (95% CI across seeds) | Buy & hold | Agent − B&H | p |
|---|---:|---:|---:|---:|
| Stock | −19.8% [−27.3%, −12.0%] | +239.5% | −259.3% | 0.0021 |
| Crypto | +26.7% [−13.8%, +70.4%] | +33.5% | −6.8% | 0.88 |

Per-seed basket-mean returns (seeds 200–209):

- **Stock:** −19.4%, +2.9%, −12.2%, −38.6%, −4.5%, −29.5%, −18.7%, −33.5%, −30.0%, −14.1%
- **Crypto:** +71.5%, +88.4%, −35.7%, −19.4%, +121.4%, −51.4%, +139.6%, −29.3%, −40.2%, +22.0%

## Predictions

| | Prediction | Result | |
|---|---|---|---|
| P1 | Stock agent below buy-and-hold | −19.8% vs +239.5% | **held** |
| P2 | Stock permutation p < 0.05, agent worse | p = 0.0021 | **held** |
| P3 | Crypto permutation p ≥ 0.05 | p = 0.88 | **held** |
| P4 | Crypto per-seed spread > 100 points | −51.4% to +139.6% (191 points) | **held** |

All four predictions held, so under the registered decision rules the README,
RESULTS and paper headlines stand unchanged.

## What did not replicate

The registration named one §5 result as the likeliest to fail, and it did:
§5's crypto mean (+56.8%) had a bootstrap interval that **excluded zero**. On
these seeds the mean is +26.7% and its interval, [−13.8%, +70.4%], **includes
zero**, and the agent is now 6.8 points *behind* buy-and-hold rather than
23.3 ahead. That is the project's argument working as intended: the positive
crypto mean was itself partly seed luck. RESULTS §5 now says so and links here.

Caveat: these runs used re-fetched data that is not byte-identical to §5's
(Yahoo re-adjusted history after August; see the registration), so the
difference could come from the data as well as the seeds.
