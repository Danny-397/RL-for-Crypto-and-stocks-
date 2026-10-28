# Pre-registration: does the §5 result replicate on fresh seeds?

**Registered:** 2026-10-28, committed and merged to `main` *before* any of the
runs below were started. The commit timestamp is the registration record; the
results file will be added in a later commit and must not edit this one.

## Why

RESULTS.md §5 reports the real-data significance study on seeds **100–109**:

| Market | Agent (95% CI across seeds) | Buy & hold | Agent − B&H | p |
|---|---:|---:|---:|---:|
| Stock | −19.1% [−26.7%, −11.2%] | +239.5% | −258.6% | 0.0021 |
| Crypto | +56.8% [+16.3%, +106.3%] | +33.5% | +23.3% | 0.79 |

Those seeds were chosen before the result was known, but the write-up was
produced after looking at them. This study fixes the predictions first, then
runs ten seeds that have never been run.

## Design (fixed)

- **Code:** `tools/real_significance.py` at the commit that registers this file;
  the recipe is unchanged apart from the `--seed-base` / `--out` options.
- **Seeds:** 200–209 (`--seed-base 200 --seeds 10`). No seed in this range has
  been run for this study before.
- **Budget:** 150,000 timesteps per seed, as in §5.
- **Data:** the same 16 tickers and date windows as §5 (stocks 2016-08-29 to
  2026-08-28; crypto full history to 2026-08-28), re-fetched on 2026-10-28 and
  pinned in [`2026-10-28-fresh-seeds.SNAPSHOT.json`](2026-10-28-fresh-seeds.SNAPSHOT.json).
  **These files are not byte-identical to `data/SNAPSHOT.json`:** Yahoo has since
  revised its history (dividend re-adjustment of stock prices; small crypto
  revisions), and the original files were not kept. Dates and row counts match.
  So this replicates on fresh seeds *and* freshly downloaded data; a difference
  could come from either.
- **Command:**

  ```bash
  python tools/fetch_data.py --verify --snapshot docs/preregistration/2026-10-28-fresh-seeds.SNAPSHOT.json
  python tools/real_significance.py --seeds 10 --timesteps 150000 --seed-base 200 \
      --out docs/preregistration/2026-10-28-fresh-seeds.results.js
  ```

## Predictions

Stated from the §5 result and the project's reading of it ("no reliable,
seed-robust edge on real markets").

- **P1 (stock, direction):** the agent's basket-mean return, averaged across the
  ten seeds, is **below** buy-and-hold.
- **P2 (stock, significance):** the paired permutation test across the held-out
  tickers gives **p < 0.05**, with the agent worse.
- **P3 (crypto, no edge):** the paired permutation test gives **p ≥ 0.05**: the
  agent is not distinguishable from buy-and-hold.
- **P4 (crypto, spread):** the per-seed returns span **more than 100 percentage
  points** (§5's spanned −28.8% to +225.5%), i.e. single-seed results remain
  unreliable.

No prediction is made about the sign of the crypto mean; §5's positive mean
with an interval excluding zero is the result this study is most likely to
fail to reproduce, and that would be reported as such.

## Decision rules

- Each prediction is scored **held / failed** and reported in
  `2026-10-28-fresh-seeds.results.md`, with every per-seed number.
- If **P1 or P2 fails**, RESULTS.md §5's stock conclusion ("significantly worse
  than buy-and-hold") is marked as not replicated, in place, with a link here.
- If **P3 fails** (crypto p < 0.05), the headline "no seed-robust edge" is
  revised in README, RESULTS and the paper, whichever direction the edge points.
- Nothing about the design above changes after the runs start. Any deviation
  (a crash, a rerun) is listed in the results file with its reason.
