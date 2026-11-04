# Paper

A short (arXiv-style) write-up of the project as a research study:
**`rl_trader.tex`** → a ~3-page, two-column PDF.

## Compile it (zero install — recommended)

1. Go to [overleaf.com](https://www.overleaf.com) → **New Project → Upload Project**
   (or **Blank Project** and paste `rl_trader.tex`).
2. Upload `rl_trader.tex`.
3. Upload `figures/` alongside it (one PNG, used by Table 1's companion figure).
4. Hit **Recompile**. It uses only standard packages.

The repository also builds the PDF on every push via
[`.github/workflows/paper.yml`](../.github/workflows/paper.yml) and commits the
result back, so `paper/rl_trader.pdf` is always current with the source.

## Compile locally

```bash
cd paper
pdflatex rl_trader.tex
pdflatex rl_trader.tex   # run twice so cross-references resolve
```

## Where each number comes from

| Table | Source file | Command |
|---|---|---|
| 1 — ablation | `docs/assets/ablation_multiseed.json` | `python tools/ablation_multiseed.py --seeds 42 43 44 45 46 47 48 49 50 51 --timesteps 60000` |
| 2 — seed robustness | `RESULTS.md` | `python tools/eval_seeds.py` |
| 3 — significance | `RESULTS.md` | `python tools/significance.py` |
| 4 — surrogate (synthetic) | `docs/assets/surrogate_synthetic.json` | `python tools/surrogate_test.py --mode synthetic --seeds 5 --timesteps 60000` |
| — surrogate (real, confirmatory null) | `docs/assets/surrogate_real.json` | `python tools/surrogate_test.py --mode real --seeds 3 --timesteps 120000` |

Both surrogate arms have been run. The real-data arm's p-values (0.4984 equities,
0.1556 crypto) are quoted in §5. Table 4 was regenerated at 5 seeds / 60k steps on
2026-08-01, which moved crypto's positive control from a marginal p ≈ 0.059 to
p ≈ 0.0032 — the control now passes in both markets rather than one.

## A caveat that is in the paper, and belongs here too

**The domain-randomized ablation arm used to be unreproducible. It is now seeded.**

Domain randomization draws a fresh path every episode. In earlier builds that
stream came from `synthetic_market_data(market)` with no seed, so `--seed` did not
determine the arm's data and no single run of it could be reproduced. Since v2.1
`tools/ablation.py` builds the stream with `seeded_path_factory(market, seed)`, which
keeps the per-episode variety while making each seed reproduce its run exactly.
Table 1 was regenerated with the seeded stream (10 seeds), and still reports
distributions with bootstrap intervals rather than single runs.

**Separately: an earlier version of Table 1 was stale.** Its numbers were
generated before commit `d4c0ef9` took the observation feature set from 19 to 28,
so they described a model the repository no longer ships. They were transcribed
into the paper rather than regenerated. Regenerate from
`docs/assets/ablation_multiseed.json` rather than copying cells by hand.
