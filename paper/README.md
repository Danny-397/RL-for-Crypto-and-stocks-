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
| 1 — ablation | `docs/assets/ablation_multiseed.json` | `python tools/ablation_multiseed.py --seeds 42 43 44 45 46 --timesteps 60000` |
| 2 — seed robustness | `RESULTS.md` | `python tools/eval_seeds.py` |
| 3 — significance | `RESULTS.md` | `python tools/significance.py` |
| 4 — surrogate (synthetic) | `docs/assets/surrogate_synthetic.json` | `python tools/surrogate_test.py --mode synthetic --seeds 5 --timesteps 60000` |
| — surrogate (real, confirmatory null) | `docs/assets/surrogate_real.json` | `python tools/surrogate_test.py --mode real --seeds 3 --timesteps 120000` |

Both surrogate arms have been run. The real-data arm's p-values (0.4984 equities,
0.1556 crypto) are quoted in §5. Table 4 was regenerated at 5 seeds / 60k steps on
2026-08-01, which moved crypto's positive control from a marginal p ≈ 0.059 to
p ≈ 0.0032 — the control now passes in both markets rather than one.

## A caveat that is in the paper, and belongs here too

**The two ablation arms are not equally reproducible, and one of them cannot be.**

In `tools/ablation.py`:

```python
if domain_random:
    factory = lambda: synthetic_market_data(market)          # no seed
else:
    fixed = synthetic_market_data(market, seed=seed)         # seeded
```

The single-path arm reproduces exactly — two runs at seed 42 / 60k agree to 0.1%
in both markets. The domain-randomized arm draws a fresh **unseeded** series every
episode, which is what domain randomization *is*, but it means `--seed` does not
determine its data stream and no single run of it is reproducible even in
principle. Comparing one run of each arm is therefore not a controlled comparison,
which is why Table 1 reports distributions over seeds with bootstrap intervals.

**Known issue, not yet fixed.** The right fix is to derive each episode's path
seed from a seeded generator — `rng = np.random.default_rng(seed)` then
`synthetic_market_data(market, seed=int(rng.integers(2**31)))` — which keeps the
variety that makes domain randomization work while making the arm reproducible.
This is deliberately left undone here because changing it would invalidate the
numbers currently in the table.

**Separately: an earlier version of Table 1 was stale.** Its numbers were
generated before commit `d4c0ef9` took the observation feature set from 19 to 28,
so they described a model the repository no longer ships. They were transcribed
into the paper rather than regenerated. Regenerate from
`docs/assets/ablation_multiseed.json` rather than copying cells by hand.
