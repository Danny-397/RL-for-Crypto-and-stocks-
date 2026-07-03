# Paper

A short (arXiv-style) write-up of the project as a research study:
**`rl_trader.tex`** → a ~3-page, two-column PDF.

## Compile it (zero install — recommended)

1. Go to [overleaf.com](https://www.overleaf.com) → **New Project → Upload Project**
   (or **Blank Project** and paste `rl_trader.tex`).
2. Upload `rl_trader.tex`.
3. Hit **Recompile**. It uses only standard packages, so it builds as-is with no
   external figures.

## Compile locally

```bash
cd paper
pdflatex rl_trader.tex
pdflatex rl_trader.tex   # run twice so cross-references resolve
```

## Before you submit / share

- **Confirm the author line** — it currently reads "Daniel Lichtenberger"; edit if
  needed.
- Table 4 (surrogate test) is filled with a real 3-seed / 30k-step run. For a
  tighter result, regenerate at a larger budget and update the cells:
  ```bash
  python tools/surrogate_test.py --mode synthetic --seeds 5 --timesteps 60000
  ```
- Optionally run the **real-data arm** (the confirmatory null) and add a sentence:
  ```bash
  python tools/surrogate_test.py --mode real --seeds 3 --timesteps 120000
  ```
- Optionally drop the figures from `docs/assets/` (e.g. `fig_ablation.png`) into
  `paper/figures/` and add `\includegraphics` blocks — the `\graphicspath` already
  points there. The paper is intentionally figure-free so it compiles with nothing
  but the `.tex`.

All numbers in the paper are the real, reproducible results from `RESULTS.md`.
