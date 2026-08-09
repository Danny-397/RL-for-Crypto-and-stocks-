# Getting this paper a DOI

Everything here is ready to paste. The paper builds clean (`tectonic -X compile
rl_trader.tex`, ~6s) and `.zenodo.json` in the repo root pre-fills Zenodo's form.

---

## Do Zenodo first — and probably only Zenodo

**arXiv requires endorsement for a first submission to `cs.LG`.** Endorsement
comes from someone who has already published in that archive, and it is not
granted automatically to submitters without an academic affiliation. You may
well not clear it on a first try, and a rejected endorsement request costs you
nothing but takes weeks.

**Zenodo has no such gate.** It is run by CERN, issues a real DOI immediately,
is indexed by Google Scholar and OpenAIRE, and is accepted everywhere as a
citable archive. For the thing you actually want — a stable DOI you can put on
an application and a citation others can use — Zenodo is equivalent and
available today.

Recommended: publish to Zenodo now. Try arXiv afterwards if you want the extra
visibility; having a DOI already does not prevent an arXiv submission.

---

## Route A — Zenodo via GitHub release (10 minutes, recommended)

This wires the repo up so every future release is archived automatically.

1. Sign in at <https://zenodo.org> with your GitHub account.
2. Go to <https://zenodo.org/account/settings/github/> and flip the switch **on**
   for `Danny-397/RL-for-Crypto-and-stocks-`.
3. Back in the repo, cut a release — this is what triggers the archive:
   ```
   gh release create v1.0.0 \
     --title "RL-Trader v1.0.0 — Generalization and Evaluation Rigor in Deep RL" \
     --notes "First archived release. Multi-seed results (5 seeds x 60k steps) with bootstrap confidence intervals; surrogate-data falsification test with positive control." \
     paper/rl_trader.pdf
   ```
4. Zenodo picks up the release within a minute or two and mints the DOI, reading
   title/authors/description/keywords from `.zenodo.json`.
5. Add the DOI badge to the top of `README.md`:
   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```
6. Add the DOI to `CITATION.cff` as a top-level key so GitHub's "Cite this
   repository" button returns a citation with the DOI in it:
   ```yaml
   doi: "10.5281/zenodo.XXXXXXX"
   ```

**Cite the *concept* DOI, not the version DOI.** Zenodo mints both; the concept
DOI always resolves to the newest version and is the one to put on applications.

---

## Route B — arXiv (if you want to try)

- **Primary category:** `cs.LG` (Machine Learning)
- **Cross-list:** `q-fin.TR` (Trading and Market Microstructure), `stat.ML`
- **License:** arXiv non-exclusive-distrib (or CC BY 4.0)
- **Comments field:** `18 pages, N figures. Code and full evaluation suite: https://github.com/Danny-397/RL-for-Crypto-and-stocks-`
- **Format:** upload `rl_trader.tex` plus `figures/` as a single `.tar.gz`. arXiv
  builds from source; do not upload only the PDF unless the source fails.

Endorsement: if prompted, arXiv names the category and gives you a code to send
to a potential endorser. A teacher, mentor, or any author who has posted to
`cs.LG` can endorse.

---

## Abstract (plain text, for pasting into either form)

Deep reinforcement learning is unusually easy to fool yourself with: agents
memorize their training trajectory, and a single lucky random seed can
masquerade as a real result. We study both failure modes from scratch — a
PyTorch implementation of Proximal Policy Optimization (PPO), two custom
Gymnasium environments, and a full evaluation suite — using financial markets as
a deliberately hard testbed: non-stationary, partially observable, with a noisy
reward and near-random-walk signal. We ask (RQ1) whether one fixed recipe
generalizes across two market regimes; (RQ2) how severely an agent overfits a
single price trajectory and whether domain randomization repairs it; (RQ3)
whether an apparent out-of-sample edge survives multi-seed significance testing;
and (RQ4) whether the same holds for cross-sectional allocation. Our headline
result is negative, and that is the point: on real markets the agent has no
seed-robust edge over buy-and-hold, and our own significance tooling catches a
single-seed "+275%" run as a false positive. We further introduce a
surrogate-data falsification test that separates "the agent is weak" from "there
is no signal to find," and validate it with a positive control on synthetic data
with known structure. In doing so the project reproduces, in a new domain, the
central methodological finding of Henderson et al. (2018) — that single-run RL
evaluation is unreliable. We then apply that finding to our own ablation, whose
domain-randomized arm is unseeded by construction and so cannot be reproduced
from a single run: we report both arms as distributions over five seeds, where
all four confidence intervals exclude zero.

---

## Before you submit — worth one pass

- [ ] Confirm the figure count in the arXiv comments line matches `figures/`.
- [ ] Confirm every number in the paper matches the current multi-seed run.
      Table 1 was stale once already (it described a 19-feature model after the
      repo moved to 28), so it is worth re-checking against `tools/ablation_multiseed.py`
      output rather than trusting the committed text.
- [ ] `date-released` in `CITATION.cff` currently says 2026-07-03 — update it to
      the release date when you cut the tag.
