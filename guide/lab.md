# The interactive research lab and its API

What the lab at [rl-for-crypto-and-stocks.vercel.app](https://rl-for-crypto-and-stocks.vercel.app/) does, what is live and what is not, and the HTTP API behind it.

[← back to the README](../README.md)

## The interactive research lab

The site in [`docs/`](../docs/) is a **laboratory**, not a slide deck. It is deployed
at **[rl-for-crypto-and-stocks.vercel.app](https://rl-for-crypto-and-stocks.vercel.app/)**
and its panels are the research questions made runnable.

| Panel | What you can do | Live? |
|---|---|---|
| **Signal or Noise?** | Before testing the agent, test yourself: tell charts carrying the agent's training-time autocorrelation from pure random walks, under a design where volatility and drift are standardised away. Scored with an exact binomial test, beside a power analysis and a one-line statistical rule run on the same charts. | ✅ live |
| **Your Turn** | Trade the same bars the agent trades, through the same environment and cost model, one bar at a time with no lookahead — then get scored against the agent and against buy-&-hold over exactly the bars you traded. | ✅ live |
| **Agent Playground** | Configure market, data source, capital, costs, reward, shorting — then run a real episode and scrub it bar by bar against buy-&-hold. | ✅ live |
| **Agent X-Ray** | At any bar, read the actual observation → policy → action → reward → position chain, all 28 features grouped as the pipeline defines them, plus the 20-bar window as a heatmap — and an occlusion pass ranking which of those inputs actually move the action. | ✅ live |
| **Can You Break It?** | The real domain-randomization ablation (Agent A vs Agent B) with per-seed points and CIs, plus a live shift test that drops the deployed policy onto controlled synthetic regimes. | mixed — see below |
| **Walk-Forward** | Disjoint chronological folds from the research code's own splitter, each scaled on its training rows alone, with the deployed policy scored on every out-of-sample block — plus a second pass with the scaler fit on everything, measuring what that shortcut costs. | mixed — see below |
| **Anything There?** | The surrogate-data falsification test: shuffle a series' returns and buy-&-hold is unchanged while every pattern is destroyed. Run with a positive control first, so the real arm's null actually means something. | precomputed — both arms are training runs |
| **Real or Luck?** | The published single-seed headline beside the ten-seed distribution, with the bootstrap and permutation machinery re-runnable at your own confidence level and resample count. | mixed — see below |
| **Notebook** | Every experiment this session, with its config, receipt, pre-registered prediction, and a Reproduce button that replays it exactly. | ✅ live |

### What is live, and what is not — and why

This matters more than any feature in the table, so the API states it explicitly
at [`/api/meta`](../server/app.py) rather than leaving the frontend to imply it.

**Live on request.** Rollouts, counterfactuals, distribution-shift sweeps, and
*all* statistical inference. These are cheap: a full episode is one NumPy forward
pass per bar and completes in seconds.

**Not live: training.** The serving container has no PyTorch and a fraction of a
CPU. More fundamentally, the seed variation this project is *about* is variation
across **training** seeds, and each of those points is a complete PPO run — a
"run 5 seeds" button that returned in two seconds would be a lie. So seed-level
and ablation results are served from the repository's **real committed
experiments**, each labelled with its source file and the command that
regenerates it. The statistics computed *over* that real data run live, which is
the honest and more instructive half: you can change the design and watch a
genuine p-value move.

**Nothing is ever fabricated.** If the backend cannot compute something it is
omitted and labelled, never filled in with a plausible number. The clearest
example: the deployed policy archives contain the actor but **no critic head**,
so the X-Ray's value slot reads *"not exported"* instead of showing an invented
estimate.

### What the agent is actually reading

The X-Ray showed all 563 inputs but never said which mattered. An occlusion pass
now answers that live: hold the observation fixed, replace one input with an
uninformative baseline — a feature by its mean over the series, an account scalar
by its value at reset — and measure how far the deterministic target position
moves. Because a feature occupies a *column* of the flattened window, occluding
it removes all 20 bars of that indicator, not one cell.

Measured on 600-bar momentum paths, the answer was not what I expected — and it
is less stable than a ranked bar chart makes it look:

<!-- BEGIN GENERATED: attribution-table -->
| | Strongest inputs | Largest account-state effect |
|---|---|---:|
| **Stock** | `volatility_10` 0.17 · `atr_norm` 0.17 · `volume_zscore` 0.15 | 0.030 |
| **Crypto** | `volatility_10` 0.23 · `return_120` 0.18 · `high_low_range` 0.17 | 0.043 |

*600-bar momentum paths, seed 1, 80 sampled bars. Fraction of equity: 0.22 means occluding the input moves the requested position by 22 percentage points of exposure.*
<!-- END GENERATED: attribution-table -->

Both policies lean on **range, volatility and long-horizon position** features
rather than the short-horizon momentum an intuitive reading would expect. And the
three account scalars move the action by roughly an order of magnitude less than
the top market features: these agents barely track their own book.

The instability is worth stating plainly. These figures were first measured
against the previous policy archives; re-measuring against the rebuilt ones left
the magnitudes almost unchanged but **changed which feature came first in both
markets** — stock moved from `high_120_dist` to `high_low_range`, crypto from
`vol_regime` to `high_120_dist`. The top few features are separated by less than
the run-to-run noise, so the ranking's *order* should not be read as a finding.
What survives re-measurement is the shape of the answer: range and volatility
inputs dominate, momentum does not, and account state barely registers. This is
why the panel calls the measurement local sensitivity rather than importance.

The panel ships its own limits alongside the chart, because a ranked bar chart is
exactly the kind of output a reader takes for causation:

- Occlusion is **local sensitivity, not causal importance**.
- The 28 features are correlated, so a feature's information survives its own
  removal through the others. Contributions are understated and a low bar is not
  proof of irrelevance.
- Replacing an input with its mean can produce a vector no real market would
  generate.

It does earn one clean structural check: on synthetic paths the four cross-asset
features have no reference index and come back at exactly zero, which the panel
names rather than leaving as four unexplained flat bars.

### Is there anything there to find?

Every other experiment shows the agent failing to beat buy-and-hold. None of them
can say **why** — a weak agent and an empty market look identical from the
outside. `tools/surrogate_test.py` separates them, using surrogate-data testing
(Theiler et al. 1992): randomly permute a series' daily log returns and
re-integrate. A sum is permutation-invariant, so the price ends in *exactly* the
same place and buy-and-hold is identical — but autocorrelation, momentum and
volatility clustering are gone. Train the same recipe on both.

**The positive control is the half that matters**, and it is presented first,
because a null from a test with unproven power says nothing:

<!-- BEGIN GENERATED: surrogate-brief -->
| Arm | Market | Structured | Surrogate | Difference | p |
|---|---|---:|---:|---:|---:|
| Synthetic (control) | crypto | +68.6% | −55.7% | **+124.3%** | **0.0075** |
| Synthetic (control) | stock | +1.8% | −47.7% | **+49.5%** | **0.0026** |
| Real | crypto | −20.6% | −977.0% | +956.4% | 0.1225 |
| Real | stock | −253.4% | −136.3% | −117.1% | 0.4150 |
<!-- END GENERATED: surrogate-brief -->

On synthetic data with a planted AR(1) signal the agent shows a clear edge and
loses it under shuffling, in both markets. That licenses reading the real arm:
**the agent does no better on real price history than on the same returns in a
random order.** On this evidence the flat performance is the market's, not the
agent's — which is what weak-form efficiency predicts, arrived at from the
inside.

The crypto real row is instructive on its own: a large positive point estimate
that is *not* significant, because the surrogate arm's interval is enormous —
reshuffling fat-tailed crypto returns occasionally creates paths on which a
leveraged agent loses catastrophically, and those dominate the mean. A large
point estimate with no power is not a result, and the panel says so rather than
quoting the headline number.

`tools/surrogate_test.py` records the per-arm values and `n_pairs`, so every row
above is re-analysable live rather than being a stored summary a p-value would
have to be re-derived from. That is what lets the median-based variant in
[RESULTS §6](../RESULTS.md) run without retraining anything: same paired null, same
pairs, a statistic chosen to be immune to the heavy tails. An artifact that
carries only summary statistics is still marked non-re-analysable by the API
instead of being silently re-analysed.

### Does it beat a coin flip?

"Beats buy-and-hold" is the bar this project discusses; it is not the first bar a
skeptical reader has in mind. Every rollout in the Playground now scores the
agent against the naive strategies from `rl_trader.evaluation.baselines` on the
same series, through the same environment, paying the same costs.

Two choices decide whether that table means anything. **The random arm is 24
independent draws**, reported as a mean with its full observed range — quoting
one random run would be exactly the single-sample mistake this project exists to
criticise — and the agent is only described as beating it when it clears the
whole range, not the mean. **And there are two buy-and-holds**: the cost-free
benchmark drawn on the charts, and the strategy that goes through the
environment and pays the entry cost. Both are shown and labelled, because
quietly swapping one for the other flatters or penalises the agent for no stated
reason.

The agent's row is highlighted but never sorted to the top. On many paths it
finishes last — behind a moving-average crossover, behind holding cash, and
behind random positions — and that is the point of running the comparison.

### Walk-forward, and the price of a leaky scaler

The leakage controls were the least visible thing in the project: a claim in this
README and a few lines in `rl_trader/evaluation/walk_forward.py`. The panel runs
that same splitter — imported, not reimplemented — over a real series, fits a
**separate feature scaler on each fold's training rows**, and scores the deployed
policy on every out-of-sample block.

**It is not a retrained walk-forward, and it says so on every response.** This
backend has no PyTorch, so the policy is fixed across folds. What that measures is
still worth seeing: on the panel's default synthetic path (crypto policy, momentum
regime, seed 3, 1,600 bars, four expanding folds), the v2.3.0 policy's excess return
over buy-and-hold ran from **−61.5% to +65.2%** across chronological blocks. A single
backtest would have reported any one of those.

Then it runs the identical folds a second time with the scaler fit on the whole
series, test block included — the standard mistake — and reports the difference
fold by fold. Same policy, same prices, only the scaler's fitting window changes.
On that series the largest single-fold gap was **9.4 percentage points**. Nothing
about that is asserted in advance; it is whatever it comes out as, and a test
fails if the two arms ever come back identical, because then the comparison would
be showing nothing and should not be displayed.

The aggregate carries the same resolution warning as everything else: a sign test
over four folds cannot produce a p-value below `2/2⁴ = 0.125`, so that design
cannot reach 0.05 whatever the effect size.

### The human baseline

A visitor can read every finding here and still assume they personally would have
done better. So let them try, on the identical series, through the identical
environment, paying the identical costs.

**No lookahead is a property of the protocol, not a promise in the copy.** The
price series lives on the server. A session hands out the warm-up window and then
exactly one new bar per decision, so the page never holds a price the visitor has
not already traded through — there is nothing to read ahead in. Sessions are
in-memory, capped, and expire after 30 minutes; the API says so rather than
implying anything durable.

**It is deliberately not a like-for-like contest, and every payload says so.** The
visitor sees a price chart and their own account; the agent sees 28 standardised
indicators over a 20-bar window plus three account scalars. The comparison that
*is* fair is the third line — buy-and-hold over the same bars through the same
cost model — which is the benchmark this whole project is about. The verdict
leads with that rather than the head-to-head, and the usual outcome, that
neither party clears it, is the finding rather than a disappointment.

### Pre-registration

Every panel here lets you run something and then read a number, which is exactly
the order in which researchers talk themselves into results. So the notebook lets
you commit first: state the question, pick a prediction, and the backend stamps
it onto the experiment record *before* the worker thread starts.

Two details make it a real commitment rather than a caption:

- **There is no endpoint that edits it.** Changing your mind after seeing the
  outcome means creating a new experiment with a new id, and the notebook lists
  both. The record of what you actually predicted survives.
- **The judging rule is a module constant, not a request field.** `MATCH_BAND` is
  ±2 percentage points of buy-and-hold, published in `/api/meta`, shown in the
  form before you choose, restated on the result, and applied identically to
  every prediction. You can disagree with the band; you cannot move it after the
  fact. A test asserts that passing `match_band` in the request does nothing.

A kind with no single agent-vs-benchmark number — a counterfactual, say —
records the prediction and reports `scorable: false` rather than scoring it
against a different quantity. And a hit is worded as weak evidence, because one
run agreeing with one prediction is one run.

### Testing the reader, not just the agent

The first panel is the project's thesis turned on the visitor. Half the charts
are drawn with the AR(1) return autocorrelation the agents were **trained
against** (`market_regime`); half are the same generator with `momentum = 0`. A
"real" condition instead pits disjoint slices of a ticker's own history against
those *same slices with their daily returns permuted* — a surrogate that keeps
the entire marginal distribution (mean, variance, skew, fat tails) and destroys
only the ordering, so the only thing left to see is time structure.

Three design choices make the result mean something:

- **Confound control.** Every chart is standardised to an identical return
  volatility and given a drift drawn from one shared distribution. Standardising
  is affine, so it leaves autocorrelation exactly intact while erasing every
  other cue. Classes are exactly balanced, then shuffled.
- **The key never leaves the server.** A quiz is a deterministic function of
  `(difficulty, source, seed, n_charts)`; scoring rebuilds it and compares. There
  is no answer to read out of the page.
- **The inference is exact and honestly underpowered.** Scoring reports an exact
  two-sided binomial test, the floor `2 / 2**n` the design can attain at all, and
  its **power**: at n = 8, a genuinely 70%-accurate observer is detected under
  10% of the time. Failing to reach significance is not evidence of no skill —
  the same distinction the seed-level tests below turn on.

The measured effect is the point: the "trending" class has a lag-1 return
autocorrelation of roughly **+0.11** against **−0.02** for the control. Real,
exploitable in principle, and invisible to the eye — while the one-line
autocorrelation rule reported beside your score typically gets 6 of 8. That gap
is the whole argument for measuring instead of looking.

### One statistical detail the lab is careful about

The project reports two numbers that live on **different axes**, and pairing on
the wrong one silently changes the claim:

- **across training seeds** (n = 10) — "how repeatable is this?" → bootstrap CI
- **across held-out tickers** (n = 10 / 6) — "is the cross-sectional edge real?"
  → paired permutation test, which is where the published p-value comes from

A two-sided sign-flip test over `n` pairs draws from only `2**n` sign
assignments, so **p can never fall below `2 / 2**n`** — 0.0625 at n = 5, and
0.002 at the n = 10 this study now runs. At n = 5 the design cannot reach
significance at 0.05 whatever the effect size; at n = 10 it can, but never below
0.002. The lab reports this resolution floor beside every test, because
*"underpowered by construction"* and *"no effect"* are different statements. It
shows up concretely in the ablation, where the held-out difference between the two
agents is decisive and the p-value sits at the floor itself: ten pairs cannot
resolve anything stronger.

<!-- BEGIN GENERATED: ablation-resolution -->
- **Stock** — held-out difference **+56.1%**, 95% CI `[+40%, +74%]`, yet **p = 0.0021** against a floor of 0.0020 at 10 pairs.
- **Crypto** — held-out difference **+230.4%**, 95% CI `[+186%, +277%]`, yet **p = 0.0021** against a floor of 0.0020 at 10 pairs.
<!-- END GENERATED: ablation-resolution -->

---

## API

The backend ([`server/`](../server/)) is the experiment engine. It imports the
research code rather than reimplementing it, so the site cannot drift from the
paper.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, loaded policies, and per-policy capabilities |
| `GET /api/meta` | What this backend can actually do — including `live.training: false` |
| `GET /api/regimes` | Synthetic distribution-shift regimes |
| `GET /api/datasets` | Real committed per-seed and per-ticker datasets, with provenance |
| `GET /api/generalization` | The real single-path vs domain-randomized ablation |
| `GET /api/surrogate` | The surrogate-data falsification test, both arms, with provenance |
| `GET /api/perception/quiz` | A balanced signal-vs-noise chart test, served without its answer key |
| `POST /api/perception/score` | Exact binomial scoring of a submission, plus power and a statistical reference |
| `POST /api/statistics` | Live bootstrap / permutation inference over that real data |
| `POST /api/experiments` | Create an experiment (async; returns an id immediately) |
| `GET /api/experiments` | Session history |
| `GET /api/experiments/<id>` | Status, progress, result, receipt |
| `GET /api/experiments/<id>/config` | The exact config needed to reproduce it |
| `GET /api/experiments/<id>/xray?step=` | The full observation at one bar |
| `GET /api/experiments/<id>/attribution` | Occlusion attribution: which inputs move the action |
| `POST /api/experiments` `kind=walk_forward` | Rolling folds, per-fold scaling, and the leakage comparison |
| `POST /api/experiments` `prediction=` | Pre-register a prediction; the backend stamps it before the run and scores it after |
| `POST /api/human/start` | Open a human-baseline session (the series stays server-side) |
| `POST /api/human/<sid>/step` | One decision; releases exactly one new bar |
| `POST /api/human/<sid>/finish` | Score it against the agent and buy-&-hold on the same bars |
| `GET /api/results`, `/api/live`, `/api/tickers` | The original dashboard endpoints (unchanged) |

### Run an experiment

```bash
# 1. Create it — returns an id like EXP-8F42A straight away
curl -s -X POST http://localhost:8000/api/experiments \
  -H 'Content-Type: application/json' \
  -d '{"kind": "rollout",
       "question": "Does the agent survive mean reversion?",
       "config": {"market": "stock", "mode": "synthetic",
                  "regime": "mean_reversion", "seed": 7}}'

# 2. Poll it
curl -s http://localhost:8000/api/experiments/EXP-8F42A

# 3. Get everything needed to reproduce it
curl -s http://localhost:8000/api/experiments/EXP-8F42A/config
```

`kind` is `rollout`, `counterfactual`, or `distribution_shift`. Experiments are
**ephemeral** — held in memory on a single worker that restarts when idle — and
the API says so rather than implying durable storage.
