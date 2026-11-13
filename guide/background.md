# Background: the question, the field, the write-up

Why the study is built the way it is: its abstract and research questions, why markets are the testbed, where it sits in the RL literature, and how the write-up is organised.

[← back to the README](../README.md)

## Abstract

Deep reinforcement learning is notoriously easy to *fool yourself* with: agents
memorise their training trajectory, and a single lucky random seed can look like a
real result. This project is a from-scratch study of both failure modes, using
financial markets as a deliberately hard environment — **non-stationary, partially
observable, with a noisy reward and near-random-walk signal**. I implement
**Proximal Policy Optimization (PPO)** from the algorithm up in PyTorch and train one
unified agent on two custom Gymnasium environments (equities and crypto) to ask three
research questions: **(1)** does the same learning recipe generalize across two
market regimes; **(2)** how badly does an RL agent overfit a single price trajectory,
and does **domain randomization** fix it; and **(3)** does an apparent out-of-sample
edge survive **multi-seed significance testing**? The headline result is negative and
that is the point: on real markets the agent has **no seed-robust edge over
buy-and-hold**, and the framework's own significance tooling is what caught it: an
earlier build of this study published the crypto agent at +275% (commit `d4c0ef9`),
and that run survived neither reseeding nor a two-month extension of the evaluation
window. In doing so the project independently reproduces, in
a new domain, the central methodological finding of Henderson et al. (2018), *Deep
Reinforcement Learning that Matters* — that single-run RL evaluations are unreliable —
and shows that domain randomization turns held-out return from **reliably negative
to reliably positive** in both markets, across ten seeds.

> **Built from scratch in PyTorch:** the PPO algorithm, both environments, the data
> pipeline, and the entire evaluation suite are implemented here — not wrapped from a
> high-level library — because owning every layer is what let me instrument and trust
> the experiments. **📄 [Paper (arXiv-style) → `paper/`](../paper/) · 📊 [Full methodology & results → `RESULTS.md`](../RESULTS.md).**

---

## Research questions

| # | Question | Where it's answered |
|---|---|---|
| **RQ1** | Does one fixed PPO recipe learn *qualitatively different* policies across two market regimes (equities vs. crypto)? | [`compare_markets.py`](../rl_trader/scripts/compare_markets.py), [RESULTS §3](../RESULTS.md) |
| **RQ2** | How severely does an RL agent overfit a single price trajectory, and does **domain randomization** restore generalization? | [`tools/ablation.py`](../tools/ablation.py), [RESULTS §1](../RESULTS.md) |
| **RQ3** | Does an apparent out-of-sample edge survive **multi-seed** resampling and a permutation test — or is it seed luck? | [`tools/real_significance.py`](../tools/real_significance.py), [RESULTS §5](../RESULTS.md) |
| **RQ4** | Do the same conclusions hold on the harder problem of **cross-sectional allocation** (long/short weights over a basket)? | [`tools/portfolio_experiment.py`](../tools/portfolio_experiment.py), [RESULTS §6](../RESULTS.md) |
| **Rigor** | Does the study survive standard deep-RL evaluation practice: a correct PPO, other implementations and algorithms, more training, no hindsight in the universe, multiple-testing correction? | [`tools/ppo_benchmark.py`](../tools/ppo_benchmark.py), [`tools/trading_rigor.py`](../tools/trading_rigor.py), [RESULTS §10](../RESULTS.md) |

**Findings in one line:** the recipe transfers across regimes (RQ1); a single-path
agent memorises catastrophically and domain randomization flips its held-out return
from reliably negative to reliably positive (RQ2); and no apparent real-market edge survives multi-seed testing (RQ3, RQ4) —
consistent with weak-form market efficiency, and a clean live demonstration of *why
you never trust a single RL run.*

---

## Why markets are the testbed

The research questions above — generalization, memorization, and honest evaluation —
are *general* problems in deep RL. Financial markets just happen to be an unusually
punishing place to study them: the environment is **non-stationary** (the data-
generating process drifts), **partially observable** (price alone hides the state),
the reward is **noisy** and near-zero-signal (weak-form efficiency; Fama 1970), and a
single historical path is *trivial* to memorise. That combination makes every RL
pathology show up loudly and quickly, which is exactly what you want from a testbed.
This framework is a disciplined attempt to study them *properly*:

- **Separation of concerns.** Data, environments, agent, training, and evaluation
  are independent, individually testable modules.
- **Leakage control.** Feature scalers are fit on the **training split only**; the
  data is split **chronologically** into train / validation / test.
- **Honest evaluation.** Agents are scored on a **held-out test set** with the
  metrics a quant actually cares about — total return, **annualised Sharpe**, and
  **maximum drawdown** — not on the data they trained on.
- **Risk-aware rewards.** Two selectable formulations: a return *net of*
  transaction costs, a drawdown penalty, and a turnover penalty; or the
  **Differential Sharpe Ratio** (Moody & Saffell, 1998), which optimises
  *risk-adjusted* return online. Both discourage reckless, over-leveraged,
  noise-trading behaviour.
- **Uncertainty quantification.** Headline claims ship with bootstrap confidence
  intervals across seeds and a paired permutation test against buy-&-hold (on
  *both* synthetic and real data), plus a rolling multi-fold walk-forward — so a
  result is a distribution, not an anecdote.
- **Stable, reproducible training.** A running observation normaliser (Welford,
  exported and applied at serve time) keeps inputs well-scaled, and full seeding of
  Torch + NumPy + the environment RNG makes a run **bit-for-bit reproducible on a
  given dataset** — the same seed over the same bars gives the same numbers.
  The dataset is the part that needs care: `tools/fetch_data.py` pulls a *relative*
  window, so an unpinned re-fetch shifts the train/test split and the figures move
  with it. `data/SNAPSHOT.json` pins the exact bars behind the published results
  (per-file sha256), `make verify-data` checks a local copy against it, and `make
  build` now refuses to run unless that check passes. Reproducing the documented
  numbers therefore means matching the snapshot first — `data/raw/` is gitignored,
  so a fresh clone fetches with `--end` set to the snapshot date rather than
  taking today's window.

---

## Where this sits in the RL literature

This project isn't inventing new algorithms — it's a careful reproduction and
cross-domain stress-test of results the field already considers important, which is
itself a research skill. The experiments are designed as concrete instances of:

- **Evaluation rigor / the reproducibility crisis in deep RL.** Henderson et al.,
  *Deep Reinforcement Learning that Matters* (AAAI 2018), showed that deep-RL results
  swing wildly across random seeds and that single-run numbers are unreliable. RQ3
  reproduces exactly this in a financial domain: an earlier published +275%
  single-seed run did not survive reseeding — across 10 seeds the individual
  returns span more than an order of magnitude, and the agent is not
  distinguishable from buy-and-hold.
  *(See [RESULTS §5](../RESULTS.md).)*
- **Generalization & overfitting in RL.** Training and testing on the same instance
  overstates performance (Cobbe et al., *Quantifying Generalization in RL*, ICML 2019).
  **Domain randomization** — the sim-to-real technique of Tobin et al. (IROS 2017) —
  is my fix: a fresh synthetic price path each episode. RQ2 measures the memorization
  gap it closes directly. *(See [RESULTS §1](../RESULTS.md).)*
- **The algorithm.** PPO (Schulman et al., 2017) with Generalized Advantage
  Estimation (Schulman et al., 2016), implemented from scratch — clipped objective,
  GAE, entropy bonus, orthogonal init, gradient clipping, and a Welford (1962) running
  observation normaliser.
- **RL for trading, specifically.** The **Differential Sharpe Ratio** reward comes
  from Moody & Saffell's direct-reinforcement trading work (1998/2001); the honest
  conclusion — no reliable edge from price alone — is what weak-form market efficiency
  (Fama, 1970) predicts, and reproducing *that* cleanly is a feature, not a
  disappointment.

<details>
<summary><b>References</b></summary>

- Schulman, Wolski, Dhariwal, Radford, Klimov (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Schulman, Moritz, Levine, Jordan, Abbeel (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438.
- Henderson, Islam, Bachman, Pineau, Precup, Meger (2018). *Deep Reinforcement Learning that Matters.* AAAI.
- Tobin, Fong, Ray, Schneider, Zaremba, Abbeel (2017). *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IROS.
- Cobbe, Klimov, Hesse, Kim, Schulman (2019). *Quantifying Generalization in Reinforcement Learning.* ICML.
- Moody, Wu, Liao, Saffell (1998); Moody & Saffell (2001). *Learning to Trade via Direct Reinforcement.* IEEE Transactions on Neural Networks.
- Fama (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance.
- Welford (1962). *Note on a Method for Calculating Corrected Sums of Squares and Products.* Technometrics.

</details>

---

## Research-style write-up

**Hypothesis.** A single PPO recipe will learn *qualitatively different* policies
on stocks versus crypto, because the two markets differ in volatility, tail risk,
and trading frictions. We expect the crypto agent to favour smaller, more defensive
position sizes (higher costs, deeper drawdowns) relative to the stock agent.

**Method.** Hold the agent architecture and PPO hyper-parameters fixed. Train one
agent per market on that market's **training split**, using market-specific
environment dynamics (`stock_config` vs `crypto_config`: cost, slippage, drawdown
penalty, exploration). Select on the **validation split** during training.

**Measurement.** Backtest each trained agent on its untouched **test split** and
report:

| Metric | What it tells us |
| --- | --- |
| **Total return** | Did the strategy make money out-of-sample? |
| **Annualised Sharpe** | Return *per unit of risk* — the real quality signal |
| **Max drawdown** | Worst peak-to-trough loss — the pain a trader would feel |
| **Action distribution** | *How* the agent traded — its learned sizing behaviour |

`scripts/compare_markets.py` prints these side by side and (with `--plot`) saves
equity curves, so the difference in learned behaviour is visible, not just asserted.

**Overfitting control — domain randomization.** A single price path is trivial
to memorise: an agent trained on one sequence reaches huge in-sample equity and
then *loses* out-of-sample. To force a *generalizable* policy, training draws a
**fresh synthetic path every episode** (`train_series_factory`), while
validation and test stay on fixed held-out paths. This single change moved the
crypto agent from catastrophic overfitting to a positive, risk-controlled
out-of-sample backtest. The effect is quantified by an **ablation study**
(`python tools/ablation.py`), which trains agents with and without the technique
and reports the in-sample vs. out-of-sample gap directly.

**Honest baselines.** The agent is benchmarked not only against buy-&-hold but
also against *random* and *moving-average-crossover* strategies
(`rl_trader/evaluation/baselines.py`), all run through the identical
cost-and-slippage environment — so any edge has to be real.

> **Note on results.** The [web dashboard](architecture.md#web-prototype-internals) and
> [`RESULTS.md`](../RESULTS.md) report **real, out-of-sample backtests** — no mock
> numbers, and no hiding the unflattering parts. The dashboard shows one favorable
> seed; the multi-seed significance study then shows that edge does **not** survive
> resampling. Reporting that — rather than the lucky backtest — is the point.
