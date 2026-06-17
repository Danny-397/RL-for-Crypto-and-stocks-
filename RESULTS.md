# Results & Findings

A short, honest research write-up of what this framework actually does when you
run it — including the parts that *don't* work, and why that's the interesting
result. All numbers here are reproducible with the commands in the final section.

---

## TL;DR

1. **The method is sound where signal exists.** On controlled synthetic markets
   with known structure, the agent learns a profitable, *generalizing* policy —
   and an ablation proves that **domain randomization is what makes it
   generalize** (it collapses the in-sample/out-of-sample overfitting gap by
   ~270×).
2. **On real markets, the learned agent does not beat passive investing.** Over
   a held-out 2021–2025 walk-forward window, PPO trails both buy-&-hold and a
   simple moving-average rule. It *does* beat a random policy, so it learned
   something — just not a tradable edge. This is consistent with weak-form market
   efficiency and is the honest, expected outcome.

The point of the project is the **methodology and the honest evaluation**, not a
fantasy money-machine.

---

## Setup

| | |
|---|---|
| **Algorithm** | PPO implemented from scratch in PyTorch (clipped objective, GAE, entropy bonus, orthogonal init, grad clipping) |
| **Environments** | Custom Gymnasium `StockTradingEnv` / `CryptoTradingEnv` over a shared base; continuous target-position actions in `[-1, 1]`; transaction costs + slippage |
| **Real data** | 10 equities + 6 crypto pairs, daily OHLCV, ~10 yrs (Yahoo Finance) |
| **Split** | Chronological walk-forward — train on the older 60%, test on the held-out recent 40% (2021–2025); scalers fit on training data only |
| **Reporting** | Mean across the basket; agent run deterministically; benchmarked vs. buy-&-hold, random, and a moving-average-crossover rule |

---

## 1. Methodology validation — the domain-randomization ablation

Training an RL agent on a **single** price series is a trap: it memorises that
one path. To show this concretely, we train two otherwise-identical agents on
synthetic data (where a real, known signal exists) and measure performance on
the training path ("in-sample") vs. 30 unseen paths ("out-of-sample").

`python tools/ablation.py --timesteps 60000`

| Market | Training | In-sample | Out-of-sample | Gap |
|---|---|---:|---:|---:|
| Stock | single-path | **+4927%** | **−37%** | +4965% |
| Stock | domain-random | +96% | **+77%** | **+18%** |
| Crypto | single-path | **+14668%** | **−69%** | +14737% |
| Crypto | domain-random | +79% | **+134%** | **−55%** |

**Reading it:** the single-path agents post absurd in-sample returns
(+4927% / +14668%) by memorising their training sequence — then **lose money** on
unseen data. Domain randomization (a fresh path every episode) collapses that
gap by two-to-three orders of magnitude and produces agents that actually
generalize. This is the project's core methodological result.

---

## 2. Real-market results (out-of-sample, walk-forward)

`python tools/build_site_data.py --real` then `python tools/baseline_report.py`

**Equities** — mean over 10 held-out tickers, 2021–2025:

| Strategy | Return | Sharpe | Max DD |
|---|---:|---:|---:|
| PPO agent | −34.6% | −0.67 | 43.6% |
| Buy & hold | **+201.6%** | **0.93** | 28.6% |
| MA crossover | +62.8% | 0.60 | **24.3%** |
| Flat (cash) | 0.0% | 0.00 | 0.0% |
| Random | −54.9% | −1.28 | 59.0% |

**Crypto** — mean over 6 held-out tickers:

| Strategy | Return | Sharpe | Max DD |
|---|---:|---:|---:|
| PPO agent | −18.8% | −0.21 | 59.3% |
| Buy & hold | +9.9% | 0.30 | 77.0% |
| MA crossover | **+11.0%** | 0.25 | 59.0% |
| Flat (cash) | 0.0% | 0.00 | 0.0% |
| Random | −78.1% | −1.22 | 80.0% |

---

## 3. Discussion — what these numbers actually mean

- **The agent beats random but not the real benchmarks.** Beating a random
  policy by ~20 percentage points shows the network learned *non-trivial*
  structure. Losing to buy-&-hold and a moving-average rule shows that structure
  isn't a tradable edge on real daily data.
- **This is what market efficiency looks like.** Raw daily OHLCV + standard
  indicators carry very little exploitable autocorrelation. An agent that trades
  on it gets whipsawed and pays costs; passive exposure to a rising market wins.
  The 2021–2025 equity test window was a historic mega-cap bull — an especially
  brutal benchmark.
- **Simple beat complex.** The hand-coded MA-crossover rule outperformed the
  learned agent and even had the *lowest drawdown* of any equity strategy
  (24.3%). That's a genuinely useful, humbling result and a reminder that model
  complexity is not a virtue by itself.
- **One bright spot:** the crypto agent's max drawdown (59%) was well below
  buy-&-hold's (77%) — it *was* more defensive, just not enough to overcome the
  return gap.

## 4. Limitations & next steps

- **Signal is the bottleneck, not the agent.** The most impactful next step is
  richer, lower-noise features (cross-sectional ranks, regime labels, alt-data,
  longer horizons) rather than a bigger network.
- **Walk-forward could be multi-fold** (rolling re-training) rather than a single
  60/40 split, with confidence intervals across seeds.
- **Cost/turnover sensitivity** and a recurrent (LSTM) policy are natural
  extensions the codebase is already structured for.

## 5. Reproduce everything

```bash
pip install -r requirements.txt
python tools/fetch_data.py                          # download the real basket
python tools/build_site_data.py --real --timesteps 200000   # real walk-forward
python tools/baseline_report.py                     # agent vs baselines (tables above)
python tools/ablation.py --timesteps 60000          # the overfitting ablation
pytest -q                                           # the test suite
```
