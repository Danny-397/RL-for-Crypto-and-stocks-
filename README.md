# RL-Trader — Reinforcement Learning for Stocks & Crypto

[![CI](https://github.com/Danny-397/RL-for-Crypto-and-stocks-/actions/workflows/ci.yml/badge.svg)](https://github.com/Danny-397/RL-for-Crypto-and-stocks-/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-d4ff3f.svg)](LICENSE)

A modular, research-grade framework for training **Proximal Policy Optimization (PPO)**
agents to trade financial markets. A single, unified agent architecture is trained
**separately** on two custom Gymnasium environments — one for **equities**, one for
**crypto** — so we can study how the *same* learning recipe adapts to two very
different market regimes.

> Built from scratch in PyTorch: the PPO algorithm, the trading environments, the
> data pipeline, and the evaluation suite are all implemented here — not wrapped from
> a high-level library. The goal is to *understand and own every layer of the stack.*

---

## Why this project

Financial markets are a uniquely hard reinforcement-learning problem: the
environment is **non-stationary**, rewards are **noisy**, and naive agents overfit
to historical price paths. This framework is a disciplined attempt to do RL trading
*properly*:

- **Separation of concerns.** Data, environments, agent, training, and evaluation
  are independent, individually testable modules.
- **Leakage control.** Feature scalers are fit on the **training split only**; the
  data is split **chronologically** into train / validation / test.
- **Honest evaluation.** Agents are scored on a **held-out test set** with the
  metrics a quant actually cares about — total return, **annualised Sharpe**, and
  **maximum drawdown** — not on the data they trained on.
- **Risk-aware rewards.** The reward is return *net of* transaction costs, a
  drawdown penalty, and a turnover penalty — discouraging reckless, over-leveraged,
  noise-trading behaviour.

---

## Architecture at a glance

```
                          ┌─────────────────────────────┐
                          │        Unified PPO Agent      │
                          │   (shared ActorCritic net)    │
                          │   policy head │ value head    │
                          └───────┬───────────────┬───────┘
            select_action(obs)    │               │   update(rollout)
                                  ▼               ▲
        ┌─────────────────────────────────────────────────────────┐
        │                     Rollout Buffer (GAE)                  │
        └─────────────────────────────────────────────────────────┘
                                  ▲               │
                          obs, reward             │ action
                                  │               ▼
        ┌───────────────────────┐     ┌───────────────────────┐
        │   StockTradingEnv     │     │   CryptoTradingEnv     │   ← BaseTradingEnv
        │  (low cost, ~252/yr)  │     │  (high cost, 365/yr)   │
        └───────────┬───────────┘     └───────────┬───────────┘
                    │                               │
        ┌───────────▼───────────────────────────────▼───────────┐
        │   Data pipeline: OHLCV → indicators → scale → split     │
        │   (synthetic GBM generator or your own CSVs)            │
        └─────────────────────────────────────────────────────────┘
```

**Observation** (per step): a rolling window of engineered features — returns over
multiple horizons, moving-average ratios, RSI, MACD, realised volatility,
high–low range, volume change — plus the agent's own account state (position
fraction, cash fraction, normalised equity).

**Action**: a single continuous value in `[-1, 1]` interpreted as the **target
position** as a fraction of equity (`+1` = fully long, `0` = flat, `-1` = fully
short). Targeting a position rather than emitting incremental buy/sell orders gives
the agent direct, stable control over its exposure and makes **position sizing**
an explicit, learnable decision.

---

## Project structure

```
rl_trader/
├── config/          # dataclass hyper-parameters + market presets
│   └── training_config.py
├── data/            # OHLCV loading, indicators, scaling, splits, synthetic data
│   └── data_loader.py
├── envs/            # Gymnasium environments
│   ├── base_env.py      # shared mechanics (accounting, costs, reward)
│   ├── stock_env.py
│   └── crypto_env.py
├── models/          # the agent and its networks
│   ├── networks.py      # shared-trunk ActorCritic (+ optional LSTM)
│   └── ppo_agent.py     # PPO: clipped objective, GAE, save/load
├── training/        # rollout collection + PPO update loop + logging
│   ├── utils.py         # RolloutBuffer (GAE), training engine, logger
│   ├── train_stock.py
│   └── train_crypto.py
├── evaluation/      # backtesting metrics and plots
│   ├── evaluate_agent.py
│   └── plots.py
└── scripts/         # command-line entry points
    ├── run_stock_training.py
    ├── run_crypto_training.py
    └── compare_markets.py
tests/               # pytest suite for envs and agent
```

---

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                      # optional: editable install

# 2. Train (uses a built-in synthetic data generator — no downloads needed)
python -m rl_trader.scripts.run_stock_training  --timesteps 50000
python -m rl_trader.scripts.run_crypto_training --timesteps 50000

# 3. Run the headline experiment: same agent, both markets, side-by-side
python -m rl_trader.scripts.compare_markets --timesteps 40000 --plot

# 4. Run the tests
pytest -q
```

### Using your own data

Drop an OHLCV CSV (`open,high,low,close,volume`, optionally `date`) anywhere and
point a trainer at it:

```bash
python -m rl_trader.scripts.run_stock_training  --data data/raw/AAPL.csv
python -m rl_trader.scripts.run_crypto_training --data data/raw/BTC-USD.csv
```

You can produce such CSVs with one line of [`yfinance`](https://github.com/ranaroussi/yfinance)
(`pip install yfinance`):

```python
import yfinance as yf
yf.download("BTC-USD", start="2019-01-01").to_csv("data/raw/BTC-USD.csv")
```

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

> **Note on results.** Numbers depend on data and seed. With the synthetic
> generator this repo is a *reproducible methodology demo*; swap in real OHLCV to
> produce real backtests. The framework deliberately makes that a one-flag change.

---

## Design decisions worth highlighting

- **Unified agent, separate environments.** The `PPOAgent` speaks only in
  observation/action tensors and is completely market-agnostic — the exact design
  the comparison experiment requires.
- **Shared `BaseTradingEnv`.** All accounting, cost, and reward logic lives in one
  place, so the stock and crypto envs cannot silently diverge.
- **From-scratch PPO** with the stabilisers that matter in practice: GAE,
  advantage normalisation, clipped value loss, entropy bonus, orthogonal init, and
  gradient clipping.
- **Extensible by construction.** Add a market by subclassing `BaseTradingEnv`;
  add an algorithm (DDPG/SAC) alongside `PPOAgent`; an LSTM actor-critic is already
  stubbed in `networks.py` as a documented next step.

---

## Extending the framework

| Want to… | Do this |
| --- | --- |
| Add a new market (e.g. FX) | Subclass `BaseTradingEnv`, register it in `envs/__init__.make_env` |
| Add a new algorithm | Implement alongside `PPOAgent` with the same `select_action`/`update` API |
| Use sequence models | Wire `RecurrentActorCritic` into a recurrent rollout buffer |
| Change the reward | Edit `RewardConfig` weights or `BaseTradingEnv._compute_reward` |
| Tune training | Edit the dataclasses in `config/training_config.py` or pass CLI flags |

---

## Web prototype

A self-contained landing-page prototype lives in [`docs/`](docs/) (dark
cyber-fintech theme with an animated agent-vs-benchmark equity chart). Open
`docs/index.html` locally, or enable **GitHub Pages → Deploy from branch →
`main` / `docs`** to publish it at
`https://danny-397.github.io/RL-for-Crypto-and-stocks-/`. It's an early
prototype — the design is meant to be iterated on.

## License

MIT — see [LICENSE](LICENSE).
