# Architecture, code layout and deployment

How the code is organised, the design decisions behind it, how to extend it, and how the site and backend are built and deployed.

[← back to the README](../README.md)

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

**Observation** (per step): a rolling window of **28 engineered features**, grouped
by what they encode — multi-horizon momentum (1/5/20/60/120-bar + log returns),
trend/mean-reversion context (10/30/50/100 SMA ratios, EMA ratio), oscillators (RSI,
MACD + signal), band/range position (Bollinger %B, Donchian position), volatility
*level and regime* (10-bar vol, ATR, a 60-bar volatility z-score, short/long vol
ratio), and a risk-regime signal (distance below the trailing 6-month high), plus
volume microstructure (high–low range, volume change, volume z-score), and
**cross-asset market context** — the asset's *relative strength* vs. the market
index (SPY for stocks, BTC for crypto) and the market's own trend/momentum — and the
agent's own account state (position fraction, cash fraction, normalised equity). The
relative-strength and market-regime features feed the policy genuinely *exogenous*
information a single ticker's OHLCV can't convey.

**Action**: a single continuous value in `[-1, 1]` interpreted as the **target
position** as a fraction of equity (`+1` = fully long, `0` = flat, `-1` = fully
short). Targeting a position rather than emitting incremental buy/sell orders gives
the agent direct, stable control over its exposure and makes **position sizing**
an explicit, learnable decision.

**Cross-sectional mode** (`PortfolioTradingEnv`): the same PPO agent generalises
from one asset to a **basket** — it observes every asset's features at once and
emits an *N*-dimensional **weight vector** (long the strong, short the weak), under
a gross-exposure budget. This is the harder, more realistic problem of *allocation*
rather than single-name timing; run it with `tools/portfolio_experiment.py`.

---

## Project structure

```
rl_trader/
├── config/          # dataclass hyper-parameters + market presets
│   └── training_config.py
├── data/            # OHLCV loading, indicators, scaling, splits, synthetic data
│   ├── data_loader.py     # single-asset pipeline
│   └── portfolio_data.py  # multi-asset, date-aligned basket pipeline
├── envs/            # Gymnasium environments
│   ├── base_env.py      # shared mechanics (accounting, costs, reward)
│   ├── stock_env.py
│   ├── crypto_env.py
│   └── portfolio_env.py # cross-sectional, N-asset weight-allocation env
├── models/          # the agent and its networks
│   ├── networks.py      # shared-trunk ActorCritic + recurrent (LSTM) ActorCritic
│   └── ppo_agent.py     # PPO: clipped objective, GAE, save/load
├── training/        # rollout collection + PPO update loop + logging
│   ├── utils.py         # RolloutBuffer (GAE), feed-forward training engine, logger
│   ├── recurrent.py     # recurrent PPO: sequence buffer + truncated-BPTT update
│   ├── portfolio.py     # cross-sectional portfolio training loop
│   ├── normalization.py # running (Welford) observation/reward normaliser
│   ├── train_stock.py
│   └── train_crypto.py
├── evaluation/      # backtesting metrics, statistics, plots
│   ├── evaluate_agent.py
│   ├── statistics.py    # bootstrap CIs + paired permutation tests
│   ├── walk_forward.py  # rolling multi-fold walk-forward splits + runner
│   ├── portfolio_eval.py # portfolio backtest + cross-sectional baselines
│   └── plots.py
└── scripts/         # command-line entry points
    ├── run_stock_training.py
    ├── run_crypto_training.py
    └── compare_markets.py
server/              # the experiment engine behind the lab (torch-free)
├── app.py               # Flask routes: dashboard + lab endpoints
├── lab.py               # config parsing, env construction, experiment runners
├── experiments.py       # async experiment registry, progress, receipts
├── policy.py            # actor(-critic) forward pass in plain NumPy
├── rollout.py           # full per-bar traces + state-restoring counterfactuals
├── regimes.py           # controlled synthetic distributions for shift testing
├── precomputed.py       # the repo's real committed results, with provenance
├── stats_api.py         # live bootstrap / permutation inference
└── models/              # exported policy archives (ppo_*.npz)
tests/               # pytest suite (envs, agent, features, reward, recurrent, stats,
                     #   normalization, portfolio, snapshot, lab backend, HTTP API)
tools/
├── fetch_data.py        # download a real OHLCV basket; --end pins a snapshot
├── smoke_lab.py         # 144 headless-browser checks against the live lab
├── build_site_data.py   # train + backtest -> docs/results.js for the dashboard
├── ablation.py          # domain-randomization overfitting study
├── baseline_report.py   # agent vs. buy-&-hold / random / momentum
├── significance.py      # multi-seed CIs + permutation test (synthetic)
├── real_significance.py # multi-seed CIs + permutation test on the real basket
├── surrogate_test.py    # surrogate-data falsification test (signal vs. noise)
├── portfolio_experiment.py # cross-sectional portfolio agent vs. quant baselines
└── make_figures.py      # render docs/assets/*.png for the README & report
docs/                # the site: dashboard + interactive lab (no build step)
├── index.html           # all views, including the five lab panels
├── app.js               # dashboard: markets explorer, charts, live widget
├── lab.js               # the lab: playground, x-ray, ablation, seeds, notebook
├── lab.css              # lab-specific styling
├── results.js           # baked real backtest output (window.RL_RESULTS)
├── significance.js      # real per-seed results (window.RL_SIGNIFICANCE)
└── config.js            # window.RL_API — set it to light up the lab
data/SNAPSHOT.json   # committed dataset pin: sha256 + date range per ticker
```

---

## Design decisions worth highlighting

- **Unified agent, separate environments.** The `PPOAgent` speaks only in
  observation/action tensors and is completely market-agnostic — the exact design
  the comparison experiment requires.
- **Shared `BaseTradingEnv`.** All accounting, cost, and reward logic lives in one
  place, so the stock and crypto envs cannot silently diverge.
- **From-scratch PPO** with the stabilisers that matter in practice: GAE,
  advantage normalisation, clipped value loss, entropy bonus, orthogonal init,
  gradient clipping, and a running observation normaliser.
- **Two policy families, one training contract.** A feed-forward shared-trunk
  `ActorCritic` and a fully-implemented **recurrent (LSTM) actor-critic** train
  through the same PPO machinery — the recurrent variant adds hidden-state
  continuity during rollout collection and replays whole sequences (truncated BPTT)
  during the update. Flip `PPOConfig.use_lstm` to switch.
- **Extensible by construction.** Add a market by subclassing `BaseTradingEnv`;
  add an algorithm (DDPG/SAC) alongside `PPOAgent` with the same
  `select_action`/`update` API.

---

## Extending the framework

| Want to… | Do this |
| --- | --- |
| Add a new market (e.g. FX) | Subclass `BaseTradingEnv`, register it in `envs/__init__.make_env` |
| Add a new algorithm | Implement alongside `PPOAgent` with the same `select_action`/`update` API |
| Use sequence models | Set `PPOConfig.use_lstm = True` — the recurrent PPO loop is built in |
| Optimise risk-adjusted return | Set `RewardConfig.kind = "dsr"` (Differential Sharpe Ratio) |
| Change the reward | Edit `RewardConfig` weights or `BaseTradingEnv._compute_reward` |
| Test significance | `tools/significance.py` (CIs + permutation) or `evaluation/walk_forward.py` |
| Tune training | Edit the dataclasses in `config/training_config.py` or pass CLI flags |

---

## Web prototype internals

The page is dependency-free — no framework, no build step. `docs/results.js` is a
baked `window.RL_RESULTS` global, so the dashboard renders with **no server at
all**, and the lab layers live experiments on top when `window.RL_API` is set.

```bash
python tools/build_site_data.py --real --timesteps 200000   # regenerate docs/results.js
python server/app.py                                        # the experiment API
python tools/smoke_lab.py                                   # 144 browser checks against both
```

`tools/smoke_lab.py` drives the real page in headless Chromium. Its most important
assertions are the negative ones: with the backend unreachable, the lab must
surface an honest error and render **nothing** — no charts, no placeholder
numbers, no fabricated results.

---

## Deploy (Render + Vercel)

Production deployment is two independent pieces — see **[DEPLOY.md](../DEPLOY.md)**
for the click-by-click guide:

- **Frontend → Vercel** (static `docs/`, zero build). Works standalone on its
  baked data.
- **Backend → Render** (`server/`) — an optional, featherweight **live-inference
  API**. The trained policy is exported to a tiny NumPy archive
  (`tools/export_policy.py`) and served with plain matmuls, so the container
  needs **no PyTorch or ONNX** and cold-starts fast on the free tier. A
  [`render.yaml`](../render.yaml) Blueprint makes it one-click.

Set `window.RL_API` in `docs/config.js` to the Render URL and the dashboard's
"Run live" widget lights up — pulling real prices and running the agent on
demand. Leave it empty and the site is fully static.
