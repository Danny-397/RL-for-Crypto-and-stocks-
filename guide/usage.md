# Running it: install, data, reproducibility

Installing, training, bringing your own data, and pinning the dataset so results reproduce.

[← back to the README](../README.md)

## Quick start

```bash
# 1. Install — `pip install -e .` is the FULL dev/training env (pulls torch etc.).
#    `requirements.txt` alone is the lightweight, torch-free *serving* set.
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .                                      # training + research (torch, matplotlib, …)

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

Or fetch the whole default basket (10 stocks + 6 crypto pairs) in one command:

```bash
pip install yfinance
python tools/fetch_data.py        # -> data/raw/{stock,crypto}/*.csv  (cached, with backoff)
python tools/build_site_data.py --real --timesteps 200000   # real walk-forward dashboard
```

`--real` trains each agent on a basket of real tickers (domain-randomized across
names) and backtests on every ticker's **held-out recent period** — a multi-asset
walk-forward evaluation. The bundled dashboard is generated exactly this way.

---

## Reproducibility

Every experiment carries a receipt: code version, dataset hash, policy hash,
the resolved environment config, and the caller's own research question if they
stated one (never invented for them). The config is a **fixed point of the round
trip** — re-submitting what `/config` returns rebuilds an identical environment
and reproduces the numbers, which the test suite asserts.

### Pin the dataset before you rebuild

`tools/fetch_data.py` requests a *relative* window (`period="10y"` / `"max"`), so
**what you get depends on when you run it**. This is not hypothetical: the v1
figures were built from data fetched 2026-06-22, and a rebuild on 2026-08-30
evaluated stocks on 2022-11-30→2026-08-28 instead of 2022-08-10→2026-06-17, moving
crypto total return 2.75 → 1.96. Same recipe, different slice of history, different
experiment. That is why the data is now pinned.

Pin a snapshot, and verify against it before rebuilding:

```bash
python tools/fetch_data.py --end 2026-08-28    # clip to a fixed date, write data/SNAPSHOT.json
make verify-data                                # fails loudly if the data has drifted
```

`data/SNAPSHOT.json` is committed and records a SHA-256, row count and date range
per ticker, so an experiment can be cited by dataset hash.

> **Note on the published figures.** The site (`docs/results.js`), the paper's
> tables and `RESULTS.md` §3 and §5 are built from the snapshot pinned at 2026-08-28,
> so the site, the paper and the DOI describe the same data. The one exception is the
> multi-fold walk-forward (`RESULTS.md` §5c), which was run on data through
> 2026-06-17 and says so.
