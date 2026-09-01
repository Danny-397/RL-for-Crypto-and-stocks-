# Convenience targets for the RL-Trader project.
# Usage: `make <target>` (requires `make`; on Windows use Git Bash or WSL).

PY ?= python
TIMESTEPS ?= 200000

.PHONY: help install test lint fetch verify-data build build-synth ablation ablation1 baselines portfolio figures all sweep attribution sync rebuild supervised costs

help:
	@echo "install     install runtime + dev dependencies"
	@echo "test        run the pytest suite"
	@echo "lint        run ruff"
	@echo "fetch       download the real OHLCV basket"
	@echo "verify-data check data/raw against the committed data/SNAPSHOT.json"
	@echo "build       train + backtest on REAL data -> docs/results.js"
	@echo "build-synth train + backtest on synthetic data -> docs/results.js"
	@echo "ablation    5-seed domain-randomization ablation -> docs/assets/ablation_multiseed.json"
	@echo "ablation1   single-seed ablation (superseded; see RESULTS.md section 1)"
	@echo "baselines   print agent vs baselines on the real test data"
	@echo "portfolio   train the cross-sectional portfolio agent vs quant baselines"
	@echo "figures     render docs/assets/*.png from results"
	@echo "sweep       hyper-parameter sensitivity sweep -> docs/assets/hyperparameter_sweep.json"
	@echo "attribution occlusion ranking of the deployed policies -> docs/assets/attribution.json"
	@echo "supervised  ridge/logistic baselines on the same features -> docs/assets/supervised.json"
	@echo "costs       held-out return vs transaction costs -> docs/assets/cost_sensitivity.json"
	@echo "sync        regenerate the result tables in README/RESULTS/paper from the artifacts"
	@echo "rebuild     every experiment, then sync (hours; see tools/rebuild_all.sh)"
	@echo "all         lint + test"

install:
	$(PY) -m pip install -e .          # full dev/training env (torch, matplotlib, …)
	$(PY) -m pip install pytest ruff

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check rl_trader tools tests server

fetch:
	$(PY) tools/fetch_data.py

# Fail loudly if the local data has drifted from the committed pin. PERIOD is a
# relative window, so an unpinned re-fetch silently shifts the train/test split
# and any rebuilt figures stop matching the published ones.
verify-data:
	$(PY) tools/fetch_data.py --verify

build: fetch
	$(PY) tools/build_site_data.py --real --timesteps $(TIMESTEPS)

build-synth:
	$(PY) tools/build_site_data.py --timesteps $(TIMESTEPS)

# The 5-seed sweep is the canonical one: the domain-randomized arm draws fresh
# unseeded data every episode, so a single run of it is not reproducible.
ablation:
	$(PY) tools/ablation_multiseed.py --seeds 42 43 44 45 46 --timesteps 60000

ablation1:
	$(PY) tools/ablation.py --timesteps 60000

baselines:
	$(PY) tools/baseline_report.py

portfolio:
	$(PY) tools/portfolio_experiment.py --market stock --timesteps $(TIMESTEPS)

figures:
	$(PY) tools/make_figures.py

sweep:
	$(PY) tools/hyperparameter_sweep.py --seeds 3 --timesteps 60000

attribution:
	$(PY) tools/attribution_report.py

# Do the learned baselines find what PPO could not, and is the loss friction?
supervised:
	$(PY) tools/supervised_report.py

costs:
	$(PY) tools/cost_sensitivity.py

# Every result table in README.md, RESULTS.md and paper/rl_trader.tex is
# generated from docs/ artifacts. `--check` is what the test suite runs, so a
# rebuild that moves a number fails the suite instead of leaving the docs wrong.
sync:
	$(PY) tools/sync_docs.py

rebuild:
	bash tools/rebuild_all.sh

all: lint test
