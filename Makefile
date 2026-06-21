# Convenience targets for the RL-Trader project.
# Usage: `make <target>` (requires `make`; on Windows use Git Bash or WSL).

PY ?= python
TIMESTEPS ?= 200000

.PHONY: help install test lint fetch build build-synth ablation baselines portfolio figures all

help:
	@echo "install     install runtime + dev dependencies"
	@echo "test        run the pytest suite"
	@echo "lint        run ruff"
	@echo "fetch       download the real OHLCV basket"
	@echo "build       train + backtest on REAL data -> docs/results.js"
	@echo "build-synth train + backtest on synthetic data -> docs/results.js"
	@echo "ablation    run the domain-randomization ablation -> docs/assets/ablation.json"
	@echo "baselines   print agent vs baselines on the real test data"
	@echo "portfolio   train the cross-sectional portfolio agent vs quant baselines"
	@echo "figures     render docs/assets/*.png from results"
	@echo "all         lint + test"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install pytest ruff yfinance

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check rl_trader tools tests

fetch:
	$(PY) tools/fetch_data.py

build: fetch
	$(PY) tools/build_site_data.py --real --timesteps $(TIMESTEPS)

build-synth:
	$(PY) tools/build_site_data.py --timesteps $(TIMESTEPS)

ablation:
	$(PY) tools/ablation.py --timesteps 60000

baselines:
	$(PY) tools/baseline_report.py

portfolio:
	$(PY) tools/portfolio_experiment.py --market stock --timesteps $(TIMESTEPS)

figures:
	$(PY) tools/make_figures.py

all: lint test
