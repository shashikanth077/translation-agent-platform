.PHONY: install dev test lint type check demo serve eval baseline docker clean

PY ?= python

install:            ## runtime install
	$(PY) -m pip install -e ".[anthropic,redis,mcp]"

dev:                ## dev install (tests, linters)
	$(PY) -m pip install -e ".[anthropic,redis,mcp,dev]"

test:               ## unit + end-to-end tests with coverage
	$(PY) -m pytest --cov=tap --cov-report=term-missing:skip-covered

lint:
	ruff check src tests && ruff format --check src tests

type:
	mypy

check: lint type test   ## everything CI runs (except the eval gate)

demo:               ## offline end-to-end demo, no API keys
	tap demo

serve:              ## API + console on http://127.0.0.1:8000/console
	tap serve --seed-demo

eval:               ## golden-set regression gate (fails on regression)
	tap eval --baseline data/golden/baseline.json --fail-on-regression

baseline:           ## regenerate the golden baseline (review the diff before committing!)
	tap eval --write-baseline data/golden/baseline.json

docker:
	docker compose up --build

clean:
	rm -rf .tap-data .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist
