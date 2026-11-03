# Errand Agent: developer entry points. `make setup && make test && make run`.

# First of 3.12 / 3.11 / 3.13 found on PATH; override with `make setup PYTHON=/path/to/python3.x`.
PYTHON ?= $(shell command -v python3.12 || command -v python3.11 || command -v python3.13 || command -v python3)
VENV   := .venv
BIN    := $(VENV)/bin
HOST   ?= 127.0.0.1
PORT   ?= 8000

.PHONY: setup run demo reset test eval eval-stress record lint format clean

setup: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	@$(PYTHON) -c 'import sys; v = sys.version_info; sys.exit(0 if v >= (3, 11) else f"Python 3.11+ is required, found {v.major}.{v.minor}. Run: make setup PYTHON=python3.12")'
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -r requirements.txt
	@touch $@

run: setup
	$(BIN)/uvicorn errand.api:create_app --factory --host $(HOST) --port $(PORT)

# A fresh world where the first booking's book() times out although it went through.
demo: setup reset
	ERRAND_SCENARIO=timeout_after_success $(BIN)/uvicorn errand.api:create_app --factory --host $(HOST) --port $(PORT)

# Forget all local state: both databases and the per-run traces (the committed demo trace stays).
reset:
	rm -rf data traces/eval
	find traces -name 'run_*.jsonl' -delete

# Unit and flow tests, then the eval suite: a failing scenario fails `make test` too.
test: setup
	$(BIN)/pytest
	$(BIN)/python -m errand.eval

# The offline eval suite (evals/scenarios/): writes reports/eval_table.md and eval_results.json,
# and each scenario's trace to traces/eval/. No network, Ollama or Vega weights needed.
eval: setup
	$(BIN)/python -m errand.eval

# The concurrency scenarios, once per world seed 1..ROUNDS, on real threads: any failure fails it.
# Writes reports/eval_stress.md and traces to traces/eval/stress/.
ROUNDS ?= 20
eval-stress: setup
	$(BIN)/python -m errand.eval --stress $(ROUNDS)

# Run scenarios against live Ollama and Vega and save their outputs for offline replay:
#   make record SCENARIOS="happy_path_plumber slot_taken"
record: setup
	$(BIN)/python -m errand.eval --record $(SCENARIOS)

lint: setup
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: setup
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -not -path './$(VENV)/*' -exec rm -rf {} +
