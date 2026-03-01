PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CONFIG ?= configs/default.yaml

.PHONY: setup convert bench plan analyze clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

convert:
	$(PY) scripts/00_env_check.py
	$(PY) scripts/01_export_torch.py --config $(CONFIG)
	$(PY) scripts/02_convert_coreml.py --config $(CONFIG)

bench:
	$(PY) scripts/03_bench.py --config $(CONFIG) --mode whole
	$(PY) scripts/03_bench.py --config $(CONFIG) --mode split --prefill-cu CPU_AND_NE --decode-cu CPU_AND_GPU
	$(PY) scripts/03_bench.py --config $(CONFIG) --mode split --prefill-cu CPU_AND_GPU --decode-cu CPU_AND_NE

plan:
	$(PY) scripts/04_computeplan_dump.py --config $(CONFIG)

analyze:
	$(PY) scripts/07_analyze_results.py

clean:
	rm -rf $(VENV) artifacts models results reports
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
