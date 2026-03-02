PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CONFIG ?= configs/default.yaml
SUITE_CONFIG ?= configs/suite.yaml
CTX ?= 64

.PHONY: setup convert convert_ctx bench bench_ctx sweep_ctx plan analyze suite suite-llm analyze-suite clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

convert:
	$(PY) scripts/00_env_check.py
	$(PY) scripts/01_export_torch.py --config $(CONFIG)
	$(PY) scripts/02_convert_coreml.py --config $(CONFIG)

convert_ctx:
	$(PY) scripts/00_env_check.py
	$(PY) scripts/01_export_torch.py --config $(CONFIG) --context-len $(CTX)
	$(PY) scripts/02_convert_coreml.py --config $(CONFIG) --context-len $(CTX)

bench:
	$(PY) scripts/03_bench.py --config $(CONFIG) --mode whole
	$(PY) scripts/03_bench.py --config $(CONFIG) --mode split --prefill-cu CPU_AND_NE --decode-cu CPU_AND_GPU
	$(PY) scripts/03_bench.py --config $(CONFIG) --mode split --prefill-cu CPU_AND_GPU --decode-cu CPU_AND_NE

bench_ctx:
	$(PY) scripts/03_bench.py --config $(CONFIG) --context-len $(CTX) --mode whole
	$(PY) scripts/03_bench.py --config $(CONFIG) --context-len $(CTX) --mode split --prefill-cu CPU_AND_NE --decode-cu CPU_AND_GPU
	$(PY) scripts/03_bench.py --config $(CONFIG) --context-len $(CTX) --mode split --prefill-cu CPU_AND_GPU --decode-cu CPU_AND_NE

sweep_ctx:
	$(PY) scripts/07_sweep_context.py --config $(CONFIG)

plan:
	$(PY) scripts/04_computeplan_dump.py --config $(CONFIG)

analyze:
	$(PY) scripts/07_analyze_results.py

suite:
	$(PY) scripts/08_run_suite.py --suite-config $(SUITE_CONFIG)

suite-llm:
	$(PY) scripts/08_run_suite.py --suite-config $(SUITE_CONFIG) --only-task llm_decode

analyze-suite:
	$(PY) scripts/07_analyze_results.py --suite-config $(SUITE_CONFIG)

clean:
	rm -rf $(VENV) artifacts models results reports
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
