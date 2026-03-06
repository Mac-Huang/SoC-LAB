PYTHON ?= python3
VENV ?= ../venv_soclab_m3
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
SOC_PY ?= $(PYTHON)
CONFIG ?= configs/default.yaml
SUITE_CONFIG ?= configs/suite.yaml
CTX ?= 64

.PHONY: setup convert convert_ctx bench bench_ctx sweep_ctx plan analyze suite suite-llm suite-qwen suite-sd15 suite-owsm analyze-suite optional-smoke optional-full optional-analyze-smoke optional-analyze-full optional-legacy-smoke optional-legacy-full optional-legacy-analyze-smoke optional-legacy-analyze-full clean-cache-local clean-cache-global-hf clean-future-tests clean

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
	$(PY) scripts/03_bench.py --config $(CONFIG) --mode whole --cu CPU_AND_NE
	$(PY) scripts/03_bench.py --config $(CONFIG) --mode whole --cu ALL
	$(PY) scripts/03_bench.py --config $(CONFIG) --mode split --prefill-cu CPU_AND_NE --decode-cu CPU_AND_GPU

bench_ctx:
	$(PY) scripts/03_bench.py --config $(CONFIG) --context-len $(CTX) --mode whole --cu CPU_AND_NE
	$(PY) scripts/03_bench.py --config $(CONFIG) --context-len $(CTX) --mode whole --cu ALL
	$(PY) scripts/03_bench.py --config $(CONFIG) --context-len $(CTX) --mode split --prefill-cu CPU_AND_NE --decode-cu CPU_AND_GPU

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

suite-qwen:
	$(PY) scripts/08_run_suite.py --suite-config $(SUITE_CONFIG) --only-task llm_decode --only-model qwen25_7b

suite-sd15:
	$(PY) scripts/08_run_suite.py --suite-config $(SUITE_CONFIG) --only-task diffusion_sd15

suite-owsm:
	$(PY) scripts/08_run_suite.py --suite-config $(SUITE_CONFIG) --only-task speech_owsm

analyze-suite:
	$(PY) scripts/07_analyze_results.py --suite-config $(SUITE_CONFIG)

optional-smoke:
	$(SOC_PY) scripts/08_run_suite.py --suite-config configs/suite_optional_tasks_smoke_sdxl_refiner_whisper_largev3.yaml --no-lm-rerun

optional-full:
	$(SOC_PY) scripts/08_run_suite.py --suite-config configs/suite_optional_tasks_full_sdxl_refiner_whisper_largev3.yaml --no-lm-rerun

optional-analyze-smoke:
	$(SOC_PY) scripts/07_analyze_results.py --suite-config configs/suite_optional_tasks_smoke_sdxl_refiner_whisper_largev3.yaml

optional-analyze-full:
	$(SOC_PY) scripts/07_analyze_results.py --suite-config configs/suite_optional_tasks_full_sdxl_refiner_whisper_largev3.yaml

optional-legacy-smoke:
	$(SOC_PY) scripts/08_run_suite.py --suite-config configs/suite_optional_tasks_smoke_split_einsum_legacy.yaml --no-lm-rerun

optional-legacy-full:
	$(SOC_PY) scripts/08_run_suite.py --suite-config configs/suite_optional_tasks_full_split_einsum_legacy.yaml --no-lm-rerun

optional-legacy-analyze-smoke:
	$(SOC_PY) scripts/07_analyze_results.py --suite-config configs/suite_optional_tasks_smoke_split_einsum_legacy.yaml

optional-legacy-analyze-full:
	$(SOC_PY) scripts/07_analyze_results.py --suite-config configs/suite_optional_tasks_full_split_einsum_legacy.yaml

clean-cache-local:
	$(SOC_PY) scripts/09_workspace_hygiene.py --local --yes

clean-cache-global-hf:
	$(SOC_PY) scripts/09_workspace_hygiene.py --global-hf --yes

clean-future-tests:
	$(SOC_PY) scripts/09_workspace_hygiene.py --local --global-hf --yes

clean:
	rm -rf $(VENV) artifacts models results reports
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
