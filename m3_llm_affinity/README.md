# m3_llm_affinity

Reproducible Core ML affinity suite focused on static-shape benchmarks, starting with decoder-only LLMs and extensible to optional task families.

## What This Measures

Primary task (`llm_decode`):

1. Whole-model compute unit placement (`CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE`, `ALL`)
2. Stage split placement (prefill vs decode)
3. MLComputePlan per-op preferred/supported devices + estimated cost
4. Performance metrics:
   - `prefill_latency_ms`
   - decode TPOT (`tpot_ms_mean`, `tpot_ms_p95`)
   - `first_decode_step_ms`
   - `ttft_ms`
   - `tokens_per_sec`
   - effective TFLOPS (prefill/decode)
   - `peak_rss_mb`

The suite is failure-tolerant: failures are recorded as data rows (`status=error`) and execution continues.

## Backward-Compatible Pipeline

Existing single-model workflow still works:

```bash
make setup
make convert
make bench
make plan
```

Single command:

```bash
make setup && make convert && make bench && make plan
```

## New Suite Runner

Run full suite config:

```bash
make suite
```

LLM-only convenience target:

```bash
make suite-llm
```

Direct invocation examples:

```bash
python scripts/08_run_suite.py --suite-config configs/suite.yaml
python scripts/08_run_suite.py --suite-config configs/suite.yaml --only-task llm_decode
python scripts/08_run_suite.py --suite-config configs/suite.yaml --only-task llm_decode --only-model gpt2
python scripts/08_run_suite.py --suite-config configs/suite.yaml --dry-run
```

## Configs

- `configs/default.yaml`: legacy/default single-context config (kept for compatibility)
- `configs/sweep_ctx.yaml`: existing context sweep config
- `configs/suite.yaml`: new multi-task suite schema

### `configs/suite.yaml` Highlights

- LLM sweep uses doubling schedule (`context_len_start`, `doubling_steps`, `context_len_max`)
- Supports multiple models (default includes GPT-2 and Llama 3.1 8B Instruct)
- Optional HF auth env per model (`hf_token_env`)
- Optional task families:
  - `diffusion_sd15` (safe stub)
  - `speech_owsm` (safe stub)

Optional tasks are non-blocking and emit one structured skip/error row when disabled or not available.

## Artifact Layout

### Variant artifacts (suite mode)

Per `(model_id, context_len)` variant:

- `artifacts/variants/<model_slug>/ctx<context_len>/torch/prefill.pt`
- `artifacts/variants/<model_slug>/ctx<context_len>/torch/decode.pt`
- `artifacts/variants/<model_slug>/ctx<context_len>/model_meta.json`

### Core ML artifacts

- `models/<model_slug>/ctx<context_len>/prefill.mlpackage`
- `models/<model_slug>/ctx<context_len>/decode.mlpackage`

### Compute plan outputs

- Runtime script output (default): `reports/computeplan_<model_alias>_ctx<context_len>_*.{csv,json}`
- Organized retained layout:
  - `reports/computeplan/<task_type>/<model_alias>/ctx<context_len>/prefill.csv`
  - `reports/computeplan/<task_type>/<model_alias>/ctx<context_len>/decode.csv`
  - `reports/computeplan/<task_type>/<model_alias>/ctx<context_len>/summary.json`

Backward-compatible fixed report names are still written:

- `reports/computeplan_prefill.csv`
- `reports/computeplan_decode.csv`
- `reports/computeplan_summary.json`

## Results Files

Suite run writes one bench JSONL per model/context. For long-term storage in this repo, files are normalized to:

- `results/<task_type>/<model_alias>/sweep_<run_id>/ctx<context_len>_bench.jsonl`

Each JSONL includes:

- identifiers: `timestamp`, `model_id`, `model_alias`, `variant_id`, `context_len`, `prefill_len`
- scenario: `mode`, `prefill_compute_units`, `decode_compute_units`
- latency/throughput: `prefill_latency_ms`, `first_decode_step_ms`, `tpot_ms_mean`, `tpot_ms_p95`, `ttft_ms`, `total_decode_latency_ms`, `tokens_per_sec`
- compute/memory: `effective_TFLOPS_prefill`, `effective_TFLOPS_decode`, `peak_rss_mb`
- status/errors: `status`, `error_type`, `error_message`, `traceback_summary`, `errors`

## Analysis and Visualization

Run suite analysis:

```bash
make analyze-suite
```

Or:

```bash
python scripts/07_analyze_results.py --suite-config configs/suite.yaml
```

By default, analysis loads the latest `N` `*_bench.jsonl` files from `results/` (`N` from `suite.pick_latest_n_jsonl`, default `50`).

Organized retained output layout:

- `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/summary.csv`
- `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/summary.json`
- `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/summary.md`
- `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/ttft_ms.png`
- `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/tokens_per_sec.png`
- `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/tflops_prefill.png`
- `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/tflops_decode.png`
- `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/peak_rss_mb.png`
- `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/ttft_vs_throughput_panel.png`

### Inspect Summary CSV Quickly

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path('reports/analysis/latest_summary.csv')
print(p.resolve())
print(pd.read_csv(p).head(20).to_string(index=False))
PY
```

For the retained GPT-2 run in this repo, use:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path('reports/analysis/llm_decode/gpt2/sweep_20260302_123805/summary.csv')
print(p.resolve())
print(pd.read_csv(p).head(20).to_string(index=False))
PY
```

## Context Length Limits

The suite checks max supported context positions from HF config (`n_positions`, `max_position_embeddings`, etc.).

- Contexts beyond max positions are skipped with a clear error row.
- If export/convert fails with hard failures (`OOM`, `model_compile_fail`, `coreml_convert_fail`), the runner stops larger contexts for that model.

## Optional Power Logging

Manual only (no sudo automation in scripts):

- `scripts/06_optional_powermetrics_instructions.md`
