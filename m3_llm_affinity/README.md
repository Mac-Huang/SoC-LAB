# m3_llm_affinity

Reproducible Core ML affinity suite focused on static-shape benchmarks, starting with decoder-only LLMs and extensible to optional task families.

Detailed design note: `docs/experiment_goal_and_design.md`.

## Hardware, Goal, and Setup

### Target Hardware / OS

- Apple Silicon M3 (target machine class)
- 24 GB unified memory
- macOS 15+

### Experiment Goal

Measure Core ML hardware affinity for decoder-only LLM inference (prefill + decode) on Apple Silicon, and determine how placement choices change with context length:

- Whole-model placement: `CPU_AND_NE` (`NE`) and `ALL`
- Stage split placement: prefill `CPU_AND_NE` + decode `CPU_AND_GPU` (`NE->GPU`)
- Per-op affinity via `MLComputePlan` (preferred/supported devices, estimated cost)
- Performance outcomes: TTFT, TPOT, decode throughput, effective TFLOPS, memory

### Experiment Setup (Default)

- Main harness: Python + `coremltools`, `torch`, `transformers`
- Default model: `Qwen/Qwen2.5-7B-Instruct`
- Static-shape Core ML artifacts per variant `(model_id, context_len)`
- Deterministic synthetic prompt tokens from fixed seed (tokenizer time excluded)
- Batch size fixed to `1`
- Default context for quick runs: `context_len=64`, `prefill_len=63`
- Sweep mode supports doubling context lengths up to model limits (for example `64...4096`)
- Failure tolerant: unsupported/failed configurations are recorded as `status=error`

## What This Measures

Task families:

1. Whole-model compute unit placement (`CPU_AND_NE`, `ALL`)
2. Stage split placement (`CPU_AND_NE -> CPU_AND_GPU`)
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

Shared schema across tasks:

- `x_label`, `x_value` (task-specific sweep axis)
- `primary_latency_ms`, `primary_throughput` (task-specific primary metrics)
- LLM maps to: `x=context_len`, `primary_latency=ttft_ms`, `primary_throughput=tokens_per_sec`
- SD15 maps to: `x=steps`, `primary_latency=total_ms`, `primary_throughput=steps_per_sec`
- OWSM maps to: `x=audio_seconds`, `primary_latency=total_ms`, `primary_throughput=audio_seconds_per_sec`

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

Per-task convenience targets:

```bash
make suite-qwen
make suite-sd15
make suite-owsm
```

Direct invocation examples:

```bash
python scripts/08_run_suite.py --suite-config configs/suite.yaml
python scripts/08_run_suite.py --suite-config configs/suite.yaml --only-task llm_decode
python scripts/08_run_suite.py --suite-config configs/suite.yaml --only-task llm_decode --only-model gpt2
python scripts/08_run_suite.py --suite-config configs/suite.yaml --dry-run
```

## Running Qwen Suite

Use the ungated decoder-only model `Qwen/Qwen2.5-7B-Instruct` with the same static-shape affinity logic:

```bash
make convert_ctx CTX=64
make bench_ctx CTX=64
make sweep_ctx CONFIG=configs/sweep_ctx.yaml
python scripts/08_run_suite.py --suite-config configs/suite_qwen_decode_ctx64.yaml
```

Notes:

- This Qwen model is ungated; `HF_TOKEN` is not required.
- Prefill/decode stage split and KV sliding-window behavior are unchanged.
- Sweep outputs are written under `results/llm_decode/qwen25_7b/sweep_<timestamp>/`.

## Configs

- `configs/default.yaml`: legacy/default single-context config (kept for compatibility)
- `configs/sweep_ctx.yaml`: existing context sweep config
- `configs/suite.yaml`: new multi-task suite schema
- `configs/suite_qwen_decode_ctx64.yaml`: Qwen decode smoke run at context 64 only
- `configs/suite_qwen_decode_ctx64_to_4096.yaml`: Qwen decode full context sweep (64 -> 4096)
- `configs/suite_optional_tasks_quick.yaml`: fast optional-task check (SD15 + OWSM)

For the full config map and intended use, see `configs/README.md`.

### `configs/suite.yaml` Highlights

- LLM sweep uses doubling schedule (`context_len_start`, `doubling_steps`, `context_len_max`)
- Supports multiple models (default includes GPT-2 and Qwen2.5-7B-Instruct)
- Optional HF auth env per model (`hf_token_env`)
- Decode mode policy is enforced in code: only `NE`, `ALL`, `NE->GPU` are valid
- `llm_decode` can attempt optional weight quantization for large models (e.g., Qwen int4 attempt with fp16 fallback)
- `diffusion_sd15` runs staged Core ML inference (`text_encoder`, `unet`, `vae_decoder`) from BYO artifacts under `assets/sd15_coreml` (or best-effort auto-download)
- `speech_owsm` runs ESPnet inference on CPU/MPS backends (`uses_coreml=false`)

All tasks are non-blocking: missing dependencies/assets produce `status=error` rows and the suite continues.

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

Suite run files are normalized to sweep directories:

- LLM decode: `results/llm_decode/<model_alias>/sweep_<run_id>/ctx<context_len>_bench.jsonl`
- Diffusion/Speech: `results/<task_type>/<model_alias>/sweep_<run_id>/<task_type>_bench.jsonl`

Each JSONL includes:

- identifiers: `timestamp`, `model_id`, `model_alias`, `variant_id`, `context_len`, `prefill_len`
- scenario: `mode`, `prefill_compute_units`, `decode_compute_units`
- latency/throughput: `prefill_latency_ms`, `first_decode_step_ms`, `tpot_ms_mean`, `tpot_ms_p95`, `ttft_ms`, `total_decode_latency_ms`, `tokens_per_sec`
- compute/memory: `effective_TFLOPS_prefill`, `effective_TFLOPS_decode`, `peak_rss_mb`
- status/errors: `status`, `error_type`, `error_message`, `traceback_summary`, `errors`
- unified axes/primary metrics: `task_type`, `x_label`, `x_value`, `primary_latency_ms`, `primary_throughput`

## Analysis and Visualization

Run suite analysis:

```bash
make analyze-suite
```

Or:

```bash
python scripts/07_analyze_results.py --suite-config configs/suite.yaml
```

Analysis generates grouped bar charts per `(task_type, model_alias)` for:

- `primary_latency_ms` vs `x_value`
- `primary_throughput` vs `x_value`
- `peak_rss_mb` vs `x_value`
- plus LLM-only TFLOPS plots when available

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
p = Path('reports/analysis/suite/latest/latest_summary.csv')
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

## Resume Script

For disk-safe long sweeps on Qwen, use:

- `scripts/run_qwen_decode_resume.sh`
