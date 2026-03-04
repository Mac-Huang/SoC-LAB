# Experiment Goal And Design

## Goal

This project benchmarks hardware affinity behavior on Apple Silicon M3 for three task families:

1. `llm_decode` (Core ML): decoder-only prefill/decode latency/throughput behavior.
2. `diffusion_sd15` (Core ML): staged text-encoder / U-Net / VAE inference behavior.
3. `speech_owsm` (ESPnet): CPU vs MPS backend behavior for speech inference.

The main objective is to compare placement strategies and detect where latency and throughput winners diverge.

## Decode Mode Policy

For decode experiments (Qwen and future decode models), only three scenarios are allowed:

1. `NE`: whole model on `CPU_AND_NE`
2. `ALL`: whole model on `ALL`
3. `NE->GPU`: split prefill `CPU_AND_NE`, decode `CPU_AND_GPU`

This policy is enforced in config and runner validation.

## Pipeline

`scripts/08_run_suite.py` orchestrates the full suite.

For `llm_decode` per `(model, context_len)`:

1. Export Torch wrappers: `scripts/01_export_torch.py`
2. Convert to Core ML: `scripts/02_convert_coreml.py`
3. Dump compute plan: `scripts/04_computeplan_dump.py`
4. Benchmark selected scenarios: `scripts/03_bench.py`

For optional tasks:

- `scripts/tasks/diffusion_sd15.py` for SD1.5 staged benchmark + compute plan.
- `scripts/tasks/speech_owsm.py` for speech benchmark (no Core ML compute plan output expected).

## Result Schema

All records are JSONL rows with shared identifiers and metrics:

- `task_type`, `model_id`, `model_alias`, `mode`, scenario compute units
- sweep axis: `x_label`, `x_value`
- primary metrics: `primary_latency_ms`, `primary_throughput`
- status fields: `status`, `error_type`, `error_message`, `errors`

Task-specific mappings:

- `llm_decode`: `x=context_len`, primary latency `ttft_ms`, throughput `tokens_per_sec`
- `diffusion_sd15`: `x=steps`, primary latency `total_ms`, throughput `steps_per_sec`
- `speech_owsm`: `x=audio_seconds`, primary latency `total_ms`, throughput `audio_seconds_per_sec`

## Storage Layout

- LLM decode:
  - `results/llm_decode/<model_alias>/sweep_<run_id>/ctx<context_len>_bench.jsonl`
- Diffusion/speech:
  - `results/<task_type>/<model_alias>/sweep_<run_id>/<task_type>_bench.jsonl`
- Compute plans:
  - `reports/computeplan/<task_type>/<model_alias>/ctx<context_len>/*`
- Analysis:
  - Suite-level: `reports/analysis/suite/latest/latest_summary.{csv,json,md}`
  - Suite-level figures: `reports/analysis/suite/latest/fig_*.png`
  - Task/model-level: `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/latest_summary.{csv,json,md}`
  - Task/model figures: `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/fig_*.png`
