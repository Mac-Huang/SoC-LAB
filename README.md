# SoC-LAB

## Hardware Information

- Target platform: Apple Silicon M3
- Memory: 24 GB unified memory
- OS: macOS 15+

## Experiment Goal

This repository benchmarks Core ML hardware affinity for decoder-only language models (GPT-style), focusing on:

- Whole-model placement (`CPU_AND_NE`, `ALL`)
- Stage-split placement (`CPU_AND_NE -> CPU_AND_GPU`)
- Per-operation device preference/support via `MLComputePlan`
- Runtime outcomes: TTFT, TPOT, tokens/sec, effective TFLOPS, and memory

## Experiment Setup

- Main project: `m3_llm_affinity/`
- Active decode model: `Qwen/Qwen2.5-7B-Instruct`
- Deterministic synthetic prompt tokens (fixed seed; tokenizer excluded from timing)
- Static-shape Core ML artifacts per `(model_id, context_len)` variant
- Context sweep support with doubling schedule (for example `64 -> 128 -> ... -> 4096`)
- Failure-tolerant execution: failing configs are recorded as data rows and sweep continues

## Quick Start

```bash
cd m3_llm_affinity
make convert && make bench && make plan
```

For suite/context sweeps and analysis, see:

- `m3_llm_affinity/README.md`
- `m3_llm_affinity/docs/experiment_goal_and_design.md`
