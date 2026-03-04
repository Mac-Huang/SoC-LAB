# Reports Layout

Pattern:
- Suite-level analysis: `reports/analysis/suite/latest/`
- Task/model analysis: `reports/analysis/<task_type>/<model_alias>/sweep_<run_id>/` (with `latest` symlink per model when available)
- Compute plan: `reports/computeplan/<task_type>/<model_alias>/ctx<context_len>/`

Current retained reports:
- `reports/analysis/suite/latest/`
- `reports/analysis/llm_decode/gpt2/sweep_<run_id>/`
- `reports/analysis/llm_decode/qwen25_7b/sweep_<run_id>/`
- `reports/computeplan/llm_decode/gpt2/ctx*/`
