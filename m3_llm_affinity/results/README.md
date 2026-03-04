# Results Layout

This folder is organized for multi-model, multi-task experiments.

Pattern:
- LLM decode: `results/llm_decode/<model_alias>/sweep_<run_id>/ctx<context_len>_bench.jsonl`
- Diffusion/Speech: `results/<task_type>/<model_alias>/sweep_<run_id>/<task_type>_bench.jsonl`

Current retained runs:
- `results/llm_decode/gpt2/sweep_<run_id>/`
- `results/llm_decode/qwen25_7b/sweep_<run_id>/`
- `results/diffusion_sd15/sd15/sweep_<run_id>/`
- `results/speech_owsm/owsm_v31_small/sweep_<run_id>/`
