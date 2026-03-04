# Config Presets

## Primary

- `default.yaml`: single-context legacy run config.
- `sweep_ctx.yaml`: context sweep for legacy pipeline scripts.
- `suite.yaml`: default multi-task suite (LLM + SD15 + OWSM).

## Qwen Decode Presets

- `suite_qwen_decode_ctx64.yaml`: small validation run at context 64.
- `suite_qwen_decode_ctx64_to_4096.yaml`: full decode sweep (64, 128, 256, 512, 1024, 2048, 4096).

## Optional Tasks Quick Check

- `suite_optional_tasks_quick.yaml`: quick smoke test for `diffusion_sd15` and `speech_owsm`.
