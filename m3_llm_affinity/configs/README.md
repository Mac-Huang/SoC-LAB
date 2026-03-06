# Config Presets

## Primary

- `default.yaml`: single-context legacy run config.
- `sweep_ctx.yaml`: context sweep for legacy pipeline scripts.
- `suite.yaml`: default multi-task suite (LLM + SD15 + OWSM).

## Qwen Decode Presets

- `suite_qwen_decode_ctx64.yaml`: small validation run at context 64.
- `suite_qwen_decode_ctx64_to_4096.yaml`: full decode sweep (64, 128, 256, 512, 1024, 2048, 4096).

## Optional Tasks (Canonical Large Models)

- `suite_optional_tasks_smoke_sdxl_refiner_whisper_largev3.yaml`:
  smoke campaign for `diffusion_sd15` + `speech_whisperkit` using
  `apple/coreml-stable-diffusion-xl-base-with-refiner` and `openai_whisper-large-v3`.
- `suite_optional_tasks_full_sdxl_refiner_whisper_largev3.yaml`:
  full optional-task campaign with the same large model pair.

## Optional Tasks (Compatibility / Legacy)

- `suite_optional_tasks_smoke_split_einsum.yaml`:
  compatibility alias for the canonical large-model smoke config.
- `suite_optional_tasks_full_split_einsum.yaml`:
  compatibility alias for the canonical large-model full config.
- `suite_optional_tasks_smoke_split_einsum_legacy.yaml`:
  legacy SD15 split-einsum + Whisper tiny smoke config.
- `suite_optional_tasks_full_split_einsum_legacy.yaml`:
  legacy SD15 split-einsum + Whisper tiny full config.
- `suite_optional_tasks_full_sdxl_whisper_medium.yaml`:
  earlier SDXL base + Whisper medium campaign.
- `suite_optional_tasks_reduced_sdxl_whisper_medium.yaml`:
  reduced SDXL base + Whisper medium campaign.
- `suite_optional_tasks_quick.yaml`:
  quick smoke test for `diffusion_sd15` and `speech_owsm`.
