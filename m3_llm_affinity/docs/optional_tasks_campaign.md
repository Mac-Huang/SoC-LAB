Preflight:
cd repo
SOC_PY=../venv_soclab_m3/bin/python
$SOC_PY scripts/00_env_check.py
$SOC_PY scripts/08_run_suite.py --suite-config configs/suite_optional_tasks_smoke_sdxl_refiner_whisper_largev3.yaml --dry-run

Phase A smoke run (canonical large models):
$SOC_PY scripts/08_run_suite.py --suite-config configs/suite_optional_tasks_smoke_sdxl_refiner_whisper_largev3.yaml --no-lm-rerun
$SOC_PY scripts/07_analyze_results.py --suite-config configs/suite_optional_tasks_smoke_sdxl_refiner_whisper_largev3.yaml

Smoke pass criteria:
- Diffusion has 2 status=ok rows (2 scenarios x 1 step x 1 run)
- Speech has 4 status=ok rows (2 scenarios x 2 audio lengths x 1 run)
- No missing_dependency, missing_asset, download_unavailable, or backend_unavailable errors
- SD compute-plan outputs exist:
  reports/computeplan_sd15_text_encoder.csv
  reports/computeplan_sd15_unet.csv
  reports/computeplan_sd15_vae_decoder.csv
  reports/computeplan_sd15_summary.json
- WhisperKit compute-plan outputs exist:
  reports/computeplan_whisperkit_mel.csv
  reports/computeplan_whisperkit_encoder.csv
  reports/computeplan_whisperkit_decoder.csv
  reports/computeplan_whisperkit_summary.json

Phase B full run (only if smoke passes):
$SOC_PY scripts/08_run_suite.py --suite-config configs/suite_optional_tasks_full_sdxl_refiner_whisper_largev3.yaml --no-lm-rerun
$SOC_PY scripts/07_analyze_results.py --suite-config configs/suite_optional_tasks_full_sdxl_refiner_whisper_largev3.yaml

Full pass criteria:
- Diffusion has 25 status=ok rows (5 scenarios x 5 step values x 1 run)
- Speech has 16 status=ok rows (4 scenarios x 4 audio lengths x 1 run)
- Scenario coverage includes all configured labels for each task
- reports/analysis/suite/latest/latest_summary.csv includes rows for diffusion_sd15 and speech_whisperkit

Legacy compatibility configs:
- configs/suite_optional_tasks_smoke_split_einsum_legacy.yaml
- configs/suite_optional_tasks_full_split_einsum_legacy.yaml
