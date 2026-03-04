#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT_DIR/../venv_soclab_m3/bin/python"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
MODEL_ALIAS="qwen25_7b"
SWEEP_ID="${1:-20260302_215458}"
OUT_DIR="$ROOT_DIR/results/llm_decode/$MODEL_ALIAS/sweep_${SWEEP_ID}"
LOG_PATH="$OUT_DIR/continue_${SWEEP_ID}.log"
TMPROOT="/private/var/folders/mq/gphz_64n6_7ct9rv014vx9fw0000gq/T"
VARIANT_BASE="${VARIANT_BASE:-/private/tmp/soclab_qwen_variants}"
SKIP_QUANT_CONVERT="${SKIP_QUANT_CONVERT:-0}"
shift || true

if [[ "$#" -gt 0 ]]; then
  CTX_LIST=("$@")
else
  CTX_LIST=(256 512 1024 2048 4096)
fi

mkdir -p "$OUT_DIR"
echo "[start] $(date) sweep_id=$SWEEP_ID out_dir=$OUT_DIR" | tee -a "$LOG_PATH"

run_cmd() {
  echo "[cmd] $*" | tee -a "$LOG_PATH"
  "$@" 2>&1 | tee -a "$LOG_PATH"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "[error] rc=$rc cmd=$*" | tee -a "$LOG_PATH"
  else
    echo "[ok] cmd=$*" | tee -a "$LOG_PATH"
  fi
  return $rc
}

has_complete_rows() {
  local path="$1"
  "$PY" - <<PY
import json
from pathlib import Path
required = {"NE", "ALL", "NE→GPU"}
p = Path(r"$path")
if not p.exists():
    raise SystemExit(1)
seen = set()
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("status") == "ok":
        seen.add(row.get("scenario_label"))
raise SystemExit(0 if required.issubset(seen) else 1)
PY
}

append_error_row() {
  local ctx="$1"
  local stage="$2"
  local msg="$3"
  "$PY" - <<PY
import json, datetime
from pathlib import Path

path = Path(r"$OUT_DIR") / f"ctx{int($ctx)}_bench.jsonl"
record = {
    "timestamp": datetime.datetime.now().isoformat(),
    "task_type": "llm_decode",
    "model_id": "$MODEL_ID",
    "model_alias": "$MODEL_ALIAS",
    "variant_id": "$MODEL_ALIAS/ctx" + str(int($ctx)),
    "context_len": int($ctx),
    "prefill_len": int($ctx) - 1,
    "gen_tokens": 1,
    "mode": None,
    "prefill_compute_units": None,
    "decode_compute_units": None,
    "scenario_label": "unknown",
    "x_label": "context_len",
    "x_value": int($ctx),
    "uses_coreml": True,
    "status": "error",
    "error_type": "$stage",
    "error_message": "$msg",
    "traceback_summary": "$msg",
    "errors": {"stage": "$stage", "message": "$msg", "failure_type": "$stage"},
}
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\\n")
print(path)
PY
}

cleanup_coreml_tmp() {
  if [[ -d "$TMPROOT" ]]; then
    find "$TMPROOT" -maxdepth 1 -mindepth 1 \
      \( -name 'tmp*' -o -name 'prefill*.mlmodelc' -o -name 'decode*.mlmodelc' -o -name 'prefill.mlmodelc' -o -name 'decode.mlmodelc' \) \
      -exec rm -rf {} + >/dev/null 2>&1 || true
  fi
  local free_space
  free_space="$(df -h "$ROOT_DIR" | awk 'NR==2 {print $4}')"
  echo "[disk] free=$free_space after tmp cleanup" | tee -a "$LOG_PATH"
}

cleanup_ctx_artifacts() {
  local torch_dir="$1"
  local model_dir="$2"
  local variant_dir="$3"
  rm -rf "$torch_dir" "$model_dir" "$variant_dir" >/dev/null 2>&1 || true
}

reset_result_path() {
  local path="$1"
  rm -f "$path" >/dev/null 2>&1 || true
}

for CTX in "${CTX_LIST[@]}"; do
  RESULT_PATH="$OUT_DIR/ctx${CTX}_bench.jsonl"
  VARIANT_DIR="$VARIANT_BASE/ctx${CTX}"
  TORCH_DIR="$VARIANT_DIR/torch"
  MODEL_DIR="$VARIANT_DIR/coreml"

  if has_complete_rows "$RESULT_PATH"; then
    echo "[skip] ctx=$CTX already has ok rows at $RESULT_PATH" | tee -a "$LOG_PATH"
    cleanup_ctx_artifacts "$TORCH_DIR" "$MODEL_DIR" "$VARIANT_DIR"
    continue
  fi

  # Stale partial/error rows from previous attempts should not mix with rerun output.
  reset_result_path "$RESULT_PATH"

  PREFILL_PT="$TORCH_DIR/prefill.pt"
  DECODE_PT="$TORCH_DIR/decode.pt"
  META_JSON="$TORCH_DIR/model_meta.json"
  PREFILL_MLPACKAGE="$MODEL_DIR/prefill.mlpackage"
  DECODE_MLPACKAGE="$MODEL_DIR/decode.mlpackage"

  if [[ -f "$PREFILL_PT" && -f "$DECODE_PT" && -f "$META_JSON" ]]; then
    echo "[skip] export exists for ctx=$CTX at $TORCH_DIR" | tee -a "$LOG_PATH"
  else
    run_cmd "$PY" "$ROOT_DIR/scripts/01_export_torch.py" \
      --config "$ROOT_DIR/configs/default.yaml" \
      --model-id "$MODEL_ID" \
      --context-len "$CTX" \
      --variant-dir "$VARIANT_DIR" \
      --batch-size 1 \
      --seed 1337
    if [[ $? -ne 0 ]]; then
      append_error_row "$CTX" "export_fail" "01_export_torch failed for ctx=$CTX"
      cleanup_ctx_artifacts "$TORCH_DIR" "$MODEL_DIR" "$VARIANT_DIR"
      continue
    fi
  fi

  cleanup_coreml_tmp
  if [[ -d "$PREFILL_MLPACKAGE" && -d "$DECODE_MLPACKAGE" ]]; then
    echo "[skip] coreml exists for ctx=$CTX at $MODEL_DIR" | tee -a "$LOG_PATH"
  else
    if [[ "$SKIP_QUANT_CONVERT" == "1" ]]; then
      run_cmd "$PY" "$ROOT_DIR/scripts/02_convert_coreml.py" \
        --config "$ROOT_DIR/configs/default.yaml" \
        --model-id "$MODEL_ID" \
        --context-len "$CTX" \
        --variant-dir "$VARIANT_DIR" \
        --skip-validate
      if [[ $? -ne 0 ]]; then
        append_error_row "$CTX" "convert_fail" "02_convert_coreml failed (non-quantized only) for ctx=$CTX"
        cleanup_ctx_artifacts "$TORCH_DIR" "$MODEL_DIR" "$VARIANT_DIR"
        continue
      fi
    else
      run_cmd "$PY" "$ROOT_DIR/scripts/02_convert_coreml.py" \
        --config "$ROOT_DIR/configs/default.yaml" \
        --model-id "$MODEL_ID" \
        --context-len "$CTX" \
        --variant-dir "$VARIANT_DIR" \
        --skip-validate \
        --allow-weight-quant \
        --weight-quant-mode int4
      if [[ $? -ne 0 ]]; then
        echo "[warn] quantized convert failed at ctx=$CTX; retrying without quantization" | tee -a "$LOG_PATH"
        run_cmd "$PY" "$ROOT_DIR/scripts/02_convert_coreml.py" \
          --config "$ROOT_DIR/configs/default.yaml" \
          --model-id "$MODEL_ID" \
          --context-len "$CTX" \
          --variant-dir "$VARIANT_DIR" \
          --skip-validate
        if [[ $? -ne 0 ]]; then
          append_error_row "$CTX" "convert_fail" "02_convert_coreml failed (quantized and non-quantized) for ctx=$CTX"
          cleanup_ctx_artifacts "$TORCH_DIR" "$MODEL_DIR" "$VARIANT_DIR"
          continue
        fi
      fi
    fi
  fi

  cleanup_coreml_tmp
  run_cmd "$PY" "$ROOT_DIR/scripts/03_bench.py" \
    --config "$ROOT_DIR/configs/default.yaml" \
    --model-id "$MODEL_ID" \
    --model-alias "$MODEL_ALIAS" \
    --context-len "$CTX" \
    --variant-dir "$VARIANT_DIR" \
    --mode whole \
    --cu CPU_AND_NE \
    --runs 1 \
    --warmup 0 \
    --gen-tokens 1 \
    --seed 1337 \
    --results-path "$RESULT_PATH"
  if [[ $? -ne 0 ]]; then
    append_error_row "$CTX" "bench_ne_fail" "03_bench NE failed for ctx=$CTX"
  fi

  cleanup_coreml_tmp
  run_cmd "$PY" "$ROOT_DIR/scripts/03_bench.py" \
    --config "$ROOT_DIR/configs/default.yaml" \
    --model-id "$MODEL_ID" \
    --model-alias "$MODEL_ALIAS" \
    --context-len "$CTX" \
    --variant-dir "$VARIANT_DIR" \
    --mode whole \
    --cu ALL \
    --runs 1 \
    --warmup 0 \
    --gen-tokens 1 \
    --seed 1337 \
    --results-path "$RESULT_PATH"
  if [[ $? -ne 0 ]]; then
    append_error_row "$CTX" "bench_all_fail" "03_bench ALL failed for ctx=$CTX"
  fi

  cleanup_coreml_tmp
  run_cmd "$PY" "$ROOT_DIR/scripts/03_bench.py" \
    --config "$ROOT_DIR/configs/default.yaml" \
    --model-id "$MODEL_ID" \
    --model-alias "$MODEL_ALIAS" \
    --context-len "$CTX" \
    --variant-dir "$VARIANT_DIR" \
    --mode split \
    --prefill-cu CPU_AND_NE \
    --decode-cu CPU_AND_GPU \
    --runs 1 \
    --warmup 0 \
    --gen-tokens 1 \
    --seed 1337 \
    --results-path "$RESULT_PATH"
  if [[ $? -ne 0 ]]; then
    append_error_row "$CTX" "bench_ne_gpu_fail" "03_bench NE->GPU failed for ctx=$CTX"
  fi
  cleanup_coreml_tmp
  cleanup_ctx_artifacts "$TORCH_DIR" "$MODEL_DIR" "$VARIANT_DIR"
done

echo "[done] $(date) sweep_id=$SWEEP_ID" | tee -a "$LOG_PATH"
