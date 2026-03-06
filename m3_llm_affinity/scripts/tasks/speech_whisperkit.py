#!/usr/bin/env python3
"""WhisperKit Core ML speech benchmark task for Apple Silicon."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import shutil
import subprocess
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import psutil

try:
    import coremltools as ct
except Exception:  # pragma: no cover - handled by dependency checks
    ct = None

MODEL_REPO = "argmaxinc/whisperkit-coreml"
DEFAULT_VARIANT = "openai_whisper-tiny.en"

CU_MAP = {
    "CPU_ONLY": getattr(ct.ComputeUnit, "CPU_ONLY", None) if ct else None,
    "CPU_AND_GPU": getattr(ct.ComputeUnit, "CPU_AND_GPU", None) if ct else None,
    "CPU_AND_NE": getattr(ct.ComputeUnit, "CPU_AND_NE", None) if ct else None,
    "ALL": getattr(ct.ComputeUnit, "ALL", None) if ct else None,
}

ABBR = {
    "CPU_ONLY": "CPU",
    "CPU_AND_GPU": "GPU",
    "CPU_AND_NE": "NE",
    "ALL": "ALL",
}


def _now() -> str:
    return dt.datetime.now().isoformat()


def _skip(reason: str, failure_type: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "failure_type": failure_type,
        "error_type": failure_type,
        "error_message": reason,
        "errors": {"message": reason},
    }


def _ok(**kwargs: Any) -> Dict[str, Any]:
    out = {"status": "ok"}
    out.update(kwargs)
    return out


def _deps_available() -> Tuple[bool, str]:
    if ct is None:
        return False, "coremltools is not installed"
    missing: List[str] = []
    if importlib.util.find_spec("numpy") is None:
        missing.append("numpy")
    if importlib.util.find_spec("psutil") is None:
        missing.append("psutil")
    if missing:
        return False, "missing dependencies: " + ", ".join(missing)
    return True, ""


def _default_alias(model_variant: str) -> str:
    tail = str(model_variant).replace("openai_whisper-", "").replace("/", "_").replace(".", "_")
    return f"whisperkit_{tail}"


def _resolve_assets_root(task_cfg: Dict[str, Any], root_dir: Path) -> Path:
    raw = str(task_cfg.get("assets_root", "assets/whisperkit_coreml"))
    path = Path(raw)
    if not path.is_absolute():
        path = (root_dir / path).resolve()
    return path


def _required_paths(variant_dir: Path) -> Dict[str, Dict[str, Path]]:
    return {
        "mlmodelc": {
            "mel": variant_dir / "MelSpectrogram.mlmodelc",
            "encoder": variant_dir / "AudioEncoder.mlmodelc",
            "decoder": variant_dir / "TextDecoder.mlmodelc",
        },
        "mlpackage": {
            "mel": variant_dir / "MelSpectrogram.mlpackage",
            "encoder": variant_dir / "AudioEncoder.mlpackage",
            "decoder": variant_dir / "TextDecoder.mlpackage",
        },
    }


def _missing_required(required: Dict[str, Dict[str, Path]]) -> List[Path]:
    missing: List[Path] = []
    for stage in ("mel", "encoder", "decoder"):
        has_mlmodelc = required["mlmodelc"][stage].exists()
        has_mlpackage = required["mlpackage"][stage].exists()
        if not has_mlmodelc and not has_mlpackage:
            missing.append(required["mlmodelc"][stage])
    return missing


def _missing_asset_message(variant_dir: Path, missing: Sequence[Path]) -> str:
    expected_text = (
        "MelSpectrogram.(mlmodelc|mlpackage), "
        "AudioEncoder.(mlmodelc|mlpackage), "
        "TextDecoder.(mlmodelc|mlpackage)"
    )
    missing_text = ", ".join(str(p.name) for p in missing)
    return (
        f"Missing WhisperKit assets under {variant_dir}. "
        f"Expected: {expected_text}. Missing: {missing_text}."
    )


def _select_stage_path(required: Dict[str, Dict[str, Path]], stage: str) -> Optional[Path]:
    mlpackage_path = required["mlpackage"][stage]
    if mlpackage_path.exists():
        return mlpackage_path
    mlmodelc_path = required["mlmodelc"][stage]
    if mlmodelc_path.exists():
        return mlmodelc_path
    return None


def _download_variant(assets_root: Path, model_variant: str) -> Tuple[bool, str]:
    if importlib.util.find_spec("huggingface_hub") is None:
        return False, "huggingface_hub is not installed"

    try:
        from huggingface_hub import snapshot_download

        assets_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=str(assets_root),
            allow_patterns=[f"{model_variant}/*"],
        )
        return True, f"downloaded {model_variant} from {MODEL_REPO}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _parse_scenarios(task_cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = task_cfg.get("scenarios")
    out: List[Dict[str, str]] = []
    if isinstance(raw, list) and raw:
        for row in raw:
            out.append(
                {
                    "label": str(row.get("label") or ""),
                    "mel": str(row.get("mel")),
                    "encoder": str(row.get("encoder")),
                    "decoder": str(row.get("decoder")),
                }
            )
        return out

    # Backward compatibility: map backend presets if scenarios are absent.
    backends = task_cfg.get("compute_units", {}).get("backends", [])
    for backend in backends:
        b = str(backend).upper()
        if b == "CPU_ONLY":
            out.append({"label": "CPU", "mel": "CPU_ONLY", "encoder": "CPU_ONLY", "decoder": "CPU_ONLY"})
        elif b == "MPS":
            out.append(
                {
                    "label": "GPU",
                    "mel": "CPU_AND_GPU",
                    "encoder": "CPU_AND_GPU",
                    "decoder": "CPU_AND_GPU",
                }
            )
    return out


def _scenario_label(mel_cu: str, enc_cu: str, dec_cu: str, explicit: str = "") -> str:
    if explicit:
        return explicit
    return f"{ABBR.get(mel_cu, mel_cu)}|{ABBR.get(enc_cu, enc_cu)}|{ABBR.get(dec_cu, dec_cu)}"


def _parse_multiarray_input(inp: Any) -> Tuple[Tuple[int, ...], np.dtype]:
    arr = inp.type.multiArrayType
    shape = tuple(int(x) if int(x) > 0 else 1 for x in arr.shape)

    dt_map = {
        65568: np.float32,  # FLOAT32
        65552: np.float16,  # FLOAT16
        131104: np.int32,   # INT32
        131072: np.float64, # DOUBLE
    }
    np_dtype = dt_map.get(int(arr.dataType), np.float32)
    return shape, np_dtype


def _dtype_from_metadata(data_type: str) -> np.dtype:
    mapping = {
        "Float16": np.float16,
        "Float32": np.float32,
        "Int32": np.int32,
        "Double": np.float64,
    }
    return mapping.get(str(data_type), np.float32)


def _parse_shape_text(text: str) -> Tuple[int, ...]:
    raw = str(text or "").strip()
    if not raw:
        return tuple()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    dims: List[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            dims.append(int(item))
        except Exception:
            dims.append(1)
    return tuple(dims)


def _metadata_schema(model: Any) -> Tuple[Dict[str, Tuple[Tuple[int, ...], np.dtype]], Dict[str, Tuple[int, ...]]]:
    path_text = str(getattr(model, "path_or_asset", ""))
    if not path_text:
        return {}, {}

    meta_path = Path(path_text) / "metadata.json"
    if not meta_path.exists():
        return {}, {}

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}

    doc = payload[0] if isinstance(payload, list) and payload else payload
    inputs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
    outputs: Dict[str, Tuple[int, ...]] = {}

    for item in doc.get("inputSchema", []) or []:
        name = str(item.get("name") or "")
        if not name:
            continue
        shape = _parse_shape_text(str(item.get("shape") or ""))
        dtype = _dtype_from_metadata(str(item.get("dataType") or "Float32"))
        inputs[name] = (shape, dtype)

    for item in doc.get("outputSchema", []) or []:
        name = str(item.get("name") or "")
        if not name:
            continue
        outputs[name] = _parse_shape_text(str(item.get("shape") or ""))

    return inputs, outputs


def _input_schema(model: Any) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
    if hasattr(model, "get_spec"):
        spec = model.get_spec()
        info: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        for inp in spec.description.input:
            if not inp.type.HasField("multiArrayType"):
                continue
            info[inp.name] = _parse_multiarray_input(inp)
        return info

    inputs, _ = _metadata_schema(model)
    return inputs


def _output_shape(model: Any, name: str) -> Optional[Tuple[int, ...]]:
    if hasattr(model, "get_spec"):
        spec = model.get_spec()
        for out in spec.description.output:
            if out.name != name or not out.type.HasField("multiArrayType"):
                continue
            shape = tuple(int(x) if int(x) > 0 else 1 for x in out.type.multiArrayType.shape)
            return shape

    _, outputs = _metadata_schema(model)
    if name in outputs:
        return outputs[name]
    return None


def _load_model(path: Path, cu_name: str):
    cu = CU_MAP.get(cu_name)
    if cu is None:
        raise ValueError(f"Unsupported compute unit for WhisperKit: {cu_name}")
    if path.suffix == ".mlmodelc" and hasattr(ct.models, "CompiledMLModel"):
        return ct.models.CompiledMLModel(str(path), compute_units=cu)
    return ct.models.MLModel(str(path), compute_units=cu)


def _make_waveform(total_seconds: int, sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(total_seconds * sample_rate)
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    wave = (
        0.6 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.02 * rng.standard_normal(size=n).astype(np.float32)
    )
    return wave.astype(np.float32, copy=False)


def _error_row(
    *,
    model_alias: str,
    model_id: str,
    scenario_label: Optional[str],
    x_value: Optional[int],
    mel_cu: Optional[str],
    encoder_cu: Optional[str],
    decoder_cu: Optional[str],
    message: str,
    error_type: str,
) -> Dict[str, Any]:
    return {
        "timestamp": _now(),
        "task_type": "speech_whisperkit",
        "model_id": model_id,
        "model_alias": model_alias,
        "variant_id": None,
        "context_len": None,
        "prefill_len": None,
        "gen_tokens": None,
        "mode": "staged",
        "prefill_compute_units": encoder_cu,
        "decode_compute_units": decoder_cu,
        "mel_compute_units": mel_cu,
        "scenario_label": scenario_label,
        "x_label": "audio_seconds",
        "x_value": x_value,
        "primary_latency_ms": None,
        "primary_throughput": None,
        "uses_coreml": True,
        "ttft_ms": None,
        "tokens_per_sec": None,
        "tpot_ms_mean": None,
        "tpot_ms_p95": None,
        "effective_TFLOPS_prefill": None,
        "effective_TFLOPS_decode": None,
        "peak_rss_mb": None,
        "status": "error",
        "error_type": error_type,
        "error_message": message,
        "traceback_summary": message,
        "errors": {"message": message},
    }


def _ok_row(
    *,
    model_alias: str,
    model_id: str,
    scenario_label: str,
    seconds: int,
    run_index: int,
    mel_cu: str,
    encoder_cu: str,
    decoder_cu: str,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "timestamp": _now(),
        "task_type": "speech_whisperkit",
        "model_id": model_id,
        "model_alias": model_alias,
        "variant_id": None,
        "context_len": None,
        "prefill_len": None,
        "gen_tokens": None,
        "mode": "staged",
        "prefill_compute_units": encoder_cu,
        "decode_compute_units": decoder_cu,
        "mel_compute_units": mel_cu,
        "scenario_label": scenario_label,
        "x_label": "audio_seconds",
        "x_value": int(seconds),
        "primary_latency_ms": float(metrics["total_ms"]),
        "primary_throughput": float(metrics["audio_seconds_per_sec"]),
        "audio_seconds_per_sec": float(metrics["audio_seconds_per_sec"]),
        "rtf": float(metrics["rtf"]),
        "uses_coreml": True,
        "mel_ms_total": float(metrics["mel_ms_total"]),
        "enc_ms_total": float(metrics["enc_ms_total"]),
        "dec_ms_total": float(metrics["dec_ms_total"]),
        "prefill_latency_ms": float(metrics["enc_ms_total"]),
        "total_decode_latency_ms": float(metrics["dec_ms_total"]),
        "tokens_per_sec": float(metrics["audio_seconds_per_sec"]),
        "ttft_ms": float(metrics["total_ms"]),
        "tpot_ms_mean": None,
        "tpot_ms_p95": None,
        "effective_TFLOPS_prefill": None,
        "effective_TFLOPS_decode": None,
        "peak_rss_mb": float(metrics["peak_rss_mb"]),
        "run_index": int(run_index),
        "status": "ok",
        "error_type": None,
        "error_message": None,
        "traceback_summary": None,
        "errors": None,
    }


def _chunk_waveform(wave: np.ndarray, start: int, chunk_samples: int, np_dtype: np.dtype) -> np.ndarray:
    part = wave[start : start + chunk_samples]
    if part.shape[0] < chunk_samples:
        out = np.zeros((chunk_samples,), dtype=np_dtype)
        out[: part.shape[0]] = part.astype(np_dtype, copy=False)
        return out
    return part[:chunk_samples].astype(np_dtype, copy=False)


def _distribute_decode_steps(total_steps: int, segments: int) -> List[int]:
    if segments <= 0:
        return []
    base = total_steps // segments
    rem = total_steps % segments
    return [base + (1 if i < rem else 0) for i in range(segments)]


def _run_once(
    *,
    mel_model: Any,
    enc_model: Any,
    dec_model: Any,
    seconds: int,
    sample_rate: int,
    decode_tokens_per_second: float,
    max_decode_tokens: int,
    seed: int,
    process: psutil.Process,
) -> Dict[str, float]:
    mel_inputs = _input_schema(mel_model)
    enc_inputs = _input_schema(enc_model)
    dec_inputs = _input_schema(dec_model)

    if "audio" not in mel_inputs:
        raise RuntimeError("MelSpectrogram model missing required input 'audio'")

    audio_shape, audio_dtype = mel_inputs["audio"]
    chunk_samples = int(np.prod(audio_shape))
    if chunk_samples <= 0:
        raise RuntimeError("MelSpectrogram audio input has invalid shape")

    total_samples = int(seconds * sample_rate)
    chunk_seconds = float(chunk_samples / float(sample_rate))
    segments = max(1, int(math.ceil(float(seconds) / chunk_seconds)))

    base_wave = _make_waveform(seconds, sample_rate, seed)

    total_steps = min(int(max_decode_tokens), int(seconds * decode_tokens_per_second))
    step_alloc = _distribute_decode_steps(max(0, total_steps), segments)

    key_shape, key_dtype = dec_inputs["key_cache"]
    val_shape, val_dtype = dec_inputs["value_cache"]
    mask_shape, mask_dtype = dec_inputs["kv_cache_update_mask"]
    pad_shape, pad_dtype = dec_inputs["decoder_key_padding_mask"]
    enc_shape, enc_dtype = dec_inputs["encoder_output_embeds"]
    input_shape, input_dtype = dec_inputs["input_ids"]
    cache_len_shape, cache_len_dtype = dec_inputs["cache_length"]

    vocab_size = 51864
    logits_shape = _output_shape(dec_model, "logits")
    if logits_shape and len(logits_shape) == 3 and int(logits_shape[-1]) > 0:
        vocab_size = int(logits_shape[-1])

    cache_width = int(key_shape[-1])
    rng = np.random.default_rng(seed)

    peak_rss = process.memory_info().rss
    mel_ms_total = 0.0
    enc_ms_total = 0.0
    dec_ms_total = 0.0

    global_step = 0

    for seg_idx in range(segments):
        start = seg_idx * chunk_samples
        audio_chunk = _chunk_waveform(base_wave, start, chunk_samples, audio_dtype).reshape(audio_shape)

        t0 = perf_counter()
        mel_out = mel_model.predict({"audio": audio_chunk})
        mel_ms_total += (perf_counter() - t0) * 1000.0
        peak_rss = max(peak_rss, process.memory_info().rss)

        mel_feat = np.asarray(mel_out.get("melspectrogram_features"), dtype=np.float16)
        enc_key = next(iter(enc_inputs.keys()))
        enc_tensor = mel_feat.astype(enc_inputs[enc_key][1], copy=False)

        t1 = perf_counter()
        enc_out = enc_model.predict({enc_key: enc_tensor})
        enc_ms_total += (perf_counter() - t1) * 1000.0
        peak_rss = max(peak_rss, process.memory_info().rss)

        encoder_embeds = np.asarray(enc_out.get("encoder_output_embeds"), dtype=np.float16)
        if tuple(encoder_embeds.shape) != tuple(enc_shape):
            encoder_embeds = np.broadcast_to(encoder_embeds, enc_shape).copy()
        encoder_embeds = encoder_embeds.astype(enc_dtype, copy=False)

        key_cache = np.zeros(key_shape, dtype=key_dtype)
        value_cache = np.zeros(val_shape, dtype=val_dtype)
        cache_len = 0

        seg_steps = int(step_alloc[seg_idx]) if seg_idx < len(step_alloc) else 0
        for _ in range(seg_steps):
            cache_idx = cache_len if cache_len < cache_width else cache_width - 1

            kv_mask = np.zeros(mask_shape, dtype=mask_dtype)
            kv_mask.reshape(-1)[cache_idx] = np.array(1.0, dtype=mask_dtype)

            dec_pad_mask = np.zeros(pad_shape, dtype=pad_dtype)
            if cache_len > 0:
                dec_pad_mask.reshape(-1)[: min(cache_len, dec_pad_mask.size)] = np.array(1.0, dtype=pad_dtype)

            token = int((global_step + 50257) % max(2, vocab_size))
            input_ids = np.full(input_shape, token, dtype=input_dtype)
            cache_length = np.full(cache_len_shape, cache_len, dtype=cache_len_dtype)

            dec_inputs_dict = {
                "input_ids": input_ids,
                "cache_length": cache_length,
                "key_cache": key_cache,
                "value_cache": value_cache,
                "kv_cache_update_mask": kv_mask,
                "encoder_output_embeds": encoder_embeds,
                "decoder_key_padding_mask": dec_pad_mask,
            }

            t2 = perf_counter()
            dec_out = dec_model.predict(dec_inputs_dict)
            dec_ms_total += (perf_counter() - t2) * 1000.0
            peak_rss = max(peak_rss, process.memory_info().rss)

            key_up = np.asarray(dec_out.get("key_cache_updates"), dtype=key_dtype)
            val_up = np.asarray(dec_out.get("value_cache_updates"), dtype=val_dtype)

            if cache_len < cache_width:
                key_cache[..., cache_len : cache_len + 1] = key_up
                value_cache[..., cache_len : cache_len + 1] = val_up
                cache_len += 1
            else:
                key_cache[..., :-1] = key_cache[..., 1:]
                value_cache[..., :-1] = value_cache[..., 1:]
                key_cache[..., -1:] = key_up
                value_cache[..., -1:] = val_up
                cache_len = cache_width

            global_step += 1

    total_ms = float(mel_ms_total + enc_ms_total + dec_ms_total)
    total_s = total_ms / 1000.0
    throughput = float(seconds / total_s) if total_s > 0 else 0.0
    rtf = float(total_s / seconds) if seconds > 0 else 0.0

    return {
        "mel_ms_total": float(mel_ms_total),
        "enc_ms_total": float(enc_ms_total),
        "dec_ms_total": float(dec_ms_total),
        "total_ms": float(total_ms),
        "audio_seconds_per_sec": float(throughput),
        "rtf": float(rtf),
        "peak_rss_mb": float(peak_rss / (1024.0 * 1024.0)),
    }


def prepare_variant(task_cfg: Dict[str, Any], root_dir: Path, **kwargs: Any) -> Dict[str, Any]:
    enabled = bool(task_cfg.get("enabled", False))
    if not enabled:
        return _skip("speech_whisperkit disabled in suite config", "task_disabled")

    ok, reason = _deps_available()
    if not ok:
        return _skip(reason, "missing_dependency")

    model_variant = str(task_cfg.get("model_variant", DEFAULT_VARIANT))
    assets_root = _resolve_assets_root(task_cfg, root_dir)
    variant_dir = assets_root / model_variant
    download_if_missing = bool(task_cfg.get("download_if_missing", False))
    dry_run = bool(kwargs.get("dry_run", False))

    required = _required_paths(variant_dir)
    missing = _missing_required(required)

    if missing and download_if_missing:
        if dry_run:
            return _ok(
                model_variant=model_variant,
                model_id=f"{MODEL_REPO}/{model_variant}",
                model_alias=str(task_cfg.get("model_alias") or _default_alias(model_variant)),
                stage_paths={},
                scenarios=_parse_scenarios(task_cfg),
                message=f"dry-run: would download {model_variant} into {assets_root}",
            )
        dl_ok, msg = _download_variant(assets_root, model_variant)
        if dl_ok:
            print(f"[speech_whisperkit] {msg}")
            required = _required_paths(variant_dir)
            missing = _missing_required(required)
        else:
            return _skip(f"WhisperKit download failed: {msg}", "download_unavailable")

    if missing:
        return _skip(_missing_asset_message(variant_dir, missing), "missing_asset")

    scenarios = _parse_scenarios(task_cfg)
    mel_model = _select_stage_path(required, "mel")
    enc_model = _select_stage_path(required, "encoder")
    dec_model = _select_stage_path(required, "decoder")
    if mel_model is None or enc_model is None or dec_model is None:
        return _skip(_missing_asset_message(variant_dir, missing), "missing_asset")

    print("[speech_whisperkit] using assets: " f"mel={mel_model} encoder={enc_model} decoder={dec_model}")

    return _ok(
        model_variant=model_variant,
        model_id=f"{MODEL_REPO}/{model_variant}",
        model_alias=str(task_cfg.get("model_alias") or _default_alias(model_variant)),
        stage_paths={
            "mel_model": str(mel_model),
            "encoder_model": str(enc_model),
            "decoder_model": str(dec_model),
            "mel_computeplan_src": str(required["mlmodelc"]["mel"] if required["mlmodelc"]["mel"].exists() else mel_model),
            "encoder_computeplan_src": str(
                required["mlmodelc"]["encoder"] if required["mlmodelc"]["encoder"].exists() else enc_model
            ),
            "decoder_computeplan_src": str(
                required["mlmodelc"]["decoder"] if required["mlmodelc"]["decoder"].exists() else dec_model
            ),
            "mel_mlmodelc": str(required["mlmodelc"]["mel"]),
            "encoder_mlmodelc": str(required["mlmodelc"]["encoder"]),
            "decoder_mlmodelc": str(required["mlmodelc"]["decoder"]),
            "mel_mlpackage": str(required["mlpackage"]["mel"]),
            "encoder_mlpackage": str(required["mlpackage"]["encoder"]),
            "decoder_mlpackage": str(required["mlpackage"]["decoder"]),
        },
        scenarios=scenarios,
        message=f"using WhisperKit assets from {variant_dir}",
    )


def run_bench(
    task_cfg: Dict[str, Any],
    prep: Dict[str, Any],
    *,
    dry_run: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    model_variant = str(task_cfg.get("model_variant", DEFAULT_VARIANT))
    model_id = str(prep.get("model_id") or f"{MODEL_REPO}/{model_variant}")
    model_alias = str(prep.get("model_alias") or task_cfg.get("model_alias") or _default_alias(model_variant))

    if prep.get("status") != "ok":
        return {
            "status": "error",
            "records": [
                _error_row(
                    model_alias=model_alias,
                    model_id=model_id,
                    scenario_label=None,
                    x_value=None,
                    mel_cu=None,
                    encoder_cu=None,
                    decoder_cu=None,
                    message=str(prep.get("error_message") or prep.get("message") or "prepare failed"),
                    error_type=str(prep.get("error_type") or prep.get("failure_type") or "prepare_failed"),
                )
            ],
        }

    sweep = task_cfg.get("sweep", {})
    seconds_list = [int(x) for x in sweep.get("audio_seconds_list", [5, 20, 40, 80])]
    sample_rate = int(sweep.get("sample_rate", 16000))
    runs = int(sweep.get("runs", 3))
    warmup = int(sweep.get("warmup", 1))
    seed = int(sweep.get("seed", 1337))
    decode_tps = float(sweep.get("decode_tokens_per_second", 2.0))
    max_decode_tokens = int(sweep.get("max_decode_tokens", 256))

    scenario_cfgs = prep.get("scenarios") or _parse_scenarios(task_cfg)
    if not scenario_cfgs:
        return {
            "status": "error",
            "records": [
                _error_row(
                    model_alias=model_alias,
                    model_id=model_id,
                    scenario_label=None,
                    x_value=None,
                    mel_cu=None,
                    encoder_cu=None,
                    decoder_cu=None,
                    message="speech_whisperkit scenarios are empty",
                    error_type="invalid_config",
                )
            ],
        }

    if dry_run:
        labels = [
            _scenario_label(str(s.get("mel")), str(s.get("encoder")), str(s.get("decoder")), str(s.get("label", "")))
            for s in scenario_cfgs
        ]
        return {
            "status": "ok",
            "records": [],
            "message": (
                "dry-run: speech_whisperkit "
                f"scenarios={len(scenario_cfgs)} labels={labels} seconds={seconds_list} "
                f"runs={runs} warmup={warmup}"
            ),
        }

    stage_paths = {k: Path(v) for k, v in prep.get("stage_paths", {}).items()}
    process = psutil.Process()

    records: List[Dict[str, Any]] = []

    for scenario in scenario_cfgs:
        mel_cu = str(scenario.get("mel"))
        enc_cu = str(scenario.get("encoder"))
        dec_cu = str(scenario.get("decoder"))

        if mel_cu not in CU_MAP or enc_cu not in CU_MAP or dec_cu not in CU_MAP:
            records.append(
                _error_row(
                    model_alias=model_alias,
                    model_id=model_id,
                    scenario_label=None,
                    x_value=None,
                    mel_cu=mel_cu,
                    encoder_cu=enc_cu,
                    decoder_cu=dec_cu,
                    message=f"Unsupported CU in scenario: {scenario}",
                    error_type="invalid_compute_unit",
                )
            )
            continue

        scenario_label = _scenario_label(mel_cu, enc_cu, dec_cu, str(scenario.get("label", "")))
        print(f"[speech_whisperkit] scenario={scenario_label} mel={mel_cu} encoder={enc_cu} decoder={dec_cu}")

        try:
            mel_model = _load_model(
                stage_paths.get("mel_model") or stage_paths.get("mel_mlpackage") or stage_paths["mel_mlmodelc"],
                mel_cu,
            )
            enc_model = _load_model(
                stage_paths.get("encoder_model") or stage_paths.get("encoder_mlpackage") or stage_paths["encoder_mlmodelc"],
                enc_cu,
            )
            dec_model = _load_model(
                stage_paths.get("decoder_model") or stage_paths.get("decoder_mlpackage") or stage_paths["decoder_mlmodelc"],
                dec_cu,
            )
        except Exception as exc:
            records.append(
                _error_row(
                    model_alias=model_alias,
                    model_id=model_id,
                    scenario_label=scenario_label,
                    x_value=None,
                    mel_cu=mel_cu,
                    encoder_cu=enc_cu,
                    decoder_cu=dec_cu,
                    message=f"{type(exc).__name__}: {exc}",
                    error_type="model_load_fail",
                )
            )
            continue

        for seconds in seconds_list:
            warmup_failed = False
            for warmup_idx in range(warmup):
                try:
                    _ = _run_once(
                        mel_model=mel_model,
                        enc_model=enc_model,
                        dec_model=dec_model,
                        seconds=seconds,
                        sample_rate=sample_rate,
                        decode_tokens_per_second=decode_tps,
                        max_decode_tokens=max_decode_tokens,
                        seed=seed + warmup_idx,
                        process=process,
                    )
                except Exception as exc:
                    records.append(
                        _error_row(
                            model_alias=model_alias,
                            model_id=model_id,
                            scenario_label=scenario_label,
                            x_value=seconds,
                            mel_cu=mel_cu,
                            encoder_cu=enc_cu,
                            decoder_cu=dec_cu,
                            message=f"warmup_{warmup_idx}: {type(exc).__name__}: {exc}",
                            error_type="warmup_fail",
                        )
                    )
                    warmup_failed = True
                    break

            if warmup_failed:
                continue

            for run_idx in range(runs):
                try:
                    metrics = _run_once(
                        mel_model=mel_model,
                        enc_model=enc_model,
                        dec_model=dec_model,
                        seconds=seconds,
                        sample_rate=sample_rate,
                        decode_tokens_per_second=decode_tps,
                        max_decode_tokens=max_decode_tokens,
                        seed=seed + run_idx,
                        process=process,
                    )
                    records.append(
                        _ok_row(
                            model_alias=model_alias,
                            model_id=model_id,
                            scenario_label=scenario_label,
                            seconds=seconds,
                            run_index=run_idx,
                            mel_cu=mel_cu,
                            encoder_cu=enc_cu,
                            decoder_cu=dec_cu,
                            metrics=metrics,
                        )
                    )
                except Exception as exc:
                    tb = traceback.format_exc().splitlines()
                    summary = "\n".join(tb[-10:]) if tb else str(exc)
                    records.append(
                        _error_row(
                            model_alias=model_alias,
                            model_id=model_id,
                            scenario_label=scenario_label,
                            x_value=seconds,
                            mel_cu=mel_cu,
                            encoder_cu=enc_cu,
                            decoder_cu=dec_cu,
                            message=summary,
                            error_type=type(exc).__name__,
                        )
                    )

    return {"status": "ok", "records": records}


def _iter_container(container: Any) -> Iterable[Any]:
    if container is None:
        return []
    if isinstance(container, dict):
        return [container[k] for k in sorted(container.keys())]
    if isinstance(container, (list, tuple)):
        return list(container)
    return [container]


def _iter_operations_from_plan(plan: Any) -> List[Any]:
    structure = plan.model_structure

    fn_container = None
    if hasattr(structure, "functions"):
        fn_container = getattr(structure, "functions")
    elif hasattr(structure, "program") and hasattr(structure.program, "functions"):
        fn_container = getattr(structure.program, "functions")

    ops: List[Any] = []
    for fn in _iter_container(fn_container):
        blocks = None
        if hasattr(fn, "block_specializations"):
            blocks = getattr(fn, "block_specializations")
        elif hasattr(fn, "blocks"):
            blocks = getattr(fn, "blocks")
        elif hasattr(fn, "block"):
            blocks = getattr(fn, "block")

        for block in _iter_container(blocks):
            op_list = getattr(block, "operations", None) or getattr(block, "ops", None)
            for op in _iter_container(op_list):
                ops.append(op)
    return ops


def _stringify_devices(devices: Any) -> str:
    if devices is None:
        return ""
    if isinstance(devices, (list, tuple, set)):
        return "|".join(sorted([str(d) for d in devices]))
    return str(devices)


def _extract_usage(plan: Any, op: Any) -> Tuple[str, str]:
    try:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    except Exception:
        return "", ""

    preferred = ""
    supported = ""
    for key in ("preferred_compute_devices", "preferred_devices", "preferred_device"):
        if hasattr(usage, key):
            preferred = _stringify_devices(getattr(usage, key))
            break
    for key in ("supported_compute_devices", "supported_devices", "supported_device"):
        if hasattr(usage, key):
            supported = _stringify_devices(getattr(usage, key))
            break
    if not preferred and not supported:
        preferred = str(usage)
    return preferred, supported


def _extract_cost(plan: Any, op: Any) -> float:
    try:
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)
    except Exception:
        return 0.0
    if isinstance(cost, (int, float)):
        return float(cost)
    for attr in ("estimated_cost", "cost", "value", "weight"):
        if hasattr(cost, attr):
            value = getattr(cost, attr)
            if isinstance(value, (int, float)):
                return float(value)
    try:
        return float(str(cost))
    except Exception:
        return 0.0


def _compile_to_mlmodelc(input_path: Path, compiled_out: Path) -> Path:
    if input_path.suffix == ".mlmodelc":
        return input_path

    compiled_out.parent.mkdir(parents=True, exist_ok=True)
    if compiled_out.exists():
        shutil.rmtree(compiled_out)

    cmd = ["xcrun", "coremlc", "compile", str(input_path), str(compiled_out.parent)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and compiled_out.exists():
        return compiled_out

    try:
        ct.utils.compile_model(str(input_path), destination_path=str(compiled_out))
    except Exception as exc:
        msg = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(
            f"coremlc compile failed: {msg}; fallback compile_model failed: {type(exc).__name__}: {exc}"
        ) from exc

    if not compiled_out.exists():
        candidates = sorted(compiled_out.parent.glob("*.mlmodelc"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No compiled .mlmodelc produced for {input_path}")
        return candidates[0]
    return compiled_out


def _device_bucket(text: str) -> Optional[str]:
    t = str(text).lower()
    if not t:
        return None
    if "neural" in t or "ane" in t:
        return "NE"
    if "gpu" in t:
        return "GPU"
    if "cpu" in t:
        return "CPU"
    return "OTHER"


def _device_counts(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts = {"CPU": 0, "GPU": 0, "NE": 0, "OTHER": 0}
    for row in rows:
        raw = str(row.get(key) or "")
        if not raw:
            continue
        buckets = set()
        for token in raw.replace(",", "|").split("|"):
            bucket = _device_bucket(token)
            if bucket:
                buckets.add(bucket)
        for bucket in buckets:
            counts[bucket] += 1
    return counts


def _dump_plan(compiled_path: Path, csv_path: Path) -> Dict[str, Any]:
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(str(compiled_path))
    operations = _iter_operations_from_plan(plan)

    rows: List[Dict[str, Any]] = []
    for idx, op in enumerate(operations):
        op_type = str(getattr(op, "operator_name", getattr(op, "op_type", type(op).__name__)))
        op_name = str(getattr(op, "name", ""))
        preferred, supported = _extract_usage(plan, op)
        cost = _extract_cost(plan, op)
        rows.append(
            {
                "op_index": idx,
                "op_type": op_type,
                "op_name": op_name,
                "preferred_devices": preferred,
                "supported_devices": supported,
                "estimated_cost": float(cost),
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "op_index",
                "op_type",
                "op_name",
                "preferred_devices",
                "supported_devices",
                "estimated_cost",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {
        "num_operations": len(rows),
        "device_counts": {
            "preferred": _device_counts(rows, "preferred_devices"),
            "supported": _device_counts(rows, "supported_devices"),
        },
        "top_10_ops_by_estimated_cost": sorted(rows, key=lambda r: float(r["estimated_cost"]), reverse=True)[:10],
        "csv": str(csv_path),
    }


def dump_computeplan(
    task_cfg: Dict[str, Any],
    prep: Dict[str, Any],
    *,
    root_dir: Path,
    dry_run: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    if not bool(task_cfg.get("enabled", False)):
        return _skip("speech_whisperkit disabled in suite config", "task_disabled")

    if prep.get("status") != "ok":
        return _skip("prepare failed, skipping WhisperKit compute plan", "prepare_failed")
    if dry_run:
        return _ok(message="dry-run: speech_whisperkit compute plan skipped")

    stage_paths = {k: Path(v) for k, v in prep.get("stage_paths", {}).items()}
    reports_dir = (root_dir / "reports").resolve()

    summary: Dict[str, Any] = {
        "model_id": str(prep.get("model_id") or f"{MODEL_REPO}/{DEFAULT_VARIANT}"),
        "model_alias": str(prep.get("model_alias") or _default_alias(DEFAULT_VARIANT)),
        "stages": {},
    }

    try:
        stage_map = {
            "mel": stage_paths.get("mel_computeplan_src") or stage_paths.get("mel_mlmodelc") or stage_paths.get("mel_model"),
            "encoder": stage_paths.get("encoder_computeplan_src")
            or stage_paths.get("encoder_mlmodelc")
            or stage_paths.get("encoder_model"),
            "decoder": stage_paths.get("decoder_computeplan_src")
            or stage_paths.get("decoder_mlmodelc")
            or stage_paths.get("decoder_model"),
        }
        for stage, src in stage_map.items():
            if src is None:
                raise FileNotFoundError(f"missing computeplan source for stage {stage}")
            compiled = _compile_to_mlmodelc(src, src)
            csv_path = reports_dir / f"computeplan_whisperkit_{stage}.csv"
            summary["stages"][stage] = _dump_plan(compiled, csv_path)
    except Exception as exc:
        return _skip(f"{type(exc).__name__}: {exc}", "computeplan_fail")

    summary_path = reports_dir / "computeplan_whisperkit_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return _ok(summary_json=str(summary_path))
