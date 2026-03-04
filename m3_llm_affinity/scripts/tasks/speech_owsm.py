#!/usr/bin/env python3
"""OWSM speech benchmark task (ESPnet/PyTorch backend, CPU or MPS)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import traceback
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch


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
    missing: List[str] = []
    if importlib.util.find_spec("espnet2") is None:
        missing.append("espnet2")
    if importlib.util.find_spec("numpy") is None:
        missing.append("numpy")
    if importlib.util.find_spec("torch") is None:
        missing.append("torch")

    if missing:
        return False, "missing dependencies: " + ", ".join(missing)
    return True, ""


def _scenario_label(prefill_cu: str, decode_cu: str) -> str:
    if prefill_cu == "CPU_ONLY" and decode_cu == "CPU_ONLY":
        return "CPU"
    if prefill_cu == "CPU_AND_GPU" and decode_cu == "CPU_AND_GPU":
        return "GPU"
    return f"{prefill_cu}->{decode_cu}"


def _error_row(
    *,
    model_alias: str,
    model_tag: str,
    scenario_label: Optional[str],
    x_value: Optional[int],
    prefill_cu: Optional[str],
    decode_cu: Optional[str],
    message: str,
    error_type: str,
) -> Dict[str, Any]:
    return {
        "timestamp": _now(),
        "task_type": "speech_owsm",
        "model_id": model_tag,
        "model_alias": model_alias,
        "variant_id": None,
        "context_len": None,
        "prefill_len": None,
        "gen_tokens": None,
        "mode": "backend",
        "prefill_compute_units": prefill_cu,
        "decode_compute_units": decode_cu,
        "scenario_label": scenario_label,
        "x_label": "audio_seconds",
        "x_value": x_value,
        "primary_latency_ms": None,
        "primary_throughput": None,
        "uses_coreml": False,
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
    model_tag: str,
    scenario_label: str,
    seconds: int,
    run_index: int,
    prefill_cu: str,
    decode_cu: str,
    total_ms: float,
    throughput: float,
    rtf: float,
    peak_rss_mb: float,
) -> Dict[str, Any]:
    return {
        "timestamp": _now(),
        "task_type": "speech_owsm",
        "model_id": model_tag,
        "model_alias": model_alias,
        "variant_id": None,
        "context_len": None,
        "prefill_len": None,
        "gen_tokens": None,
        "mode": "backend",
        "prefill_compute_units": prefill_cu,
        "decode_compute_units": decode_cu,
        "scenario_label": scenario_label,
        "x_label": "audio_seconds",
        "x_value": int(seconds),
        "primary_latency_ms": float(total_ms),
        "primary_throughput": float(throughput),
        "audio_seconds_per_sec": float(throughput),
        "rtf": float(rtf),
        "uses_coreml": False,
        "prefill_latency_ms": None,
        "total_decode_latency_ms": None,
        "tokens_per_sec": float(throughput),
        "ttft_ms": float(total_ms),
        "tpot_ms_mean": None,
        "tpot_ms_p95": None,
        "effective_TFLOPS_prefill": None,
        "effective_TFLOPS_decode": None,
        "peak_rss_mb": float(peak_rss_mb),
        "run_index": int(run_index),
        "status": "ok",
        "error_type": None,
        "error_message": None,
        "traceback_summary": None,
        "errors": None,
    }


def _make_waveform(total_seconds: int, sample_rate: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(total_seconds * sample_rate)
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    # Deterministic synthetic speech-like signal.
    wave = (
        0.5 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.1 * np.sin(2.0 * np.pi * 880.0 * t)
        + 0.02 * rng.standard_normal(size=n).astype(np.float32)
    )
    return wave.astype(np.float32, copy=False)


def _backend_to_device(backend: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    b = str(backend).upper()
    if b == "CPU_ONLY":
        return "cpu", "CPU_ONLY", "CPU_ONLY"
    if b == "MPS":
        if not torch.backends.mps.is_available():
            return None, None, None
        return "mps", "CPU_AND_GPU", "CPU_AND_GPU"
    if b == "CPU_AND_NE":
        return None, "CPU_AND_NE", "CPU_AND_NE"
    return None, None, None


def _make_s2t(model_tag: str, device: str):
    from espnet2.bin.s2t_inference import Speech2Text

    # Prefer the modern classmethod path; older init() signatures may not take model_tag.
    if hasattr(Speech2Text, "from_pretrained"):
        return Speech2Text.from_pretrained(model_tag=model_tag, device=device)

    kwargs: Dict[str, Any] = {
        "model_tag": model_tag,
        "device": device,
    }
    return Speech2Text(**kwargs)


def prepare_variant(task_cfg: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    enabled = bool(task_cfg.get("enabled", False))
    if not enabled:
        return _skip("speech_owsm disabled in suite config", "task_disabled")

    ok, reason = _deps_available()
    if not ok:
        return _skip(reason, "missing_dependency")

    return _ok(message="speech_owsm dependencies available")


def run_bench(
    task_cfg: Dict[str, Any],
    prep: Dict[str, Any],
    *,
    dry_run: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    model_alias = str(task_cfg.get("model_alias", "owsm_v31_small"))
    model_tag = str(task_cfg.get("model_tag", "espnet/owsm_v3.1_ebf_small"))

    if prep.get("status") != "ok":
        return {
            "status": "error",
            "records": [
                _error_row(
                    model_alias=model_alias,
                    model_tag=model_tag,
                    scenario_label=None,
                    x_value=None,
                    prefill_cu=None,
                    decode_cu=None,
                    message=str(prep.get("error_message") or prep.get("message") or "prepare failed"),
                    error_type=str(prep.get("error_type") or prep.get("failure_type") or "prepare_failed"),
                )
            ],
        }

    sweep = task_cfg.get("sweep", {})
    seconds_list = [int(x) for x in sweep.get("audio_seconds_list", [5, 10, 20, 40])]
    sample_rate = int(sweep.get("sample_rate", 16000))
    runs = int(sweep.get("runs", 3))
    warmup = int(sweep.get("warmup", 1))
    seed = int(sweep.get("seed", 1337))

    backends = task_cfg.get("compute_units", {}).get("backends", ["CPU_ONLY", "MPS"])

    if dry_run:
        return {
            "status": "ok",
            "records": [],
            "message": f"dry-run: speech_owsm backends={backends} seconds={seconds_list}",
        }

    max_seconds = max(seconds_list) if seconds_list else 1
    base_wave = _make_waveform(max_seconds, sample_rate, seed)
    process = psutil.Process()

    records: List[Dict[str, Any]] = []

    for backend in backends:
        device, prefill_cu, decode_cu = _backend_to_device(str(backend))

        if backend == "CPU_AND_NE":
            records.append(
                _error_row(
                    model_alias=model_alias,
                    model_tag=model_tag,
                    scenario_label="NE",
                    x_value=None,
                    prefill_cu="CPU_AND_NE",
                    decode_cu="CPU_AND_NE",
                    message="OWSM task uses ESPnet/PyTorch backend; CPU_AND_NE is unsupported",
                    error_type="unsupported_backend",
                )
            )
            continue

        if device is None or prefill_cu is None or decode_cu is None:
            records.append(
                _error_row(
                    model_alias=model_alias,
                    model_tag=model_tag,
                    scenario_label=str(backend),
                    x_value=None,
                    prefill_cu=None,
                    decode_cu=None,
                    message=f"backend unavailable: {backend}",
                    error_type="backend_unavailable",
                )
            )
            continue

        try:
            s2t = _make_s2t(model_tag=model_tag, device=device)
        except Exception as exc:
            records.append(
                _error_row(
                    model_alias=model_alias,
                    model_tag=model_tag,
                    scenario_label=str(backend),
                    x_value=None,
                    prefill_cu=prefill_cu,
                    decode_cu=decode_cu,
                    message=f"{type(exc).__name__}: {exc}",
                    error_type="model_load_fail",
                )
            )
            continue

        scenario = _scenario_label(prefill_cu, decode_cu)

        for seconds in seconds_list:
            n = int(seconds * sample_rate)
            speech = base_wave[:n]

            warmup_failed = False
            for warmup_idx in range(warmup):
                try:
                    _ = s2t(speech)
                except Exception as exc:
                    records.append(
                        _error_row(
                            model_alias=model_alias,
                            model_tag=model_tag,
                            scenario_label=scenario,
                            x_value=seconds,
                            prefill_cu=prefill_cu,
                            decode_cu=decode_cu,
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
                    t0 = perf_counter()
                    _ = s2t(speech)
                    total_s = perf_counter() - t0
                    total_ms = total_s * 1000.0
                    throughput = float(seconds / total_s) if total_s > 0 else 0.0
                    rtf = float(total_s / seconds) if seconds > 0 else 0.0
                    peak_rss_mb = float(process.memory_info().rss / (1024.0 * 1024.0))

                    records.append(
                        _ok_row(
                            model_alias=model_alias,
                            model_tag=model_tag,
                            scenario_label=scenario,
                            seconds=seconds,
                            run_index=run_idx,
                            prefill_cu=prefill_cu,
                            decode_cu=decode_cu,
                            total_ms=total_ms,
                            throughput=throughput,
                            rtf=rtf,
                            peak_rss_mb=peak_rss_mb,
                        )
                    )
                except Exception as exc:
                    tb = traceback.format_exc().splitlines()
                    summary = "\n".join(tb[-10:]) if tb else str(exc)
                    records.append(
                        _error_row(
                            model_alias=model_alias,
                            model_tag=model_tag,
                            scenario_label=scenario,
                            x_value=seconds,
                            prefill_cu=prefill_cu,
                            decode_cu=decode_cu,
                            message=summary,
                            error_type=type(exc).__name__,
                        )
                    )

    return {"status": "ok", "records": records}


def dump_computeplan(task_cfg: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    if not bool(task_cfg.get("enabled", False)):
        return _skip("speech_owsm disabled in suite config", "task_disabled")
    return _ok(message="speech_owsm uses_coreml=false; compute plan not applicable")
