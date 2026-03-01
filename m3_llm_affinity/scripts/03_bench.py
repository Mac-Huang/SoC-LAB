#!/usr/bin/env python3
"""Benchmark Core ML prefill/decode affinity under different compute-unit settings."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import coremltools as ct
import numpy as np
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]

CU_MAP = {
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "ALL": ct.ComputeUnit.ALL,
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_flops_module() -> Any:
    flops_path = Path(__file__).with_name("05_flops.py")
    spec = importlib.util.spec_from_file_location("flops_module", str(flops_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/05_flops.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_by_name(outputs: Dict[str, Any], preferred_names: Sequence[str]) -> np.ndarray:
    for name in preferred_names:
        if name in outputs:
            return np.asarray(outputs[name])

    for name, value in outputs.items():
        lower = name.lower()
        for preferred in preferred_names:
            if preferred.lower() in lower:
                return np.asarray(value)

    if len(outputs) == 1:
        return np.asarray(next(iter(outputs.values())))

    raise KeyError(f"Could not find output among names={preferred_names}. got={list(outputs.keys())}")


def parse_error(exc: BaseException) -> Dict[str, str]:
    tb = traceback.format_exc().strip().splitlines()
    summary = "\n".join(tb[-10:]) if tb else ""
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_summary": summary,
    }


def p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def run_single_benchmark(
    prefill_model: ct.models.MLModel,
    decode_model: ct.models.MLModel,
    prompt_tokens: np.ndarray,
    prefill_len: int,
    gen_tokens: int,
    model_meta: Dict[str, Any],
    flops_module: Any,
    process: psutil.Process,
) -> Dict[str, Any]:
    peak_rss = process.memory_info().rss

    prefill_inputs = {
        "input_ids": prompt_tokens.astype(np.int32, copy=False),
        "attention_mask": np.ones((1, prefill_len), dtype=np.int32),
    }

    t0 = perf_counter()
    prefill_outputs = prefill_model.predict(prefill_inputs)
    prefill_latency_ms = (perf_counter() - t0) * 1000.0

    peak_rss = max(peak_rss, process.memory_info().rss)

    logits = output_by_name(prefill_outputs, ["logits_last", "logits"]).astype(np.float16, copy=False)
    past_key = output_by_name(prefill_outputs, ["past_key"]).astype(np.float16, copy=False)
    past_value = output_by_name(prefill_outputs, ["past_value"]).astype(np.float16, copy=False)

    decode_step_latencies: List[float] = []

    for step in range(gen_tokens):
        next_input_id = np.argmax(logits, axis=-1).astype(np.int32).reshape(1, 1)
        decode_inputs = {
            "input_id": next_input_id,
            "attention_mask": np.ones((1, prefill_len + 1), dtype=np.int32),
            "position_id": np.array([[prefill_len + step]], dtype=np.int32),
            "past_key": past_key,
            "past_value": past_value,
        }

        t1 = perf_counter()
        decode_outputs = decode_model.predict(decode_inputs)
        step_ms = (perf_counter() - t1) * 1000.0
        decode_step_latencies.append(step_ms)

        logits = output_by_name(decode_outputs, ["logits"]).astype(np.float16, copy=False)
        present_key = output_by_name(decode_outputs, ["present_key", "past_key"]).astype(np.float16, copy=False)
        present_value = output_by_name(decode_outputs, ["present_value", "past_value"]).astype(np.float16, copy=False)

        # Sliding-window KV cache approximation keeps decode input shapes fixed.
        past_key = present_key[:, :, -prefill_len:, :]
        past_value = present_value[:, :, -prefill_len:, :]

        peak_rss = max(peak_rss, process.memory_info().rss)

    total_decode_ms = float(sum(decode_step_latencies))
    tokens_per_sec = float(gen_tokens / (total_decode_ms / 1000.0)) if total_decode_ms > 0 else 0.0

    decode_stats = {
        "mean": float(np.mean(np.asarray(decode_step_latencies, dtype=np.float64))),
        "median": float(np.median(np.asarray(decode_step_latencies, dtype=np.float64))),
        "p95": p95(decode_step_latencies),
    }

    prefill_flops = float(flops_module.flops_prefill(prefill_len, model_meta))
    decode_step_flops = float(flops_module.flops_decode_step(prefill_len, model_meta))
    decode_total_flops = decode_step_flops * float(gen_tokens)

    tflops_prefill = flops_module.effective_tflops(prefill_flops, prefill_latency_ms)
    tflops_decode = flops_module.effective_tflops(decode_total_flops, total_decode_ms)

    return {
        "prefill_latency_ms": float(prefill_latency_ms),
        "decode_step_latency_ms_stats": decode_stats,
        "total_decode_latency_ms": total_decode_ms,
        "tokens_per_sec": tokens_per_sec,
        "effective_TFLOPS_prefill": float(tflops_prefill) if tflops_prefill is not None else None,
        "effective_TFLOPS_decode": float(tflops_decode) if tflops_decode is not None else None,
        "peak_rss_mb": float(peak_rss / (1024.0 * 1024.0)),
    }


def scenario_list(
    mode: str,
    config_compute_units: Sequence[str],
    cu: Optional[str],
    prefill_cu: Optional[str],
    decode_cu: Optional[str],
) -> List[Tuple[str, str]]:
    if mode == "whole":
        if cu is not None:
            return [(cu, cu)]
        return [(name, name) for name in config_compute_units]

    if prefill_cu is None or decode_cu is None:
        raise ValueError("split mode requires --prefill-cu and --decode-cu")
    return [(prefill_cu, decode_cu)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["whole", "split"], required=True)
    parser.add_argument("--prefill-cu", choices=tuple(CU_MAP.keys()))
    parser.add_argument("--decode-cu", choices=tuple(CU_MAP.keys()))
    parser.add_argument("--cu", choices=tuple(CU_MAP.keys()))
    parser.add_argument("--runs", type=int)
    parser.add_argument("--warmup", type=int)
    args = parser.parse_args()

    cfg = load_yaml((ROOT / args.config).resolve())
    meta = load_json(ROOT / "artifacts" / "model_meta.json")
    flops_module = load_flops_module()

    model_id = str(cfg["hf_model_id"])
    context_len = int(cfg["context_len"])
    prefill_len = int(cfg["prefill_len"])
    gen_tokens = int(cfg["gen_tokens"])
    runs = int(args.runs if args.runs is not None else cfg["runs"])
    warmup = int(args.warmup if args.warmup is not None else cfg["warmup"])
    seed = int(cfg["seed"])

    if prefill_len != context_len - 1:
        raise ValueError("prefill_len must equal context_len - 1")

    for cu_name in cfg["compute_units_list"]:
        if cu_name not in CU_MAP:
            raise ValueError(f"Unsupported compute unit in config: {cu_name}")

    scenarios = scenario_list(
        mode=args.mode,
        config_compute_units=cfg["compute_units_list"],
        cu=args.cu,
        prefill_cu=args.prefill_cu,
        decode_cu=args.decode_cu,
    )

    vocab_size = int(meta["vocab_size"])
    rng = np.random.default_rng(seed)
    prompt_tokens = rng.integers(0, vocab_size, size=(1, prefill_len), dtype=np.int32)

    prefill_path = ROOT / "models" / "prefill.mlpackage"
    decode_path = ROOT / "models" / "decode.mlpackage"
    if not prefill_path.exists() or not decode_path.exists():
        raise FileNotFoundError("Core ML models not found. Run make convert first.")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = results_dir / f"{ts}_bench.jsonl"

    process = psutil.Process()

    with result_path.open("a", encoding="utf-8") as out:
        for prefill_cu_name, decode_cu_name in scenarios:
            base = {
                "timestamp": dt.datetime.now().isoformat(),
                "model_id": model_id,
                "context_len": context_len,
                "prefill_len": prefill_len,
                "gen_tokens": gen_tokens,
                "mode": args.mode,
                "prefill_compute_units": prefill_cu_name,
                "decode_compute_units": decode_cu_name,
            }

            try:
                prefill_model = ct.models.MLModel(
                    str(prefill_path),
                    compute_units=CU_MAP[prefill_cu_name],
                )
                decode_model = ct.models.MLModel(
                    str(decode_path),
                    compute_units=CU_MAP[decode_cu_name],
                )
            except Exception as exc:
                err = parse_error(exc)
                record = {
                    **base,
                    "run_index": None,
                    "prefill_latency_ms": None,
                    "decode_step_latency_ms_stats": None,
                    "total_decode_latency_ms": None,
                    "tokens_per_sec": None,
                    "effective_TFLOPS_prefill": None,
                    "effective_TFLOPS_decode": None,
                    "peak_rss_mb": float(process.memory_info().rss / (1024.0 * 1024.0)),
                    "status": "error",
                    "errors": err,
                    "error_type": err["type"],
                    "error_message": err["message"],
                    "traceback_summary": err["traceback_summary"],
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(
                    "scenario failed during model load "
                    f"prefill={prefill_cu_name} decode={decode_cu_name}: {err['type']}"
                )
                continue

            warmup_failed = False
            for warmup_idx in range(warmup):
                try:
                    _ = run_single_benchmark(
                        prefill_model=prefill_model,
                        decode_model=decode_model,
                        prompt_tokens=prompt_tokens,
                        prefill_len=prefill_len,
                        gen_tokens=gen_tokens,
                        model_meta=meta,
                        flops_module=flops_module,
                        process=process,
                    )
                except Exception as exc:
                    err = parse_error(exc)
                    record = {
                        **base,
                        "run_index": None,
                        "prefill_latency_ms": None,
                        "decode_step_latency_ms_stats": None,
                        "total_decode_latency_ms": None,
                        "tokens_per_sec": None,
                        "effective_TFLOPS_prefill": None,
                        "effective_TFLOPS_decode": None,
                        "peak_rss_mb": float(process.memory_info().rss / (1024.0 * 1024.0)),
                        "status": "error",
                        "errors": {
                            "stage": f"warmup_{warmup_idx}",
                            **err,
                        },
                        "error_type": err["type"],
                        "error_message": err["message"],
                        "traceback_summary": err["traceback_summary"],
                    }
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    print(
                        "scenario failed during warmup "
                        f"prefill={prefill_cu_name} decode={decode_cu_name}: {err['type']}"
                    )
                    warmup_failed = True
                    break

            if warmup_failed:
                continue

            for run_idx in range(runs):
                try:
                    metrics = run_single_benchmark(
                        prefill_model=prefill_model,
                        decode_model=decode_model,
                        prompt_tokens=prompt_tokens,
                        prefill_len=prefill_len,
                        gen_tokens=gen_tokens,
                        model_meta=meta,
                        flops_module=flops_module,
                        process=process,
                    )
                    record = {
                        **base,
                        "run_index": run_idx,
                        **metrics,
                        "status": "ok",
                        "errors": None,
                        "error_type": None,
                        "error_message": None,
                        "traceback_summary": None,
                    }
                except Exception as exc:
                    err = parse_error(exc)
                    record = {
                        **base,
                        "run_index": run_idx,
                        "prefill_latency_ms": None,
                        "decode_step_latency_ms_stats": None,
                        "total_decode_latency_ms": None,
                        "tokens_per_sec": None,
                        "effective_TFLOPS_prefill": None,
                        "effective_TFLOPS_decode": None,
                        "peak_rss_mb": float(process.memory_info().rss / (1024.0 * 1024.0)),
                        "status": "error",
                        "errors": err,
                        "error_type": err["type"],
                        "error_message": err["message"],
                        "traceback_summary": err["traceback_summary"],
                    }

                out.write(json.dumps(record) + "\n")
                out.flush()

    print(f"results: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
