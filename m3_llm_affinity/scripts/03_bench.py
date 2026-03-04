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

from lib_paths import coreml_paths, default_model_alias, slugify_model_id, torch_paths

ROOT = Path(__file__).resolve().parents[1]

CU_MAP = {
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "ALL": ct.ComputeUnit.ALL,
}
ALLOWED_SCENARIOS = {
    ("CPU_AND_NE", "CPU_AND_NE"),
    ("ALL", "ALL"),
    ("CPU_AND_NE", "CPU_AND_GPU"),
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


def resolve_context_len(raw_cfg: Dict[str, Any], override: int | None) -> int:
    if override is not None:
        return int(override)

    context_len = int(raw_cfg["context_len"])
    cfg_prefill = raw_cfg.get("prefill_len")
    if cfg_prefill is not None and int(cfg_prefill) != context_len - 1:
        raise ValueError("prefill_len must equal context_len - 1")
    return context_len


def resolve_model_id(raw_cfg: Dict[str, Any], override: str | None, meta: Dict[str, Any] | None = None) -> str:
    if override:
        return str(override)
    if meta and meta.get("hf_model_id"):
        return str(meta["hf_model_id"])
    return str(raw_cfg["hf_model_id"])


def resolve_model_alias(
    override: str | None,
    raw_cfg: Dict[str, Any],
    model_id: str,
    meta: Dict[str, Any] | None = None,
) -> str:
    if override:
        return str(override)
    if raw_cfg.get("model_alias"):
        return str(raw_cfg["model_alias"])
    if meta and meta.get("model_alias"):
        return str(meta["model_alias"])
    return default_model_alias(model_id)


def resolve_result_path(path_arg: str | None) -> Path:
    if path_arg is not None:
        out_path = Path(path_arg)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / f"{ts}_bench.jsonl"


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
        "position_ids": np.arange(prefill_len, dtype=np.int32).reshape(1, prefill_len),
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

        logits = output_by_name(decode_outputs, ["logits", "logits_last"]).astype(np.float16, copy=False)
        present_key = output_by_name(decode_outputs, ["present_key", "past_key"]).astype(np.float16, copy=False)
        present_value = output_by_name(decode_outputs, ["present_value", "past_value"]).astype(np.float16, copy=False)

        # Sliding-window KV cache approximation keeps decode input shapes fixed.
        past_key = present_key[:, :, -prefill_len:, :]
        past_value = present_value[:, :, -prefill_len:, :]

        peak_rss = max(peak_rss, process.memory_info().rss)

    total_decode_ms = float(sum(decode_step_latencies))
    tokens_per_sec = float(gen_tokens / (total_decode_ms / 1000.0)) if total_decode_ms > 0 else 0.0

    decode_stats = {
        "mean": float(np.mean(np.asarray(decode_step_latencies, dtype=np.float64)))
        if decode_step_latencies
        else 0.0,
        "median": float(np.median(np.asarray(decode_step_latencies, dtype=np.float64)))
        if decode_step_latencies
        else 0.0,
        "p95": p95(decode_step_latencies),
    }

    first_decode_step_ms = float(decode_step_latencies[0]) if decode_step_latencies else 0.0
    ttft_ms = float(prefill_latency_ms + first_decode_step_ms)

    prefill_flops = float(flops_module.flops_prefill(prefill_len, model_meta))
    decode_step_flops = float(flops_module.flops_decode_step(prefill_len, model_meta))
    decode_total_flops = decode_step_flops * float(gen_tokens)

    tflops_prefill = flops_module.effective_tflops(prefill_flops, prefill_latency_ms)
    tflops_decode = flops_module.effective_tflops(decode_total_flops, total_decode_ms)

    return {
        "prefill_latency_ms": float(prefill_latency_ms),
        "decode_step_latency_ms_stats": decode_stats,
        "first_decode_step_ms": first_decode_step_ms,
        "tpot_ms_mean": float(decode_stats["mean"]),
        "tpot_ms_p95": float(decode_stats["p95"]),
        "ttft_ms": ttft_ms,
        "total_decode_latency_ms": total_decode_ms,
        "tokens_per_sec": tokens_per_sec,
        "effective_TFLOPS_prefill": float(tflops_prefill) if tflops_prefill is not None else None,
        "effective_TFLOPS_decode": float(tflops_decode) if tflops_decode is not None else None,
        "peak_rss_mb": float(peak_rss / (1024.0 * 1024.0)),
    }


def scenario_label_for_record(mode: str, prefill_cu: str, decode_cu: str) -> str:
    whole = {
        "CPU_ONLY": "CPU",
        "CPU_AND_GPU": "GPU",
        "CPU_AND_NE": "NE",
        "ALL": "ALL",
    }
    split = {
        ("CPU_AND_NE", "CPU_AND_GPU"): "NE→GPU",
        ("CPU_AND_GPU", "CPU_AND_NE"): "GPU→NE",
    }
    if mode == "whole":
        return whole.get(prefill_cu, prefill_cu)
    return split.get((prefill_cu, decode_cu), f"{prefill_cu}->{decode_cu}")


def scenario_list(
    mode: str,
    config_compute_units: Sequence[str],
    cu: Optional[str],
    prefill_cu: Optional[str],
    decode_cu: Optional[str],
) -> List[Tuple[str, str]]:
    if mode == "whole":
        if cu is not None:
            scenarios = [(cu, cu)]
        else:
            scenarios = [(name, name) for name in config_compute_units]
    else:
        if prefill_cu is None or decode_cu is None:
            raise ValueError("split mode requires --prefill-cu and --decode-cu")
        scenarios = [(prefill_cu, decode_cu)]

    disallowed = [f"{p}->{d}" for p, d in scenarios if (p, d) not in ALLOWED_SCENARIOS]
    if disallowed:
        raise ValueError(
            "Decode mode policy only allows scenarios: "
            "CPU_AND_NE->CPU_AND_NE, ALL->ALL, CPU_AND_NE->CPU_AND_GPU. "
            f"Got disallowed: {', '.join(disallowed)}"
        )
    return scenarios


def error_record(base: Dict[str, Any], process: psutil.Process, err: Dict[str, str], run_index: int | None) -> Dict[str, Any]:
    return {
        **base,
        "run_index": run_index,
        "prefill_latency_ms": None,
        "decode_step_latency_ms_stats": None,
        "first_decode_step_ms": None,
        "tpot_ms_mean": None,
        "tpot_ms_p95": None,
        "ttft_ms": None,
        "total_decode_latency_ms": None,
        "tokens_per_sec": None,
        "effective_TFLOPS_prefill": None,
        "effective_TFLOPS_decode": None,
        "peak_rss_mb": float(process.memory_info().rss / (1024.0 * 1024.0)),
        "primary_latency_ms": None,
        "primary_throughput": None,
        "status": "error",
        "errors": err,
        "error_type": err["type"],
        "error_message": err["message"],
        "traceback_summary": err["traceback_summary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--context-len", type=int)
    parser.add_argument("--model-id")
    parser.add_argument("--model-alias")
    parser.add_argument("--variant-dir")
    parser.add_argument("--mode", choices=["whole", "split"], required=True)
    parser.add_argument("--prefill-cu", choices=tuple(CU_MAP.keys()))
    parser.add_argument("--decode-cu", choices=tuple(CU_MAP.keys()))
    parser.add_argument("--cu", choices=tuple(CU_MAP.keys()))
    parser.add_argument("--runs", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--gen-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results-path")
    args = parser.parse_args()

    cfg = load_yaml((ROOT / args.config).resolve())
    context_len = resolve_context_len(cfg, args.context_len)
    prefill_len = int(context_len - 1)

    model_id_hint = str(args.model_id or cfg.get("hf_model_id"))
    torch_map = torch_paths(
        model_id=model_id_hint,
        context_len=context_len,
        variant_dir=args.variant_dir,
        legacy=args.variant_dir is None,
    )
    meta_path = torch_map["model_meta_json"]
    if not meta_path.exists():
        legacy_meta = ROOT / "artifacts" / "model_meta.json"
        if args.variant_dir is None and legacy_meta.exists():
            meta_path = legacy_meta
        else:
            raise FileNotFoundError(
                f"model_meta not found at {meta_path}. "
                "Run scripts/01_export_torch.py for this context/model first."
            )

    meta = load_json(meta_path)
    model_id = resolve_model_id(cfg, args.model_id, meta=meta)
    model_alias = resolve_model_alias(args.model_alias, cfg, model_id, meta=meta)
    variant_id = f"{slugify_model_id(model_id)}/ctx{context_len}"

    flops_module = load_flops_module()

    gen_tokens = int(args.gen_tokens if args.gen_tokens is not None else cfg.get("gen_tokens", 32))
    runs = int(args.runs if args.runs is not None else cfg.get("runs", 20))
    warmup = int(args.warmup if args.warmup is not None else cfg.get("warmup", 3))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1337))

    compute_units_list = [str(x) for x in cfg.get("compute_units_list", list(CU_MAP.keys()))]
    for cu_name in compute_units_list:
        if cu_name not in CU_MAP:
            raise ValueError(f"Unsupported compute unit in config: {cu_name}")

    scenarios = scenario_list(
        mode=args.mode,
        config_compute_units=compute_units_list,
        cu=args.cu,
        prefill_cu=args.prefill_cu,
        decode_cu=args.decode_cu,
    )

    vocab_size = int(meta["vocab_size"])
    rng = np.random.default_rng(seed)
    prompt_tokens = rng.integers(0, vocab_size, size=(1, prefill_len), dtype=np.int32)

    model_paths = coreml_paths(model_id=model_id, context_len=context_len, legacy=False)
    prefill_path = model_paths["prefill_mlpackage"]
    decode_path = model_paths["decode_mlpackage"]

    # Optional variant-local coreml path override.
    if args.variant_dir:
        local_paths = coreml_paths(model_id=model_id, context_len=context_len, variant_dir=args.variant_dir, legacy=False)
        if local_paths["prefill_mlpackage"].exists() and local_paths["decode_mlpackage"].exists():
            prefill_path = local_paths["prefill_mlpackage"]
            decode_path = local_paths["decode_mlpackage"]

    if not prefill_path.exists() or not decode_path.exists():
        legacy_paths = coreml_paths(model_id=model_id, context_len=context_len, legacy=True)
        if legacy_paths["prefill_mlpackage"].exists() and legacy_paths["decode_mlpackage"].exists():
            prefill_path = legacy_paths["prefill_mlpackage"]
            decode_path = legacy_paths["decode_mlpackage"]
        else:
            raise FileNotFoundError(
                f"Core ML models not found for context_len={context_len}. "
                f"Expected {prefill_path} and {decode_path}. Run conversion first."
            )

    result_path = resolve_result_path(args.results_path)

    process = psutil.Process()

    with result_path.open("a", encoding="utf-8") as out:
        for prefill_cu_name, decode_cu_name in scenarios:
            scenario_label = scenario_label_for_record(args.mode, prefill_cu_name, decode_cu_name)
            base = {
                "timestamp": dt.datetime.now().isoformat(),
                "task_type": "llm_decode",
                "model_id": model_id,
                "model_alias": model_alias,
                "variant_id": variant_id,
                "context_len": context_len,
                "prefill_len": prefill_len,
                "gen_tokens": gen_tokens,
                "mode": args.mode,
                "mode_label": scenario_label,
                "prefill_compute_units": prefill_cu_name,
                "decode_compute_units": decode_cu_name,
                "scenario_label": scenario_label,
                "x_label": "context_len",
                "x_value": int(context_len),
                "uses_coreml": True,
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
                out.write(json.dumps(error_record(base, process, err, run_index=None)) + "\n")
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
                    err["stage"] = f"warmup_{warmup_idx}"
                    out.write(json.dumps(error_record(base, process, err, run_index=None)) + "\n")
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
                        "primary_latency_ms": metrics.get("ttft_ms"),
                        "primary_throughput": metrics.get("tokens_per_sec"),
                        "status": "ok",
                        "errors": None,
                        "error_type": None,
                        "error_message": None,
                        "traceback_summary": None,
                    }
                except Exception as exc:
                    err = parse_error(exc)
                    record = error_record(base, process, err, run_index=run_idx)

                out.write(json.dumps(record) + "\n")
                out.flush()

    print(f"results: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
