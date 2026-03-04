#!/usr/bin/env python3
"""Stable Diffusion v1.5 Core ML staged benchmark task."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
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
    return True, ""


def _find_artifact(base_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    def _is_valid_path(path: Path) -> bool:
        if ".cache" in path.parts:
            return False
        if path.suffix == ".mlpackage":
            manifest = path / "Manifest.json"
            return manifest.exists()
        if path.suffix == ".mlmodelc":
            return True
        return False

    for stem in candidates:
        for ext in (".mlpackage", ".mlmodelc"):
            p = base_dir / f"{stem}{ext}"
            if p.exists() and _is_valid_path(p):
                return p

    for stem in candidates:
        matches = [p for p in sorted(base_dir.rglob(f"*{stem}*.mlpackage")) if _is_valid_path(p)]
        if matches:
            return matches[0]
        matches = [p for p in sorted(base_dir.rglob(f"*{stem}*.mlmodelc")) if _is_valid_path(p)]
        if matches:
            return matches[0]
    return None


def _locate_stage_models(base_dir: Path) -> Dict[str, Path]:
    stage_map = {
        "text_encoder": _find_artifact(base_dir, ["text_encoder", "text-encoder", "textencoder"]),
        "unet": _find_artifact(base_dir, ["unet", "u_net"]),
        "vae_decoder": _find_artifact(base_dir, ["vae_decoder", "vae-decoder", "vaedecoder", "vae"]),
    }
    missing = [name for name, path in stage_map.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"Missing SD15 stage artifacts in {base_dir}: {', '.join(missing)}. "
            "Expected text_encoder/unet/vae_decoder (.mlpackage or .mlmodelc)."
        )
    return {k: v for k, v in stage_map.items() if v is not None}


def _try_download_preconverted(base_dir: Path) -> Tuple[bool, str]:
    """Best-effort helper for BYO assets. Returns (ok, message)."""
    if importlib.util.find_spec("huggingface_hub") is None:
        return False, "huggingface_hub is not installed"

    try:
        from huggingface_hub import snapshot_download

        base_dir.mkdir(parents=True, exist_ok=True)

        # Try a commonly used repo containing pre-converted SD15 assets.
        snapshot_download(
            repo_id="apple/coreml-stable-diffusion-v1-5",
            local_dir=str(base_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["**/*text*encoder*", "**/*unet*", "**/*vae*decoder*", "**/*.mlpackage", "**/*.mlmodelc"],
        )
        return True, "downloaded candidate SD15 Core ML assets"
    except Exception as exc:
        return False, f"automatic SD15 asset download failed: {type(exc).__name__}: {exc}"


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


def _make_stage_inputs(
    model: Any,
    *,
    stage: str,
    seed: int,
    step_index: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + step_index)
    spec = model.get_spec()
    inputs: Dict[str, np.ndarray] = {}

    for inp in spec.description.input:
        if not inp.type.HasField("multiArrayType"):
            continue

        shape, np_dtype = _parse_multiarray_input(inp)
        name = inp.name.lower()

        if np.issubdtype(np_dtype, np.integer):
            high = 49408 if "token" in name or "input" in name else 32000
            arr = rng.integers(0, max(2, high), size=shape, dtype=np_dtype)
            if "timestep" in name:
                arr = np.full(shape, step_index, dtype=np_dtype)
        else:
            arr = rng.standard_normal(size=shape).astype(np_dtype, copy=False)
            if "timestep" in name:
                arr = np.full(shape, float(step_index), dtype=np_dtype)
            if stage == "unet" and ("sample" in name or "latent" in name):
                arr *= 0.18215

        inputs[inp.name] = arr

    return inputs


def _load_model(path: Path, cu_name: str):
    cu = CU_MAP.get(cu_name)
    if cu is None:
        raise ValueError(f"Unsupported compute unit for SD15: {cu_name}")
    return ct.models.MLModel(str(path), compute_units=cu)


def _safe_percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _benchmark_once(
    text_encoder_model: Any,
    unet_model: Any,
    vae_model: Any,
    *,
    steps: int,
    seed: int,
    process: psutil.Process,
) -> Dict[str, float]:
    peak_rss = process.memory_info().rss

    text_inputs = _make_stage_inputs(text_encoder_model, stage="text_encoder", seed=seed, step_index=0)
    t0 = perf_counter()
    _ = text_encoder_model.predict(text_inputs)
    textenc_ms = (perf_counter() - t0) * 1000.0
    peak_rss = max(peak_rss, process.memory_info().rss)

    unet_steps: List[float] = []
    for step in range(int(steps)):
        unet_inputs = _make_stage_inputs(unet_model, stage="unet", seed=seed, step_index=step)
        t1 = perf_counter()
        _ = unet_model.predict(unet_inputs)
        unet_steps.append((perf_counter() - t1) * 1000.0)
        peak_rss = max(peak_rss, process.memory_info().rss)

    vae_inputs = _make_stage_inputs(vae_model, stage="vae_decoder", seed=seed, step_index=steps)
    t2 = perf_counter()
    _ = vae_model.predict(vae_inputs)
    vae_ms = (perf_counter() - t2) * 1000.0
    peak_rss = max(peak_rss, process.memory_info().rss)

    unet_total_ms = float(sum(unet_steps))
    total_ms = float(textenc_ms + unet_total_ms + vae_ms)
    unet_total_s = unet_total_ms / 1000.0

    return {
        "textenc_ms": float(textenc_ms),
        "unet_total_ms": float(unet_total_ms),
        "unet_step_ms_mean": float(np.mean(np.asarray(unet_steps, dtype=np.float64))) if unet_steps else 0.0,
        "unet_step_ms_p95": _safe_percentile(unet_steps, 95),
        "vae_ms": float(vae_ms),
        "total_ms": total_ms,
        "steps_per_sec": float(steps / unet_total_s) if unet_total_s > 0 else 0.0,
        "peak_rss_mb": float(peak_rss / (1024.0 * 1024.0)),
    }


def _error_row(
    *,
    model_alias: str,
    scenario_label: Optional[str],
    x_value: Optional[int],
    prefill_cu: Optional[str],
    decode_cu: Optional[str],
    vae_cu: Optional[str],
    message: str,
    error_type: str,
) -> Dict[str, Any]:
    return {
        "timestamp": _now(),
        "task_type": "diffusion_sd15",
        "model_id": "stable-diffusion-v1-5-coreml",
        "model_alias": model_alias,
        "variant_id": None,
        "context_len": None,
        "prefill_len": None,
        "gen_tokens": None,
        "mode": "staged",
        "prefill_compute_units": prefill_cu,
        "decode_compute_units": decode_cu,
        "vae_compute_units": vae_cu,
        "scenario_label": scenario_label,
        "x_label": "steps",
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
    scenario_label: str,
    steps: int,
    run_index: int,
    prefill_cu: str,
    decode_cu: str,
    vae_cu: str,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "timestamp": _now(),
        "task_type": "diffusion_sd15",
        "model_id": "stable-diffusion-v1-5-coreml",
        "model_alias": model_alias,
        "variant_id": None,
        "context_len": None,
        "prefill_len": None,
        "gen_tokens": None,
        "mode": "staged",
        "prefill_compute_units": prefill_cu,
        "decode_compute_units": decode_cu,
        "vae_compute_units": vae_cu,
        "scenario_label": scenario_label,
        "x_label": "steps",
        "x_value": int(steps),
        "primary_latency_ms": float(metrics["total_ms"]),
        "primary_throughput": float(metrics["steps_per_sec"]),
        "uses_coreml": True,
        "textenc_ms": float(metrics["textenc_ms"]),
        "unet_total_ms": float(metrics["unet_total_ms"]),
        "unet_step_ms_mean": float(metrics["unet_step_ms_mean"]),
        "unet_step_ms_p95": float(metrics["unet_step_ms_p95"]),
        "vae_ms": float(metrics["vae_ms"]),
        "prefill_latency_ms": float(metrics["textenc_ms"]),
        "total_decode_latency_ms": float(metrics["unet_total_ms"]),
        "tokens_per_sec": float(metrics["steps_per_sec"]),
        "tpot_ms_mean": float(metrics["unet_step_ms_mean"]),
        "tpot_ms_p95": float(metrics["unet_step_ms_p95"]),
        "ttft_ms": float(metrics["textenc_ms"] + metrics["unet_step_ms_mean"]),
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


def prepare_variant(task_cfg: Dict[str, Any], root_dir: Path, **_: Any) -> Dict[str, Any]:
    enabled = bool(task_cfg.get("enabled", False))
    if not enabled:
        return _skip("diffusion_sd15 disabled in suite config", "task_disabled")

    ok, reason = _deps_available()
    if not ok:
        return _skip(reason, "missing_dependency")

    base_dir = Path(str(task_cfg.get("byop_coreml_dir", "assets/sd15_coreml")))
    if not base_dir.is_absolute():
        base_dir = (root_dir / base_dir).resolve()

    convert_if_missing = bool(task_cfg.get("convert_if_missing", False))

    try:
        stage_paths = _locate_stage_models(base_dir)
        return _ok(assets_dir=str(base_dir), stage_paths={k: str(v) for k, v in stage_paths.items()})
    except Exception as exc:
        if convert_if_missing:
            dl_ok, msg = _try_download_preconverted(base_dir)
            if dl_ok:
                try:
                    stage_paths = _locate_stage_models(base_dir)
                    return _ok(
                        assets_dir=str(base_dir),
                        stage_paths={k: str(v) for k, v in stage_paths.items()},
                        message=msg,
                    )
                except Exception as exc2:
                    return _skip(
                        f"download attempt completed but required artifacts still missing: {type(exc2).__name__}: {exc2}",
                        "missing_asset",
                    )
            return _skip(
                f"SD15 assets missing and auto-download failed: {msg}",
                "missing_asset",
            )

        return _skip(f"{type(exc).__name__}: {exc}", "missing_asset")


def run_bench(
    task_cfg: Dict[str, Any],
    prep: Dict[str, Any],
    *,
    dry_run: bool = False,
    **_: Any,
) -> Dict[str, Any]:
    model_alias = str(task_cfg.get("model_alias", "sd15"))
    if prep.get("status") != "ok":
        return {
            "status": "error",
            "records": [
                _error_row(
                    model_alias=model_alias,
                    scenario_label=None,
                    x_value=None,
                    prefill_cu=None,
                    decode_cu=None,
                    vae_cu=None,
                    message=str(prep.get("error_message") or prep.get("message") or "prepare failed"),
                    error_type=str(prep.get("error_type") or prep.get("failure_type") or "prepare_failed"),
                )
            ],
        }

    sweep = task_cfg.get("sweep", {})
    steps_list = [int(x) for x in sweep.get("steps_list", [10, 20, 30, 50])]
    runs = int(sweep.get("runs", 3))
    warmup = int(sweep.get("warmup", 1))
    seed = int(sweep.get("seed", 1337))

    stage_cfgs = task_cfg.get("compute_units", {}).get("stages", [])
    if not stage_cfgs:
        return {
            "status": "error",
            "records": [
                _error_row(
                    model_alias=model_alias,
                    scenario_label=None,
                    x_value=None,
                    prefill_cu=None,
                    decode_cu=None,
                    vae_cu=None,
                    message="diffusion_sd15 compute_units.stages is empty",
                    error_type="invalid_config",
                )
            ],
        }

    if dry_run:
        return {
            "status": "ok",
            "records": [],
            "message": f"dry-run: diffusion_sd15 scenarios={len(stage_cfgs)} steps={steps_list}",
        }

    stage_paths = {k: Path(v) for k, v in prep.get("stage_paths", {}).items()}
    process = psutil.Process()

    records: List[Dict[str, Any]] = []

    for scenario in stage_cfgs:
        te_cu = str(scenario.get("text_encoder"))
        unet_cu = str(scenario.get("unet"))
        vae_cu = str(scenario.get("vae"))

        if te_cu not in CU_MAP or unet_cu not in CU_MAP or vae_cu not in CU_MAP:
            records.append(
                _error_row(
                    model_alias=model_alias,
                    scenario_label=None,
                    x_value=None,
                    prefill_cu=te_cu,
                    decode_cu=unet_cu,
                    vae_cu=vae_cu,
                    message=f"Unsupported CU in scenario: {scenario}",
                    error_type="invalid_compute_unit",
                )
            )
            continue

        scenario_label = f"TE:{ABBR.get(te_cu, te_cu)}|UN:{ABBR.get(unet_cu, unet_cu)}|VAE:{ABBR.get(vae_cu, vae_cu)}"

        try:
            text_model = _load_model(stage_paths["text_encoder"], te_cu)
            unet_model = _load_model(stage_paths["unet"], unet_cu)
            vae_model = _load_model(stage_paths["vae_decoder"], vae_cu)
        except Exception as exc:
            records.append(
                _error_row(
                    model_alias=model_alias,
                    scenario_label=scenario_label,
                    x_value=None,
                    prefill_cu=te_cu,
                    decode_cu=unet_cu,
                    vae_cu=vae_cu,
                    message=f"{type(exc).__name__}: {exc}",
                    error_type="model_load_fail",
                )
            )
            continue

        for steps in steps_list:
            warmup_failed = False
            for warmup_idx in range(warmup):
                try:
                    _ = _benchmark_once(
                        text_encoder_model=text_model,
                        unet_model=unet_model,
                        vae_model=vae_model,
                        steps=steps,
                        seed=seed + warmup_idx,
                        process=process,
                    )
                except Exception as exc:
                    records.append(
                        _error_row(
                            model_alias=model_alias,
                            scenario_label=scenario_label,
                            x_value=steps,
                            prefill_cu=te_cu,
                            decode_cu=unet_cu,
                            vae_cu=vae_cu,
                            message=f"warmup_{warmup_idx}: {type(exc).__name__}: {exc}",
                            error_type="warmup_fail",
                        )
                    )
                    warmup_failed = True
                    break

            if warmup_failed:
                continue

            for run_index in range(runs):
                try:
                    metrics = _benchmark_once(
                        text_encoder_model=text_model,
                        unet_model=unet_model,
                        vae_model=vae_model,
                        steps=steps,
                        seed=seed + run_index,
                        process=process,
                    )
                    records.append(
                        _ok_row(
                            model_alias=model_alias,
                            scenario_label=scenario_label,
                            steps=steps,
                            run_index=run_index,
                            prefill_cu=te_cu,
                            decode_cu=unet_cu,
                            vae_cu=vae_cu,
                            metrics=metrics,
                        )
                    )
                except Exception as exc:
                    tb = traceback.format_exc().splitlines()
                    summary = "\n".join(tb[-10:]) if tb else str(exc)
                    records.append(
                        _error_row(
                            model_alias=model_alias,
                            scenario_label=scenario_label,
                            x_value=steps,
                            prefill_cu=te_cu,
                            decode_cu=unet_cu,
                            vae_cu=vae_cu,
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
            v = getattr(cost, attr)
            if isinstance(v, (int, float)):
                return float(v)
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

    # Fallback for environments where `xcrun coremlc` is unavailable.
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
        "top_20_ops_by_estimated_cost": sorted(rows, key=lambda r: float(r["estimated_cost"]), reverse=True)[:20],
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
    model_alias = str(task_cfg.get("model_alias", "sd15"))
    if prep.get("status") != "ok":
        return _skip("prepare failed, skipping SD15 compute plan", "prepare_failed")
    if dry_run:
        return _ok(message="dry-run: diffusion_sd15 compute plan skipped")

    stage_paths = {k: Path(v) for k, v in prep.get("stage_paths", {}).items()}
    reports_dir = (root_dir / "reports").resolve()
    compiled_root = (root_dir / "artifacts" / "compiled" / "diffusion_sd15" / model_alias).resolve()

    summary: Dict[str, Any] = {"model_alias": model_alias, "stages": {}}

    try:
        for stage, src in stage_paths.items():
            compiled = _compile_to_mlmodelc(src, compiled_root / f"{stage}.mlmodelc")
            csv_path = reports_dir / f"computeplan_sd15_{stage}.csv"
            stage_summary = _dump_plan(compiled, csv_path)
            summary["stages"][stage] = stage_summary
    except Exception as exc:
        return _skip(f"{type(exc).__name__}: {exc}", "computeplan_fail")

    summary_path = reports_dir / "computeplan_sd15_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return _ok(summary_json=str(summary_path))
