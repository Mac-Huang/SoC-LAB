#!/usr/bin/env python3
"""Dump MLComputePlan operation-level device affinity and estimated costs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import coremltools as ct
import yaml

from lib_paths import compiled_paths, coreml_paths, default_model_alias, slugify_model_id

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_context_len(raw_cfg: Dict[str, Any], override: int | None) -> int:
    if override is not None:
        return int(override)

    context_len = int(raw_cfg["context_len"])
    cfg_prefill = raw_cfg.get("prefill_len")
    if cfg_prefill is not None and int(cfg_prefill) != context_len - 1:
        raise ValueError("prefill_len must equal context_len - 1")
    return context_len


def _iter_container(container: Any) -> Iterable[Any]:
    if container is None:
        return []
    if isinstance(container, dict):
        return [container[key] for key in sorted(container.keys())]
    if isinstance(container, (list, tuple)):
        return list(container)
    return [container]


def _iter_operations_from_plan(plan: Any) -> List[Any]:
    structure = plan.model_structure

    function_container = None
    if hasattr(structure, "functions"):
        function_container = getattr(structure, "functions")
    elif hasattr(structure, "program") and hasattr(structure.program, "functions"):
        function_container = getattr(structure.program, "functions")

    operations: List[Any] = []
    for function in _iter_container(function_container):
        blocks = None
        if hasattr(function, "block_specializations"):
            blocks = getattr(function, "block_specializations")
        elif hasattr(function, "blocks"):
            blocks = getattr(function, "blocks")
        elif hasattr(function, "block"):
            blocks = getattr(function, "block")

        for block in _iter_container(blocks):
            op_list = None
            if hasattr(block, "operations"):
                op_list = getattr(block, "operations")
            elif hasattr(block, "ops"):
                op_list = getattr(block, "ops")

            for operation in _iter_container(op_list):
                operations.append(operation)

    return operations


def _device_name(device: Any) -> str:
    if device is None:
        return ""
    if hasattr(device, "name"):
        return str(getattr(device, "name"))
    return str(device)


def _stringify_devices(devices: Any) -> str:
    if devices is None:
        return ""
    if isinstance(devices, (list, tuple, set)):
        names = [_device_name(dev) for dev in devices]
        names = sorted(name for name in names if name)
        return "|".join(names)
    return _device_name(devices)


def _extract_usage_fields(usage: Any) -> Tuple[str, str]:
    if usage is None:
        return "", ""

    preferred = None
    supported = None

    preferred_candidates = [
        "preferred_compute_devices",
        "preferred_devices",
        "preferred_compute_device",
        "preferred_device",
    ]
    supported_candidates = [
        "supported_compute_devices",
        "supported_devices",
        "supported_compute_device",
        "supported_device",
    ]

    for key in preferred_candidates:
        if hasattr(usage, key):
            preferred = getattr(usage, key)
            break
        if isinstance(usage, dict) and key in usage:
            preferred = usage[key]
            break

    for key in supported_candidates:
        if hasattr(usage, key):
            supported = getattr(usage, key)
            break
        if isinstance(usage, dict) and key in usage:
            supported = usage[key]
            break

    if preferred is None and supported is None:
        text = str(usage)
        return text, ""

    return _stringify_devices(preferred), _stringify_devices(supported)


def _extract_numeric_cost(cost_obj: Any) -> float:
    if cost_obj is None:
        return 0.0

    if isinstance(cost_obj, (int, float)):
        return float(cost_obj)

    for attr in ("estimated_cost", "cost", "value", "weight"):
        if hasattr(cost_obj, attr):
            candidate = getattr(cost_obj, attr)
            if isinstance(candidate, (int, float)):
                return float(candidate)

    try:
        return float(str(cost_obj))
    except Exception:
        return 0.0


def compile_mlpackage(mlpackage: Path, compiled_target: Path) -> Path:
    compiled_target.parent.mkdir(parents=True, exist_ok=True)
    if compiled_target.exists():
        shutil.rmtree(compiled_target)

    cmd = ["xcrun", "coremlc", "compile", str(mlpackage), str(compiled_target.parent)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and compiled_target.exists():
        return compiled_target

    coremlc_error = (
        f"coremlc compile failed for {mlpackage.name}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    print(f"WARNING: {coremlc_error}")
    print("WARNING: falling back to coremltools.utils.compile_model(...)")

    try:
        ct.utils.compile_model(str(mlpackage), destination_path=str(compiled_target))
    except Exception as fallback_exc:
        raise RuntimeError(
            f"{coremlc_error}\nFallback compile_model failed: {type(fallback_exc).__name__}: {fallback_exc}"
        ) from fallback_exc

    if compiled_target.exists():
        return compiled_target

    candidates = sorted(compiled_target.parent.glob("*.mlmodelc"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Compiled .mlmodelc not found for {mlpackage}")
    return candidates[0]


def dump_plan_for_model(compiled_path: Path, csv_path: Path) -> Dict[str, Any]:
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(str(compiled_path))
    operations = _iter_operations_from_plan(plan)

    rows: List[Dict[str, Any]] = []
    for idx, operation in enumerate(operations):
        op_type = ""
        for attr in ("operator_name", "op_type", "type", "kind"):
            if hasattr(operation, attr):
                op_type = str(getattr(operation, attr))
                break
        if not op_type:
            op_type = type(operation).__name__

        op_name = ""
        if hasattr(operation, "name"):
            op_name = str(getattr(operation, "name"))

        usage = None
        try:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        except Exception:
            usage = None

        preferred_devices, supported_devices = _extract_usage_fields(usage)

        cost_obj = None
        try:
            cost_obj = plan.get_estimated_cost_for_mlprogram_operation(operation)
        except Exception:
            cost_obj = None
        estimated_cost = _extract_numeric_cost(cost_obj)

        rows.append(
            {
                "op_index": idx,
                "op_type": op_type,
                "op_name": op_name,
                "preferred_devices": preferred_devices,
                "supported_devices": supported_devices,
                "estimated_cost": estimated_cost,
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
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

    counts = {"NE": 0, "GPU": 0, "CPU": 0}
    for row in rows:
        pref = str(row["preferred_devices"]).upper()
        if "NE" in pref or "ANE" in pref or "NEURAL" in pref:
            counts["NE"] += 1
        if "GPU" in pref:
            counts["GPU"] += 1
        if "CPU" in pref:
            counts["CPU"] += 1

    top20 = sorted(rows, key=lambda r: float(r["estimated_cost"]), reverse=True)[:20]
    top20 = [
        {
            "op_index": int(item["op_index"]),
            "op_type": item["op_type"],
            "op_name": item["op_name"],
            "estimated_cost": float(item["estimated_cost"]),
        }
        for item in top20
    ]

    return {
        "num_operations": len(rows),
        "preferred_device_counts": counts,
        "top_20_ops_by_estimated_cost": top20,
        "csv": str(csv_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--context-len", type=int)
    parser.add_argument("--model-id")
    parser.add_argument("--model-alias")
    parser.add_argument("--variant-dir")
    args = parser.parse_args()

    cfg = load_yaml((ROOT / args.config).resolve())
    context_len = resolve_context_len(cfg, args.context_len)
    model_id = str(args.model_id or cfg["hf_model_id"])
    model_alias = str(args.model_alias or default_model_alias(model_id))
    model_slug = slugify_model_id(model_id)

    package_paths = coreml_paths(model_id=model_id, context_len=context_len, legacy=False)
    prefill_mlpackage = package_paths["prefill_mlpackage"]
    decode_mlpackage = package_paths["decode_mlpackage"]

    if args.variant_dir:
        local_package_paths = coreml_paths(model_id=model_id, context_len=context_len, variant_dir=args.variant_dir, legacy=False)
        if local_package_paths["prefill_mlpackage"].exists() and local_package_paths["decode_mlpackage"].exists():
            prefill_mlpackage = local_package_paths["prefill_mlpackage"]
            decode_mlpackage = local_package_paths["decode_mlpackage"]

    if not prefill_mlpackage.exists() or not decode_mlpackage.exists():
        legacy_packages = coreml_paths(model_id=model_id, context_len=context_len, legacy=True)
        if legacy_packages["prefill_mlpackage"].exists() and legacy_packages["decode_mlpackage"].exists():
            prefill_mlpackage = legacy_packages["prefill_mlpackage"]
            decode_mlpackage = legacy_packages["decode_mlpackage"]
        else:
            raise FileNotFoundError(
                f"Core ML packages not found. Expected {prefill_mlpackage} and {decode_mlpackage}. "
                "Run conversion first."
            )

    compiled_map = compiled_paths(
        model_id=model_id,
        context_len=context_len,
        variant_dir=args.variant_dir,
        legacy=args.variant_dir is None,
    )

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Compiling prefill mlpackage...")
    prefill_compiled = compile_mlpackage(prefill_mlpackage, compiled_map["prefill_mlmodelc"])
    print("Compiling decode mlpackage...")
    decode_compiled = compile_mlpackage(decode_mlpackage, compiled_map["decode_mlmodelc"])

    prefill_csv = reports_dir / f"computeplan_{model_alias}_ctx{context_len}_prefill.csv"
    decode_csv = reports_dir / f"computeplan_{model_alias}_ctx{context_len}_decode.csv"

    print("Dumping prefill compute plan...")
    prefill_summary = dump_plan_for_model(prefill_compiled, prefill_csv)
    print("Dumping decode compute plan...")
    decode_summary = dump_plan_for_model(decode_compiled, decode_csv)

    summary = {
        "model_id": model_id,
        "model_slug": model_slug,
        "model_alias": model_alias,
        "context_len": context_len,
        "prefill": prefill_summary,
        "decode": decode_summary,
    }

    summary_path = reports_dir / f"computeplan_{model_alias}_ctx{context_len}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    # Backward-compatible report names for tooling that expects fixed paths.
    legacy_prefill_csv = reports_dir / "computeplan_prefill.csv"
    legacy_decode_csv = reports_dir / "computeplan_decode.csv"
    legacy_summary_path = reports_dir / "computeplan_summary.json"
    shutil.copyfile(prefill_csv, legacy_prefill_csv)
    shutil.copyfile(decode_csv, legacy_decode_csv)
    shutil.copyfile(summary_path, legacy_summary_path)

    print(f"saved: {prefill_csv}")
    print(f"saved: {decode_csv}")
    print(f"saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
