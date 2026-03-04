#!/usr/bin/env python3
"""Convert TorchScript wrappers to fixed-shape Core ML ML Programs."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import coremltools as ct
import numpy as np
import torch
import yaml

from lib_paths import coreml_paths, torch_paths

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def assert_shape(array: np.ndarray, expected: Iterable[int], name: str) -> None:
    got = tuple(int(x) for x in array.shape)
    want = tuple(int(x) for x in expected)
    if got != want:
        raise ValueError(f"{name} shape mismatch. expected={want}, got={got}")


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


def best_macos_target() -> Any:
    for name in ("macOS15", "macOS14", "macOS13"):
        if hasattr(ct.target, name):
            return getattr(ct.target, name)
    return None


def _quantize_linear_weights(mlmodel: ct.models.MLModel, nbits: int) -> ct.models.MLModel:
    optimize = getattr(ct, "optimize", None)
    if optimize is None or not hasattr(optimize, "coreml"):
        raise RuntimeError("coremltools optimize.coreml API is unavailable")

    coreml_opt = optimize.coreml
    quant_fn = getattr(coreml_opt, "linear_quantize_weights", None)
    if quant_fn is None:
        raise RuntimeError("linear_quantize_weights is unavailable in coremltools.optimize.coreml")

    # Try function signatures across coremltools versions.
    errors = []
    try:
        return quant_fn(mlmodel, nbits=nbits)
    except Exception as exc:
        errors.append(f"nbits arg failed: {type(exc).__name__}: {exc}")

    try:
        return quant_fn(mlmodel)
    except Exception as exc:
        errors.append(f"no-arg call failed: {type(exc).__name__}: {exc}")

    opt_cfg_cls = getattr(coreml_opt, "OptimizationConfig", None)
    op_cfg_cls = getattr(coreml_opt, "OpLinearQuantizerConfig", None)
    if opt_cfg_cls is not None and op_cfg_cls is not None:
        try:
            op_sig = inspect.signature(op_cfg_cls)
            op_kwargs: Dict[str, Any] = {}
            if "mode" in op_sig.parameters:
                op_kwargs["mode"] = "linear_symmetric"
            if "nbits" in op_sig.parameters:
                op_kwargs["nbits"] = int(nbits)
            if "weight_threshold" in op_sig.parameters:
                op_kwargs["weight_threshold"] = 0
            op_cfg = op_cfg_cls(**op_kwargs)

            cfg_sig = inspect.signature(opt_cfg_cls)
            cfg_kwargs: Dict[str, Any] = {}
            if "global_config" in cfg_sig.parameters:
                cfg_kwargs["global_config"] = op_cfg
            config = opt_cfg_cls(**cfg_kwargs)

            return quant_fn(mlmodel, config=config)
        except Exception as exc:
            errors.append(f"config-based call failed: {type(exc).__name__}: {exc}")

    raise RuntimeError("weight quantization failed: " + " | ".join(errors))


def maybe_quantize(
    prefill_mlmodel: ct.models.MLModel,
    decode_mlmodel: ct.models.MLModel,
    *,
    enabled: bool,
    mode: str,
) -> Tuple[ct.models.MLModel, ct.models.MLModel, Optional[str]]:
    if not enabled:
        return prefill_mlmodel, decode_mlmodel, None

    mode_l = str(mode).lower().strip()
    nbits = 4 if mode_l == "int4" else 8

    prefill_q = _quantize_linear_weights(prefill_mlmodel, nbits=nbits)
    decode_q = _quantize_linear_weights(decode_mlmodel, nbits=nbits)
    return prefill_q, decode_q, f"linear_int{nbits}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--context-len", type=int)
    parser.add_argument("--model-id")
    parser.add_argument("--variant-dir")
    parser.add_argument("--allow-weight-quant", action="store_true")
    parser.add_argument("--weight-quant-mode", default=None)
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml((ROOT / args.config).resolve())
    context_len = resolve_context_len(cfg, args.context_len)
    prefill_len = int(context_len - 1)

    use_legacy_layout = args.variant_dir is None
    model_id_hint = str(args.model_id or cfg["hf_model_id"])
    torch_map = torch_paths(
        model_id=model_id_hint,
        context_len=context_len,
        variant_dir=args.variant_dir,
        legacy=use_legacy_layout,
    )

    meta_path = torch_map["model_meta_json"]
    if not meta_path.exists():
        legacy_meta = ROOT / "artifacts" / "model_meta.json"
        if use_legacy_layout and legacy_meta.exists():
            meta_path = legacy_meta
        else:
            raise FileNotFoundError(
                f"Context-specific model metadata not found: {meta_path}. "
                "Run scripts/01_export_torch.py for this context first."
            )
    meta = load_json(meta_path)
    model_id = resolve_model_id(cfg, args.model_id, meta=meta)

    meta_context = int(meta.get("context_len", context_len))
    if meta_context != context_len:
        raise ValueError(
            f"model_meta context_len={meta_context} does not match requested context_len={context_len}"
        )

    n_layers = int(meta["n_layers"])
    n_heads = int(meta["n_heads"])
    n_kv_heads = int(meta.get("n_kv_heads", n_heads))
    head_dim = int(meta["head_dim"])
    vocab_size = int(meta["vocab_size"])

    prefill_ts_path = torch_map["prefill_pt"]
    decode_ts_path = torch_map["decode_pt"]
    if not prefill_ts_path.exists() or not decode_ts_path.exists():
        raise FileNotFoundError(
            f"TorchScript artifacts not found under {prefill_ts_path.parent}. "
            "Run scripts/01_export_torch.py first."
        )

    prefill_ts = torch.jit.load(str(prefill_ts_path))
    decode_ts = torch.jit.load(str(decode_ts_path))

    prefill_inputs = [
        ct.TensorType(name="input_ids", shape=(1, prefill_len), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, prefill_len), dtype=np.int32),
        ct.TensorType(name="position_ids", shape=(1, prefill_len), dtype=np.int32),
    ]
    prefill_outputs = [
        ct.TensorType(name="logits_last", dtype=np.float16),
        ct.TensorType(name="past_key", dtype=np.float16),
        ct.TensorType(name="past_value", dtype=np.float16),
    ]

    decode_inputs = [
        ct.TensorType(name="input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, prefill_len + 1), dtype=np.int32),
        ct.TensorType(name="position_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType(
            name="past_key",
            shape=(n_layers, n_kv_heads, prefill_len, head_dim),
            dtype=np.float16,
        ),
        ct.TensorType(
            name="past_value",
            shape=(n_layers, n_kv_heads, prefill_len, head_dim),
            dtype=np.float16,
        ),
    ]
    decode_outputs = [
        ct.TensorType(name="logits", dtype=np.float16),
        ct.TensorType(name="present_key", dtype=np.float16),
        ct.TensorType(name="present_value", dtype=np.float16),
    ]

    min_target = best_macos_target()
    if min_target is not None:
        print(f"Using minimum deployment target: {min_target}")
    else:
        print("WARNING: unable to determine explicit macOS deployment target from coremltools.")

    print("Converting prefill wrapper...")
    prefill_mlmodel = ct.convert(
        prefill_ts,
        source="pytorch",
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        inputs=prefill_inputs,
        outputs=prefill_outputs,
        minimum_deployment_target=min_target,
    )

    print("Converting decode wrapper...")
    decode_mlmodel = ct.convert(
        decode_ts,
        source="pytorch",
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        inputs=decode_inputs,
        outputs=decode_outputs,
        minimum_deployment_target=min_target,
    )

    quant_mode = args.weight_quant_mode or cfg.get("weight_quant_mode") or "int4"
    quant_enabled = bool(args.allow_weight_quant or cfg.get("allow_weight_quant", False))

    quant_applied: Optional[str] = None
    if quant_enabled:
        try:
            prefill_mlmodel, decode_mlmodel, quant_applied = maybe_quantize(
                prefill_mlmodel,
                decode_mlmodel,
                enabled=True,
                mode=quant_mode,
            )
            print(f"Applied weight quantization: {quant_applied}")
        except Exception as exc:
            print(
                "WARNING: weight quantization failed; falling back to fp16 conversion. "
                f"reason={type(exc).__name__}: {exc}"
            )
            quant_applied = None

    coreml_map = coreml_paths(
        model_id=model_id,
        context_len=context_len,
        variant_dir=args.variant_dir,
        legacy=False,
    )
    prefill_model_path = coreml_map["prefill_mlpackage"]
    decode_model_path = coreml_map["decode_mlpackage"]
    prefill_model_path.parent.mkdir(parents=True, exist_ok=True)
    decode_model_path.parent.mkdir(parents=True, exist_ok=True)

    prefill_mlmodel.save(str(prefill_model_path))
    decode_mlmodel.save(str(decode_model_path))

    if use_legacy_layout:
        legacy_coreml = coreml_paths(model_id=model_id, context_len=context_len, legacy=True)
        legacy_coreml["prefill_mlpackage"].parent.mkdir(parents=True, exist_ok=True)
        try:
            prefill_mlmodel.save(str(legacy_coreml["prefill_mlpackage"]))
            decode_mlmodel.save(str(legacy_coreml["decode_mlpackage"]))
        except Exception as exc:
            print(
                "WARNING: failed to write legacy Core ML package copies; continuing with "
                f"context-specific packages only. reason={type(exc).__name__}: {exc}"
            )

    print(f"saved: {prefill_model_path}")
    print(f"saved: {decode_model_path}")

    if args.skip_validate:
        if quant_applied is not None:
            print(f"quantization: {quant_applied}")
        print("Core ML conversion validation: skipped")
        return 0

    print("Validating prefill model with CPU_ONLY predict...")
    prefill_cpu = ct.models.MLModel(str(prefill_model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    prefill_input = {
        "input_ids": np.zeros((1, prefill_len), dtype=np.int32),
        "attention_mask": np.ones((1, prefill_len), dtype=np.int32),
        "position_ids": np.arange(prefill_len, dtype=np.int32).reshape(1, prefill_len),
    }
    prefill_out = prefill_cpu.predict(prefill_input)
    prefill_logits_last = output_by_name(prefill_out, ["logits_last", "logits"])
    prefill_past_key = output_by_name(prefill_out, ["past_key"])
    prefill_past_value = output_by_name(prefill_out, ["past_value"])

    assert_shape(prefill_logits_last, (1, vocab_size), "logits_last")
    assert_shape(
        prefill_past_key,
        (n_layers, n_kv_heads, prefill_len, head_dim),
        "past_key",
    )
    assert_shape(
        prefill_past_value,
        (n_layers, n_kv_heads, prefill_len, head_dim),
        "past_value",
    )

    print("Validating decode model with CPU_ONLY predict...")
    decode_cpu = ct.models.MLModel(str(decode_model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    decode_input = {
        "input_id": np.zeros((1, 1), dtype=np.int32),
        "attention_mask": np.ones((1, prefill_len + 1), dtype=np.int32),
        "position_id": np.array([[prefill_len]], dtype=np.int32),
        "past_key": np.zeros((n_layers, n_kv_heads, prefill_len, head_dim), dtype=np.float16),
        "past_value": np.zeros((n_layers, n_kv_heads, prefill_len, head_dim), dtype=np.float16),
    }
    decode_out = decode_cpu.predict(decode_input)
    decode_logits = output_by_name(decode_out, ["logits", "logits_last"])
    decode_present_key = output_by_name(decode_out, ["present_key", "past_key"])
    decode_present_value = output_by_name(decode_out, ["present_value", "past_value"])

    assert_shape(decode_logits, (1, vocab_size), "logits")
    assert_shape(
        decode_present_key,
        (n_layers, n_kv_heads, prefill_len + 1, head_dim),
        "present_key",
    )
    assert_shape(
        decode_present_value,
        (n_layers, n_kv_heads, prefill_len + 1, head_dim),
        "present_value",
    )

    if quant_applied is not None:
        print(f"quantization: {quant_applied}")
    print("Core ML conversion validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
