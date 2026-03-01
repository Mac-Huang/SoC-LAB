#!/usr/bin/env python3
"""Convert TorchScript wrappers to fixed-shape Core ML ML Programs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import coremltools as ct
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_yaml((ROOT / args.config).resolve())
    meta = load_json(ROOT / "artifacts" / "model_meta.json")

    prefill_len = int(cfg["prefill_len"])
    if prefill_len != int(cfg["context_len"]) - 1:
        raise ValueError("prefill_len must equal context_len - 1")

    n_layers = int(meta["n_layers"])
    n_heads = int(meta["n_heads"])
    head_dim = int(meta["head_dim"])
    vocab_size = int(meta["vocab_size"])

    prefill_ts_path = ROOT / "artifacts" / "torch" / "prefill.pt"
    decode_ts_path = ROOT / "artifacts" / "torch" / "decode.pt"
    if not prefill_ts_path.exists() or not decode_ts_path.exists():
        raise FileNotFoundError("TorchScript artifacts not found. Run scripts/01_export_torch.py first.")

    prefill_ts = torch.jit.load(str(prefill_ts_path))
    decode_ts = torch.jit.load(str(decode_ts_path))

    prefill_inputs = [
        ct.TensorType(name="input_ids", shape=(1, prefill_len), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, prefill_len), dtype=np.int32),
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
            shape=(n_layers, n_heads, prefill_len, head_dim),
            dtype=np.float16,
        ),
        ct.TensorType(
            name="past_value",
            shape=(n_layers, n_heads, prefill_len, head_dim),
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

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    prefill_model_path = models_dir / "prefill.mlpackage"
    decode_model_path = models_dir / "decode.mlpackage"

    prefill_mlmodel.save(str(prefill_model_path))
    decode_mlmodel.save(str(decode_model_path))

    print(f"saved: {prefill_model_path}")
    print(f"saved: {decode_model_path}")

    print("Validating prefill model with CPU_ONLY predict...")
    prefill_cpu = ct.models.MLModel(str(prefill_model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    prefill_input = {
        "input_ids": np.zeros((1, prefill_len), dtype=np.int32),
        "attention_mask": np.ones((1, prefill_len), dtype=np.int32),
    }
    prefill_out = prefill_cpu.predict(prefill_input)
    prefill_logits_last = output_by_name(prefill_out, ["logits_last", "logits"])
    prefill_past_key = output_by_name(prefill_out, ["past_key"])
    prefill_past_value = output_by_name(prefill_out, ["past_value"])

    assert_shape(prefill_logits_last, (1, vocab_size), "logits_last")
    assert_shape(
        prefill_past_key,
        (n_layers, n_heads, prefill_len, head_dim),
        "past_key",
    )
    assert_shape(
        prefill_past_value,
        (n_layers, n_heads, prefill_len, head_dim),
        "past_value",
    )

    print("Validating decode model with CPU_ONLY predict...")
    decode_cpu = ct.models.MLModel(str(decode_model_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    decode_input = {
        "input_id": np.zeros((1, 1), dtype=np.int32),
        "attention_mask": np.ones((1, prefill_len + 1), dtype=np.int32),
        "position_id": np.array([[prefill_len]], dtype=np.int32),
        "past_key": np.zeros((n_layers, n_heads, prefill_len, head_dim), dtype=np.float16),
        "past_value": np.zeros((n_layers, n_heads, prefill_len, head_dim), dtype=np.float16),
    }
    decode_out = decode_cpu.predict(decode_input)
    decode_logits = output_by_name(decode_out, ["logits", "logits_last"])
    decode_present_key = output_by_name(decode_out, ["present_key", "past_key"])
    decode_present_value = output_by_name(decode_out, ["present_value", "past_value"])

    assert_shape(decode_logits, (1, vocab_size), "logits")
    assert_shape(
        decode_present_key,
        (n_layers, n_heads, prefill_len + 1, head_dim),
        "present_key",
    )
    assert_shape(
        decode_present_value,
        (n_layers, n_heads, prefill_len + 1, head_dim),
        "present_value",
    )

    print("Core ML conversion validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
