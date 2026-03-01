#!/usr/bin/env python3
"""Approximate FLOP estimators for decoder-only Transformer prefill/decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _get_meta_int(meta: Dict[str, Any], key: str) -> int:
    value = meta.get(key)
    if value is None:
        raise KeyError(f"Missing required model_meta key: {key}")
    return int(value)


def flops_prefill(prompt_len: int, meta: Dict[str, Any]) -> int:
    """Approximate FLOPs for one prefill pass over `prompt_len` tokens."""
    s = int(prompt_len)
    if s < 0:
        raise ValueError("prompt_len must be non-negative")

    n_layers = _get_meta_int(meta, "n_layers")
    heads = _get_meta_int(meta, "n_heads")
    h = _get_meta_int(meta, "hidden_size")
    d = _get_meta_int(meta, "head_dim")
    i = _get_meta_int(meta, "intermediate_size")

    qkv = 2 * s * h * (3 * h)
    qk_t = 2 * heads * s * s * d
    av = 2 * heads * s * s * d
    out = 2 * s * h * h
    ffn = 2 * s * h * i + 2 * s * i * h

    per_layer = qkv + qk_t + av + out + ffn
    return int(n_layers * per_layer)


def flops_decode_step(kv_len: int, meta: Dict[str, Any]) -> int:
    """Approximate FLOPs for one decode step with KV cache length `kv_len`."""
    s = int(kv_len)
    if s < 0:
        raise ValueError("kv_len must be non-negative")

    n_layers = _get_meta_int(meta, "n_layers")
    heads = _get_meta_int(meta, "n_heads")
    h = _get_meta_int(meta, "hidden_size")
    d = _get_meta_int(meta, "head_dim")
    i = _get_meta_int(meta, "intermediate_size")

    qkv = 2 * 1 * h * (3 * h)
    qk_t = 2 * heads * 1 * s * d
    av = 2 * heads * 1 * s * d
    out = 2 * 1 * h * h
    ffn = 2 * 1 * h * i + 2 * 1 * i * h

    per_layer = qkv + qk_t + av + out + ffn
    return int(n_layers * per_layer)


def effective_tflops(flops: float, latency_ms: float) -> Optional[float]:
    if latency_ms <= 0:
        return None
    return float(flops) / (latency_ms / 1000.0) / 1e12


def load_model_meta(meta_path: Path) -> Dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_sanity_tests() -> None:
    meta = {
        "n_layers": 2,
        "n_heads": 4,
        "hidden_size": 32,
        "head_dim": 8,
        "intermediate_size": 128,
    }

    p16 = flops_prefill(16, meta)
    p32 = flops_prefill(32, meta)
    d8 = flops_decode_step(8, meta)
    d16 = flops_decode_step(16, meta)

    assert p16 >= 0
    assert p32 >= p16
    assert d8 >= 0
    assert d16 >= d8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="artifacts/model_meta.json")
    parser.add_argument("--prefill-len", type=int, default=63)
    parser.add_argument("--kv-len", type=int, default=63)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    if args.run_tests:
        run_sanity_tests()
        print("flops_sanity_tests: OK")
        return 0

    meta_path = Path(args.meta)
    if not meta_path.exists():
        raise FileNotFoundError(f"model_meta not found: {meta_path}")

    meta = load_model_meta(meta_path)
    prefill = flops_prefill(args.prefill_len, meta)
    decode = flops_decode_step(args.kv_len, meta)

    payload = {
        "prefill_len": args.prefill_len,
        "kv_len": args.kv_len,
        "flops_prefill": prefill,
        "flops_decode_step": decode,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
