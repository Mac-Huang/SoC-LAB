#!/usr/bin/env python3
"""Shared path helpers for model/context variant artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]

_MODEL_SLUG_OVERRIDES = {
    "qwen/qwen2.5-7b-instruct": "qwen25_7b",
    "qwen_qwen2.5-7b-instruct": "qwen25_7b",
}


def _normalized_model_key(model_id: str) -> str:
    return str(model_id).strip().lower().replace("\\", "/")


def slugify_model_id(model_id: str) -> str:
    override = _MODEL_SLUG_OVERRIDES.get(_normalized_model_key(model_id))
    if override:
        return override
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_id)).strip("_")


def default_model_alias(model_id: str) -> str:
    override = _MODEL_SLUG_OVERRIDES.get(_normalized_model_key(model_id))
    if override:
        return override
    tail = str(model_id).rstrip("/").split("/")[-1]
    alias = re.sub(r"[^A-Za-z0-9._-]+", "_", tail).strip("_").lower()
    return alias or "model"


def llm_variant_dir(model_id: str, context_len: int) -> Path:
    model_slug = slugify_model_id(model_id)
    return ROOT / "artifacts" / "variants" / model_slug / f"ctx{int(context_len)}"


def _resolve_variant_dir(
    model_id: str,
    context_len: int,
    variant_dir: Optional[str] = None,
) -> Path:
    if variant_dir:
        given = Path(variant_dir)
        if not given.is_absolute():
            given = (ROOT / given).resolve()
        return given
    return llm_variant_dir(model_id, context_len)


def torch_paths(
    model_id: str,
    context_len: int,
    variant_dir: Optional[str] = None,
    legacy: bool = False,
) -> Dict[str, Path]:
    context_len = int(context_len)
    if legacy:
        ctx_dir = ROOT / "artifacts" / "torch" / f"ctx{context_len}"
        return {
            "prefill_pt": ctx_dir / "prefill.pt",
            "decode_pt": ctx_dir / "decode.pt",
            "model_meta_json": ctx_dir / "model_meta.json",
        }

    base = _resolve_variant_dir(model_id=model_id, context_len=context_len, variant_dir=variant_dir)
    return {
        "prefill_pt": base / "torch" / "prefill.pt",
        "decode_pt": base / "torch" / "decode.pt",
        "model_meta_json": base / "model_meta.json",
    }


def coreml_paths(
    model_id: str,
    context_len: int,
    variant_dir: Optional[str] = None,
    legacy: bool = False,
) -> Dict[str, Path]:
    context_len = int(context_len)
    if legacy:
        return {
            "prefill_mlpackage": ROOT / "models" / "prefill.mlpackage",
            "decode_mlpackage": ROOT / "models" / "decode.mlpackage",
        }

    if variant_dir:
        base = _resolve_variant_dir(model_id=model_id, context_len=context_len, variant_dir=variant_dir) / "coreml"
    else:
        base = ROOT / "models" / slugify_model_id(model_id) / f"ctx{context_len}"
    return {
        "prefill_mlpackage": base / "prefill.mlpackage",
        "decode_mlpackage": base / "decode.mlpackage",
    }


def compiled_paths(
    model_id: str,
    context_len: int,
    variant_dir: Optional[str] = None,
    legacy: bool = False,
) -> Dict[str, Path]:
    context_len = int(context_len)
    if legacy:
        base = ROOT / "artifacts" / "compiled"
    elif variant_dir:
        base = _resolve_variant_dir(model_id=model_id, context_len=context_len, variant_dir=variant_dir) / "compiled"
    else:
        base = ROOT / "artifacts" / "compiled" / slugify_model_id(model_id) / f"ctx{context_len}"

    return {
        "prefill_mlmodelc": base / "prefill.mlmodelc",
        "decode_mlmodelc": base / "decode.mlmodelc",
    }


def results_prefix(model_alias: str, context_len: int) -> str:
    alias = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_alias)).strip("_").lower()
    if not alias:
        alias = "model"
    return f"{alias}_ctx{int(context_len)}"
