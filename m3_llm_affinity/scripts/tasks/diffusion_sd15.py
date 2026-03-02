#!/usr/bin/env python3
"""Optional diffusion task integration points for suite orchestration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict


def _skip(reason: str, failure_type: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "failure_type": failure_type,
        "error_type": failure_type,
        "error_message": reason,
        "errors": {"message": reason},
    }


def _deps_available() -> bool:
    return (
        importlib.util.find_spec("coremltools") is not None
        and importlib.util.find_spec("diffusers") is not None
        and importlib.util.find_spec("torch") is not None
    )


def prepare_variant(task_cfg: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    if not bool(task_cfg.get("enabled", False)):
        return _skip("diffusion_sd15 disabled in suite config", "task_disabled")

    if not _deps_available():
        return _skip(
            "diffusion_sd15 requires diffusers + torch + coremltools; dependencies missing",
            "missing_dependency",
        )

    preconverted_dir = task_cfg.get("preconverted_coreml_dir")
    if preconverted_dir:
        path = Path(str(preconverted_dir)).expanduser().resolve()
        if not path.exists():
            return _skip(f"preconverted_coreml_dir not found: {path}", "missing_asset")

    return {
        "status": "ok",
        "message": "diffusion_sd15 preparation hook ready (benchmark implementation optional).",
    }


def run_bench(task_cfg: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    if not bool(task_cfg.get("enabled", False)):
        return _skip("diffusion_sd15 disabled in suite config", "task_disabled")

    return _skip(
        "diffusion_sd15 bench is not implemented in this suite revision; provide preconverted assets to extend.",
        "not_implemented",
    )


def dump_computeplan(task_cfg: Dict[str, Any], **_: Any) -> Dict[str, Any]:
    if not bool(task_cfg.get("enabled", False)):
        return _skip("diffusion_sd15 disabled in suite config", "task_disabled")

    return _skip(
        "diffusion_sd15 compute plan dump is not implemented in this suite revision.",
        "not_implemented",
    )
