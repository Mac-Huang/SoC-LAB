#!/usr/bin/env python3
"""Environment checks for Core ML affinity experiments."""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print(f"python: {sys.version.split()[0]}")

    if sys.platform != "darwin" or platform.machine() != "arm64":
        print("ERROR: This suite requires macOS on Apple Silicon (arm64).")
        return 1

    try:
        import coremltools as ct
    except Exception as exc:  # pragma: no cover - runtime dependency path
        print(f"ERROR: coremltools import failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"coremltools: {ct.__version__}")

    try:
        devices = ct.models.MLModel.get_available_compute_devices()
        print("available_compute_devices:")
        for idx, device in enumerate(devices, start=1):
            print(f"  {idx}. {device}")
    except Exception as exc:  # pragma: no cover - API variation by version
        print(f"WARNING: unable to query available compute devices: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
