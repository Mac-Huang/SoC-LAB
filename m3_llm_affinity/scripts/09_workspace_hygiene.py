#!/usr/bin/env python3
"""Workspace hygiene utility for cache cleanup without touching benchmark outputs."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()


@dataclass
class Target:
    path: Path
    kind: str  # "dir" or "file"
    scope: str  # "local" or "global-hf"


def _iter_local_targets(root: Path) -> Iterable[Target]:
    seen = set()

    def _preserved(path: Path) -> bool:
        try:
            rel = path.resolve().relative_to(root.resolve())
        except Exception:
            return False
        return bool(rel.parts and rel.parts[0] in {"results", "reports"})

    for p in root.glob("assets/**/.cache"):
        rp = p.resolve()
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        yield Target(path=rp, kind="dir", scope="local")

    for rel in ("artifacts/torch", "artifacts/compiled"):
        rp = (root / rel).resolve()
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        yield Target(path=rp, kind="dir", scope="local")

    for p in root.rglob("__pycache__"):
        rp = p.resolve()
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        yield Target(path=rp, kind="dir", scope="local")

    for p in root.rglob(".DS_Store"):
        if _preserved(p):
            continue
        rp = p.resolve()
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        yield Target(path=rp, kind="file", scope="local")


def _iter_global_hf_targets(home: Path) -> Iterable[Target]:
    base = home / ".cache" / "huggingface"
    for rel in ("hub", "xet", "assets"):
        p = (base / rel).resolve()
        kind = "dir" if p.is_dir() else "file"
        yield Target(path=p, kind=kind, scope="global-hf")


def _size_bytes(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        if path.is_file() or path.is_symlink():
            return int(path.lstat().st_size)
    except Exception:
        return 0

    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                fp = Path(root) / name
                try:
                    total += int(fp.lstat().st_size)
                except Exception:
                    continue
    except Exception:
        return 0
    return total


def _fmt_bytes(n: int) -> str:
    value = float(max(0, n))
    units = ["B", "KB", "MB", "GB", "TB"]
    for u in units:
        if value < 1024.0 or u == units[-1]:
            return f"{value:.2f} {u}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _collect_targets(local: bool, global_hf: bool) -> List[Target]:
    out: List[Target] = []
    if local:
        out.extend(_iter_local_targets(ROOT))
    if global_hf:
        out.extend(_iter_global_hf_targets(HOME))

    dedup = {}
    for t in out:
        dedup[str(t.path)] = t
    return list(dedup.values())


def _delete_target(target: Target) -> Tuple[bool, str]:
    p = target.path
    if not p.exists() and not p.is_symlink():
        return True, "missing"

    try:
        if target.kind == "file" or p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
        else:
            shutil.rmtree(p)
        return True, "deleted"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="clean repo-local caches/temp artifacts")
    parser.add_argument("--global-hf", action="store_true", help="clean ~/.cache/huggingface/{hub,xet,assets}")
    parser.add_argument("--yes", action="store_true", help="apply deletions (default is preview-only)")
    parser.add_argument("--dry-run", action="store_true", help="force preview mode")
    args = parser.parse_args()

    if not args.local and not args.global_hf:
        parser.error("select at least one scope: --local and/or --global-hf")

    do_delete = bool(args.yes and not args.dry_run)
    targets = _collect_targets(local=args.local, global_hf=args.global_hf)

    existing = [t for t in targets if t.path.exists() or t.path.is_symlink()]
    total_before = sum(_size_bytes(t.path) for t in existing)

    print(f"mode: {'apply' if do_delete else 'dry-run'}")
    print(f"targets_selected: {len(targets)}")
    print(f"targets_existing: {len(existing)}")
    print(f"bytes_selected_before: {total_before} ({_fmt_bytes(total_before)})")

    # Safety statement for accidental misuse.
    print("safety: results/ and reports/ are never targeted by this script")

    failures = 0
    if do_delete:
        for t in existing:
            ok, msg = _delete_target(t)
            if not ok:
                failures += 1
                print(f"[FAIL] {t.scope} {t.path} :: {msg}")
            else:
                print(f"[OK] {t.scope} {t.path} :: {msg}")

    post_existing = [t for t in targets if t.path.exists() or t.path.is_symlink()]
    total_after = sum(_size_bytes(t.path) for t in post_existing)
    reclaimed = max(0, total_before - total_after)

    print(f"bytes_selected_after: {total_after} ({_fmt_bytes(total_after)})")
    print(f"bytes_reclaimed_estimate: {reclaimed} ({_fmt_bytes(reclaimed)})")

    if failures:
        print(f"status: partial ({failures} failed deletions)")
        return 2

    print("status: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
