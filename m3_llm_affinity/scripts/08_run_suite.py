#!/usr/bin/env python3
"""Run multi-task benchmark suite with model/context sweeps."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from transformers import AutoConfig

from lib_paths import default_model_alias, llm_variant_dir, results_prefix, slugify_model_id

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FOR_SUBSCRIPTS = "configs/default.yaml"

HARD_FAIL_TYPES = {"OOM", "model_compile_fail", "coreml_convert_fail"}


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_max_positions(cfg_obj: Any) -> int:
    candidates = [
        getattr(cfg_obj, "n_positions", None),
        getattr(cfg_obj, "max_position_embeddings", None),
        getattr(cfg_obj, "max_sequence_length", None),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    return 0


def build_context_schedule(start: int, max_len: int, doubling_steps: int) -> List[int]:
    contexts: List[int] = []
    value = int(start)
    for _ in range(int(doubling_steps) + 1):
        contexts.append(value)
        value *= 2

    unique: List[int] = []
    seen = set()
    for ctx in contexts:
        if ctx in seen:
            continue
        seen.add(ctx)
        if ctx <= int(max_len):
            unique.append(int(ctx))
    return unique


def classify_failure(message: str) -> str:
    text = message.lower()
    if "out of memory" in text or "oom" in text:
        return "OOM"
    if "coreml" in text and ("convert" in text or "mlprogram" in text):
        return "coreml_convert_fail"
    if "coremlc" in text or "mlmodelc" in text or "compile" in text:
        return "model_compile_fail"
    return "runtime_error"


def run_command(cmd: List[str], cwd: Path, dry_run: bool = False) -> Tuple[bool, str]:
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return True, ""

    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    ok = proc.returncode == 0
    output = ""
    if proc.stdout:
        output += proc.stdout.strip()
    if proc.stderr:
        if output:
            output += "\n"
        output += proc.stderr.strip()
    return ok, output


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def append_error_record(
    result_path: Path,
    *,
    model_id: str,
    model_alias: str,
    context_len: Optional[int],
    stage: str,
    message: str,
    failure_type: Optional[str] = None,
    mode: Optional[str] = None,
    prefill_cu: Optional[str] = None,
    decode_cu: Optional[str] = None,
) -> None:
    ctx = int(context_len) if context_len is not None else None
    prefill_len = int(ctx - 1) if ctx is not None else None
    variant_id = None
    if ctx is not None:
        variant_id = f"{slugify_model_id(model_id)}/ctx{ctx}"

    failure = failure_type or classify_failure(message)
    record = {
        "timestamp": dt.datetime.now().isoformat(),
        "model_id": model_id,
        "model_alias": model_alias,
        "variant_id": variant_id,
        "context_len": ctx,
        "prefill_len": prefill_len,
        "gen_tokens": None,
        "mode": mode,
        "prefill_compute_units": prefill_cu,
        "decode_compute_units": decode_cu,
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
        "peak_rss_mb": None,
        "status": "error",
        "error_type": failure,
        "error_message": message,
        "traceback_summary": message,
        "errors": {
            "stage": stage,
            "message": message,
            "failure_type": failure,
        },
        "failure_type": failure,
    }
    append_jsonl(result_path, record)


def run_llm_task(
    task_cfg: Dict[str, Any],
    out_dir: Path,
    timestamp: str,
    only_model: Optional[str],
    dry_run: bool,
) -> None:
    sweep_cfg = task_cfg.get("sweep", {})
    compute_cfg = task_cfg.get("compute_units", {})

    context_len_start = int(sweep_cfg.get("context_len_start", 64))
    context_len_max = int(sweep_cfg.get("context_len_max", 4096))
    doubling_steps = int(sweep_cfg.get("doubling_steps", 6))
    gen_tokens = int(sweep_cfg.get("gen_tokens", 32))
    batch_size = int(sweep_cfg.get("batch_size", 1))
    runs = int(sweep_cfg.get("runs", 10))
    warmup = int(sweep_cfg.get("warmup", 2))
    seed = int(sweep_cfg.get("seed", 1337))

    whole_cus = [str(x) for x in compute_cfg.get("whole", ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"])]
    split_pairs = compute_cfg.get("split", [])

    contexts = build_context_schedule(
        start=context_len_start,
        max_len=context_len_max,
        doubling_steps=doubling_steps,
    )

    models = task_cfg.get("models", [])
    for model_cfg in models:
        model_id = str(model_cfg["model_id"])
        model_alias = str(model_cfg.get("model_alias") or default_model_alias(model_id))

        if only_model and model_alias != only_model:
            continue

        hf_token_env = model_cfg.get("hf_token_env")
        token = None
        if hf_token_env:
            token = os.getenv(str(hf_token_env))
        if token is None:
            token = os.getenv("HF_TOKEN")

        try:
            model_conf = AutoConfig.from_pretrained(model_id, token=token)
            max_positions = get_max_positions(model_conf)
        except Exception as exc:
            result_path = out_dir / f"{timestamp}_{results_prefix(model_alias, 0)}_bench.jsonl"
            append_error_record(
                result_path,
                model_id=model_id,
                model_alias=model_alias,
                context_len=None,
                stage="load_hf_config",
                message=f"{type(exc).__name__}: {exc}",
                failure_type="hf_config_load_fail",
            )
            print(f"[{model_alias}] failed to load HF config: {exc}")
            continue

        stop_on_hard_fail = False
        for ctx in contexts:
            if stop_on_hard_fail:
                break

            result_path = out_dir / f"{timestamp}_{results_prefix(model_alias, ctx)}_bench.jsonl"
            variant_dir = llm_variant_dir(model_id=model_id, context_len=ctx)

            if max_positions and ctx > max_positions:
                msg = (
                    f"context_len={ctx} exceeds model max positions={max_positions} for {model_id}; skipping"
                )
                append_error_record(
                    result_path,
                    model_id=model_id,
                    model_alias=model_alias,
                    context_len=ctx,
                    stage="context_check",
                    message=msg,
                    failure_type="context_exceeds_max_positions",
                )
                print(f"[{model_alias} ctx{ctx}] {msg}")
                continue

            export_cmd = [
                sys.executable,
                "scripts/01_export_torch.py",
                "--config",
                DEFAULT_CONFIG_FOR_SUBSCRIPTS,
                "--model-id",
                model_id,
                "--context-len",
                str(ctx),
                "--variant-dir",
                str(variant_dir),
                "--batch-size",
                str(batch_size),
                "--seed",
                str(seed),
            ]
            if hf_token_env:
                export_cmd.extend(["--hf-token-env", str(hf_token_env)])

            convert_cmd = [
                sys.executable,
                "scripts/02_convert_coreml.py",
                "--config",
                DEFAULT_CONFIG_FOR_SUBSCRIPTS,
                "--model-id",
                model_id,
                "--context-len",
                str(ctx),
                "--variant-dir",
                str(variant_dir),
            ]

            plan_cmd = [
                sys.executable,
                "scripts/04_computeplan_dump.py",
                "--config",
                DEFAULT_CONFIG_FOR_SUBSCRIPTS,
                "--model-id",
                model_id,
                "--model-alias",
                model_alias,
                "--context-len",
                str(ctx),
                "--variant-dir",
                str(variant_dir),
            ]

            print(f"[{model_alias} ctx{ctx}] export")
            ok, out = run_command(export_cmd, ROOT, dry_run=dry_run)
            if not ok:
                failure_type = classify_failure(out)
                append_error_record(
                    result_path,
                    model_id=model_id,
                    model_alias=model_alias,
                    context_len=ctx,
                    stage="export",
                    message=out,
                    failure_type=failure_type,
                )
                print(out)
                if failure_type in HARD_FAIL_TYPES:
                    stop_on_hard_fail = True
                continue

            print(f"[{model_alias} ctx{ctx}] convert")
            ok, out = run_command(convert_cmd, ROOT, dry_run=dry_run)
            if not ok:
                failure_type = classify_failure(out)
                append_error_record(
                    result_path,
                    model_id=model_id,
                    model_alias=model_alias,
                    context_len=ctx,
                    stage="convert",
                    message=out,
                    failure_type=failure_type,
                )
                print(out)
                if failure_type in HARD_FAIL_TYPES:
                    stop_on_hard_fail = True
                continue

            print(f"[{model_alias} ctx{ctx}] compute plan")
            ok, out = run_command(plan_cmd, ROOT, dry_run=dry_run)
            if not ok:
                append_error_record(
                    result_path,
                    model_id=model_id,
                    model_alias=model_alias,
                    context_len=ctx,
                    stage="computeplan",
                    message=out,
                    failure_type=classify_failure(out),
                )
                print(out)

            for cu in whole_cus:
                bench_cmd = [
                    sys.executable,
                    "scripts/03_bench.py",
                    "--config",
                    DEFAULT_CONFIG_FOR_SUBSCRIPTS,
                    "--model-id",
                    model_id,
                    "--model-alias",
                    model_alias,
                    "--context-len",
                    str(ctx),
                    "--variant-dir",
                    str(variant_dir),
                    "--mode",
                    "whole",
                    "--cu",
                    cu,
                    "--runs",
                    str(runs),
                    "--warmup",
                    str(warmup),
                    "--gen-tokens",
                    str(gen_tokens),
                    "--seed",
                    str(seed),
                    "--results-path",
                    str(result_path),
                ]
                print(f"[{model_alias} ctx{ctx}] bench whole {cu}")
                ok, out = run_command(bench_cmd, ROOT, dry_run=dry_run)
                if not ok:
                    append_error_record(
                        result_path,
                        model_id=model_id,
                        model_alias=model_alias,
                        context_len=ctx,
                        stage="bench_whole",
                        message=out,
                        failure_type=classify_failure(out),
                        mode="whole",
                        prefill_cu=cu,
                        decode_cu=cu,
                    )
                    print(out)

            for pair in split_pairs:
                prefill_cu = str(pair["prefill"])
                decode_cu = str(pair["decode"])
                bench_cmd = [
                    sys.executable,
                    "scripts/03_bench.py",
                    "--config",
                    DEFAULT_CONFIG_FOR_SUBSCRIPTS,
                    "--model-id",
                    model_id,
                    "--model-alias",
                    model_alias,
                    "--context-len",
                    str(ctx),
                    "--variant-dir",
                    str(variant_dir),
                    "--mode",
                    "split",
                    "--prefill-cu",
                    prefill_cu,
                    "--decode-cu",
                    decode_cu,
                    "--runs",
                    str(runs),
                    "--warmup",
                    str(warmup),
                    "--gen-tokens",
                    str(gen_tokens),
                    "--seed",
                    str(seed),
                    "--results-path",
                    str(result_path),
                ]
                print(f"[{model_alias} ctx{ctx}] bench split {prefill_cu}->{decode_cu}")
                ok, out = run_command(bench_cmd, ROOT, dry_run=dry_run)
                if not ok:
                    append_error_record(
                        result_path,
                        model_id=model_id,
                        model_alias=model_alias,
                        context_len=ctx,
                        stage="bench_split",
                        message=out,
                        failure_type=classify_failure(out),
                        mode="split",
                        prefill_cu=prefill_cu,
                        decode_cu=decode_cu,
                    )
                    print(out)


def load_optional_task(task_type: str):
    if task_type == "diffusion_sd15":
        from tasks import diffusion_sd15 as module

        return module
    if task_type == "speech_owsm":
        from tasks import speech_owsm as module

        return module
    return None


def run_optional_task(task_cfg: Dict[str, Any], out_dir: Path, timestamp: str, dry_run: bool) -> None:
    task_type = str(task_cfg.get("task_type"))
    module = load_optional_task(task_type)
    model_alias = str(task_cfg.get("model_alias") or task_type)
    result_path = out_dir / f"{timestamp}_{model_alias}_bench.jsonl"

    if module is None:
        append_error_record(
            result_path,
            model_id=task_type,
            model_alias=model_alias,
            context_len=None,
            stage="task_loader",
            message=f"No optional task module implemented for {task_type}",
            failure_type="task_module_missing",
        )
        return

    for stage_name, fn in (
        ("prepare_variant", module.prepare_variant),
        ("run_bench", module.run_bench),
        ("dump_computeplan", module.dump_computeplan),
    ):
        if dry_run:
            print(f"[dry-run] optional task {task_type} stage {stage_name}")
            continue

        result = fn(task_cfg=task_cfg, out_dir=out_dir)
        record = {
            "timestamp": dt.datetime.now().isoformat(),
            "model_id": task_type,
            "model_alias": model_alias,
            "variant_id": None,
            "context_len": None,
            "prefill_len": None,
            "gen_tokens": None,
            "mode": None,
            "prefill_compute_units": None,
            "decode_compute_units": None,
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
            "peak_rss_mb": None,
            "status": result.get("status", "error"),
            "error_type": result.get("error_type"),
            "error_message": result.get("error_message") or result.get("message"),
            "traceback_summary": result.get("error_message") or result.get("message"),
            "errors": {
                "stage": stage_name,
                **result,
            },
            "failure_type": result.get("failure_type"),
        }
        append_jsonl(result_path, record)

        if result.get("status") != "ok":
            # optional tasks should emit one record and skip.
            break


def iter_selected_tasks(tasks: Iterable[Dict[str, Any]], only_task: Optional[str]) -> Iterable[Dict[str, Any]]:
    for task in tasks:
        if only_task and str(task.get("task_type")) != only_task:
            continue
        yield task


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-config", default="configs/suite.yaml")
    parser.add_argument("--only-task", choices=["llm_decode", "diffusion_sd15", "speech_owsm"])
    parser.add_argument("--only-model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml((ROOT / args.suite_config).resolve())
    suite_cfg = cfg.get("suite", {})
    tasks = cfg.get("tasks", [])

    out_dir = (ROOT / str(suite_cfg.get("out_dir", "results"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    for task in iter_selected_tasks(tasks, args.only_task):
        task_type = str(task.get("task_type"))
        enabled = bool(task.get("enabled", True))

        if task_type == "llm_decode":
            if not enabled:
                path = out_dir / f"{timestamp}_llm_decode_disabled_bench.jsonl"
                append_error_record(
                    path,
                    model_id="llm_decode",
                    model_alias="llm_decode",
                    context_len=None,
                    stage="suite",
                    message="llm_decode task is disabled",
                    failure_type="task_disabled",
                )
                continue

            run_llm_task(
                task_cfg=task,
                out_dir=out_dir,
                timestamp=timestamp,
                only_model=args.only_model,
                dry_run=args.dry_run,
            )
            continue

        run_optional_task(task_cfg=task, out_dir=out_dir, timestamp=timestamp, dry_run=args.dry_run)

    print(f"suite_results_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
