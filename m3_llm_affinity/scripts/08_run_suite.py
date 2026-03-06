#!/usr/bin/env python3
"""Run multi-task benchmark suite with model/context sweeps."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from transformers import AutoConfig

from lib_paths import default_model_alias, llm_variant_dir, slugify_model_id

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FOR_SUBSCRIPTS = "configs/default.yaml"
HARD_FAIL_TYPES = {"OOM", "model_compile_fail", "coreml_convert_fail"}
ALLOWED_LLM_WHOLE = ("CPU_AND_NE", "ALL")
ALLOWED_LLM_SPLIT = (("CPU_AND_NE", "CPU_AND_GPU"),)
OPTIONAL_HARD_FAIL_TYPES = {
    "missing_dependency",
    "missing_asset",
    "download_unavailable",
    "backend_unavailable",
}


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
    text = str(message).lower()
    if "out of memory" in text or "oom" in text:
        return "OOM"
    if "coreml" in text and ("convert" in text or "mlprogram" in text):
        return "coreml_convert_fail"
    if "coremlc" in text or "mlmodelc" in text or "compile" in text:
        return "model_compile_fail"
    if "token" in text and ("hf" in text or "hugging" in text or "401" in text or "gated" in text):
        return "missing_hf_token"
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


def append_jsonl_many(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    n = 0
    for record in records:
        append_jsonl(path, record)
        n += 1
    return n


def append_error_record(
    result_path: Path,
    *,
    task_type: str,
    model_id: str,
    model_alias: str,
    context_len: Optional[int],
    stage: str,
    message: str,
    failure_type: Optional[str] = None,
    mode: Optional[str] = None,
    prefill_cu: Optional[str] = None,
    decode_cu: Optional[str] = None,
    x_label: Optional[str] = None,
    x_value: Optional[float] = None,
    uses_coreml: Optional[bool] = None,
) -> None:
    ctx = int(context_len) if context_len is not None else None
    prefill_len = int(ctx - 1) if ctx is not None else None
    variant_id = None
    if ctx is not None and task_type == "llm_decode":
        variant_id = f"{slugify_model_id(model_id)}/ctx{ctx}"

    failure = failure_type or classify_failure(message)
    record = {
        "timestamp": dt.datetime.now().isoformat(),
        "task_type": task_type,
        "model_id": model_id,
        "model_alias": model_alias,
        "variant_id": variant_id,
        "context_len": ctx,
        "prefill_len": prefill_len,
        "gen_tokens": None,
        "mode": mode,
        "prefill_compute_units": prefill_cu,
        "decode_compute_units": decode_cu,
        "scenario_label": None,
        "x_label": x_label,
        "x_value": x_value,
        "primary_latency_ms": None,
        "primary_throughput": None,
        "uses_coreml": uses_coreml,
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


def sweep_dir_for(out_dir: Path, task_type: str, model_alias: str, sweep_id: str) -> Path:
    return out_dir / task_type / model_alias / f"sweep_{sweep_id}"


def llm_result_path_for(
    out_dir: Path,
    model_alias: str,
    sweep_id: str,
    context_len: int,
) -> Path:
    return sweep_dir_for(out_dir, "llm_decode", model_alias, sweep_id) / f"ctx{int(context_len)}_bench.jsonl"


def task_result_path_for(
    out_dir: Path,
    task_type: str,
    model_alias: str,
    sweep_id: str,
) -> Path:
    return sweep_dir_for(out_dir, task_type, model_alias, sweep_id) / f"{task_type}_bench.jsonl"


def _normalize_split_pairs(raw_pairs: Iterable[Dict[str, Any]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for pair in raw_pairs:
        out.append((str(pair["prefill"]), str(pair["decode"])))
    return out


def validate_llm_mode_policy(whole_cus: Iterable[str], split_pairs: Iterable[Dict[str, Any]]) -> None:
    whole_set = set(str(x) for x in whole_cus)
    split_set = set(_normalize_split_pairs(split_pairs))
    allowed_whole_set = set(ALLOWED_LLM_WHOLE)
    allowed_split_set = set(ALLOWED_LLM_SPLIT)

    if whole_set != allowed_whole_set or split_set != allowed_split_set:
        raise ValueError(
            "llm_decode compute_units must be exactly "
            f"whole={list(ALLOWED_LLM_WHOLE)} and split={list(ALLOWED_LLM_SPLIT)}; "
            f"got whole={sorted(whole_set)} split={sorted(split_set)}"
        )


def run_llm_task(
    task_cfg: Dict[str, Any],
    out_dir: Path,
    timestamp: str,
    only_model: Optional[str],
    dry_run: bool,
    skip_convert: bool,
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

    whole_cus = [str(x) for x in compute_cfg.get("whole", list(ALLOWED_LLM_WHOLE))]
    split_pairs = compute_cfg.get("split", [])
    validate_llm_mode_policy(whole_cus, split_pairs)

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
            if not token:
                missing_token_path = llm_result_path_for(
                    out_dir=out_dir,
                    model_alias=model_alias,
                    sweep_id=timestamp,
                    context_len=0,
                )
                append_error_record(
                    missing_token_path,
                    task_type="llm_decode",
                    model_id=model_id,
                    model_alias=model_alias,
                    context_len=None,
                    stage="token_check",
                    message=f"Missing required env var '{hf_token_env}' for model {model_id}",
                    failure_type="missing_hf_token",
                    x_label="context_len",
                    uses_coreml=True,
                )
                print(f"[{model_alias}] missing required token env: {hf_token_env}; skipping model")
                continue

        if token is None:
            token = os.getenv("HF_TOKEN")

        try:
            model_conf = AutoConfig.from_pretrained(model_id, token=token)
            max_positions = get_max_positions(model_conf)
        except Exception as exc:
            result_path = llm_result_path_for(
                out_dir=out_dir,
                model_alias=model_alias,
                sweep_id=timestamp,
                context_len=0,
            )
            append_error_record(
                result_path,
                task_type="llm_decode",
                model_id=model_id,
                model_alias=model_alias,
                context_len=None,
                stage="load_hf_config",
                message=f"{type(exc).__name__}: {exc}",
                failure_type="hf_config_load_fail",
                x_label="context_len",
                uses_coreml=True,
            )
            print(f"[{model_alias}] failed to load HF config: {exc}")
            continue

        allow_weight_quant = bool(model_cfg.get("allow_weight_quant", False))
        weight_quant_mode = str(model_cfg.get("weight_quant_mode", "int4"))

        stop_on_hard_fail = False
        for ctx in contexts:
            if stop_on_hard_fail:
                break

            result_path = llm_result_path_for(
                out_dir=out_dir,
                model_alias=model_alias,
                sweep_id=timestamp,
                context_len=ctx,
            )
            variant_dir = llm_variant_dir(model_id=model_id, context_len=ctx)

            if max_positions and ctx > max_positions:
                msg = f"context_len={ctx} exceeds model max positions={max_positions} for {model_id}; skipping"
                append_error_record(
                    result_path,
                    task_type="llm_decode",
                    model_id=model_id,
                    model_alias=model_alias,
                    context_len=ctx,
                    stage="context_check",
                    message=msg,
                    failure_type="context_exceeds_max_positions",
                    x_label="context_len",
                    x_value=float(ctx),
                    uses_coreml=True,
                )
                print(f"[{model_alias} ctx{ctx}] {msg}")
                continue

            model_prefill = ROOT / "models" / slugify_model_id(model_id) / f"ctx{ctx}" / "prefill.mlpackage"
            model_decode = ROOT / "models" / slugify_model_id(model_id) / f"ctx{ctx}" / "decode.mlpackage"
            models_exist = model_prefill.exists() and model_decode.exists()

            need_convert = not (skip_convert and models_exist)
            if skip_convert and not models_exist:
                append_error_record(
                    result_path,
                    task_type="llm_decode",
                    model_id=model_id,
                    model_alias=model_alias,
                    context_len=ctx,
                    stage="convert",
                    message=(
                        "--skip-convert was set but Core ML packages are missing at "
                        f"{model_prefill.parent}. Provide BYO mlpackage artifacts or rerun without --skip-convert."
                    ),
                    failure_type="missing_asset",
                    x_label="context_len",
                    x_value=float(ctx),
                    uses_coreml=True,
                )
                continue

            if need_convert:
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
                if allow_weight_quant:
                    convert_cmd.append("--allow-weight-quant")
                    convert_cmd.extend(["--weight-quant-mode", weight_quant_mode])

                print(f"[{model_alias} ctx{ctx}] export")
                ok, out = run_command(export_cmd, ROOT, dry_run=dry_run)
                if not ok:
                    failure_type = classify_failure(out)
                    append_error_record(
                        result_path,
                        task_type="llm_decode",
                        model_id=model_id,
                        model_alias=model_alias,
                        context_len=ctx,
                        stage="export",
                        message=out,
                        failure_type=failure_type,
                        x_label="context_len",
                        x_value=float(ctx),
                        uses_coreml=True,
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
                        task_type="llm_decode",
                        model_id=model_id,
                        model_alias=model_alias,
                        context_len=ctx,
                        stage="convert",
                        message=out,
                        failure_type=failure_type,
                        x_label="context_len",
                        x_value=float(ctx),
                        uses_coreml=True,
                    )
                    print(out)
                    if failure_type in HARD_FAIL_TYPES:
                        stop_on_hard_fail = True
                    continue

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
            print(f"[{model_alias} ctx{ctx}] compute plan")
            ok, out = run_command(plan_cmd, ROOT, dry_run=dry_run)
            if not ok:
                append_error_record(
                    result_path,
                    task_type="llm_decode",
                    model_id=model_id,
                    model_alias=model_alias,
                    context_len=ctx,
                    stage="computeplan",
                    message=out,
                    failure_type=classify_failure(out),
                    x_label="context_len",
                    x_value=float(ctx),
                    uses_coreml=True,
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
                        task_type="llm_decode",
                        model_id=model_id,
                        model_alias=model_alias,
                        context_len=ctx,
                        stage="bench_whole",
                        message=out,
                        failure_type=classify_failure(out),
                        mode="whole",
                        prefill_cu=cu,
                        decode_cu=cu,
                        x_label="context_len",
                        x_value=float(ctx),
                        uses_coreml=True,
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
                        task_type="llm_decode",
                        model_id=model_id,
                        model_alias=model_alias,
                        context_len=ctx,
                        stage="bench_split",
                        message=out,
                        failure_type=classify_failure(out),
                        mode="split",
                        prefill_cu=prefill_cu,
                        decode_cu=decode_cu,
                        x_label="context_len",
                        x_value=float(ctx),
                        uses_coreml=True,
                    )
                    print(out)


def load_optional_task(task_type: str):
    if task_type == "diffusion_sd15":
        from tasks import diffusion_sd15 as module

        return module
    if task_type == "speech_whisperkit":
        from tasks import speech_whisperkit as module

        return module
    if task_type == "speech_owsm":
        from tasks import speech_owsm as module

        return module
    return None


def _optional_model_id(task_cfg: Dict[str, Any], task_type: str) -> str:
    if task_type == "speech_owsm":
        return str(task_cfg.get("model_tag") or task_type)
    if task_type == "speech_whisperkit":
        variant = str(task_cfg.get("model_variant", "openai_whisper-tiny.en"))
        return f"argmaxinc/whisperkit-coreml/{variant}"
    return str(task_cfg.get("model_id") or task_cfg.get("model_tag") or task_type)


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        p = path.resolve()
        if not p.exists() or not p.is_file():
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _resolve_globs(patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        text = str(pattern)
        if not text:
            continue
        if Path(text).is_absolute():
            hits = glob.glob(text, recursive=True)
        else:
            hits = glob.glob(str((ROOT / text).resolve()), recursive=True)
        out.extend(Path(p) for p in hits)
    return _dedupe_paths(out)


def _collect_run_jsonls(out_dir: Path, sweep_id: str) -> List[Path]:
    return _dedupe_paths(sorted(out_dir.glob(f"**/sweep_{sweep_id}/*_bench.jsonl")))


def _build_analyzer_inputs(out_dir: Path, sweep_id: str, historical_globs: Iterable[str]) -> List[Path]:
    current = _collect_run_jsonls(out_dir, sweep_id)
    historical = _resolve_globs(historical_globs)
    return _dedupe_paths([*current, *historical])


def run_optional_task(
    task_cfg: Dict[str, Any],
    out_dir: Path,
    timestamp: str,
    dry_run: bool,
) -> Dict[str, Any]:
    task_type = str(task_cfg.get("task_type"))
    module = load_optional_task(task_type)
    model_alias = str(task_cfg.get("model_alias") or task_type)
    result_path = task_result_path_for(out_dir, task_type, model_alias, timestamp)
    model_id = _optional_model_id(task_cfg, task_type)
    hard_fail = False

    if module is None:
        append_error_record(
            result_path,
            task_type=task_type,
            model_id=model_id,
            model_alias=model_alias,
            context_len=None,
            stage="task_loader",
            message=f"No optional task module implemented for {task_type}",
            failure_type="task_module_missing",
            x_label="x",
            uses_coreml=False,
        )
        return {"result_path": result_path, "hard_fail": False}

    prep = module.prepare_variant(task_cfg=task_cfg, root_dir=ROOT, dry_run=dry_run)
    if prep.get("status") != "ok":
        failure = str(prep.get("error_type") or prep.get("failure_type") or "")
        if task_type == "speech_whisperkit" and failure in OPTIONAL_HARD_FAIL_TYPES:
            hard_fail = True
    if dry_run and prep.get("message"):
        print(f"[dry-run] {task_type}: {prep.get('message')}")

    bench_result = module.run_bench(
        task_cfg=task_cfg,
        prep=prep,
        root_dir=ROOT,
        out_dir=out_dir,
        dry_run=dry_run,
    )

    records = bench_result.get("records", [])
    if records:
        append_jsonl_many(result_path, records)
        if task_type == "speech_whisperkit":
            for row in records:
                if str(row.get("status") or "ok") != "ok":
                    failure = str(row.get("error_type") or row.get("failure_type") or "")
                    if failure in OPTIONAL_HARD_FAIL_TYPES:
                        hard_fail = True
    elif bench_result.get("status") != "ok":
        failure = str(bench_result.get("error_type") or bench_result.get("failure_type") or "bench_failed")
        append_error_record(
            result_path,
            task_type=task_type,
            model_id=model_id,
            model_alias=model_alias,
            context_len=None,
            stage="run_bench",
            message=str(bench_result.get("error_message") or bench_result.get("message") or "bench failed"),
            failure_type=failure,
            x_label="x",
            uses_coreml=bool(task_type != "speech_owsm"),
        )
        if task_type == "speech_whisperkit" and failure in OPTIONAL_HARD_FAIL_TYPES:
            hard_fail = True
    if dry_run and bench_result.get("message"):
        print(f"[dry-run] {task_type}: {bench_result.get('message')}")

    plan_result = module.dump_computeplan(
        task_cfg=task_cfg,
        prep=prep,
        root_dir=ROOT,
        out_dir=out_dir,
        dry_run=dry_run,
    )
    if plan_result.get("status") != "ok":
        failure = str(plan_result.get("error_type") or plan_result.get("failure_type") or "computeplan_fail")
        append_error_record(
            result_path,
            task_type=task_type,
            model_id=model_id,
            model_alias=model_alias,
            context_len=None,
            stage="dump_computeplan",
            message=str(plan_result.get("error_message") or plan_result.get("message") or "compute plan failed"),
            failure_type=failure,
            x_label="x",
            uses_coreml=bool(task_type != "speech_owsm"),
        )
        if task_type == "speech_whisperkit" and failure in OPTIONAL_HARD_FAIL_TYPES:
            hard_fail = True
    if dry_run and plan_result.get("message"):
        print(f"[dry-run] {task_type}: {plan_result.get('message')}")

    return {
        "result_path": result_path,
        "hard_fail": bool(hard_fail),
    }


def iter_selected_tasks(tasks: Iterable[Dict[str, Any]], only_task: Optional[str]) -> Iterable[Dict[str, Any]]:
    for task in tasks:
        if only_task and str(task.get("task_type")) != only_task:
            continue
        yield task


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-config", default="configs/suite.yaml")
    parser.add_argument("--only-task", choices=["llm_decode", "diffusion_sd15", "speech_owsm", "speech_whisperkit"])
    parser.add_argument("--only-model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--no-lm-rerun", action="store_true")
    parser.add_argument("--analyze-after-run", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml((ROOT / args.suite_config).resolve())
    suite_cfg = cfg.get("suite", {})
    tasks = cfg.get("tasks", [])

    out_dir = (ROOT / str(suite_cfg.get("out_dir", "results"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    hard_fail_hit = False

    for task in iter_selected_tasks(tasks, args.only_task):
        task_type = str(task.get("task_type"))
        enabled = bool(task.get("enabled", True))

        if not enabled:
            path = task_result_path_for(out_dir, task_type, task_type, timestamp)
            append_error_record(
                path,
                task_type=task_type,
                model_id=task_type,
                model_alias=task_type,
                context_len=None,
                stage="suite",
                message=f"{task_type} task is disabled",
                failure_type="task_disabled",
                x_label="x",
                uses_coreml=bool(task_type != "speech_owsm"),
            )
            continue

        if task_type == "llm_decode":
            if args.no_lm_rerun:
                print("[suite] skipping llm_decode due to --no-lm-rerun")
                continue
            run_llm_task(
                task_cfg=task,
                out_dir=out_dir,
                timestamp=timestamp,
                only_model=args.only_model,
                dry_run=args.dry_run,
                skip_convert=args.skip_convert,
            )
            continue

        optional_result = run_optional_task(task_cfg=task, out_dir=out_dir, timestamp=timestamp, dry_run=args.dry_run)
        if optional_result.get("hard_fail") and not args.dry_run:
            print(f"[suite] hard-fail prerequisite unmet for {task_type}; stopping campaign.")
            hard_fail_hit = True
            break

    print(f"suite_results_dir: {out_dir}")

    include_globs = [str(x) for x in suite_cfg.get("include_historical_results_globs", [])]
    analyze_requested = bool(args.analyze_after_run or args.dry_run)
    if analyze_requested:
        analyzer_inputs = _build_analyzer_inputs(out_dir=out_dir, sweep_id=timestamp, historical_globs=include_globs)
        analyzer_cmd = [
            sys.executable,
            "scripts/07_analyze_results.py",
            "--suite-config",
            args.suite_config,
        ]
        if analyzer_inputs:
            analyzer_cmd.append("--inputs")
            analyzer_cmd.extend(str(path) for path in analyzer_inputs)
        print(
            "[suite] analyzer inputs: "
            f"{len(analyzer_inputs)} files ({len(_collect_run_jsonls(out_dir, timestamp))} current run + "
            f"{max(0, len(analyzer_inputs) - len(_collect_run_jsonls(out_dir, timestamp)))} historical)"
        )
        if args.dry_run:
            print("[dry-run] " + " ".join(shlex.quote(x) for x in analyzer_cmd))
        else:
            ok, out = run_command(analyzer_cmd, ROOT, dry_run=False)
            if out:
                print(out)
            if not ok:
                print("[suite] analyzer failed")
                return 3

    if hard_fail_hit:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
