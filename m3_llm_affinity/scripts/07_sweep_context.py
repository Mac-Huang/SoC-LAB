#!/usr/bin/env python3
"""Run context-length sweep: export, convert, benchmark, and aggregate summary outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from transformers import AutoConfig

from lib_paths import default_model_alias, slugify_model_id

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_LLM_WHOLE = ("CPU_AND_NE", "ALL")
ALLOWED_LLM_SPLIT = (("CPU_AND_NE", "CPU_AND_GPU"),)


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


def run_command(cmd: Sequence[str], cwd: Path, timeout_sec: Optional[int] = None) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return False, (
            f"TimeoutExpired: command exceeded {timeout_sec}s\n"
            f"command: {' '.join(cmd)}\n"
            f"partial_stdout:\n{(exc.stdout or '').strip()}\n"
            f"partial_stderr:\n{(exc.stderr or '').strip()}"
        )
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


def append_error_row(
    path: Path,
    *,
    model_id: str,
    model_alias: str,
    context_len: Optional[int],
    mode: Optional[str],
    prefill_compute_units: Optional[str],
    decode_compute_units: Optional[str],
    stage: str,
    message: str,
    error_type: Optional[str] = None,
) -> None:
    ctx = int(context_len) if context_len is not None else None
    prefill_len = int(ctx - 1) if ctx is not None else None
    record = {
        "timestamp": dt.datetime.now().isoformat(),
        "task_type": "llm_decode",
        "model_id": model_id,
        "model_alias": model_alias,
        "context_len": ctx,
        "prefill_len": prefill_len,
        "mode": mode,
        "prefill_compute_units": prefill_compute_units,
        "decode_compute_units": decode_compute_units,
        "ttft_ms": None,
        "tokens_per_sec": None,
        "effective_TFLOPS_prefill": None,
        "effective_TFLOPS_decode": None,
        "peak_rss_mb": None,
        "status": "error",
        "error_type": error_type or classify_failure(message),
        "error_message": message,
        "errors": {
            "stage": stage,
            "message": message,
            "failure_type": error_type or classify_failure(message),
        },
    }
    append_jsonl(path, record)


def collect_jsonl_rows(sweep_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(sweep_dir.glob("ctx*/*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_source_jsonl"] = str(path)
                rows.append(record)
    return rows


def to_summary_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        decode_stats = row.get("decode_step_latency_ms_stats") or {}
        status = str(row.get("status") or "ok")
        err_type = row.get("error_type")
        err_msg = row.get("error_message")
        error = ""
        if status != "ok":
            et = str(err_type or "")
            em = str(err_msg or "")
            error = f"{et}: {em}".strip(": ")

        out_rows.append(
            {
                "timestamp": row.get("timestamp"),
                "model_id": row.get("model_id"),
                "model_alias": row.get("model_alias"),
                "context_len": row.get("context_len"),
                "prefill_len": row.get("prefill_len"),
                "gen_tokens": row.get("gen_tokens"),
                "mode": row.get("mode"),
                "prefill_compute_units": row.get("prefill_compute_units"),
                "decode_compute_units": row.get("decode_compute_units"),
                "status": status,
                "error_type": err_type,
                "ttft_ms": row.get("ttft_ms", row.get("prefill_latency_ms")),
                "tpot_ms_mean": row.get("tpot_ms_mean", decode_stats.get("mean")),
                "tokens_per_sec": row.get("tokens_per_sec"),
                "effective_tflops_prefill": row.get("effective_TFLOPS_prefill"),
                "effective_tflops_decode": row.get("effective_TFLOPS_decode"),
                "peak_rss_mb": row.get("peak_rss_mb"),
                "error": error,
            }
        )

    columns = [
        "timestamp",
        "model_id",
        "model_alias",
        "context_len",
        "prefill_len",
        "gen_tokens",
        "mode",
        "prefill_compute_units",
        "decode_compute_units",
        "status",
        "error_type",
        "ttft_ms",
        "tpot_ms_mean",
        "tokens_per_sec",
        "effective_tflops_prefill",
        "effective_tflops_decode",
        "peak_rss_mb",
        "error",
    ]

    if not out_rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(out_rows, columns=columns)


def scenario_key(mode: str, prefill: str, decode: str) -> str:
    if mode == "whole":
        return f"whole:{prefill}"
    return f"split:{prefill}->{decode}"


def validate_llm_mode_policy(whole_cus: Sequence[str], split_pairs: Sequence[Dict[str, Any]]) -> None:
    whole_set = set(str(x) for x in whole_cus)
    split_set = set((str(p["prefill"]), str(p["decode"])) for p in split_pairs)
    allowed_whole_set = set(ALLOWED_LLM_WHOLE)
    allowed_split_set = set(ALLOWED_LLM_SPLIT)
    if whole_set != allowed_whole_set or split_set != allowed_split_set:
        raise ValueError(
            "sweep_modes must be exactly "
            f"whole={list(ALLOWED_LLM_WHOLE)} and split_pairs={list(ALLOWED_LLM_SPLIT)}; "
            f"got whole={sorted(whole_set)} split_pairs={sorted(split_set)}"
        )


def compute_ctx_best(summary_df: pd.DataFrame) -> Dict[str, Any]:
    if summary_df.empty:
        return {"best_ttft": [], "best_tokens_per_sec": []}

    ok = summary_df[summary_df["status"] == "ok"].copy()
    if ok.empty:
        return {"best_ttft": [], "best_tokens_per_sec": []}

    ok["scenario"] = ok.apply(
        lambda r: scenario_key(str(r["mode"]), str(r["prefill_compute_units"]), str(r["decode_compute_units"])),
        axis=1,
    )

    grouped = (
        ok.groupby(["context_len", "scenario", "mode", "prefill_compute_units", "decode_compute_units"], as_index=False)
        .agg(
            ttft_ms_mean=("ttft_ms", "mean"),
            tokens_per_sec_mean=("tokens_per_sec", "mean"),
        )
        .sort_values(["context_len", "scenario"])
    )

    best_ttft: List[Dict[str, Any]] = []
    best_tok: List[Dict[str, Any]] = []
    for ctx, part in grouped.groupby("context_len"):
        ttft_row = part.sort_values("ttft_ms_mean", ascending=True).iloc[0]
        tok_row = part.sort_values("tokens_per_sec_mean", ascending=False).iloc[0]
        best_ttft.append(
            {
                "context_len": int(ctx),
                "scenario": str(ttft_row["scenario"]),
                "ttft_ms_mean": float(ttft_row["ttft_ms_mean"]),
            }
        )
        best_tok.append(
            {
                "context_len": int(ctx),
                "scenario": str(tok_row["scenario"]),
                "tokens_per_sec_mean": float(tok_row["tokens_per_sec_mean"]),
            }
        )

    return {
        "best_ttft": best_ttft,
        "best_tokens_per_sec": best_tok,
    }


def detect_flip(summary_df: pd.DataFrame) -> Dict[str, Any]:
    if summary_df.empty:
        return {"flip_detected": False, "deltas_by_context": []}

    ok = summary_df[summary_df["status"] == "ok"].copy()
    if ok.empty:
        return {"flip_detected": False, "deltas_by_context": []}

    a = ok[
        (ok["mode"] == "split")
        & (ok["prefill_compute_units"] == "CPU_AND_NE")
        & (ok["decode_compute_units"] == "CPU_AND_GPU")
    ]
    b = ok[
        (ok["mode"] == "split")
        & (ok["prefill_compute_units"] == "CPU_AND_GPU")
        & (ok["decode_compute_units"] == "CPU_AND_NE")
    ]

    if a.empty or b.empty:
        return {
            "flip_detected": False,
            "deltas_by_context": [],
            "note": "insufficient rows to compare split pair directions",
        }

    a_mean = a.groupby("context_len", as_index=False)["tokens_per_sec"].mean()
    b_mean = b.groupby("context_len", as_index=False)["tokens_per_sec"].mean()
    merged = a_mean.merge(b_mean, on="context_len", suffixes=("_ne_gpu", "_gpu_ne"))
    if merged.empty:
        return {
            "flip_detected": False,
            "deltas_by_context": [],
            "note": "no overlapping contexts for split pair direction comparison",
        }

    merged["delta"] = merged["tokens_per_sec_ne_gpu"] - merged["tokens_per_sec_gpu_ne"]
    signs: List[int] = []
    deltas: List[Dict[str, Any]] = []
    for _, row in merged.sort_values("context_len").iterrows():
        delta = float(row["delta"])
        signs.append(1 if delta > 0 else -1 if delta < 0 else 0)
        deltas.append(
            {
                "context_len": int(row["context_len"]),
                "tokens_per_sec_ne_gpu": float(row["tokens_per_sec_ne_gpu"]),
                "tokens_per_sec_gpu_ne": float(row["tokens_per_sec_gpu_ne"]),
                "delta": delta,
            }
        )

    nonzero = [s for s in signs if s != 0]
    flip = False
    if len(nonzero) >= 2:
        first = nonzero[0]
        flip = any(s != first for s in nonzero[1:])

    return {
        "flip_detected": bool(flip),
        "deltas_by_context": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep_ctx.yaml")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--only-bench", action="store_true")
    parser.add_argument("--cmd-timeout-sec", type=int, default=900)
    args = parser.parse_args()

    cfg = load_yaml((ROOT / args.config).resolve())
    hf_model_id = str(cfg["hf_model_id"])
    model_alias = str(cfg.get("model_alias") or default_model_alias(hf_model_id))
    model_slug = slugify_model_id(hf_model_id)
    runs = int(cfg.get("runs", 20))
    warmup = int(cfg.get("warmup", 3))

    context_sweep = cfg.get("context_len_sweep")
    if context_sweep is None:
        base = int(cfg["context_len"])
        context_sweep = [base, base * 2, base * 4, base * 8]
    contexts = [int(x) for x in context_sweep]

    sweep_modes = cfg.get("sweep_modes", {})
    whole_cus = [str(x) for x in sweep_modes.get("whole", cfg.get("compute_units_list", []))]
    split_pairs = sweep_modes.get("split_pairs", [])
    validate_llm_mode_policy(whole_cus, split_pairs)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = ROOT / "results" / "llm_decode" / model_alias / f"sweep_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_cfg = AutoConfig.from_pretrained(hf_model_id)
        max_positions = get_max_positions(model_cfg)
    except Exception as exc:
        err_path = sweep_dir / "ctx0" / "suite_error.jsonl"
        append_error_row(
            err_path,
            model_id=hf_model_id,
            model_alias=model_alias,
            context_len=None,
            mode=None,
            prefill_compute_units=None,
            decode_compute_units=None,
            stage="load_hf_config",
            message=f"{type(exc).__name__}: {exc}",
            error_type="hf_config_load_fail",
        )
        summary_csv = sweep_dir / "summary.csv"
        summary_json = sweep_dir / "summary.json"
        df = to_summary_frame(collect_jsonl_rows(sweep_dir))
        df.to_csv(summary_csv, index=False)
        payload = {
            "timestamp": ts,
            "model_id": hf_model_id,
            "model_alias": model_alias,
            "model_slug": model_slug,
            "contexts": contexts,
            "rows": int(df.shape[0]),
            "summary_csv": str(summary_csv),
            "error": f"{type(exc).__name__}: {exc}",
        }
        with summary_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"saved: {summary_csv}")
        print(f"saved: {summary_json}")
        return 0

    orchestration_errors: List[Dict[str, Any]] = []
    skipped_contexts: List[Dict[str, Any]] = []

    for ctx in contexts:
        ctx_dir = sweep_dir / f"ctx{ctx}"
        ctx_dir.mkdir(parents=True, exist_ok=True)

        if max_positions and ctx > max_positions:
            msg = (
                f"context_len={ctx} exceeds model max positions={max_positions} "
                f"for {hf_model_id}; skipping context."
            )
            print(f"WARNING: {msg}")
            skipped_contexts.append({"context_len": ctx, "reason": msg})
            append_error_row(
                ctx_dir / "skip_context.jsonl",
                model_id=hf_model_id,
                model_alias=model_alias,
                context_len=ctx,
                mode=None,
                prefill_compute_units=None,
                decode_compute_units=None,
                stage="context_check",
                message=msg,
                error_type="context_exceeds_max_positions",
            )
            continue

        models_dir = ROOT / "models" / model_slug / f"ctx{ctx}"
        prefill_model = models_dir / "prefill.mlpackage"
        decode_model = models_dir / "decode.mlpackage"

        if not args.only_bench:
            must_convert = True
            if args.skip_convert and prefill_model.exists() and decode_model.exists():
                must_convert = False

            if must_convert:
                export_cmd = [
                    sys.executable,
                    "scripts/01_export_torch.py",
                    "--config",
                    args.config,
                    "--model-id",
                    hf_model_id,
                    "--context-len",
                    str(ctx),
                ]
                convert_cmd = [
                    sys.executable,
                    "scripts/02_convert_coreml.py",
                    "--config",
                    args.config,
                    "--model-id",
                    hf_model_id,
                    "--context-len",
                    str(ctx),
                ]

                print(f"[{ctx}] export torch wrappers")
                ok, output = run_command(export_cmd, ROOT, timeout_sec=args.cmd_timeout_sec)
                if not ok:
                    orchestration_errors.append(
                        {
                            "context_len": ctx,
                            "stage": "export",
                            "command": " ".join(export_cmd),
                            "error": output,
                        }
                    )
                    append_error_row(
                        ctx_dir / "export_error.jsonl",
                        model_id=hf_model_id,
                        model_alias=model_alias,
                        context_len=ctx,
                        mode=None,
                        prefill_compute_units=None,
                        decode_compute_units=None,
                        stage="export",
                        message=output,
                    )
                    print(output)
                    continue

                print(f"[{ctx}] convert to coreml")
                ok, output = run_command(convert_cmd, ROOT, timeout_sec=args.cmd_timeout_sec)
                if not ok:
                    orchestration_errors.append(
                        {
                            "context_len": ctx,
                            "stage": "convert",
                            "command": " ".join(convert_cmd),
                            "error": output,
                        }
                    )
                    append_error_row(
                        ctx_dir / "convert_error.jsonl",
                        model_id=hf_model_id,
                        model_alias=model_alias,
                        context_len=ctx,
                        mode=None,
                        prefill_compute_units=None,
                        decode_compute_units=None,
                        stage="convert",
                        message=output,
                    )
                    print(output)
                    continue

        if not prefill_model.exists() or not decode_model.exists():
            msg = (
                f"missing models for context_len={ctx} at {models_dir}; "
                "skipping benchmark for this context."
            )
            print(f"WARNING: {msg}")
            orchestration_errors.append(
                {"context_len": ctx, "stage": "precheck", "command": None, "error": msg}
            )
            append_error_row(
                ctx_dir / "precheck_error.jsonl",
                model_id=hf_model_id,
                model_alias=model_alias,
                context_len=ctx,
                mode=None,
                prefill_compute_units=None,
                decode_compute_units=None,
                stage="precheck",
                message=msg,
                error_type="missing_model_artifact",
            )
            continue

        plan_cmd = [
            sys.executable,
            "scripts/04_computeplan_dump.py",
            "--config",
            args.config,
            "--model-id",
            hf_model_id,
            "--model-alias",
            model_alias,
            "--context-len",
            str(ctx),
        ]
        print(f"[{ctx}] compute plan")
        ok, output = run_command(plan_cmd, ROOT, timeout_sec=args.cmd_timeout_sec)
        if not ok:
            orchestration_errors.append(
                {
                    "context_len": ctx,
                    "stage": "computeplan",
                    "command": " ".join(plan_cmd),
                    "error": output,
                }
            )
            append_error_row(
                ctx_dir / "computeplan_error.jsonl",
                model_id=hf_model_id,
                model_alias=model_alias,
                context_len=ctx,
                mode=None,
                prefill_compute_units=None,
                decode_compute_units=None,
                stage="computeplan",
                message=output,
                error_type="computeplan_fail",
            )
            print(output)

        for cu in whole_cus:
            out_path = ctx_dir / f"whole_{cu}.jsonl"
            bench_cmd = [
                sys.executable,
                "scripts/03_bench.py",
                "--config",
                args.config,
                "--model-id",
                hf_model_id,
                "--model-alias",
                model_alias,
                "--context-len",
                str(ctx),
                "--mode",
                "whole",
                "--cu",
                cu,
                "--runs",
                str(runs),
                "--warmup",
                str(warmup),
                "--results-path",
                str(out_path),
            ]
            print(f"[{ctx}] bench whole {cu}")
            ok, output = run_command(bench_cmd, ROOT, timeout_sec=args.cmd_timeout_sec)
            if not ok:
                orchestration_errors.append(
                    {
                        "context_len": ctx,
                        "stage": "bench_whole",
                        "command": " ".join(bench_cmd),
                        "error": output,
                    }
                )
                append_error_row(
                    out_path,
                    model_id=hf_model_id,
                    model_alias=model_alias,
                    context_len=ctx,
                    mode="whole",
                    prefill_compute_units=cu,
                    decode_compute_units=cu,
                    stage="bench_whole",
                    message=output,
                )
                print(output)

        for pair in split_pairs:
            prefill = str(pair["prefill"])
            decode = str(pair["decode"])
            out_path = ctx_dir / f"split_{prefill}__{decode}.jsonl"
            bench_cmd = [
                sys.executable,
                "scripts/03_bench.py",
                "--config",
                args.config,
                "--model-id",
                hf_model_id,
                "--model-alias",
                model_alias,
                "--context-len",
                str(ctx),
                "--mode",
                "split",
                "--prefill-cu",
                prefill,
                "--decode-cu",
                decode,
                "--runs",
                str(runs),
                "--warmup",
                str(warmup),
                "--results-path",
                str(out_path),
            ]
            print(f"[{ctx}] bench split {prefill}->{decode}")
            ok, output = run_command(bench_cmd, ROOT, timeout_sec=args.cmd_timeout_sec)
            if not ok:
                orchestration_errors.append(
                    {
                        "context_len": ctx,
                        "stage": "bench_split",
                        "command": " ".join(bench_cmd),
                        "error": output,
                    }
                )
                append_error_row(
                    out_path,
                    model_id=hf_model_id,
                    model_alias=model_alias,
                    context_len=ctx,
                    mode="split",
                    prefill_compute_units=prefill,
                    decode_compute_units=decode,
                    stage="bench_split",
                    message=output,
                )
                print(output)

    rows = collect_jsonl_rows(sweep_dir)
    summary_df = to_summary_frame(rows)
    summary_csv = sweep_dir / "summary.csv"
    summary_json = sweep_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False)

    best = compute_ctx_best(summary_df)
    flip = detect_flip(summary_df)

    payload = {
        "timestamp": ts,
        "model_id": hf_model_id,
        "model_alias": model_alias,
        "model_slug": model_slug,
        "contexts": contexts,
        "max_positions": max_positions,
        "rows": int(summary_df.shape[0]),
        "summary_csv": str(summary_csv),
        "skipped_contexts": skipped_contexts,
        "orchestration_errors": orchestration_errors,
        "best_by_context": best,
        "split_flip_analysis": flip,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(f"saved: {summary_csv}")
    print(f"saved: {summary_json}")

    if best["best_ttft"]:
        print("Best TTFT per context:")
        for row in best["best_ttft"]:
            print(
                f"  ctx={row['context_len']}: {row['scenario']} "
                f"ttft_ms_mean={row['ttft_ms_mean']:.3f}"
            )

    if best["best_tokens_per_sec"]:
        print("Best tokens/sec per context:")
        for row in best["best_tokens_per_sec"]:
            print(
                f"  ctx={row['context_len']}: {row['scenario']} "
                f"tokens_per_sec_mean={row['tokens_per_sec_mean']:.3f}"
            )

    if "deltas_by_context" in flip and flip["deltas_by_context"]:
        direction = "flips" if flip.get("flip_detected") else "does not flip"
        print(
            "Split comparison (prefill NE + decode GPU vs prefill GPU + decode NE) "
            f"{direction} as context grows."
        )
        for row in flip["deltas_by_context"]:
            print(
                f"  ctx={row['context_len']}: delta_tokens_per_sec="
                f"{row['delta']:.3f} (NE->GPU minus GPU->NE)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
