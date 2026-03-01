#!/usr/bin/env python3
"""Aggregate benchmark JSONL outputs, generate summary tables and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_latest_files(results_dir: Path, count: int = 3) -> List[Path]:
    files = sorted(results_dir.glob("*_bench.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return list(reversed(files[:count]))


def scenario_name(mode: str, prefill_cu: str, decode_cu: str) -> str:
    if mode == "whole":
        return f"whole:{prefill_cu}"
    return f"split:{prefill_cu}->{decode_cu}"


def ci95(series: pd.Series) -> float:
    n = int(series.shape[0])
    if n <= 1:
        return 0.0
    return float(1.96 * series.std(ddof=1) / np.sqrt(n))


def write_markdown(
    summary: pd.DataFrame,
    computeplan_summary_path: Path,
    files: List[Path],
    out_md: Path,
) -> None:
    with computeplan_summary_path.open("r", encoding="utf-8") as handle:
        cp = json.load(handle)

    prefill = cp.get("prefill", {})
    decode = cp.get("decode", {})

    best_row = summary.sort_values("tokens_per_sec_mean", ascending=False).iloc[0]
    worst_row = summary.sort_values("tokens_per_sec_mean", ascending=True).iloc[0]

    lines: List[str] = []
    lines.append("# Benchmark Analysis")
    lines.append("")
    lines.append("## Inputs")
    for path in files:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Key Findings")
    lines.append(
        f"- Best decode throughput: `{best_row['scenario']}` at {best_row['tokens_per_sec_mean']:.2f} tok/s "
        f"(95% CI +/- {best_row['tokens_per_sec_ci95']:.2f})."
    )
    lines.append(
        f"- Slowest decode throughput: `{worst_row['scenario']}` at {worst_row['tokens_per_sec_mean']:.2f} tok/s."
    )
    lines.append(
        "- Prefill is fastest on configurations using NE, but end-to-end decode throughput depends more on decode-stage placement."
    )
    lines.append(
        f"- Compute plan preference split: prefill has {prefill.get('preferred_device_counts', {}).get('NE', 0)} NE-preferred ops; "
        f"decode has {decode.get('preferred_device_counts', {}).get('GPU', 0)} GPU-preferred ops."
    )
    lines.append("")

    display_cols = [
        "scenario",
        "n_runs",
        "prefill_latency_ms_mean",
        "total_decode_latency_ms_mean",
        "tokens_per_sec_mean",
        "effective_TFLOPS_prefill_mean",
        "effective_TFLOPS_decode_mean",
        "peak_rss_mb_mean",
    ]

    table = summary[display_cols].copy().round(3)
    lines.append("## Aggregated Table")
    lines.append("")
    header = "| " + " | ".join(display_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(display_cols)) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in table.iterrows():
        vals = [str(row[col]) for col in display_cols]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def plot_metrics(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered = summary.sort_values("tokens_per_sec_mean", ascending=False).reset_index(drop=True)
    x = np.arange(len(ordered))
    labels = ordered["scenario"].tolist()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, ordered["tokens_per_sec_mean"], yerr=ordered["tokens_per_sec_ci95"], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Decode tokens/s")
    ax.set_title("Decode Throughput by Scenario (mean +/- 95% CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "decode_tokens_per_sec.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.4
    ax.bar(x - width / 2, ordered["prefill_latency_ms_mean"], width=width, label="Prefill latency (ms)")
    ax.bar(x + width / 2, ordered["total_decode_latency_ms_mean"], width=width, label="Total decode latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Split by Scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_breakdown.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, ordered["peak_rss_mb_mean"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("Memory Footprint by Scenario")
    fig.tight_layout()
    fig.savefig(out_dir / "peak_rss_mb.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--inputs", nargs="*")
    parser.add_argument("--computeplan-summary", default="reports/computeplan_summary.json")
    parser.add_argument("--output-dir", default="reports/analysis")
    args = parser.parse_args()

    results_dir = (ROOT / args.results_dir).resolve()
    if args.inputs:
        files = [(ROOT / item).resolve() for item in args.inputs]
    else:
        files = pick_latest_files(results_dir, count=3)

    if not files:
        raise FileNotFoundError(f"No benchmark JSONL files found under {results_dir}")

    records: List[Dict[str, Any]] = []
    for path in files:
        for row in load_jsonl(path):
            row["source_file"] = str(path)
            records.append(row)

    if not records:
        raise RuntimeError("No records loaded from benchmark files")

    df = pd.DataFrame(records)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise RuntimeError("No successful benchmark rows (status=ok) in selected files")

    df["scenario"] = df.apply(
        lambda r: scenario_name(r["mode"], r["prefill_compute_units"], r["decode_compute_units"]),
        axis=1,
    )

    grouped = (
        df.groupby("scenario", as_index=False)
        .agg(
            n_runs=("scenario", "size"),
            prefill_latency_ms_mean=("prefill_latency_ms", "mean"),
            prefill_latency_ms_std=("prefill_latency_ms", "std"),
            total_decode_latency_ms_mean=("total_decode_latency_ms", "mean"),
            total_decode_latency_ms_std=("total_decode_latency_ms", "std"),
            tokens_per_sec_mean=("tokens_per_sec", "mean"),
            tokens_per_sec_std=("tokens_per_sec", "std"),
            effective_TFLOPS_prefill_mean=("effective_TFLOPS_prefill", "mean"),
            effective_TFLOPS_decode_mean=("effective_TFLOPS_decode", "mean"),
            peak_rss_mb_mean=("peak_rss_mb", "mean"),
        )
        .sort_values("tokens_per_sec_mean", ascending=False)
        .reset_index(drop=True)
    )

    grouped["tokens_per_sec_ci95"] = (
        df.groupby("scenario")["tokens_per_sec"].apply(ci95).reindex(grouped["scenario"]).to_numpy()
    )

    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "latest_summary.csv"
    grouped.to_csv(summary_csv, index=False)

    plot_metrics(grouped, out_dir / "figures")

    analysis_md = out_dir / "latest_summary.md"
    write_markdown(
        summary=grouped,
        computeplan_summary_path=(ROOT / args.computeplan_summary).resolve(),
        files=files,
        out_md=analysis_md,
    )

    print(f"saved: {summary_csv}")
    print(f"saved: {analysis_md}")
    print(f"saved: {out_dir / 'figures' / 'decode_tokens_per_sec.png'}")
    print(f"saved: {out_dir / 'figures' / 'latency_breakdown.png'}")
    print(f"saved: {out_dir / 'figures' / 'peak_rss_mb.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
