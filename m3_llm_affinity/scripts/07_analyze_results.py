#!/usr/bin/env python3
"""Aggregate benchmark JSONL outputs and generate suite-level plots/reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from lib_paths import default_model_alias

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ORDER = ["CPU", "GPU", "NE", "ALL", "NE→GPU", "GPU→NE"]


METRIC_COLUMNS = [
    "ttft_ms",
    "tokens_per_sec",
    "tpot_ms_mean",
    "peak_rss_mb",
    "effective_TFLOPS_prefill",
    "effective_TFLOPS_decode",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pick_latest_files(results_dir: Path, count: int) -> List[Path]:
    files = sorted(results_dir.rglob("*_bench.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    selected = files[: int(count)]
    return list(reversed(selected))


def scenario_label(mode: Any, prefill_cu: Any, decode_cu: Any) -> str:
    mode_s = str(mode or "")
    prefill_s = str(prefill_cu or "")
    decode_s = str(decode_cu or "")

    whole_map = {
        "CPU_ONLY": "CPU",
        "CPU_AND_GPU": "GPU",
        "CPU_AND_NE": "NE",
        "ALL": "ALL",
    }

    if mode_s == "whole":
        return whole_map.get(prefill_s, prefill_s)

    split_map = {
        ("CPU_AND_NE", "CPU_AND_GPU"): "NE→GPU",
        ("CPU_AND_GPU", "CPU_AND_NE"): "GPU→NE",
    }
    if (prefill_s, decode_s) in split_map:
        return split_map[(prefill_s, decode_s)]

    if prefill_s and decode_s:
        return f"{prefill_s}->{decode_s}"
    return "unknown"


def ci95(values: pd.Series) -> float:
    s = values.dropna().astype(float)
    n = int(s.shape[0])
    if n <= 1:
        return 0.0
    return float(1.96 * s.std(ddof=1) / np.sqrt(n))


def flatten_rows(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    for row in rows:
        decode_stats = row.get("decode_step_latency_ms_stats") or {}
        model_id = str(row.get("model_id") or "unknown_model")
        model_alias = str(row.get("model_alias") or default_model_alias(model_id))

        flat = dict(row)
        flat["model_id"] = model_id
        flat["model_alias"] = model_alias
        flat["scenario_label"] = scenario_label(
            row.get("mode"),
            row.get("prefill_compute_units"),
            row.get("decode_compute_units"),
        )

        if "tpot_ms_mean" not in flat or flat.get("tpot_ms_mean") is None:
            flat["tpot_ms_mean"] = decode_stats.get("mean")
        if "tpot_ms_p95" not in flat or flat.get("tpot_ms_p95") is None:
            flat["tpot_ms_p95"] = decode_stats.get("p95")

        flat["decode_step_latency_ms_stats_mean"] = decode_stats.get("mean")
        flat["decode_step_latency_ms_stats_median"] = decode_stats.get("median")
        flat["decode_step_latency_ms_stats_p95"] = decode_stats.get("p95")

        out.append(flat)

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out)
    for col in ("context_len", "prefill_len"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in METRIC_COLUMNS + ["ttft_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def summary_from_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    str_key_cols = [
        "model_id",
        "model_alias",
        "mode",
        "prefill_compute_units",
        "decode_compute_units",
        "scenario_label",
    ]
    for col in str_key_cols:
        work[col] = work[col].fillna("NA").astype(str)
    work["context_len"] = pd.to_numeric(work["context_len"], errors="coerce").astype("Int64")

    key_cols = [
        "model_id",
        "model_alias",
        "context_len",
        "mode",
        "prefill_compute_units",
        "decode_compute_units",
        "scenario_label",
    ]

    error_counts = work[work["status"] != "ok"].groupby(key_cols, dropna=False).size().rename("error_count").reset_index()

    ok = work[work["status"] == "ok"].copy()
    if ok.empty:
        merged = error_counts.copy()
        for col in [
            "n_runs",
            "ttft_ms_mean",
            "ttft_ms_ci95",
            "tokens_per_sec_mean",
            "tokens_per_sec_ci95",
            "tpot_ms_mean_mean",
            "effective_TFLOPS_prefill_mean",
            "effective_TFLOPS_decode_mean",
            "peak_rss_mb_mean",
        ]:
            merged[col] = np.nan
        return merged

    grouped = (
        ok.groupby(key_cols, dropna=False)
        .agg(
            n_runs=("status", "size"),
            ttft_ms_mean=("ttft_ms", "mean"),
            ttft_ms_std=("ttft_ms", "std"),
            tokens_per_sec_mean=("tokens_per_sec", "mean"),
            tokens_per_sec_std=("tokens_per_sec", "std"),
            tpot_ms_mean_mean=("tpot_ms_mean", "mean"),
            tpot_ms_mean_std=("tpot_ms_mean", "std"),
            effective_TFLOPS_prefill_mean=("effective_TFLOPS_prefill", "mean"),
            effective_TFLOPS_prefill_std=("effective_TFLOPS_prefill", "std"),
            effective_TFLOPS_decode_mean=("effective_TFLOPS_decode", "mean"),
            effective_TFLOPS_decode_std=("effective_TFLOPS_decode", "std"),
            peak_rss_mb_mean=("peak_rss_mb", "mean"),
            peak_rss_mb_std=("peak_rss_mb", "std"),
        )
        .reset_index()
    )

    ttft_ci = ok.groupby(key_cols, dropna=False)["ttft_ms"].apply(ci95).rename("ttft_ms_ci95").reset_index()
    tps_ci = (
        ok.groupby(key_cols, dropna=False)["tokens_per_sec"].apply(ci95).rename("tokens_per_sec_ci95").reset_index()
    )

    merged = grouped.merge(ttft_ci, on=key_cols, how="left")
    merged = merged.merge(tps_ci, on=key_cols, how="left")
    merged = merged.merge(error_counts, on=key_cols, how="outer")
    merged["error_count"] = merged["error_count"].fillna(0).astype(int)
    if "n_runs" in merged.columns:
        merged["n_runs"] = merged["n_runs"].fillna(0).astype(int)

    merged["scenario_order"] = merged["scenario_label"].apply(
        lambda x: SCENARIO_ORDER.index(x) if x in SCENARIO_ORDER else len(SCENARIO_ORDER)
    )
    merged = merged.sort_values(["model_alias", "context_len", "scenario_order", "scenario_label"]).reset_index(drop=True)
    merged = merged.drop(columns=["scenario_order"])
    return merged


def _ordered_scenarios(model_df: pd.DataFrame) -> List[str]:
    present = list(model_df["scenario_label"].dropna().unique())
    ordered = [s for s in SCENARIO_ORDER if s in present]
    tail = sorted([s for s in present if s not in SCENARIO_ORDER])
    return ordered + tail


def draw_grouped_bars(
    ax: plt.Axes,
    model_df: pd.DataFrame,
    *,
    value_col: str,
    error_col: Optional[str],
    ylabel: str,
    title: str,
    show_legend: bool,
) -> None:
    contexts = sorted([int(x) for x in model_df["context_len"].dropna().unique()])
    scenarios = _ordered_scenarios(model_df)

    if not contexts or not scenarios:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No successful runs", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        return

    x = np.arange(len(contexts), dtype=float)
    width = 0.82 / max(1, len(scenarios))

    for idx, scenario in enumerate(scenarios):
        part = model_df[model_df["scenario_label"] == scenario]
        y_vals: List[float] = []
        err_vals: List[float] = []
        for ctx in contexts:
            match = part[part["context_len"] == ctx]
            if match.empty:
                y_vals.append(np.nan)
                err_vals.append(0.0)
            else:
                y = float(match.iloc[0][value_col]) if pd.notna(match.iloc[0][value_col]) else np.nan
                y_vals.append(y)
                if error_col and error_col in match.columns and pd.notna(match.iloc[0][error_col]):
                    err_vals.append(float(match.iloc[0][error_col]))
                else:
                    err_vals.append(0.0)

        offset = (idx - (len(scenarios) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            y_vals,
            width=width,
            label=scenario,
            yerr=err_vals if error_col else None,
            capsize=3 if error_col else 0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in contexts])
    ax.set_xlabel("Context Length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper right", ncol=2)


def save_model_figures(summary: pd.DataFrame, out_dir: Path) -> Dict[str, List[Path]]:
    out_paths: Dict[str, List[Path]] = {}
    if summary.empty:
        return out_paths

    for model_alias, model_df in summary.groupby("model_alias"):
        model_df = model_df.sort_values(["context_len", "scenario_label"])
        files: List[Path] = []

        plots = [
            (
                "ttft_ms_mean",
                "ttft_ms_ci95",
                "TTFT (ms)",
                f"{model_alias} TTFT by Context and Scenario",
                f"fig_{model_alias}_ttft_ms.png",
            ),
            (
                "tokens_per_sec_mean",
                "tokens_per_sec_ci95",
                "Decode Tokens/s",
                f"{model_alias} Throughput by Context and Scenario",
                f"fig_{model_alias}_tokens_per_sec.png",
            ),
            (
                "effective_TFLOPS_prefill_mean",
                None,
                "Effective TFLOPS (Prefill)",
                f"{model_alias} Prefill Effective TFLOPS",
                f"fig_{model_alias}_tflops_prefill.png",
            ),
            (
                "effective_TFLOPS_decode_mean",
                None,
                "Effective TFLOPS (Decode)",
                f"{model_alias} Decode Effective TFLOPS",
                f"fig_{model_alias}_tflops_decode.png",
            ),
            (
                "peak_rss_mb_mean",
                None,
                "Peak RSS (MB)",
                f"{model_alias} Peak RSS by Context and Scenario",
                f"fig_{model_alias}_peak_rss_mb.png",
            ),
        ]

        for value_col, error_col, ylabel, title, filename in plots:
            fig, ax = plt.subplots(figsize=(11, 5))
            draw_grouped_bars(
                ax,
                model_df,
                value_col=value_col,
                error_col=error_col,
                ylabel=ylabel,
                title=title,
                show_legend=True,
            )
            fig.tight_layout()
            output = out_dir / filename
            fig.savefig(output, dpi=170)
            plt.close(fig)
            files.append(output)

        out_paths[str(model_alias)] = files

    return out_paths


def save_combined_figure(summary: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    if summary.empty:
        return None

    models = list(summary["model_alias"].dropna().unique())
    if not models:
        return None

    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, max(4, 4 * n_rows)))
    if n_rows == 1:
        axes = np.array([axes])

    legend_handles = None
    legend_labels = None

    for idx, model_alias in enumerate(models):
        row_df = summary[summary["model_alias"] == model_alias]
        ax_ttft = axes[idx, 0]
        ax_tps = axes[idx, 1]

        draw_grouped_bars(
            ax_ttft,
            row_df,
            value_col="ttft_ms_mean",
            error_col="ttft_ms_ci95",
            ylabel="TTFT (ms)",
            title=f"{model_alias} TTFT",
            show_legend=False,
        )
        draw_grouped_bars(
            ax_tps,
            row_df,
            value_col="tokens_per_sec_mean",
            error_col="tokens_per_sec_ci95",
            ylabel="Decode Tokens/s",
            title=f"{model_alias} Throughput",
            show_legend=False,
        )

        if legend_handles is None:
            handles, labels = ax_ttft.get_legend_handles_labels()
            if handles:
                legend_handles = handles
                legend_labels = labels

    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc="upper right", ncol=min(len(legend_labels), 6))

    fig.tight_layout(rect=[0, 0, 0.97, 1])
    output = out_dir / "fig_all_models_ttft_vs_throughput.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def top_k_table(df: pd.DataFrame, model_alias: str, metric_col: str, ascending: bool, k: int = 3) -> pd.DataFrame:
    part = df[df["model_alias"] == model_alias].copy()
    if part.empty:
        return pd.DataFrame(columns=["context_len", "scenario_label", metric_col])
    part = part.sort_values(metric_col, ascending=ascending).head(k)
    return part[["context_len", "scenario_label", metric_col]].reset_index(drop=True)


def detect_tradeoff(summary: pd.DataFrame) -> List[str]:
    notes: List[str] = []
    if summary.empty:
        return notes

    for model_alias, model_df in summary.groupby("model_alias"):
        findings: List[str] = []
        for ctx, ctx_df in model_df.groupby("context_len"):
            if "NE→GPU" not in set(ctx_df["scenario_label"]):
                continue
            valid_ttft = ctx_df.dropna(subset=["ttft_ms_mean"])
            valid_tps = ctx_df.dropna(subset=["tokens_per_sec_mean"])
            if valid_ttft.empty or valid_tps.empty:
                continue

            best_ttft = valid_ttft.sort_values("ttft_ms_mean", ascending=True).iloc[0]["scenario_label"]
            best_tps = valid_tps.sort_values("tokens_per_sec_mean", ascending=False).iloc[0]["scenario_label"]
            if best_ttft == "NE→GPU" and best_tps != "NE→GPU":
                findings.append(f"ctx {int(ctx)}: TTFT winner=NE→GPU, throughput winner={best_tps}")

        if findings:
            notes.append(f"- {model_alias}: " + "; ".join(findings))
        else:
            notes.append(f"- {model_alias}: no NE→GPU TTFT-vs-throughput rank inversion detected")

    return notes


def write_markdown_report(
    *,
    out_md: Path,
    files: Sequence[Path],
    raw_df: pd.DataFrame,
    summary: pd.DataFrame,
    fig_paths: Dict[str, List[Path]],
    combined_fig: Optional[Path],
) -> None:
    lines: List[str] = []
    lines.append("# Suite Summary")
    lines.append("")

    lines.append("## What We Ran")
    lines.append("")
    for f in files:
        lines.append(f"- `{f}`")

    models = sorted(summary["model_alias"].dropna().unique().tolist()) if not summary.empty else []
    contexts = sorted([int(x) for x in summary["context_len"].dropna().unique().tolist()]) if not summary.empty else []
    lines.append("")
    lines.append(f"Detected models: `{', '.join(models) if models else 'none'}`")
    lines.append("")
    lines.append(f"Detected context lengths: `{', '.join(str(c) for c in contexts) if contexts else 'none'}`")
    lines.append("")

    if combined_fig is not None:
        lines.append(f"![All models TTFT vs Throughput]({combined_fig.name})")
        lines.append("")

    lines.append("## Top-3 Fastest TTFT Per Model")
    lines.append("")
    for model_alias in models:
        top = top_k_table(summary, model_alias, "ttft_ms_mean", ascending=True, k=3)
        lines.append(f"### {model_alias}")
        if top.empty:
            lines.append("No successful runs.")
            lines.append("")
            continue
        lines.append("| context_len | scenario_label | ttft_ms_mean |")
        lines.append("| --- | --- | --- |")
        for _, row in top.iterrows():
            lines.append(f"| {int(row['context_len'])} | {row['scenario_label']} | {row['ttft_ms_mean']:.3f} |")
        lines.append("")

    lines.append("## Top-3 Throughput Per Model")
    lines.append("")
    for model_alias in models:
        top = top_k_table(summary, model_alias, "tokens_per_sec_mean", ascending=False, k=3)
        lines.append(f"### {model_alias}")
        if top.empty:
            lines.append("No successful runs.")
            lines.append("")
            continue
        lines.append("| context_len | scenario_label | tokens_per_sec_mean |")
        lines.append("| --- | --- | --- |")
        for _, row in top.iterrows():
            lines.append(f"| {int(row['context_len'])} | {row['scenario_label']} | {row['tokens_per_sec_mean']:.3f} |")
        lines.append("")

    lines.append("## Tradeoff Note")
    lines.append("")
    for line in detect_tradeoff(summary):
        lines.append(line)
    lines.append("")

    lines.append("## Error Summary")
    lines.append("")
    error_df = raw_df[raw_df["status"] != "ok"].copy() if not raw_df.empty else pd.DataFrame()
    if error_df.empty:
        lines.append("No error records detected.")
    else:
        err_group = (
            error_df.groupby(["model_alias", "context_len", "scenario_label"], dropna=False)
            .size()
            .reset_index(name="error_count")
            .sort_values(["model_alias", "context_len", "scenario_label"])
        )
        lines.append("| model_alias | context_len | scenario_label | error_count |")
        lines.append("| --- | --- | --- | --- |")
        for _, row in err_group.iterrows():
            ctx = "NA" if pd.isna(row["context_len"]) else int(row["context_len"])
            lines.append(
                f"| {row['model_alias']} | {ctx} | {row['scenario_label']} | {int(row['error_count'])} |"
            )

    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for model_alias in models:
        lines.append(f"### {model_alias}")
        for p in fig_paths.get(model_alias, []):
            lines.append(f"![{p.stem}]({p.name})")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--suite-config", default="configs/suite.yaml")
    parser.add_argument("--pick-latest-n", type=int)
    parser.add_argument("--inputs", nargs="*")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    suite_cfg = {}
    suite_cfg_path = (ROOT / args.suite_config).resolve()
    if suite_cfg_path.exists():
        suite_cfg = load_yaml(suite_cfg_path).get("suite", {})

    results_dir = (ROOT / args.results_dir).resolve()
    if args.inputs:
        files = [((ROOT / item).resolve() if not Path(item).is_absolute() else Path(item).resolve()) for item in args.inputs]
    else:
        pick_n = int(args.pick_latest_n if args.pick_latest_n is not None else suite_cfg.get("pick_latest_n_jsonl", 50))
        files = pick_latest_files(results_dir, count=pick_n)

    if not files:
        raise FileNotFoundError(f"No benchmark JSONL files found under {results_dir}")

    rows: List[Dict[str, Any]] = []
    for path in files:
        for row in load_jsonl(path):
            row["source_file"] = str(path)
            rows.append(row)

    if not rows:
        raise RuntimeError("No rows were loaded from selected JSONL files")

    raw_df = flatten_rows(rows)
    if raw_df.empty:
        raise RuntimeError("No usable rows in selected JSONL files")

    summary = summary_from_df(raw_df)

    reports_dir = args.output_dir or suite_cfg.get("reports_dir", "reports/analysis")
    out_dir = (ROOT / str(reports_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "latest_summary.csv"
    summary_json = out_dir / "latest_summary.json"
    summary_md = out_dir / "latest_summary.md"

    summary.to_csv(summary_csv, index=False)

    payload = {
        "inputs": [str(p) for p in files],
        "rows_loaded": int(raw_df.shape[0]),
        "rows_ok": int((raw_df["status"] == "ok").sum()),
        "rows_error": int((raw_df["status"] != "ok").sum()),
        "models": sorted(raw_df["model_alias"].dropna().unique().tolist()),
        "contexts": sorted([int(x) for x in raw_df["context_len"].dropna().unique().tolist()]),
        "summary_csv": str(summary_csv),
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    fig_paths = save_model_figures(summary, out_dir)
    combined_fig = save_combined_figure(summary, out_dir)

    write_markdown_report(
        out_md=summary_md,
        files=files,
        raw_df=raw_df,
        summary=summary,
        fig_paths=fig_paths,
        combined_fig=combined_fig,
    )

    print(f"saved: {summary_csv}")
    print(f"saved: {summary_json}")
    print(f"saved: {summary_md}")
    if combined_fig is not None:
        print(f"saved: {combined_fig}")
    for model_alias, paths in fig_paths.items():
        for path in paths:
            print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
