#!/usr/bin/env python3
"""Aggregate benchmark JSONL outputs and generate multi-task plots/reports."""

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
LLM_SCENARIO_ORDER = ["CPU", "GPU", "NE", "ALL", "NE→GPU"]
LLM_ALLOWED_SCENARIOS = set(LLM_SCENARIO_ORDER)
SPEECH_SCENARIO_ORDER = ["CPU", "GPU"]
SCENARIO_COLORS = {
    "NE": "#1f77b4",
    "ALL": "#ff7f0e",
    "NE→GPU": "#2ca02c",
    "CPU": "#d62728",
    "GPU": "#9467bd",
    "GPU→NE": "#8c564b",
}


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


def _cu_abbr(cu: Any) -> str:
    m = {
        "CPU_ONLY": "CPU",
        "CPU_AND_GPU": "GPU",
        "CPU_AND_NE": "NE",
        "ALL": "ALL",
        "MPS": "GPU",
    }
    s = str(cu or "")
    return m.get(s, s)


def scenario_label(row: Dict[str, Any]) -> str:
    explicit = row.get("scenario_label")
    if explicit not in (None, ""):
        return str(explicit)

    task_type = str(row.get("task_type") or "llm_decode")
    mode = str(row.get("mode") or "")
    prefill = str(row.get("prefill_compute_units") or "")
    decode = str(row.get("decode_compute_units") or "")

    if task_type == "llm_decode":
        if mode == "whole":
            return _cu_abbr(prefill)
        split_map = {
            ("CPU_AND_NE", "CPU_AND_GPU"): "NE→GPU",
            ("CPU_AND_GPU", "CPU_AND_NE"): "GPU→NE",
        }
        if (prefill, decode) in split_map:
            return split_map[(prefill, decode)]
        if prefill and decode:
            return f"{_cu_abbr(prefill)}→{_cu_abbr(decode)}"
        return "unknown"

    if task_type == "diffusion_sd15":
        vae = str(row.get("vae_compute_units") or "")
        if prefill or decode or vae:
            return f"TE:{_cu_abbr(prefill)}|UN:{_cu_abbr(decode)}|VAE:{_cu_abbr(vae)}"
        return "unknown"

    if task_type == "speech_owsm":
        if prefill == "CPU_AND_GPU" or decode == "CPU_AND_GPU":
            return "GPU"
        if prefill == "CPU_ONLY" or decode == "CPU_ONLY":
            return "CPU"
        return _cu_abbr(prefill or decode or "unknown")

    if prefill and decode:
        return f"{prefill}->{decode}"
    return "unknown"


def infer_x(row: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    x_label = row.get("x_label")
    x_value = row.get("x_value")
    if x_label not in (None, "") and x_value not in (None, ""):
        try:
            return str(x_label), float(x_value)
        except Exception:
            pass

    task_type = str(row.get("task_type") or "llm_decode")
    if task_type == "llm_decode":
        return "context_len", _safe_float(row.get("context_len"))
    if task_type == "diffusion_sd15":
        return "steps", _safe_float(row.get("steps") or row.get("x_value"))
    if task_type == "speech_owsm":
        return "audio_seconds", _safe_float(row.get("audio_seconds") or row.get("x_value"))
    return "x", _safe_float(row.get("x_value"))


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


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
        task_type = str(row.get("task_type") or "llm_decode")

        x_label, x_value = infer_x(row)

        flat = dict(row)
        flat["task_type"] = task_type
        flat["model_id"] = model_id
        flat["model_alias"] = model_alias
        flat["scenario_label"] = scenario_label(row)
        flat["x_label"] = x_label
        flat["x_value"] = x_value
        if task_type == "llm_decode" and str(flat.get("status") or "ok") == "ok":
            if str(flat["scenario_label"]) not in LLM_ALLOWED_SCENARIOS:
                continue

        if "tpot_ms_mean" not in flat or flat.get("tpot_ms_mean") is None:
            flat["tpot_ms_mean"] = decode_stats.get("mean")
        if "tpot_ms_p95" not in flat or flat.get("tpot_ms_p95") is None:
            flat["tpot_ms_p95"] = decode_stats.get("p95")

        if "ttft_ms" not in flat or flat.get("ttft_ms") is None:
            prefill = _safe_float(flat.get("prefill_latency_ms")) or 0.0
            first = _safe_float(flat.get("first_decode_step_ms"))
            if first is not None:
                flat["ttft_ms"] = prefill + first

        if "primary_latency_ms" not in flat or flat.get("primary_latency_ms") is None:
            flat["primary_latency_ms"] = flat.get("ttft_ms")
        if "primary_throughput" not in flat or flat.get("primary_throughput") is None:
            flat["primary_throughput"] = flat.get("tokens_per_sec")

        out.append(flat)

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out)

    numeric_cols = [
        "context_len",
        "prefill_len",
        "x_value",
        "ttft_ms",
        "tokens_per_sec",
        "tpot_ms_mean",
        "peak_rss_mb",
        "effective_TFLOPS_prefill",
        "effective_TFLOPS_decode",
        "primary_latency_ms",
        "primary_throughput",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "status" not in df.columns:
        df["status"] = "ok"

    return df


def summary_from_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    key_cols = [
        "task_type",
        "model_id",
        "model_alias",
        "x_label",
        "x_value",
        "mode",
        "prefill_compute_units",
        "decode_compute_units",
        "scenario_label",
    ]
    for col in key_cols:
        if col not in work.columns:
            work[col] = "NA"
        if col != "x_value":
            work[col] = work[col].fillna("NA").astype(str)
    work["x_value"] = pd.to_numeric(work["x_value"], errors="coerce")

    error_counts = work[work["status"] != "ok"].groupby(key_cols, dropna=False).size().rename("error_count").reset_index()

    ok = work[work["status"] == "ok"].copy()
    if ok.empty:
        merged = error_counts.copy()
        for col in [
            "n_runs",
            "primary_latency_ms_mean",
            "primary_latency_ms_ci95",
            "primary_throughput_mean",
            "primary_throughput_ci95",
            "ttft_ms_mean",
            "tokens_per_sec_mean",
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
            primary_latency_ms_mean=("primary_latency_ms", "mean"),
            primary_latency_ms_std=("primary_latency_ms", "std"),
            primary_throughput_mean=("primary_throughput", "mean"),
            primary_throughput_std=("primary_throughput", "std"),
            ttft_ms_mean=("ttft_ms", "mean"),
            tokens_per_sec_mean=("tokens_per_sec", "mean"),
            tpot_ms_mean_mean=("tpot_ms_mean", "mean"),
            effective_TFLOPS_prefill_mean=("effective_TFLOPS_prefill", "mean"),
            effective_TFLOPS_decode_mean=("effective_TFLOPS_decode", "mean"),
            peak_rss_mb_mean=("peak_rss_mb", "mean"),
        )
        .reset_index()
    )

    lat_ci = ok.groupby(key_cols, dropna=False)["primary_latency_ms"].apply(ci95).rename("primary_latency_ms_ci95").reset_index()
    tput_ci = ok.groupby(key_cols, dropna=False)["primary_throughput"].apply(ci95).rename("primary_throughput_ci95").reset_index()

    merged = grouped.merge(lat_ci, on=key_cols, how="left")
    merged = merged.merge(tput_ci, on=key_cols, how="left")
    merged = merged.merge(error_counts, on=key_cols, how="outer")
    merged["error_count"] = merged["error_count"].fillna(0).astype(int)
    merged["n_runs"] = merged["n_runs"].fillna(0).astype(int)

    merged = merged.sort_values(["task_type", "model_alias", "x_value", "scenario_label"]).reset_index(drop=True)
    return merged


def _ordered_scenarios(part: pd.DataFrame, task_type: str) -> List[str]:
    present = [str(x) for x in part["scenario_label"].dropna().unique().tolist()]
    if task_type == "llm_decode":
        order = [s for s in LLM_SCENARIO_ORDER if s in present]
        tail = sorted([s for s in present if s not in LLM_SCENARIO_ORDER])
        return order + tail
    if task_type == "speech_owsm":
        order = [s for s in SPEECH_SCENARIO_ORDER if s in present]
        tail = sorted([s for s in present if s not in SPEECH_SCENARIO_ORDER])
        return order + tail
    return sorted(present)


def draw_grouped_bars(
    ax: plt.Axes,
    part: pd.DataFrame,
    *,
    x_label: str,
    value_col: str,
    error_col: Optional[str],
    ylabel: str,
    title: str,
    show_legend: bool,
    task_type: str,
) -> None:
    valid = part.dropna(subset=[value_col]).copy() if value_col in part.columns else pd.DataFrame()
    x_vals = sorted([float(x) for x in valid["x_value"].dropna().unique()]) if not valid.empty else []
    scenarios = _ordered_scenarios(valid if not valid.empty else part, task_type)

    if not x_vals or not scenarios:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No successful runs", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        return

    x = np.arange(len(x_vals), dtype=float)
    width = 0.82 / max(1, len(scenarios))

    for idx, scenario in enumerate(scenarios):
        scen = part[part["scenario_label"] == scenario]
        y_vals: List[float] = []
        err_vals: List[float] = []

        for xv in x_vals:
            match = scen[scen["x_value"] == xv]
            if match.empty:
                y_vals.append(np.nan)
                err_vals.append(0.0)
                continue

            row = match.iloc[0]
            y = float(row[value_col]) if pd.notna(row[value_col]) else np.nan
            y_vals.append(y)

            if error_col and error_col in match.columns and pd.notna(row[error_col]):
                err_vals.append(float(row[error_col]))
            else:
                err_vals.append(0.0)

        offset = (idx - (len(scenarios) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            y_vals,
            width=width,
            label=scenario,
            color=SCENARIO_COLORS.get(scenario),
            yerr=err_vals if error_col else None,
            capsize=3 if error_col else 0,
        )

    ax.set_xticks(x)
    # Show integer-like labels as ints.
    labels = [str(int(v)) if abs(v - int(v)) < 1e-9 else f"{v:g}" for v in x_vals]
    ax.set_xticklabels(labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper right", ncol=2)


def _task_model_groups(summary: pd.DataFrame) -> List[Tuple[str, str]]:
    if summary.empty:
        return []
    pairs = summary[["task_type", "model_alias"]].drop_duplicates()
    return [(str(r.task_type), str(r.model_alias)) for r in pairs.itertuples(index=False)]


def save_task_model_figures(summary: pd.DataFrame, out_dir: Path) -> Dict[Tuple[str, str], List[Path]]:
    out_paths: Dict[Tuple[str, str], List[Path]] = {}
    if summary.empty:
        return out_paths

    for task_type, model_alias in _task_model_groups(summary):
        part = summary[(summary["task_type"] == task_type) & (summary["model_alias"] == model_alias)].copy()
        if part.empty:
            continue
        part = part.sort_values(["x_value", "scenario_label"])
        files: List[Path] = []
        x_label = str(part["x_label"].dropna().iloc[0]) if not part["x_label"].dropna().empty else "x"

        base_plots = [
            (
                "primary_latency_ms_mean",
                "primary_latency_ms_ci95",
                "Primary Latency (ms)",
                f"{task_type}/{model_alias} Primary Latency",
                f"fig_{task_type}_{model_alias}_primary_latency_ms.png",
            ),
            (
                "primary_throughput_mean",
                "primary_throughput_ci95",
                "Primary Throughput",
                f"{task_type}/{model_alias} Primary Throughput",
                f"fig_{task_type}_{model_alias}_primary_throughput.png",
            ),
            (
                "peak_rss_mb_mean",
                None,
                "Peak RSS (MB)",
                f"{task_type}/{model_alias} Peak RSS",
                f"fig_{task_type}_{model_alias}_peak_rss_mb.png",
            ),
        ]

        llm_extra = []
        if task_type == "llm_decode":
            llm_extra = [
                (
                    "effective_TFLOPS_prefill_mean",
                    None,
                    "Effective TFLOPS (Prefill)",
                    f"{task_type}/{model_alias} Prefill Effective TFLOPS",
                    f"fig_{task_type}_{model_alias}_tflops_prefill.png",
                ),
                (
                    "effective_TFLOPS_decode_mean",
                    None,
                    "Effective TFLOPS (Decode)",
                    f"{task_type}/{model_alias} Decode Effective TFLOPS",
                    f"fig_{task_type}_{model_alias}_tflops_decode.png",
                ),
            ]

        for value_col, error_col, ylabel, title, filename in base_plots + llm_extra:
            if value_col not in part.columns:
                continue

            if part[value_col].notna().sum() == 0:
                continue

            fig, ax = plt.subplots(figsize=(11, 5))
            draw_grouped_bars(
                ax,
                part,
                x_label=x_label,
                value_col=value_col,
                error_col=error_col,
                ylabel=ylabel,
                title=title,
                show_legend=True,
                task_type=task_type,
            )
            fig.tight_layout()
            output = out_dir / filename
            fig.savefig(output, dpi=170)
            plt.close(fig)
            files.append(output)

        out_paths[(task_type, model_alias)] = files

    return out_paths


def save_combined_figure(summary: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    groups = _task_model_groups(summary)
    if not groups:
        return None

    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, max(4, 4 * n_rows)))
    if n_rows == 1:
        axes = np.array([axes])

    legend_handles = None
    legend_labels = None

    for idx, (task_type, model_alias) in enumerate(groups):
        part = summary[(summary["task_type"] == task_type) & (summary["model_alias"] == model_alias)].copy()
        x_label = str(part["x_label"].dropna().iloc[0]) if not part["x_label"].dropna().empty else "x"

        ax_lat = axes[idx, 0]
        ax_thr = axes[idx, 1]

        draw_grouped_bars(
            ax_lat,
            part,
            x_label=x_label,
            value_col="primary_latency_ms_mean",
            error_col="primary_latency_ms_ci95",
            ylabel="Primary Latency (ms)",
            title=f"{task_type}/{model_alias} Latency",
            show_legend=False,
            task_type=task_type,
        )
        draw_grouped_bars(
            ax_thr,
            part,
            x_label=x_label,
            value_col="primary_throughput_mean",
            error_col="primary_throughput_ci95",
            ylabel="Primary Throughput",
            title=f"{task_type}/{model_alias} Throughput",
            show_legend=False,
            task_type=task_type,
        )

        if legend_handles is None:
            handles, labels = ax_lat.get_legend_handles_labels()
            if handles:
                legend_handles = handles
                legend_labels = labels

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=min(len(legend_labels), 6),
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output = out_dir / "fig_all_task_models_primary_latency_vs_throughput.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def top_k_table(
    summary: pd.DataFrame,
    task_type: str,
    model_alias: str,
    metric_col: str,
    ascending: bool,
    k: int = 3,
) -> pd.DataFrame:
    part = summary[(summary["task_type"] == task_type) & (summary["model_alias"] == model_alias)].copy()
    if part.empty or metric_col not in part.columns:
        return pd.DataFrame(columns=["x_value", "scenario_label", metric_col])
    part = part.dropna(subset=[metric_col]).sort_values(metric_col, ascending=ascending).head(k)
    return part[["x_value", "scenario_label", metric_col]].reset_index(drop=True)


def detect_llm_tradeoff(summary: pd.DataFrame) -> List[str]:
    notes: List[str] = []
    llm = summary[summary["task_type"] == "llm_decode"].copy()
    if llm.empty:
        return notes

    for model_alias, model_df in llm.groupby("model_alias"):
        findings: List[str] = []
        for xv, ctx_df in model_df.groupby("x_value"):
            if "NE→GPU" not in set(ctx_df["scenario_label"]):
                continue
            valid_lat = ctx_df.dropna(subset=["primary_latency_ms_mean"])
            valid_thr = ctx_df.dropna(subset=["primary_throughput_mean"])
            if valid_lat.empty or valid_thr.empty:
                continue

            best_lat = valid_lat.sort_values("primary_latency_ms_mean", ascending=True).iloc[0]["scenario_label"]
            best_thr = valid_thr.sort_values("primary_throughput_mean", ascending=False).iloc[0]["scenario_label"]
            if best_lat == "NE→GPU" and best_thr != "NE→GPU":
                xv_text = int(xv) if pd.notna(xv) and abs(float(xv) - int(xv)) < 1e-9 else xv
                findings.append(f"x={xv_text}: latency winner=NE→GPU, throughput winner={best_thr}")

        if findings:
            notes.append(f"- {model_alias}: " + "; ".join(findings))
        else:
            notes.append(f"- {model_alias}: no NE→GPU latency-vs-throughput rank inversion detected")

    return notes


def write_markdown_report(
    *,
    out_md: Path,
    files: Sequence[Path],
    raw_df: pd.DataFrame,
    summary: pd.DataFrame,
    fig_paths: Dict[Tuple[str, str], List[Path]],
    combined_fig: Optional[Path],
) -> None:
    lines: List[str] = []
    lines.append("# Suite Summary")
    lines.append("")

    lines.append("## What We Ran")
    lines.append("")
    for f in files:
        lines.append(f"- `{f}`")

    if summary.empty:
        lines.append("")
        lines.append("No summary rows available.")
        out_md.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("")
    detected = (
        summary[["task_type", "model_alias", "x_label", "x_value"]]
        .drop_duplicates()
        .sort_values(["task_type", "model_alias", "x_value"])
    )
    for (task_type, model_alias), part in detected.groupby(["task_type", "model_alias"]):
        x_label = str(part["x_label"].dropna().iloc[0]) if not part["x_label"].dropna().empty else "x"
        x_vals = [
            str(int(v)) if abs(float(v) - int(float(v))) < 1e-9 else f"{float(v):g}"
            for v in part["x_value"].dropna().tolist()
        ]
        lines.append(f"- `{task_type}/{model_alias}`: {x_label} = [{', '.join(x_vals)}]")

    lines.append("")
    if combined_fig is not None:
        lines.append(f"![All task/model latency vs throughput]({combined_fig.name})")
        lines.append("")

    for task_type, model_alias in _task_model_groups(summary):
        lines.append(f"## {task_type} / {model_alias}")
        lines.append("")

        top_lat = top_k_table(summary, task_type, model_alias, "primary_latency_ms_mean", ascending=True, k=3)
        lines.append("### Top-3 Fastest Primary Latency")
        if top_lat.empty:
            lines.append("No successful runs.")
        else:
            lines.append("| x_value | scenario_label | primary_latency_ms_mean |")
            lines.append("| --- | --- | --- |")
            for _, row in top_lat.iterrows():
                xv = row["x_value"]
                xv_text = int(xv) if abs(float(xv) - int(float(xv))) < 1e-9 else f"{float(xv):g}"
                lines.append(f"| {xv_text} | {row['scenario_label']} | {row['primary_latency_ms_mean']:.3f} |")

        lines.append("")
        top_thr = top_k_table(summary, task_type, model_alias, "primary_throughput_mean", ascending=False, k=3)
        lines.append("### Top-3 Primary Throughput")
        if top_thr.empty:
            lines.append("No successful runs.")
        else:
            lines.append("| x_value | scenario_label | primary_throughput_mean |")
            lines.append("| --- | --- | --- |")
            for _, row in top_thr.iterrows():
                xv = row["x_value"]
                xv_text = int(xv) if abs(float(xv) - int(float(xv))) < 1e-9 else f"{float(xv):g}"
                lines.append(f"| {xv_text} | {row['scenario_label']} | {row['primary_throughput_mean']:.3f} |")

        lines.append("")
        lines.append("### Figures")
        for p in fig_paths.get((task_type, model_alias), []):
            lines.append(f"![{p.stem}]({p.name})")
        lines.append("")

    lines.append("## LLM Tradeoff Note")
    lines.append("")
    notes = detect_llm_tradeoff(summary)
    if notes:
        lines.extend(notes)
    else:
        lines.append("No LLM rows found for tradeoff check.")

    lines.append("")
    lines.append("## Error Summary")
    lines.append("")
    error_df = raw_df[raw_df["status"] != "ok"].copy() if not raw_df.empty else pd.DataFrame()
    if error_df.empty:
        lines.append("No error records detected.")
    else:
        err_group = (
            error_df.groupby(["task_type", "model_alias", "x_value", "scenario_label"], dropna=False)
            .size()
            .reset_index(name="error_count")
            .sort_values(["task_type", "model_alias", "x_value", "scenario_label"])
        )
        lines.append("| task_type | model_alias | x_value | scenario_label | error_count |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in err_group.iterrows():
            xv = "NA" if pd.isna(row["x_value"]) else (str(int(row["x_value"])) if abs(float(row["x_value"]) - int(float(row["x_value"]))) < 1e-9 else f"{float(row['x_value']):g}")
            lines.append(
                f"| {row['task_type']} | {row['model_alias']} | {xv} | {row['scenario_label']} | {int(row['error_count'])} |"
            )

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
        "task_types": sorted(raw_df["task_type"].dropna().unique().tolist()) if "task_type" in raw_df.columns else [],
        "models": sorted(raw_df["model_alias"].dropna().unique().tolist()),
        "summary_csv": str(summary_csv),
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    fig_paths = save_task_model_figures(summary, out_dir)
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
    for (_, _), paths in fig_paths.items():
        for path in paths:
            print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
