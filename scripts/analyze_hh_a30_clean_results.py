#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "outputs" / "hh_a30_canonical"
DATE_TAG = "20260428"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a final clean-results analysis package from the completed A30 campaign.")
    ap.add_argument("--campaign-root", default=str(DEFAULT_ROOT))
    return ap.parse_args()


def safe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    try:
        out = float(str(value).strip())
    except Exception:
        return None
    return out if math.isfinite(out) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def metric_value(obj: dict[str, Any], key: str, split: str = "test") -> float | None:
    value = obj.get(key)
    if isinstance(value, dict):
        for candidate in (split, "test", "validate", "train", "value"):
            if candidate in value:
                parsed = safe_float(value[candidate])
                if parsed is not None:
                    return parsed
        return None
    return safe_float(value)


def metric_from_summary(metrics_obj: dict[str, Any], key: str) -> float | None:
    direct = metric_value(metrics_obj, key)
    if direct is not None:
        return direct
    nested = metrics_obj.get("test_metrics", {})
    if isinstance(nested, dict):
        nested_value = metric_value(nested, key)
        if nested_value is not None:
            return nested_value
    return None


def task_run_root(task: dict[str, Any]) -> Path | None:
    raw_paths = task.get("output_paths", [])
    if not isinstance(raw_paths, list) or not raw_paths:
        return None
    for raw in raw_paths:
        path = Path(str(raw))
        if not path.suffix:
            return path
    return Path(str(raw_paths[0])).parent


def find_first(root: Path, name: str) -> Path | None:
    if not root.exists():
        return None
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def resource_summary_for_task(task: dict[str, Any]) -> dict[str, Any] | None:
    run_root = task_run_root(task)
    if run_root is None:
        return None
    resource_dir = Path(str(task.get("resource_dir", ""))) if task.get("resource_dir") else None
    candidates = []
    if resource_dir:
        candidates.append(resource_dir / "resource_summary.json")
    candidates.extend(
        [
            run_root / "resource_monitor" / str(task["task_id"]) / "resource_summary.json",
            run_root / "resource_monitor" / "resource_summary.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    return None


def infer_budget_label(run_name: str) -> str:
    lowered = run_name.lower()
    if "smoke" in lowered:
        return "smoke"
    if "extended" in lowered:
        return "extended"
    if "primary" in lowered:
        return "primary"
    if "secondary" in lowered:
        return "secondary"
    if "full" in lowered or "default" in lowered:
        return "full"
    return "other"


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    tables = campaign_root / "tables"
    reports = campaign_root / "reports"
    manifest = json.loads((tables / "hh_track4_a30_literal_rerun_manifest_20260427.json").read_text())
    tasks = {str(t["task_id"]): t for t in manifest["tasks"]}
    registry = {row["task_id"]: row for row in read_csv(tables / "hh_track4_a30_literal_rerun_20260427_registry.csv")}

    if any(row["status"] != "COMPLETED" for row in registry.values()):
        raise SystemExit("The clean campaign is not fully completed yet.")

    # Direct task aggregation from run artifacts
    direct_rows: list[dict[str, object]] = []
    direct_summary_by_strategy: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for task_id, task in tasks.items():
        phase = str(task["phase"])
        if not phase.startswith("phaseA") or phase.startswith("phaseANALYSIS"):
            continue
        run_root = task_run_root(task)
        if run_root is None:
            continue
        metrics_path = find_first(run_root, "metrics_summary.json")
        if metrics_path is None:
            continue
        metrics = json.loads(metrics_path.read_text())
        mae = metric_from_summary(metrics, "mae")
        mse = metric_from_summary(metrics, "mse")
        r2 = metric_from_summary(metrics, "r2")
        runtime = safe_float((resource_summary_for_task(task) or {}).get("elapsed_sec"))
        strategy_id = str(task.get("strategy_id", ""))
        entry = {
            "task_id": task_id,
            "phase": phase,
            "strategy_id": strategy_id,
            "baseline_name": str(task.get("baseline_name", "")),
            "manifest_category": str(task.get("manifest_category", "")),
            "mae": mae,
            "mse": mse,
            "r2": r2,
            "runtime_sec": runtime,
            "metrics_path": str(metrics_path),
        }
        direct_rows.append(entry)
        direct_summary_by_strategy[(phase, strategy_id)].append(entry)

    direct_rows.sort(key=lambda row: (row["phase"], safe_float(row["mae"]) or float("inf"), row["task_id"]))
    direct_summary_rows: list[dict[str, object]] = []
    for (phase, strategy_id), items in sorted(direct_summary_by_strategy.items()):
        maes = [float(item["mae"]) for item in items if item["mae"] is not None]
        r2s = [float(item["r2"]) for item in items if item["r2"] is not None]
        runtimes = [float(item["runtime_sec"]) for item in items if item["runtime_sec"] is not None]
        direct_summary_rows.append(
            {
                "phase": phase,
                "strategy_id": strategy_id,
                "n_runs": len(items),
                "mean_mae": st.mean(maes) if maes else "",
                "std_mae": st.pstdev(maes) if len(maes) > 1 else 0.0 if maes else "",
                "mean_r2": st.mean(r2s) if r2s else "",
                "mean_runtime_sec": st.mean(runtimes) if runtimes else "",
                "task_ids": ",".join(item["task_id"] for item in items),
            }
        )

    # Active-design ladder table from refreshed clean A30 literature table
    active_table = read_csv(tables / "hh_track4_a30_literature_active_policy_comparison_20260427_20260427_hh_track4_a30_literal_rerun.csv")
    active_rows: list[dict[str, object]] = []
    for row in active_table:
        active_rows.append(
            {
                "run_name": row["run_name"],
                "task_id": row["task_id"],
                "acquisition_policy": row["acquisition_policy"] or "legacy_default",
                "budget_label": infer_budget_label(row["run_name"]),
                "mean_abs_log10_error_params_mean": safe_float(row["mean_abs_log10_error_params_mean"]),
                "mean_relative_error_params_mean": safe_float(row["mean_relative_error_params_mean"]),
                "posterior_contraction_mean_mean": safe_float(row["posterior_contraction_mean_mean"]),
                "ess_mean": safe_float(row["ess_mean"]),
                "runtime_sec": safe_float(row["runtime_sec"]),
                "source": row["source"],
            }
        )
    active_rows.sort(key=lambda row: (str(row["acquisition_policy"]), str(row["budget_label"]), safe_float(row["mean_abs_log10_error_params_mean"]) or float("inf")))

    # Framework / decision / hybrid tables
    framework_rows = read_csv(tables / "hh_track4_a30_literature_framework_snapshot_20260427_20260427_hh_track4_a30_literal_rerun.csv")
    decision_rows = read_csv(tables / "hh_track4_a30_literature_posterior_decision_rules_20260427_20260427_hh_track4_a30_literal_rerun.csv")
    hybrid_rows = read_csv(tables / "hh_track4_a30_hybrid_initializer_comparison_20260427_20260427_hh_track4_a30_literal_rerun.csv")
    method_rows = read_csv(tables / "hh_track4_a30_literature_method_comparison_20260427_20260427_hh_track4_a30_literal_rerun.csv")

    # Resource summary by phase
    phase_resource_rows: list[dict[str, object]] = []
    phase_resource = defaultdict(lambda: {"elapsed": [], "cpu_pct": [], "rss_gb": [], "gpu_samples": [], "gpu_util_max": [], "count": 0})
    for task_id, task in tasks.items():
        summary = resource_summary_for_task(task)
        if summary is None:
            continue
        phase = str(task["phase"])
        time_metrics = summary.get("time_metrics", {})
        gpu_metrics = summary.get("gpu_metrics", {})
        pct = str(time_metrics.get("Percent of CPU this job got", "")).strip().rstrip("%")
        rss = safe_float(time_metrics.get("Maximum resident set size (kbytes)"))
        elapsed = safe_float(summary.get("elapsed_sec"))
        gpu_sample_count = safe_float(gpu_metrics.get("gpu_sample_count"))
        per_gpu = gpu_metrics.get("per_gpu", []) if isinstance(gpu_metrics.get("per_gpu"), list) else []
        gpu_util_max = max((safe_float(item.get("utilization_gpu_max")) or 0.0) for item in per_gpu) if per_gpu else None
        bucket = phase_resource[phase]
        bucket["count"] += 1
        if pct:
            bucket["cpu_pct"].append(float(pct))
        if rss is not None:
            bucket["rss_gb"].append(rss / 1024 / 1024)
        if elapsed is not None:
            bucket["elapsed"].append(elapsed)
        if gpu_sample_count is not None:
            bucket["gpu_samples"].append(gpu_sample_count)
        if gpu_util_max is not None:
            bucket["gpu_util_max"].append(gpu_util_max)

    for phase, data in sorted(phase_resource.items()):
        phase_resource_rows.append(
            {
                "phase": phase,
                "n_completed": data["count"],
                "elapsed_sec_mean": st.mean(data["elapsed"]) if data["elapsed"] else "",
                "cpu_pct_mean": st.mean(data["cpu_pct"]) if data["cpu_pct"] else "",
                "cpu_pct_max": max(data["cpu_pct"]) if data["cpu_pct"] else "",
                "rss_gb_mean": st.mean(data["rss_gb"]) if data["rss_gb"] else "",
                "rss_gb_max": max(data["rss_gb"]) if data["rss_gb"] else "",
                "gpu_sample_count_median": st.median(data["gpu_samples"]) if data["gpu_samples"] else "",
                "gpu_util_max_mean": st.mean(data["gpu_util_max"]) if data["gpu_util_max"] else "",
            }
        )

    # Key findings
    direct_best = min(
        (row for row in direct_summary_rows if row["mean_mae"] != ""),
        key=lambda row: float(row["mean_mae"]),
    )
    active_best = min(active_rows, key=lambda row: float(row["mean_abs_log10_error_params_mean"]))
    heuristic_rows = [row for row in active_rows if row["acquisition_policy"] != "asnpe_style"]
    heuristic_best = min(heuristic_rows, key=lambda row: float(row["mean_abs_log10_error_params_mean"]))
    asnpe_rows = [row for row in active_rows if row["acquisition_policy"] == "asnpe_style"]
    asnpe_best = min(asnpe_rows, key=lambda row: float(row["mean_abs_log10_error_params_mean"]))
    framework_best = min(framework_rows, key=lambda row: float(row["mae"]))
    comparable_decision_rows = [
        row
        for row in decision_rows
        if str(row.get("comparability_class", "")) != "upstream_hh_benchmark_not_directly_comparable"
    ]
    decision_best = min((comparable_decision_rows or decision_rows), key=lambda row: float(row["mae"]))
    simformer_decision_rows = [row for row in decision_rows if str(row.get("comparability_class", "")) == "upstream_hh_benchmark_not_directly_comparable"]
    simformer_decision_best = min(simformer_decision_rows, key=lambda row: float(row["mae"])) if simformer_decision_rows else None
    hybrid_best = min(hybrid_rows, key=lambda row: float(row["objective"]))

    key_rows = [
        {
            "section": "direct_clean",
            "name": direct_best["strategy_id"],
            "metric_name": "mean_mae",
            "metric_value": direct_best["mean_mae"],
            "note": direct_best["phase"],
        },
        {
            "section": "aligned_framework",
            "name": framework_best["run_name"],
            "metric_name": "mae",
            "metric_value": framework_best["mae"],
            "note": framework_best["framework"],
        },
        {
            "section": "decision_rule",
            "name": f"{decision_best['run_name']}::{decision_best['decision_rule']}",
            "metric_name": "mae",
            "metric_value": decision_best["mae"],
            "note": decision_best["family"],
        },
        *(
            [
                {
                    "section": "decision_rule_noncomparable_upstream",
                    "name": f"{simformer_decision_best['run_name']}::{simformer_decision_best['decision_rule']}",
                    "metric_name": "mae",
                    "metric_value": simformer_decision_best["mae"],
                    "note": "upstream_hh_benchmark_not_directly_comparable",
                }
            ]
            if simformer_decision_best is not None
            else []
        ),
        {
            "section": "active_policy_heuristic",
            "name": heuristic_best["run_name"],
            "metric_name": "mean_abs_log10_error_params_mean",
            "metric_value": heuristic_best["mean_abs_log10_error_params_mean"],
            "note": heuristic_best["acquisition_policy"],
        },
        {
            "section": "active_policy_asnpe",
            "name": asnpe_best["run_name"],
            "metric_name": "mean_abs_log10_error_params_mean",
            "metric_value": asnpe_best["mean_abs_log10_error_params_mean"],
            "note": "asnpe_style",
        },
        {
            "section": "hybrid_initializer",
            "name": hybrid_best["run_name"],
            "metric_name": "objective",
            "metric_value": hybrid_best["objective"],
            "note": hybrid_best["initializer"],
        },
    ]

    # Write outputs
    out_methods = tables / f"hh_track4_a30_clean_methods_{DATE_TAG}.csv"
    out_direct = tables / f"hh_track4_a30_clean_direct_summary_{DATE_TAG}.csv"
    out_active = tables / f"hh_track4_a30_clean_active_policy_ladder_{DATE_TAG}.csv"
    out_resource = tables / f"hh_track4_a30_clean_resource_phase_summary_{DATE_TAG}.csv"
    out_key = tables / f"hh_track4_a30_clean_key_findings_{DATE_TAG}.csv"
    out_report = reports / f"hh_track4_a30_clean_results_analysis_{DATE_TAG}.md"

    methods_rows = [
        {"item": "campaign_root", "value": str(campaign_root), "note": "fresh homogeneous A30 campaign"},
        {"item": "completed_tasks", "value": len(registry), "note": "registry closure"},
        {"item": "direct_task_count", "value": sum(1 for t in tasks.values() if str(t['phase']).startswith('phaseA') and not str(t['phase']).startswith('phaseANALYSIS')), "note": "direct task coverage"},
        {"item": "framework_rows", "value": len(framework_rows), "note": "A30-local aligned framework rows"},
        {"item": "decision_rows", "value": len(decision_rows), "note": "A30-local decision-rule rows"},
        {"item": "hybrid_rows", "value": len(hybrid_rows), "note": "A30-local hybrid initializer rows"},
        {"item": "active_rows", "value": len(active_rows), "note": "A30-local active-policy rows, including ASNPE"},
    ]

    write_csv(out_methods, methods_rows)
    write_csv(out_direct, direct_summary_rows)
    write_csv(out_active, active_rows)
    write_csv(out_resource, phase_resource_rows)
    write_csv(out_key, key_rows)

    report_lines = [
        "# HH Track4 Clean A30 Results Analysis",
        "",
        "## Methods",
        "",
        f"- campaign root: `{campaign_root}`",
        f"- task count completed under the clean homogeneous campaign: `{len(registry)}`",
        f"- direct task phases completed: `{sum(1 for t in tasks.values() if str(t['phase']).startswith('phaseA') and not str(t['phase']).startswith('phaseANALYSIS'))}`",
        "- all result interpretation below is based on the fresh A30-local campaign root, its manifest and registry, task-local `metrics_summary.json` artifacts, task-local `resource_summary.json` sidecars, and the clean A30-local literature, decision-rule, hybrid, and active-policy tables regenerated inside that root",
        "",
        "## Coverage",
        "",
        f"- direct summary rows: `{len(direct_summary_rows)}`",
        f"- active-policy rows: `{len(active_rows)}`",
        f"- aligned framework rows: `{len(framework_rows)}`",
        f"- decision-rule rows: `{len(decision_rows)}`",
        f"- hybrid initializer rows: `{len(hybrid_rows)}`",
        "",
        "## Findings",
        "",
        f"- best direct strategy by clean mean MAE: `{direct_best['strategy_id']}` with `{float(direct_best['mean_mae']):.6f}` in `{direct_best['phase']}`",
        f"- best aligned framework row: `{framework_best['run_name']}` with `MAE {float(framework_best['mae']):.6f}` and `R2 {float(framework_best['r2']):.6f}`",
        f"- best comparable posterior decision-rule row: `{decision_best['run_name']}::{decision_best['decision_rule']}` with `MAE {float(decision_best['mae']):.6f}`",
        f"- best hybrid initializer row: `{hybrid_best['run_name']}` with `objective {float(hybrid_best['objective']):.6f}`",
        f"- best heuristic active-design row: `{heuristic_best['run_name']}` with `mean_abs_log10_error_params_mean {float(heuristic_best['mean_abs_log10_error_params_mean']):.6f}` under policy `{heuristic_best['acquisition_policy']}`",
        f"- best ASNPE-style row: `{asnpe_best['run_name']}` with `mean_abs_log10_error_params_mean {float(asnpe_best['mean_abs_log10_error_params_mean']):.6f}`",
        *(
            [
                f"- best non-comparable upstream Simformer decision-rule row: `{simformer_decision_best['run_name']}::{simformer_decision_best['decision_rule']}` with `MAE {float(simformer_decision_best['mae']):.6f}`"
            ]
            if simformer_decision_best is not None
            else []
        ),
        "",
        "## Active Design Ladder",
        "",
        "| run | policy | budget | mean_abs_log10_error_params_mean | runtime_sec |",
        "|---|---|---|---:|---:|",
    ]
    for row in active_rows:
        report_lines.append(
            f"| `{row['run_name']}` | `{row['acquisition_policy']}` | `{row['budget_label']}` | {float(row['mean_abs_log10_error_params_mean']):.6f} | {float(row['runtime_sec']):.6f} |"
        )

    report_lines.extend(
        [
            "",
            "## Resource Findings",
            "",
            "| phase | n_completed | elapsed_sec_mean | cpu_pct_mean | rss_gb_max | gpu_util_max_mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in phase_resource_rows:
        report_lines.append(
            f"| `{row['phase']}` | {row['n_completed']} | "
            f"{float(row['elapsed_sec_mean']):.3f} | "
            f"{float(row['cpu_pct_mean']) if row['cpu_pct_mean'] != '' else 0.0:.3f} | "
            f"{float(row['rss_gb_max']) if row['rss_gb_max'] != '' else 0.0:.3f} | "
            f"{float(row['gpu_util_max_mean']) if row['gpu_util_max_mean'] != '' else 0.0:.3f} |"
        )

    report_lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- direct fixed-array rows, aligned posterior rows, surrogate direct-fitting rows, active-design rows, and upstream Simformer rows should not be pooled into one numerical leaderboard without comparability labels",
            "- the active-design ladder is broader than before, but each rung is still represented by a single clean A30 rerun, so budget-by-seed variance is not fully characterized",
            "- the ASNPE rows use their own manifest schema, and the clean A30 analysis normalizes them into the active-policy table using top-level posterior error fields rather than a shared aggregate-metrics block",
            "",
            "## Output Files",
            "",
            f"- `{out_methods}`",
            f"- `{out_direct}`",
            f"- `{out_active}`",
            f"- `{out_resource}`",
            f"- `{out_key}`",
            f"- `{out_report}`",
        ]
    )

    write_text(out_report, "\n".join(report_lines) + "\n")
    print(out_report)


if __name__ == "__main__":
    main()
