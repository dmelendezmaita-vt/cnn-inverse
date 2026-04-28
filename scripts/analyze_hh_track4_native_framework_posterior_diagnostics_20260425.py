#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
ROOT = REPO / "data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan"
RUNS = ROOT / "runs"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize_interval(npz_path: Path) -> dict:
    arr = np.load(npz_path)
    y_true = np.asarray(arr["test_true"], dtype=np.float32)
    q05 = np.asarray(arr["posterior_q05"], dtype=np.float32)
    q95 = np.asarray(arr["posterior_q95"], dtype=np.float32)
    covered = (y_true >= q05) & (y_true <= q95)
    width = q95 - q05
    return {
        "coverage90_overall": float(np.mean(covered)),
        "coverage90_per_target": np.mean(covered, axis=0).astype(float).tolist(),
        "interval_width_mean": float(np.mean(width)),
        "interval_width_per_target": np.mean(width, axis=0).astype(float).tolist(),
    }


def maybe_load_bayesflow_diag(run_dir: Path) -> dict:
    diag_path = run_dir / "diagnostics.csv"
    if not diag_path.exists():
        return {}
    df = pd.read_csv(diag_path)
    out = {}
    for col in ["NRMSE", "Log Gamma", "Calibration Error", "Posterior Contraction"]:
        if col in df.columns:
            out[col] = float(df[col].mean())
    return out


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    specs = [
        ("swyft_native_meanstd", RUNS / "swyft_native_v100_20260425"),
        ("swyft_native_concat_light", RUNS / "swyft_native_concat_small_v100_20260425"),
        ("swyft_tmnre_style_native", RUNS / "swyft_tmnre_native_v100_20260425"),
        ("bayesflow_native_concat", RUNS / "bayesflow_native_v100_20260425"),
        ("bayesflow_native_meanstd", RUNS / "bayesflow_native_meanstd_v100_20260425"),
    ]

    rows = []
    for name, run_dir in specs:
        metrics_path = run_dir / "metrics_summary.json"
        pred_path = run_dir / "predictions_test.npz"
        if not metrics_path.exists() or not pred_path.exists():
            continue
        metrics = load_json(metrics_path)
        interval = summarize_interval(pred_path)
        diag = maybe_load_bayesflow_diag(run_dir)
        test_metrics = metrics["test_metrics"] if "test_metrics" in metrics else metrics.get("metrics_test", {})
        rows.append(
            {
                "representative": name,
                "framework": metrics.get("framework"),
                "aggregation": metrics.get("aggregation"),
                "mae": test_metrics.get("mae"),
                "mse": test_metrics.get("mse"),
                "r2": test_metrics.get("r2"),
                "coverage90_overall": interval["coverage90_overall"],
                "interval_width_mean": interval["interval_width_mean"],
                "bayesflow_nrmse_mean": diag.get("NRMSE"),
                "bayesflow_log_gamma_mean": diag.get("Log Gamma"),
                "bayesflow_calibration_error_mean": diag.get("Calibration Error"),
                "bayesflow_posterior_contraction_mean": diag.get("Posterior Contraction"),
                "source": str(metrics_path),
            }
        )

    out_csv = TABLES / "hh_track4_native_framework_posterior_diagnostics_20260425.csv"
    if rows:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    report_path = REPORTS / "hh_track4_native_framework_posterior_diagnostics_20260425.md"
    lines = [
        "# HH Track4 Native Framework Posterior Diagnostics",
        "",
        f"- summary table: `{out_csv}`" if rows else "- no rows available",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| representative | MAE | MSE | R2 | coverage90_overall | interval_width_mean |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['representative']}` | {row['mae']:.4f} | {row['mse']:.4f} | {row['r2']:.4f} | "
                f"{row['coverage90_overall']:.4f} | {row['interval_width_mean']:.4f} |"
            )
        lines.append("")
        bayesflow_rows = [r for r in rows if str(r["framework"]).startswith("bayesflow")]
        if bayesflow_rows:
            lines.append("BayesFlow diagnostic columns are mean values from the exported `diagnostics.csv` files.")
            lines.append("")

    report_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
