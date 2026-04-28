#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SURR = REPO / "data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_active_rows() -> list[dict[str, object]]:
    rows = []
    for manifest_path in sorted((SURR / "runs/active_sequential").glob("*/active_sequential_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        aggregate = manifest.get("aggregate_metrics", {})
        rows.append(
            {
                "run_name": manifest_path.parent.name,
                "acquisition_policy": manifest.get("acquisition_policy", ""),
                "mean_abs_log10_error_params_mean": aggregate.get("mean_abs_log10_error_params_mean"),
                "mean_relative_error_params_mean": aggregate.get("mean_relative_error_params_mean"),
                "posterior_contraction_mean_mean": aggregate.get("posterior_contraction_mean_mean"),
                "ess_mean": aggregate.get("ess_mean"),
                "runtime_sec": manifest.get("runtime_sec"),
                "source": str(manifest_path),
            }
        )
    return rows


def load_asnpe_rows() -> list[dict[str, object]]:
    rows = []
    for manifest_path in sorted((SURR / "runs/active_sequential").glob("*/asnpe_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        aggregate = manifest.get("aggregate_metrics", {})
        rows.append(
            {
                "run_name": manifest_path.parent.name,
                "acquisition_policy": "asnpe_style",
                "mean_abs_log10_error_params_mean": aggregate.get("mean_abs_log10_error_params_mean"),
                "mean_relative_error_params_mean": aggregate.get("mean_relative_error_params_mean"),
                "posterior_contraction_mean_mean": aggregate.get("posterior_std_norm_mean_mean"),
                "ess_mean": "",
                "runtime_sec": manifest.get("runtime_sec"),
                "source": str(manifest_path),
            }
        )
    return rows


def main() -> None:
    rows = load_active_rows() + load_asnpe_rows()
    rows.sort(key=lambda r: (float(r["mean_abs_log10_error_params_mean"]), float(r["runtime_sec"])))
    out_csv = SURR / "tables" / "hh_track4_active_sequential_policy_comparison_20260425.csv"
    write_csv(out_csv, rows)
    report = [
        "# HH Track4 Active Sequential Policy Comparison",
        "",
        f"- summary csv: `{out_csv}`",
        "",
        "| run | policy | mean_abs_log10_error_params_mean | mean_relative_error_params_mean | runtime_sec |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        ess = row["mean_relative_error_params_mean"]
        report.append(
            f"| `{row['run_name']}` | `{row['acquisition_policy']}` | {float(row['mean_abs_log10_error_params_mean']):.6f} | {float(ess):.6f} | {float(row['runtime_sec']):.6f} |"
        )
    (SURR / "reports" / "hh_track4_active_sequential_policy_comparison_20260425.md").write_text("\n".join(report) + "\n")
    print(out_csv)


if __name__ == "__main__":
    main()
