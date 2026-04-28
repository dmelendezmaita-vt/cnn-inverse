#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Refresh the A30-local hybrid initializer comparison table from clean campaign direct-fitting outputs."
    )
    ap.add_argument("--campaign-root", required=True)
    return ap.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def initializer_label(summary: dict[str, object], run_name: str) -> str:
    if str(summary.get("init_mode")) == "midpoint":
        return "midpoint"
    pred = summary.get("prediction_file")
    if pred:
        return Path(str(pred)).stem
    return run_name


def campaign_tag(campaign_root: Path) -> str:
    name = campaign_root.name
    return name[len("optimization_track_") :] if name.startswith("optimization_track_") else name


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    tag = campaign_tag(campaign_root)
    runs_root = campaign_root / "runs" / "phaseSURR_direct_fitting"
    rows: list[dict[str, object]] = []
    for manifest_path in sorted(runs_root.glob("*/hybrid_refinement_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        summary = manifest.get("summary", {})
        rows.append(
            {
                "run_name": manifest_path.parent.name,
                "initializer": initializer_label(summary, manifest_path.parent.name),
                "objective": float(summary.get("objective", float("nan"))),
                "mean_trace_rmse_z": float(summary.get("mean_trace_rmse_z", float("nan"))),
                "mean_summary_rel_l1": float(summary.get("mean_summary_rel_l1", float("nan"))),
                "mean_range_rel_error": float(summary.get("mean_range_rel_error", float("nan"))),
                "mean_mean_rel_error": float(summary.get("mean_mean_rel_error", float("nan"))),
                "mean_std_rel_error": float(summary.get("mean_std_rel_error", float("nan"))),
                "current_gain": float(summary.get("current_gain", float("nan"))),
                "source": str(manifest_path),
            }
        )

    rows.sort(key=lambda r: (r["objective"], r["mean_trace_rmse_z"]))
    tables_dir = campaign_root / "tables"
    reports_dir = campaign_root / "reports"
    out_csv = tables_dir / f"hh_track4_a30_hybrid_initializer_comparison_20260427_{tag}.csv"
    out_md = reports_dir / f"hh_track4_a30_hybrid_initializer_comparison_20260427_{tag}.md"
    write_csv(out_csv, rows)

    report_lines = [
        "# HH Track4 A30 Hybrid Initializer Comparison",
        "",
        f"- summary csv: `{out_csv}`",
        "",
        "| rank | run | initializer | objective | mean_trace_rmse_z | mean_summary_rel_l1 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        report_lines.append(
            f"| {idx} | `{row['run_name']}` | `{row['initializer']}` | {row['objective']:.6f} | {row['mean_trace_rmse_z']:.6f} | {row['mean_summary_rel_l1']:.6f} |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(report_lines) + "\n")
    print(out_csv)


if __name__ == "__main__":
    main()
