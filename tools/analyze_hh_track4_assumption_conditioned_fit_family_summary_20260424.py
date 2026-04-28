#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNS_ROOT = (
    REPO
    / "data"
    / "important_notes"
    / "optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_fit_search"
    / "runs"
)
OUT_ROOT = (
    REPO
    / "data"
    / "important_notes"
    / "optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_fit_search"
    / "reports"
)
OUT_TABLES = (
    REPO
    / "data"
    / "important_notes"
    / "optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_fit_search"
    / "tables"
)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def maybe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def maybe_load_csv(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def classify_run(name: str) -> tuple[str, str]:
    objective_mode = "feature_only" if "featureonly" in name else "hybrid_trace_feature"
    optimizer_family = "adam_bptt" if name.startswith("bptt_") else "differential_evolution"
    return optimizer_family, objective_mode


def summarize_run(run_dir: Path) -> dict[str, object] | None:
    name = run_dir.name
    optimizer_family, objective_mode = classify_run(name)
    de_manifest = maybe_load_json(run_dir / "assumption_conditioned_compact_hh_fit_search_manifest.json")
    bptt_manifest = maybe_load_json(run_dir / "assumption_conditioned_compact_hh_bptt_fit_manifest.json")
    if de_manifest is None and bptt_manifest is None:
        return None

    row: dict[str, object] = {
        "run_name": name,
        "optimizer_family": optimizer_family,
        "objective_mode": objective_mode,
        "status": "completed",
        "pulse_window": "",
        "support_variant": "narrow" if "narrow" in name else "broad",
        "trace_length": "",
        "current_gain_bounds": "",
        "trace_weight": "",
        "mean_objective": "",
        "best_objective": "",
        "mean_trace_rmse_z": "",
        "best_trace_rmse_z": "",
        "mean_summary_rel_l1": "",
        "best_summary_rel_l1": "",
        "notes": "",
    }

    if de_manifest is not None:
        agg = de_manifest.get("aggregate_metrics", {})
        row.update(
            {
                "pulse_window": f"{de_manifest.get('pulse_start_frac','')} - {de_manifest.get('pulse_end_frac','')}",
                "trace_length": de_manifest.get("trace_length", ""),
                "current_gain_bounds": de_manifest.get("current_gain_bounds", ""),
                "trace_weight": de_manifest.get("trace_weight", 1.0),
                "mean_objective": agg.get("mean_objective", ""),
                "best_objective": agg.get("best_objective", ""),
                "mean_trace_rmse_z": agg.get("mean_trace_rmse_z", ""),
                "mean_summary_rel_l1": agg.get("mean_summary_rel_l1", ""),
                "notes": "differential_evolution aggregate metrics",
            }
        )
        summary_rows = maybe_load_csv(run_dir / "assumption_conditioned_compact_hh_fit_search_summary.csv") or []
        if summary_rows:
            best = min(summary_rows, key=lambda r: float(r["objective"]))
            row["best_trace_rmse_z"] = best.get("mean_trace_rmse_z", "")
            row["best_summary_rel_l1"] = best.get("mean_summary_rel_l1", "")

    if bptt_manifest is not None:
        best = bptt_manifest.get("best", {})
        row.update(
            {
                "pulse_window": f"{bptt_manifest.get('pulse_start_frac','')} - {bptt_manifest.get('pulse_end_frac','')}",
                "trace_length": bptt_manifest.get("trace_length", ""),
                "current_gain_bounds": bptt_manifest.get("current_gain_bounds", ""),
                "trace_weight": bptt_manifest.get("trace_weight", 1.0),
                "best_objective": best.get("loss", ""),
                "best_trace_rmse_z": best.get("trace_rmse_z", ""),
                "best_summary_rel_l1": best.get("summary_rel_l1", ""),
                "notes": "adam_bptt best-step metrics",
            }
        )
    return row


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in sorted(p for p in RUNS_ROOT.iterdir() if p.is_dir()):
        row = summarize_run(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No completed assumption-conditioned fit-family runs found.")
        return

    fieldnames = list(rows[0].keys())
    table_path = OUT_TABLES / "hh_track4_assumption_conditioned_fit_family_summary_20260424.csv"
    write_csv(table_path, rows, fieldnames)

    lines = [
        "# HH Track4 Assumption-Conditioned Fit Family Summary",
        "",
        "This table collects completed fit-family runs under the assumption-conditioned compact-HH lane.",
        "",
        f"- source runs root: `{RUNS_ROOT}`",
        f"- table: `{table_path}`",
        "",
        "## Runs",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['run_name']}` | `{row['optimizer_family']}` | `{row['objective_mode']}` | best objective `{row['best_objective']}` | best trace RMSE z `{row['best_trace_rmse_z']}`"
        )
    report_path = OUT_ROOT / "hh_track4_assumption_conditioned_fit_family_summary_20260424.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {table_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
