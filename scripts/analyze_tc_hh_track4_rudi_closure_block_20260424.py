#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

DATE_TAG = "20260424_tc_hh_track4_rudi_closure_block"

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

REGISTRY_CSV = TABLES / f"tc_hh_track4_rudi_closure_block_registry_{DATE_TAG}.csv"
SUMMARY_CSV = TABLES / f"tc_hh_track4_rudi_closure_block_summary_{DATE_TAG}.csv"
SUMMARY_MD = NOTES / f"tc_hh_track4_rudi_closure_block_summary_{DATE_TAG}.md"


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def metrics_path(run_root: Path) -> Path | None:
    candidates = sorted(run_root.glob("*/metrics_summary.csv"))
    if candidates:
        return candidates[0]
    candidate = run_root / "metrics_summary.csv"
    return candidate if candidate.exists() else None


def read_test_metrics(path: Path) -> Dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("split") == "test":
            return row
    raise ValueError(f"No test row in {path}")


def main() -> None:
    registry = load_csv(REGISTRY_CSV)
    if not registry:
        raise SystemExit(f"No registry rows found at {REGISTRY_CSV}")

    summary_rows: List[Dict[str, str]] = []
    for row in registry:
        run_root = Path(row["run_output_root"])
        metrics = {"mse": "", "mae": "", "r2": ""}
        metric_file = metrics_path(run_root)
        if metric_file is not None:
            try:
                metrics = read_test_metrics(metric_file)
            except Exception:
                pass

        summary_rows.append(
            {
                "row_id": row["row_id"],
                "strategy_id": row["strategy_id"],
                "seed": row["seed"],
                "status": row["status"],
                "return_code": row["return_code"],
                "elapsed_sec": row["elapsed_sec"],
                "test_mse": metrics.get("mse", ""),
                "test_mae": metrics.get("mae", ""),
                "test_r2": metrics.get("r2", ""),
                "run_output_root": row["run_output_root"],
                "stderr_log": row["stderr_log"],
            }
        )

    fieldnames = list(summary_rows[0].keys())
    write_csv(SUMMARY_CSV, summary_rows, fieldnames)

    completed = [r for r in summary_rows if r["status"] == "COMPLETED" and r["test_mae"]]
    running = [r for r in summary_rows if r["status"] == "RUNNING"]
    failed = [r for r in summary_rows if r["status"] == "FAILED"]
    notes: List[str] = [
        f"# TC HH Track4 Rudi Closure Block Summary ({DATE_TAG})",
        "",
        f"- registry: `{REGISTRY_CSV}`",
        f"- summary table: `{SUMMARY_CSV}`",
        f"- rows total: `{len(summary_rows)}`",
        f"- running: `{len(running)}`",
        f"- completed: `{len(completed)}`",
        f"- failed: `{len(failed)}`",
    ]

    if completed:
        best = min(completed, key=lambda r: float(r["test_mae"]))
        notes.extend(
            [
                "",
                "## Best Completed Row",
                f"- `strategy_id={best['strategy_id']}`",
                f"- `seed={best['seed']}`",
                f"- `test_mae={best['test_mae']}`",
                f"- `test_mse={best['test_mse']}`",
                f"- `test_r2={best['test_r2']}`",
            ]
        )

    if failed:
        notes.extend(
            [
                "",
                "## Failed Rows",
                *[
                    f"- `{r['row_id']} {r['strategy_id']} seed={r['seed']} return_code={r['return_code']} stderr={r['stderr_log']}`"
                    for r in failed
                ],
            ]
        )

    SUMMARY_MD.write_text("\n".join(notes) + "\n")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
