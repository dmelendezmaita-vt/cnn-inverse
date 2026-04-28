#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SURR = REPO / "data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle"


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_report(registry_path: Path, report_path: Path) -> None:
    rows = load_csv(registry_path)
    lines = [
        "# HH Track4 Assumption-Conditioned Backlog Queue",
        "",
        f"- registry: `{registry_path}`",
        "",
    ]
    if not rows:
        lines.append("No rows recorded.")
        report_path.write_text("\n".join(lines) + "\n")
        return
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "")
        status_counts[status] = status_counts.get(status, 0) + 1
    lines.append("## Status Counts")
    lines.append("")
    for key in sorted(status_counts):
        lines.append(f"- `{key}`: `{status_counts[key]}`")
    lines.append("")
    lines.extend(
        [
            "| row_id | label | status | node | started_at | ended_at | return_code |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row.get('row_id','')}` | `{row.get('label','')}` | `{row.get('status','')}` | `{row.get('node','')}` | "
            f"`{row.get('started_at','')}` | `{row.get('ended_at','')}` | `{row.get('return_code','')}` |"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    tables = SURR / "tables"
    reports = SURR / "reports"
    for registry_name in [
        "hh_track4_remaining_backlog_registry_20260425.csv",
        "hh_track4_remaining_backlog_round2_registry_20260425.csv",
    ]:
        registry_path = tables / registry_name
        report_path = reports / registry_name.replace(".csv", ".md")
        write_report(registry_path, report_path)
        print(report_path)


if __name__ == "__main__":
    main()
