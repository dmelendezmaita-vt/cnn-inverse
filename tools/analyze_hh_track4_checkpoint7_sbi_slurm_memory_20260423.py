#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
EFF_ROOT = IMPORTANT / "optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency"
TABLES = EFF_ROOT / "tables"

SBI_SEED_ROWS = TABLES / "hh_track4_checkpoint7_standardized_efficiency_sbi_seed_rows_20260423.csv"
FINAL_REPS = TABLES / "hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv"
MEMORY_AUDIT = TABLES / "hh_track4_checkpoint7_sbi_slurm_memory_audit_20260423.csv"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def extract_row_id(metrics_path: str) -> str:
    parts = Path(metrics_path).parts
    for part in parts:
        if part.startswith("v100cp"):
            return part
    raise ValueError(f"Could not infer row_id from metrics_path={metrics_path}")


def registry_rows_for(row_id: str) -> list[dict[str, str]]:
    matches = []
    for registry in IMPORTANT.glob("optimization_track_2026042*_v100_hh_track4_checkpoint2_*/tables/*registry*.csv"):
        for row in load_rows(registry):
            if row.get("row_id") == row_id:
                out = dict(row)
                out["_registry"] = str(registry)
                matches.append(out)
    return matches


def parse_step_id(stderr_log: str) -> str:
    path = Path(stderr_log)
    if not path.is_absolute():
        path = REPO / path
    text = path.read_text(errors="ignore")
    match = re.search(r"StepId=(\d+\.\d+)", text)
    if not match:
        return ""
    return match.group(1)


def parse_metrics_written_time(stderr_log: str) -> str:
    path = Path(stderr_log)
    if not path.is_absolute():
        path = REPO / path
    text = path.read_text(errors="ignore")
    matches = re.findall(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+ INFO Wrote metrics summary", text)
    if not matches:
        return ""
    date, time = matches[-1]
    return f"{date}T{time}"


def parse_kib(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    match = re.fullmatch(r"([0-9.]+)([KMGTP]?)", value)
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2)
    scale = {
        "": 1.0 / 1024.0,
        "K": 1.0 / 1024.0,
        "M": 1.0,
        "G": 1024.0,
        "T": 1024.0 * 1024.0,
        "P": 1024.0 * 1024.0 * 1024.0,
    }[suffix]
    return number * scale


def sacct_for_jobs(job_ids: set[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    result: dict[str, dict[str, str]] = {}
    for job_id in sorted(job_ids):
        cmd = [
            "ssh",
            "falcon",
            (
                "sacct "
                f"-j {job_id} "
                "--format=JobID,JobName%40,State,Start,End,Elapsed,MaxRSS,AveRSS,ReqMem,NodeList,AllocTRES%80 "
                "-P"
            ),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            continue
        header = lines[0].split("|")
        for line in lines[1:]:
            values = line.split("|")
            row = dict(zip(header, values))
            result[row["JobID"]] = row
    return result


def parse_sacct_time(value: str) -> datetime | None:
    if not value or value == "Unknown":
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def main() -> None:
    seed_rows = load_rows(SBI_SEED_ROWS)
    enriched: list[dict[str, object]] = []
    job_ids: set[str] = set()

    for seed_row in seed_rows:
        row_id = extract_row_id(seed_row["metrics_path"])
        matches = registry_rows_for(row_id)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one registry row for {row_id}, found {len(matches)}")
        registry_row = matches[0]
        step_id = parse_step_id(registry_row["stderr_log"])
        metrics_written_at = parse_metrics_written_time(registry_row["stderr_log"])
        job_ids.add(registry_row["alloc_job_id"])
        enriched.append(
            {
                **seed_row,
                "row_id": row_id,
                "alloc_job_id": registry_row["alloc_job_id"],
                "assigned_nodes": registry_row["assigned_nodes"],
                "stderr_log": registry_row["stderr_log"],
                "slurm_step_id": step_id,
                "metrics_written_at": metrics_written_at,
                "registry": registry_row["_registry"],
            }
        )

    sacct_rows = sacct_for_jobs(job_ids)
    sacct_steps = [
        row
        for row in sacct_rows.values()
        if "." in row.get("JobID", "") and row.get("State") == "COMPLETED"
    ]
    sacct_by_end: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sacct_steps:
        end_time = row.get("End", "")
        if end_time and end_time != "Unknown":
            sacct_by_end[end_time].append(row)

    audit_rows: list[dict[str, object]] = []
    for row in enriched:
        step_id = str(row["slurm_step_id"])
        match_method = "stderr_step_id" if step_id else ""
        sacct_row = sacct_rows.get(step_id, {})
        if not sacct_row:
            candidates = sacct_by_end.get(str(row["metrics_written_at"]), [])
            candidates = [
                c for c in candidates if c.get("NodeList", "") == str(row["assigned_nodes"])
            ]
            if len(candidates) == 1:
                sacct_row = candidates[0]
                step_id = sacct_row.get("JobID", "")
                match_method = "metrics_written_timestamp"
            elif len(candidates) > 1:
                candidates = [c for c in candidates if c.get("State") == "COMPLETED"]
                if len(candidates) == 1:
                    sacct_row = candidates[0]
                    step_id = sacct_row.get("JobID", "")
                    match_method = "metrics_written_timestamp_completed_filter"
        if not sacct_row:
            target_time = parse_sacct_time(str(row["metrics_written_at"]))
            if target_time is not None:
                candidates = [
                    c
                    for c in sacct_steps
                    if c.get("NodeList", "") == str(row["assigned_nodes"])
                    and parse_sacct_time(c.get("End", "")) is not None
                ]
                close = []
                for cand in candidates:
                    end_time = parse_sacct_time(cand.get("End", ""))
                    assert end_time is not None
                    delta = abs((end_time - target_time).total_seconds())
                    if delta <= 3:
                        close.append((delta, cand))
                close.sort(key=lambda item: (item[0], item[1].get("JobID", "")))
                if close:
                    sacct_row = close[0][1]
                    step_id = sacct_row.get("JobID", "")
                    match_method = f"metrics_written_timestamp_nearest_{close[0][0]:.0f}s"

        max_rss_mb = parse_kib(sacct_row.get("MaxRSS", ""))
        ave_rss_mb = parse_kib(sacct_row.get("AveRSS", ""))
        audit_rows.append(
            {
                "representative": row["representative"],
                "row_id": row["row_id"],
                "seed": row["seed"],
                "method": row["method"],
                "eval_baseline_drift_std": row["eval_baseline_drift_std"],
                "eval_mask_fraction": row["eval_mask_fraction"],
                "slurm_step_id": step_id,
                "slurm_match_method": match_method,
                "metrics_written_at": row["metrics_written_at"],
                "assigned_nodes": row["assigned_nodes"],
                "max_rss_mb": "" if max_rss_mb is None else max_rss_mb,
                "ave_rss_mb": "" if ave_rss_mb is None else ave_rss_mb,
                "sacct_end": sacct_row.get("End", ""),
                "sacct_nodelist": sacct_row.get("NodeList", ""),
                "sacct_elapsed": sacct_row.get("Elapsed", ""),
                "sacct_state": sacct_row.get("State", ""),
                "alloc_tres": sacct_row.get("AllocTRES", ""),
                "metrics_path": row["metrics_path"],
            }
        )

    write_rows(
        MEMORY_AUDIT,
        audit_rows,
        [
            "representative",
            "row_id",
            "seed",
            "method",
            "eval_baseline_drift_std",
            "eval_mask_fraction",
            "slurm_step_id",
            "slurm_match_method",
            "metrics_written_at",
            "assigned_nodes",
            "max_rss_mb",
            "ave_rss_mb",
            "sacct_end",
            "sacct_nodelist",
            "sacct_elapsed",
            "sacct_state",
            "alloc_tres",
            "metrics_path",
        ],
    )

    rss_by_rep: dict[str, list[float]] = defaultdict(list)
    for row in audit_rows:
        if row["max_rss_mb"] != "":
            rss_by_rep[str(row["representative"])].append(float(row["max_rss_mb"]))

    final_rows = load_rows(FINAL_REPS)
    for row in final_rows:
        rep = row["representative"]
        if rep in rss_by_rep:
            row["peak_rss_mb_mean"] = f"{mean(rss_by_rep[rep]):.6f}"
            note = row.get("notes", "")
            row["notes"] = (
                note.replace(
                    "peak memory was not logged for these SBI rows",
                    "host peak RSS joined from Slurm step accounting; GPU VRAM was not logged",
                )
                if note
                else "Host peak RSS joined from Slurm step accounting; GPU VRAM was not logged."
            )

    write_rows(FINAL_REPS, final_rows, list(final_rows[0].keys()))

    print(f"Wrote {MEMORY_AUDIT}")
    print(f"Updated {FINAL_REPS}")
    for rep in sorted(rss_by_rep):
        print(f"{rep}: mean_max_rss_mb={mean(rss_by_rep[rep]):.3f} n={len(rss_by_rep[rep])}")


if __name__ == "__main__":
    main()
