#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DATE_TAG = "20260418_v100_hh_default_policy_stability"
REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

REGISTRY_CSV = TABLES / f"v100_hh_default_policy_stability_registry_{DATE_TAG}.csv"
SUMMARY_JSON = NOTES / f"v100_hh_default_policy_stability_summary_{DATE_TAG}.json"
SUMMARY_MD = NOTES / f"v100_hh_default_policy_stability_summary_{DATE_TAG}.md"


def load_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def read_runtime(info_log: Path, label: str):
    if not info_log.exists():
        return None
    txt = info_log.read_text(errors="ignore")
    mt = re.search(rf"Runtime - {label} \[sec\]:\s+([0-9.eE+-]+)", txt)
    return float(mt.group(1)) if mt else None


def median_or_none(vals):
    vals = list(vals)
    return statistics.median(vals) if vals else None


def main():
    rows = load_csv(REGISTRY_CSV)
    if not rows:
        raise SystemExit(f"Missing or empty registry: {REGISTRY_CSV}")

    completed = [row for row in rows if row["status"] == "COMPLETED"]
    state_counts = Counter(row["status"] for row in rows)

    enriched = []
    for row in completed:
        run_root = Path(row["run_output_root"]) / f"{row['alloc_job_id']}_{row['row_id']}"
        metrics = run_root / "metrics_summary.json"
        info = run_root / "run_dnn" / "rank0_info.log"
        if not metrics.exists():
            continue
        js = json.loads(metrics.read_text())
        enriched.append(
            {
                **row,
                "train_sec": read_runtime(info, "train"),
                "eval_sec": read_runtime(info, "evaluate") or read_runtime(info, "eval"),
                "test_r2": float(js["r2"]["test"]),
                "test_mse": float(js["mse"]["test"]),
            }
        )

    grouped = defaultdict(list)
    dedicated = defaultdict(list)
    for row in enriched:
        key = (row["track_key"], row["strategy_id"])
        if row["launch_group"]:
            grouped[key].append(row)
        else:
            dedicated[key].append(row)

    group_summary = []
    for key, vals in sorted(grouped.items()):
        track_key, strategy_id = key
        start = min(parse_iso(v["started_at"]) for v in vals)
        end = max(parse_iso(v["ended_at"]) for v in vals)
        wallclock_sec = (end - start).total_seconds()
        group_summary.append(
            {
                "track_key": track_key,
                "strategy_id": strategy_id,
                "mode": "concurrent_4x1n",
                "rows_completed": len(vals),
                "wallclock_sec": wallclock_sec,
                "completed_runs_per_hour": 3600.0 * len(vals) / wallclock_sec if wallclock_sec > 0 else None,
                "median_elapsed_sec": median_or_none(float(v["elapsed_sec"]) for v in vals if v["elapsed_sec"]),
                "median_train_sec": median_or_none(v["train_sec"] for v in vals if v["train_sec"] is not None),
                "median_eval_sec": median_or_none(v["eval_sec"] for v in vals if v["eval_sec"] is not None),
                "median_test_r2": median_or_none(v["test_r2"] for v in vals),
                "median_test_mse": median_or_none(v["test_mse"] for v in vals),
            }
        )

    dedicated_summary = []
    for key, vals in sorted(dedicated.items()):
        track_key, strategy_id = key
        dedicated_summary.append(
            {
                "track_key": track_key,
                "strategy_id": strategy_id,
                "mode": "dedicated",
                "rows_completed": len(vals),
                "median_elapsed_sec": median_or_none(float(v["elapsed_sec"]) for v in vals if v["elapsed_sec"]),
                "median_train_sec": median_or_none(v["train_sec"] for v in vals if v["train_sec"] is not None),
                "median_eval_sec": median_or_none(v["eval_sec"] for v in vals if v["eval_sec"] is not None),
                "median_test_r2": median_or_none(v["test_r2"] for v in vals),
                "median_test_mse": median_or_none(v["test_mse"] for v in vals),
            }
        )

    summary = {
        "date_tag": DATE_TAG,
        "registry_csv": str(REGISTRY_CSV),
        "status_counts": dict(state_counts),
        "completed_rows": len(completed),
        "group_summary": group_summary,
        "dedicated_summary": dedicated_summary,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        f"# V100 HH Default Policy Stability Summary ({DATE_TAG})",
        "",
        f"- registry: `{REGISTRY_CSV}`",
        f"- status counts: `{dict(state_counts)}`",
        f"- completed rows: `{len(completed)}`",
        "",
        "## Grouped Defaults",
        "",
        "| track | strategy | completed | wallclock (s) | runs/hour | median elapsed (s) | median train (s) | median eval (s) | median test R2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in group_summary:
        md.append(
            f"| {row['track_key']} | {row['strategy_id']} | {row['rows_completed']} | "
            f"{round(row['wallclock_sec'], 3)} | "
            f"{'' if row['completed_runs_per_hour'] is None else round(row['completed_runs_per_hour'], 3)} | "
            f"{'' if row['median_elapsed_sec'] is None else round(row['median_elapsed_sec'], 3)} | "
            f"{'' if row['median_train_sec'] is None else round(row['median_train_sec'], 3)} | "
            f"{'' if row['median_eval_sec'] is None else round(row['median_eval_sec'], 3)} | "
            f"{'' if row['median_test_r2'] is None else round(row['median_test_r2'], 6)} |"
        )
    md.extend(
        [
            "",
            "## Targeted References",
            "",
            "| track | strategy | completed | median elapsed (s) | median train (s) | median eval (s) | median test R2 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dedicated_summary:
        md.append(
            f"| {row['track_key']} | {row['strategy_id']} | {row['rows_completed']} | "
            f"{'' if row['median_elapsed_sec'] is None else round(row['median_elapsed_sec'], 3)} | "
            f"{'' if row['median_train_sec'] is None else round(row['median_train_sec'], 3)} | "
            f"{'' if row['median_eval_sec'] is None else round(row['median_eval_sec'], 3)} | "
            f"{'' if row['median_test_r2'] is None else round(row['median_test_r2'], 6)} |"
        )
    SUMMARY_MD.write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
