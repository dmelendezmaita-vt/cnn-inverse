#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DATE_TAG = "20260413_v100_hh_postlocalsgd_r3"
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

REGISTRY_CSV = TABLES / f"v100_hh_postlocalsgd_r3_registry_{DATE_TAG}.csv"
SUMMARY_JSON = NOTES / f"v100_hh_postlocalsgd_r3_summary_{DATE_TAG}.json"
SUMMARY_MD = NOTES / f"v100_hh_postlocalsgd_r3_summary_{DATE_TAG}.md"


def load_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def median_or_none(vals):
    vals = list(vals)
    return statistics.median(vals) if vals else None


def main():
    rows = load_csv(REGISTRY_CSV)
    if not rows:
        raise SystemExit(f"Missing or empty registry: {REGISTRY_CSV}")

    completed = [row for row in rows if row["status"] == "COMPLETED"]
    state_counts = Counter(row["status"] for row in rows)
    agg = defaultdict(list)

    for row in completed:
        run_root = Path(row["run_output_root"])
        subdirs = [p for p in run_root.iterdir() if p.is_dir()] if run_root.exists() else []
        if not subdirs:
            continue
        task_dir = sorted(subdirs)[0]
        metrics_path = task_dir / "metrics_summary.json"
        info_log = task_dir / "run_dnn" / "rank0_info.log"
        if not metrics_path.exists():
            continue
        js = json.loads(metrics_path.read_text())
        train_sec = eval_sec = None
        if info_log.exists():
            txt = info_log.read_text(errors="ignore")
            mt = re.search(r"Runtime - train \[sec\]:\s+([0-9.eE+-]+)", txt)
            me = re.search(r"Runtime - eval \[sec\]:\s+([0-9.eE+-]+)", txt)
            if mt:
                train_sec = float(mt.group(1))
            if me:
                eval_sec = float(me.group(1))
        key = (row["phase"], row["strategy_id"], row["track_key"], row["policy"], row["nodes"])
        agg[key].append(
            {
                "elapsed_sec": float(row["elapsed_sec"]) if row["elapsed_sec"] else None,
                "train_sec": train_sec,
                "eval_sec": eval_sec,
                "test_mse": float(js["mse"]["test"]),
                "test_r2": float(js["r2"]["test"]),
            }
        )

    summary_rows = []
    for key in sorted(agg):
        phase, strategy_id, track_key, policy, nodes = key
        vals = agg[key]
        summary_rows.append(
            {
                "phase": phase,
                "strategy_id": strategy_id,
                "track_key": track_key,
                "policy": policy,
                "nodes": nodes,
                "n_completed": len(vals),
                "median_elapsed_sec": median_or_none(v["elapsed_sec"] for v in vals if v["elapsed_sec"] is not None),
                "median_train_sec": median_or_none(v["train_sec"] for v in vals if v["train_sec"] is not None),
                "median_eval_sec": median_or_none(v["eval_sec"] for v in vals if v["eval_sec"] is not None),
                "median_test_mse": median_or_none(v["test_mse"] for v in vals),
                "median_test_r2": median_or_none(v["test_r2"] for v in vals),
            }
        )

    summary = {
        "date_tag": DATE_TAG,
        "registry_csv": str(REGISTRY_CSV),
        "status_counts": dict(state_counts),
        "completed_rows": len(completed),
        "aggregates": summary_rows,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        f"# V100 HH Post-Local SGD R3 Summary ({DATE_TAG})",
        "",
        f"- registry: `{REGISTRY_CSV}`",
        f"- status counts: `{dict(state_counts)}`",
        f"- completed rows: `{len(completed)}`",
        "",
        "| phase | strategy | track | policy | nodes | completed | median elapsed (s) | median train (s) | median eval (s) | median test MSE | median test R2 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['phase']} | {row['strategy_id']} | {row['track_key']} | {row['policy']} | {row['nodes']} | "
            f"{row['n_completed']} | "
            f"{'' if row['median_elapsed_sec'] is None else round(row['median_elapsed_sec'], 3)} | "
            f"{'' if row['median_train_sec'] is None else round(row['median_train_sec'], 3)} | "
            f"{'' if row['median_eval_sec'] is None else round(row['median_eval_sec'], 3)} | "
            f"{'' if row['median_test_mse'] is None else round(row['median_test_mse'], 3)} | "
            f"{'' if row['median_test_r2'] is None else round(row['median_test_r2'], 6)} |"
        )
    SUMMARY_MD.write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
