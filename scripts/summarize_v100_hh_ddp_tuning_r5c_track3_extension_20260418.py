#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DATE_TAG = "20260418_v100_hh_ddp_tuning_r5c_track3_extension"
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"

REGISTRY_CSV = TABLES / f"v100_hh_ddp_tuning_r5c_track3_extension_registry_{DATE_TAG}.csv"
SUMMARY_JSON = NOTES / f"v100_hh_ddp_tuning_r5c_track3_extension_summary_{DATE_TAG}.json"
SUMMARY_MD = NOTES / f"v100_hh_ddp_tuning_r5c_track3_extension_summary_{DATE_TAG}.md"


def load_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def median_or_none(vals):
    vals = list(vals)
    return statistics.median(vals) if vals else None


def read_runtime(info_log: Path, label: str):
    if not info_log.exists():
        return None
    txt = info_log.read_text(errors="ignore")
    mt = re.search(rf"Runtime - {label} \[sec\]:\s+([0-9.eE+-]+)", txt)
    return float(mt.group(1)) if mt else None


def main():
    rows = load_csv(REGISTRY_CSV)
    if not rows:
        raise SystemExit(f"Missing or empty registry: {REGISTRY_CSV}")

    completed = [row for row in rows if row["status"] == "COMPLETED"]
    state_counts = Counter(row["status"] for row in rows)
    by_track_seed = defaultdict(list)

    enriched = []
    for row in completed:
        run_root = Path(row["run_output_root"]) / f"{row['alloc_job_id']}_{row['row_id']}"
        metrics_path = run_root / "metrics_summary.json"
        info_log = run_root / "run_dnn" / "rank0_info.log"
        if not metrics_path.exists():
            continue
        js = json.loads(metrics_path.read_text())
        item = {
            **row,
            "data_load_sec": read_runtime(info_log, "load_data"),
            "train_sec": read_runtime(info_log, "train"),
            "eval_sec": read_runtime(info_log, "evaluate") or read_runtime(info_log, "eval"),
            "test_mse": float(js["mse"]["test"]),
            "test_r2": float(js["r2"]["test"]),
        }
        enriched.append(item)
        by_track_seed[(row["track_key"], row["seed"])].append(item)

    aggregates = []
    pairwise = []
    for (track_key, seed), items in sorted(by_track_seed.items()):
        base = next((r for r in items if r["strategy_id"] == "baseline"), None)
        tuned = next((r for r in items if r["strategy_id"] != "baseline"), None)
        if base and tuned and base["train_sec"] and tuned["train_sec"]:
            speedup_pct = 100.0 * (base["train_sec"] - tuned["train_sec"]) / base["train_sec"]
            pairwise.append(
                {
                    "track_key": track_key,
                    "seed": int(seed),
                    "baseline_strategy": base["strategy_id"],
                    "tuned_strategy": tuned["strategy_id"],
                    "baseline_train_sec": base["train_sec"],
                    "tuned_train_sec": tuned["train_sec"],
                    "train_speedup_pct": speedup_pct,
                    "baseline_test_r2": base["test_r2"],
                    "tuned_test_r2": tuned["test_r2"],
                }
            )

    grouped = defaultdict(list)
    for row in enriched:
        grouped[(row["track_key"], row["strategy_id"])].append(row)
    for key in sorted(grouped):
        track_key, strategy_id = key
        vals = grouped[key]
        aggregates.append(
            {
                "track_key": track_key,
                "strategy_id": strategy_id,
                "n_completed": len(vals),
                "median_train_sec": median_or_none(v["train_sec"] for v in vals if v["train_sec"] is not None),
                "median_eval_sec": median_or_none(v["eval_sec"] for v in vals if v["eval_sec"] is not None),
                "median_data_load_sec": median_or_none(v["data_load_sec"] for v in vals if v["data_load_sec"] is not None),
                "median_test_mse": median_or_none(v["test_mse"] for v in vals),
                "median_test_r2": median_or_none(v["test_r2"] for v in vals),
            }
        )

    summary = {
        "date_tag": DATE_TAG,
        "registry_csv": str(REGISTRY_CSV),
        "status_counts": dict(state_counts),
        "completed_rows": len(completed),
        "aggregates": aggregates,
        "pairwise_by_track_seed": pairwise,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    md = [
        f"# V100 HH DDP Tuning R5c Track3 Extension Summary ({DATE_TAG})",
        "",
        f"- registry: `{REGISTRY_CSV}`",
        f"- status counts: `{dict(state_counts)}`",
        f"- completed rows: `{len(completed)}`",
        "",
        "## Pairwise Results",
        "",
        "| track | seed | tuned strategy | baseline train (s) | tuned train (s) | train speedup (%) | baseline test R2 | tuned test R2 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pairwise:
        md.append(
            f"| {row['track_key']} | {row['seed']} | {row['tuned_strategy']} | "
            f"{row['baseline_train_sec']:.3f} | {row['tuned_train_sec']:.3f} | {row['train_speedup_pct']:.3f} | "
            f"{row['baseline_test_r2']:.6f} | {row['tuned_test_r2']:.6f} |"
        )
    md.extend(
        [
            "",
            "## Aggregates",
            "",
            "| track | strategy | completed | median train (s) | median eval (s) | median data-load (s) | median test MSE | median test R2 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        md.append(
            f"| {row['track_key']} | {row['strategy_id']} | {row['n_completed']} | "
            f"{'' if row['median_train_sec'] is None else round(row['median_train_sec'], 3)} | "
            f"{'' if row['median_eval_sec'] is None else round(row['median_eval_sec'], 3)} | "
            f"{'' if row['median_data_load_sec'] is None else round(row['median_data_load_sec'], 3)} | "
            f"{'' if row['median_test_mse'] is None else round(row['median_test_mse'], 3)} | "
            f"{'' if row['median_test_r2'] is None else round(row['median_test_r2'], 6)} |"
        )
    SUMMARY_MD.write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
