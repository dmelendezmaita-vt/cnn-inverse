#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
PACKAGE_SPECS = [
    (
        "optimization_track_*_v100_hh_track4_checkpoint7_vram_chain*",
        "tables/v100_hh_track4_checkpoint7_vram_chain_registry_*.csv",
    ),
    (
        "optimization_track_*_v100_hh_track4_checkpoint7_vram_parallel*",
        "tables/v100_hh_track4_checkpoint7_vram_parallel_registry_*.csv",
    ),
]
VRAM_AUDIT = (
    IMPORTANT
    / "optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency"
    / "tables"
    / "hh_track4_checkpoint7_vram_chain_audit_20260423.csv"
)
FINAL_TABLE = (
    IMPORTANT
    / "optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency"
    / "tables"
    / "hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv"
)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def discover_vram_registries() -> list[Path]:
    registries: list[Path] = []
    for package_glob, registry_glob in PACKAGE_SPECS:
        for package_root in sorted(IMPORTANT.glob(package_glob)):
            if not package_root.is_dir():
                continue
            registries.extend(sorted(package_root.glob(registry_glob)))
    return sorted({path.resolve() for path in registries})


def latest_metrics(run_output_root: str) -> tuple[Path, dict]:
    root = Path(run_output_root)
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No run dir under {root}")
    metrics_path = subdirs[-1] / "metrics_summary.json"
    return metrics_path, json.loads(metrics_path.read_text())


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> None:
    chain_registries = discover_vram_registries()
    registry_rows: list[dict[str, str]] = []
    for registry_path in chain_registries:
        for row in load_rows(registry_path):
            if row.get("status") == "COMPLETED":
                registry_rows.append(row)

    audit_rows: list[dict[str, object]] = []
    by_rep: dict[str, list[dict[str, float]]] = defaultdict(list)

    for row in sorted(
        registry_rows,
        key=lambda row: (
            row.get("representative", ""),
            row.get("started_at", ""),
            row.get("row_id", ""),
        ),
    ):
        metrics_path, metrics = latest_metrics(row["run_output_root"])
        gpu = metrics.get("gpu_memory") or {}
        record = {
            "representative": row["representative"],
            "row_id": row["row_id"],
            "assigned_node": row["assigned_node"],
            "metrics_path": str(metrics_path),
            "mae": float(metrics["posterior_mean_metrics"]["mae"]),
            "mse": float(metrics["posterior_mean_metrics"]["mse"]),
            "r2": float(metrics["posterior_mean_metrics"]["r2"]),
            "training_runtime_sec": float(metrics["training_runtime_sec"]),
            "evaluation_runtime_sec": float(metrics["evaluation_runtime_sec"]),
            "total_runtime_sec": float(metrics["total_runtime_sec"]),
            "gpu_device_name": gpu.get("device_name", ""),
            "gpu_max_memory_allocated_mb": float(gpu["max_memory_allocated_mb"]) if gpu.get("max_memory_allocated_mb") is not None else "",
            "gpu_max_memory_reserved_mb": float(gpu["max_memory_reserved_mb"]) if gpu.get("max_memory_reserved_mb") is not None else "",
            "gpu_memory_allocated_mb": float(gpu["memory_allocated_mb"]) if gpu.get("memory_allocated_mb") is not None else "",
            "gpu_memory_reserved_mb": float(gpu["memory_reserved_mb"]) if gpu.get("memory_reserved_mb") is not None else "",
        }
        audit_rows.append(record)
        if record["gpu_max_memory_allocated_mb"] != "":
            by_rep[row["representative"]].append(
                {
                    "alloc": float(record["gpu_max_memory_allocated_mb"]),
                    "reserved": float(record["gpu_max_memory_reserved_mb"]),
                    "mae": float(record["mae"]),
                    "mse": float(record["mse"]),
                    "r2": float(record["r2"]),
                    "train": float(record["training_runtime_sec"]),
                    "eval": float(record["evaluation_runtime_sec"]),
                    "total": float(record["total_runtime_sec"]),
                }
            )

    write_rows(
        VRAM_AUDIT,
        audit_rows,
        [
            "representative",
            "row_id",
            "assigned_node",
            "metrics_path",
            "mae",
            "mse",
            "r2",
            "training_runtime_sec",
            "evaluation_runtime_sec",
            "total_runtime_sec",
            "gpu_device_name",
            "gpu_max_memory_allocated_mb",
            "gpu_max_memory_reserved_mb",
            "gpu_memory_allocated_mb",
            "gpu_memory_reserved_mb",
        ],
    )

    final_rows = load_rows(FINAL_TABLE)
    if not final_rows:
        raise FileNotFoundError(f"Missing final table: {FINAL_TABLE}")

    new_fields = [
        "gpu_peak_allocated_mb_mean",
        "gpu_peak_reserved_mb_mean",
        "gpu_memory_observed_runs",
        "gpu_memory_audit_source",
    ]
    fieldnames = list(final_rows[0].keys())
    for field in new_fields:
        if field not in fieldnames:
            insert_at = fieldnames.index("peak_rss_mb_mean") + 1
            fieldnames.insert(insert_at, field)

    for row in final_rows:
        rep = row["representative"]
        if rep not in by_rep:
            row.setdefault("gpu_peak_allocated_mb_mean", "")
            row.setdefault("gpu_peak_reserved_mb_mean", "")
            row.setdefault("gpu_memory_observed_runs", "")
            row.setdefault("gpu_memory_audit_source", "")
            continue
        values = by_rep[rep]
        row["gpu_peak_allocated_mb_mean"] = fmt(mean(v["alloc"] for v in values))
        row["gpu_peak_reserved_mb_mean"] = fmt(mean(v["reserved"] for v in values))
        row["gpu_memory_observed_runs"] = str(len(values))
        row["gpu_memory_audit_source"] = str(VRAM_AUDIT)
        notes = row.get("notes", "")
        notes = notes.replace(
            "GPU VRAM was not logged.",
            "original seed block lacked GPU VRAM telemetry.",
        )
        clause = "checkpoint7 rerun package added V100 GPU memory telemetry."
        if clause not in notes:
            row["notes"] = f"{notes} {clause}".strip()
        else:
            row["notes"] = notes

    write_rows(FINAL_TABLE, final_rows, fieldnames)
    print(f"Wrote {VRAM_AUDIT}")
    print(f"Updated {FINAL_TABLE}")
    print(f"Discovered {len(chain_registries)} chain package registries")
    for rep, values in sorted(by_rep.items()):
        print(
            rep,
            f"runs={len(values)}",
            f"gpu_peak_allocated_mb_mean={mean(v['alloc'] for v in values):.3f}",
            f"gpu_peak_reserved_mb_mean={mean(v['reserved'] for v in values):.3f}",
        )


if __name__ == "__main__":
    main()
