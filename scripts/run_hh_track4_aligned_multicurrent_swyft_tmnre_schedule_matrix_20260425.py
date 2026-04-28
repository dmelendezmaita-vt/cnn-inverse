#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
BASE_SCRIPT = REPO / "scripts" / "run_hh_track4_aligned_multicurrent_swyft_tmnre_20260425.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a bounded TMNRE-style schedule matrix for native Swyft.")
    ap.add_argument("--params", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--stage1-run-dir", required=True)
    ap.add_argument(
        "--data-dir",
        default="/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260411_v100_interactive/shared_data/track4_hh_full/concatenated_data",
    )
    ap.add_argument("--data-prefix", default="concatenated_data")
    ap.add_argument("--threshold-grid", default="1e-2,3e-3,1e-3,3e-4")
    ap.add_argument("--min-retained-grid", default="0.10,0.25,0.40")
    ap.add_argument("--bound-observations-grid", default="64,128")
    ap.add_argument("--aggregation", default="meanstd")
    ap.add_argument("--feature-mode", default="raw_plus_fft256_summary12")
    ap.add_argument("--max-schedules", type=int, default=8)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--n-train", type=int, default=4096)
    ap.add_argument("--n-validate", type=int, default=1024)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    return ap.parse_args()


def parse_list(text: str, cast):
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_list(args.threshold_grid, float)
    min_retained = parse_list(args.min_retained_grid, float)
    n_obs_grid = parse_list(args.bound_observations_grid, int)
    schedules = []
    for thr in thresholds:
        for keep in min_retained:
            for n_obs in n_obs_grid:
                schedules.append((thr, keep, n_obs))
    schedules = schedules[: args.max_schedules]

    rows = []
    for idx, (thr, keep, n_obs) in enumerate(schedules, start=1):
        run_dir = save_dir / f"tmnre_sched_{idx:03d}"
        cmd = [
            sys.executable,
            str(BASE_SCRIPT),
            "--params",
            args.params,
            "--save-dir",
            str(run_dir),
            "--stage1-run-dir",
            args.stage1_run_dir,
            "--feature-mode",
            args.feature_mode,
            "--aggregation",
            args.aggregation,
            "--data-dir",
            args.data_dir,
            "--data-prefix",
            args.data_prefix,
            "--device",
            args.device,
            "--n-train",
            str(args.n_train),
            "--n-validate",
            str(args.n_validate),
            "--n-test",
            str(args.n_test),
            "--max-epochs",
            str(args.max_epochs),
            "--batch-size",
            str(args.batch_size),
            "--bound-threshold",
            str(thr),
            "--min-retained-frac",
            str(keep),
            "--bound-observations",
            str(n_obs),
        ]
        subprocess.run(cmd, check=True)
        summary = json.loads((run_dir / "metrics_summary.json").read_text())
        bounds = summary["bounds"]
        metrics = summary["test_metrics"]
        rows.append(
            {
                "schedule_id": f"tmnre_sched_{idx:03d}",
                "bound_threshold": float(thr),
                "min_retained_frac": float(keep),
                "bound_observations": int(n_obs),
                "retained_frac": float(bounds["retained_frac"]),
                "q_low": float(bounds["q_low"]),
                "q_high": float(bounds["q_high"]),
                "n_train_truncated": int(summary["n_train_truncated"]),
                "n_validate_truncated": int(summary["n_validate_truncated"]),
                "posterior_bank_size_truncated": int(summary["posterior_bank_size_truncated"]),
                "mae": float(metrics["mae"]),
                "mse": float(metrics["mse"]),
                "r2": float(metrics["r2"]),
                "run_dir": str(run_dir),
            }
        )

    rows.sort(key=lambda row: (row["mae"], -row["retained_frac"]))
    if rows:
        best_run_dir = Path(str(rows[0]["run_dir"]))
        for name in ("predictions_test.npz", "predictions_mean.npz"):
            src = best_run_dir / name
            dst = save_dir / name
            if src.exists():
                shutil.copy2(src, dst)
    write_csv(save_dir / "tmnre_schedule_matrix_summary.csv", rows)
    report = [
        "# HH Track4 Swyft TMNRE Schedule Matrix",
        "",
        f"- schedules run: `{len(rows)}`",
        f"- summary csv: `{save_dir / 'tmnre_schedule_matrix_summary.csv'}`",
        "",
        "| schedule | thr | min_keep | retained | MAE | R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| `{row['schedule_id']}` | {row['bound_threshold']:.4g} | {row['min_retained_frac']:.2f} | "
            f"{row['retained_frac']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} |"
        )
    (save_dir / "tmnre_schedule_matrix_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"n_schedules": len(rows), "best": rows[0] if rows else None}, indent=2))


if __name__ == "__main__":
    main()
