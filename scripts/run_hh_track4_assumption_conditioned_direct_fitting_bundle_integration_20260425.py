#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
HYBRID_SCRIPT = REPO / "scripts" / "run_hh_track4_assumption_conditioned_hybrid_refinement_20260425.py"
ABC_SCRIPT = REPO / "scripts" / "run_hh_track4_assumption_conditioned_wasserstein_abc_20260425.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run surrogate direct-fitting hooks under the surrogate bundle root and standardize the manifests.")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--run-hybrid", action="store_true")
    ap.add_argument("--run-abc", action="store_true")
    ap.add_argument("--n-exemplars", type=int, default=3)
    ap.add_argument("--trace-length", type=int, default=2000)
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--n-particles", type=int, default=256)
    ap.add_argument("--keep-top", type=int, default=24)
    return ap.parse_args()


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
    do_hybrid = args.run_hybrid or (not args.run_hybrid and not args.run_abc)
    do_abc = args.run_abc or (not args.run_hybrid and not args.run_abc)
    rows = []

    if do_hybrid:
        hybrid_dir = save_dir / "hybrid_refinement_default"
        subprocess.run(
            [
                sys.executable,
                str(HYBRID_SCRIPT),
                "--save-dir",
                str(hybrid_dir),
                "--init-mode",
                "midpoint",
                "--n-exemplars",
                str(args.n_exemplars),
                "--trace-length",
                str(args.trace_length),
                "--maxiter",
                str(args.maxiter),
            ],
            check=True,
        )
        manifest = json.loads((hybrid_dir / "hybrid_refinement_manifest.json").read_text())
        summary = manifest["summary"]
        rows.append(
            {
                "method_family": "hybrid_refinement",
                "objective": float(summary["objective"]),
                "trace_rmse_z_mean": float(summary["trace_rmse_z_mean"]),
                "summary_rel_l1_mean": float(summary["summary_rel_l1_mean"]),
                "current_gain": float(summary["current_gain"]),
                "run_dir": str(hybrid_dir),
            }
        )

    if do_abc:
        abc_dir = save_dir / "wasserstein_abc_default"
        subprocess.run(
            [
                sys.executable,
                str(ABC_SCRIPT),
                "--save-dir",
                str(abc_dir),
                "--n-particles",
                str(args.n_particles),
                "--keep-top",
                str(args.keep_top),
                "--n-exemplars",
                str(args.n_exemplars),
                "--trace-length",
                str(args.trace_length),
            ],
            check=True,
        )
        manifest = json.loads((abc_dir / "wasserstein_abc_manifest.json").read_text())
        rows.append(
            {
                "method_family": "wasserstein_abc",
                "objective": float(manifest["best_distance"]),
                "trace_rmse_z_mean": None,
                "summary_rel_l1_mean": float(manifest["median_top_distance"]),
                "current_gain": float(manifest["top_particles"][0]["current_gain"]),
                "run_dir": str(abc_dir),
            }
        )

    write_csv(save_dir / "direct_fitting_bundle_summary.csv", rows)
    (save_dir / "direct_fitting_bundle_manifest.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    report = [
        "# HH Track4 Assumption-Conditioned Direct Fitting Bundle Integration",
        "",
        f"- summary csv: `{save_dir / 'direct_fitting_bundle_summary.csv'}`",
        "",
    ]
    if rows:
        report.extend(
            [
                "| method | objective | trace_rmse_z_mean | summary_or_distance |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            report.append(
                f"| `{row['method_family']}` | {row['objective']:.4f} | "
                f"{'' if row['trace_rmse_z_mean'] is None else format(row['trace_rmse_z_mean'], '.4f')} | "
                f"{row['summary_rel_l1_mean']:.4f} |"
            )
    (save_dir / "direct_fitting_bundle_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
