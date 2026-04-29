#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE_SCRIPT = REPO / "scripts" / "run_hh_track4_assumption_conditioned_active_sequential_design_20260425.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a bounded active-sequential policy suite under the surrogate HH contract.")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--n-episodes", type=int, default=4)
    ap.add_argument("--n-particles", type=int, default=128)
    ap.add_argument("--n-rounds", type=int, default=4)
    ap.add_argument("--initial-currents", default="0.30")
    ap.add_argument("--candidate-currents", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--policies", default="uncertainty,disagreement,wasserstein")
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
    rows = []
    for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
        run_dir = save_dir / f"policy_{policy}"
        cmd = [
            sys.executable,
            str(BASE_SCRIPT),
            "--save-dir",
            str(run_dir),
            "--n-episodes",
            str(args.n_episodes),
            "--n-particles",
            str(args.n_particles),
            "--n-rounds",
            str(args.n_rounds),
            "--initial-currents",
            args.initial_currents,
            "--candidate-currents",
            args.candidate_currents,
            "--acquisition-policy",
            policy,
        ]
        subprocess.run(cmd, check=True)
        summary = json.loads((run_dir / "active_sequential_manifest.json").read_text())
        rows.append(
            {
                "policy": policy,
                "mean_abs_log10_error_params_mean": float(summary["aggregate_metrics"]["mean_abs_log10_error_params_mean"]),
                "mean_relative_error_params_mean": float(summary["aggregate_metrics"]["mean_relative_error_params_mean"]),
                "posterior_contraction_mean_mean": float(summary["aggregate_metrics"]["posterior_contraction_mean_mean"]),
                "ess_mean": float(summary["aggregate_metrics"]["ess_mean"]),
                "run_dir": str(run_dir),
            }
        )
    rows.sort(key=lambda row: row["mean_abs_log10_error_params_mean"])
    write_csv(save_dir / "active_sequential_policy_comparison.csv", rows)
    lines = [
        "# HH Track4 Active Sequential Policy Suite",
        "",
        f"- policies run: `{len(rows)}`",
        f"- summary csv: `{save_dir / 'active_sequential_policy_comparison.csv'}`",
        "",
        "| policy | mean abs log10 error | mean relative error | contraction | ESS |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['policy']}` | {row['mean_abs_log10_error_params_mean']:.4f} | {row['mean_relative_error_params_mean']:.4f} | "
            f"{row['posterior_contraction_mean_mean']:.4f} | {row['ess_mean']:.4f} |"
        )
    (save_dir / "active_sequential_policy_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"best": rows[0] if rows else None}, indent=2))


if __name__ == "__main__":
    main()
