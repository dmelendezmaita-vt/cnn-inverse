#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
RUNALL = REPO / "data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan"
SURR = REPO / "data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a remaining hybrid-rule backlog from decision-rule outputs.")
    ap.add_argument(
        "--decision-csv",
        default=str(RUNALL / "tables/hh_track4_posterior_decision_rule_comparison_20260425.csv"),
    )
    ap.add_argument(
        "--out-json",
        default=str(SURR / "tables/hh_track4_remaining_backlog_round4_20260426.json"),
    )
    ap.add_argument("--row-prefix", default="acdhh_4")
    ap.add_argument("--mae-threshold", type=float, default=800.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    decision_csv = Path(args.decision_csv)
    used = set()
    for p in (SURR / "runs/direct_fitting").glob("*/hybrid_refinement_manifest.json"):
        obj = json.loads(p.read_text())
        pred = obj["summary"].get("prediction_file", "")
        if pred:
            used.add(str(Path(pred).resolve()))
    rows = []
    with decision_csv.open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            pred = str(Path(row["prediction_file"]).resolve())
            mae = float(row["mae"])
            r2 = float(row["r2"])
            if pred in used:
                continue
            if not (mae < args.mae_threshold):
                continue
            label = f"prediction_{Path(pred).parent.name}_{row['decision_rule']}"
            rows.append(
                {
                    "row_id": f"{args.row_prefix}{i:03d}",
                    "label": label,
                    "command": (
                        "cd /projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch && "
                        "/projects/neuro-collab/conda/neuro-collab-env/bin/python "
                        "scripts/run_hh_track4_assumption_conditioned_hybrid_refinement_surrogate_bundle_integration_20260425.py "
                        f"--bundle-label {label} --init-mode prediction --prediction-file {pred} "
                        "--n-exemplars 3 --trace-length 2000 --maxiter 120"
                    ),
                    "mae": mae,
                    "r2": r2,
                    "prediction_file": pred,
                }
            )
    rows.sort(key=lambda r: (r["mae"], -r["r2"]))
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2) + "\n")
    print(out_json)


if __name__ == "__main__":
    main()
