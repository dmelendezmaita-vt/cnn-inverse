#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
SCRIPT = REPO / "scripts" / "analyze_hh_track4_checkpoint6_shift_stability_20260423.py"

CLASSICAL_REGISTRY = (
    REPO
    / "data/important_notes/optimization_track_20260423_hh_track4_checkpoint6_drift_mask_followup_classical/tables"
    / "hh_track4_checkpoint6_drift_mask_followup_classical_registry_20260423_hh_track4_checkpoint6_drift_mask_followup_classical.csv"
)
NEURAL_REGISTRY = (
    REPO
    / "data/important_notes/optimization_track_20260423_v100_hh_track4_checkpoint6_drift_mask_followup_neural/tables"
    / "v100_hh_track4_checkpoint6_drift_mask_followup_neural_registry_20260423_v100_hh_track4_checkpoint6_drift_mask_followup_neural.csv"
)
OUT_ROOT = (
    REPO
    / "data/important_notes/optimization_track_20260423_hh_track4_checkpoint6_drift_mask_followup_analysis"
)


def main() -> None:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--classical-registry",
        str(CLASSICAL_REGISTRY),
        "--neural-registry",
        str(NEURAL_REGISTRY),
        "--random-forest-base",
        "random_forest_500",
        "--knn-base",
        "knn_k11",
        "--effnet-base",
        "effnet_beta005_invvar_e500",
        "--out-root",
        str(OUT_ROOT),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
