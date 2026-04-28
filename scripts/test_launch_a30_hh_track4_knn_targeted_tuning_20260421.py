#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from launch_a30_hh_track4_knn_targeted_tuning_20260421 import build_env, build_srun_cmd

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")


def load_row(path: Path, row_id: str) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(newline="")))
    matches = [row for row in rows if row["row_id"] == row_id]
    if len(matches) != 1:
        raise AssertionError(f"Expected one row for {row_id}, found {len(matches)}")
    return matches[0]


def assert_flag(args: list[str], expected: str) -> None:
    if expected not in args:
        raise AssertionError(f"Missing expected flag: {expected}\nargs={args}")


def main() -> None:
    matrix = (
        REPO
        / "data/important_notes/optimization_track_20260421_a30_hh_track4_knn_targeted_tuning/tables"
        / "a30_hh_track4_knn_targeted_tuning_matrix_20260421_a30_hh_track4_knn_targeted_tuning.csv"
    )
    row = load_row(matrix, "a30t4knn_0001")

    env = build_env(row, Path("/tmp/a30t4knntest"))
    if env.get("BASELINE_NAME") != "knn_k3":
        raise AssertionError("expected knn baseline")
    if "DATA_DIR" not in env:
        raise AssertionError("expected shared data dir in environment")

    cmd = build_srun_cmd("falcon", "351880", ["fal011"], row)
    assert_flag(cmd, "-M")
    assert_flag(cmd, "falcon")
    assert_flag(cmd, "--ntasks=1")
    assert_flag(cmd, "--cpus-per-task=64")
    print("A30_HH_TRACK4_KNN_TARGETED_TUNING_LAUNCH_TEST_OK")


if __name__ == "__main__":
    main()
