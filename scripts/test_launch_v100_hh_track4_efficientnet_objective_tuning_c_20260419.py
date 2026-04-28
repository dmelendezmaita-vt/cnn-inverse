#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from launch_v100_hh_track4_efficientnet_objective_tuning_c_20260419 import build_env, build_srun_cmd

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
        / "data/important_notes/optimization_track_20260419_v100_hh_track4_efficientnet_objective_tuning_c/tables"
        / "v100_hh_track4_efficientnet_objective_tuning_c_matrix_20260419_v100_hh_track4_efficientnet_objective_tuning_c.csv"
    )
    row = load_row(matrix, "v100t4effo_0001")

    env = build_env(row, "355055", "fal130", 29671, Path("/tmp/t4effotest"))
    if env.get("DATA_ACCESS_MODE") != "direct_tar":
        raise AssertionError("expected direct_tar access mode")
    if "SHARED_DATA_DIR" not in env:
        raise AssertionError("expected shared data dir in environment")

    cmd = build_srun_cmd("falcon", "355055", ["fal101"], row)
    assert_flag(cmd, "-M")
    assert_flag(cmd, "falcon")
    assert_flag(cmd, "--ntasks=1")
    assert_flag(cmd, "--ntasks-per-node=1")
    assert_flag(cmd, "--cpus-per-task=24")
    assert_flag(cmd, "--gres=gpu:v100:2")
    print("V100_HH_TRACK4_EFFICIENTNET_OBJECTIVE_TUNING_C_LAUNCH_TEST_OK")


if __name__ == "__main__":
    main()
