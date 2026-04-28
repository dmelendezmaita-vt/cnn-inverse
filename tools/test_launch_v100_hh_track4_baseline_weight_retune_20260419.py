#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from launch_v100_hh_track4_baseline_weight_retune_20260419 import build_env, build_srun_cmd

REPO = Path(__file__).resolve().parents[1]


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
        / "data/important_notes/optimization_track_20260419_v100_hh_track4_baseline_weight_retune/tables"
        / "v100_hh_track4_baseline_weight_retune_matrix_20260419_v100_hh_track4_baseline_weight_retune.csv"
    )
    row_1n = load_row(matrix, "v100t4bw_0001")

    env = build_env(row_1n, "355055", "fal101", 29671, "/tmp/t4baseweighttest")
    if env.get("DATA_ACCESS_MODE") != "direct_tar":
        raise AssertionError("expected direct_tar access mode")
    if "SHARED_DATA_DIR" not in env:
        raise AssertionError("expected shared data dir in environment")

    cmd_1n = build_srun_cmd("355055", ["fal101"], row_1n)
    assert_flag(cmd_1n, "--ntasks=1")
    assert_flag(cmd_1n, "--ntasks-per-node=1")
    assert_flag(cmd_1n, "--cpus-per-task=24")
    assert_flag(cmd_1n, "--gres=gpu:v100:2")
    print("V100_HH_TRACK4_BASELINE_WEIGHT_RETUNE_LAUNCH_TEST_OK")


if __name__ == "__main__":
    main()
