#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from launch_a30_hh_track4_gpu_profile_20260420 import build_env, build_srun_cmd

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
        / "data/important_notes/optimization_track_20260420_a30_hh_track4_gpu_profile/tables"
        / "a30_hh_track4_gpu_profile_matrix_20260420_a30_hh_track4_gpu_profile.csv"
    )
    row = load_row(matrix, "a30t4gpu_0001")

    env = build_env(row, "351880", "fal011", 29671, Path("/tmp/a30t4gputest"))
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
    assert_flag(cmd, "--gres=gpu:a30:2")
    print("A30_HH_TRACK4_GPU_PROFILE_LAUNCH_TEST_OK")


if __name__ == "__main__":
    main()
