#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from launch_v100_hh_accuracy_policy_20260419 import build_env, build_srun_cmd

REPO = Path(__file__).resolve().parents[1]


def load_row(path: Path, row_id: str) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(newline="")))
    matches = [row for row in rows if row["row_id"] == row_id]
    if len(matches) != 1:
        raise AssertionError(f"Expected one row for {row_id}, found {len(matches)}")
    return matches[0]


def assert_in(cmd: list[str], token: str) -> None:
    if token not in cmd:
        raise AssertionError(f"Missing token {token!r} in {cmd!r}")


def main() -> None:
    matrix = (
        REPO
        / "data/important_notes/optimization_track_20260419_v100_hh_accuracy_policy/tables"
        / "v100_hh_accuracy_policy_matrix_20260419_v100_hh_accuracy_policy.csv"
    )
    row = load_row(matrix, "v100accpol_0001")

    env = build_env(row, "355055", "fal101", 29671, Path("/tmp/accpolicytest"))
    if env.get("DATA_ACCESS_MODE") != "direct_tar":
        raise AssertionError("expected direct_tar access mode")
    if "SHARED_DATA_DIR" not in env:
        raise AssertionError("expected shared data dir in environment")

    cmd = build_srun_cmd("falcon", "355055", ["fal101"], row)
    assert_in(cmd, "-M")
    assert_in(cmd, "falcon")
    assert_in(cmd, "--ntasks=1")
    assert_in(cmd, "--ntasks-per-node=1")
    assert_in(cmd, "--cpus-per-task=24")
    assert_in(cmd, "--gres=gpu:v100:2")
    print("V100_HH_ACCURACY_POLICY_LAUNCH_TEST_OK")


if __name__ == "__main__":
    main()
