#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from launch_a100_hh_datapath_redesign_r4_20260416 import (
    build_env as build_a100_env,
    build_srun_cmd as build_a100_srun_cmd,
)
from launch_v100_hh_datapath_redesign_r4_20260416 import (
    build_env as build_v100_env,
    build_srun_cmd as build_v100_srun_cmd,
)

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
    v100_matrix = (
        REPO
        / "data/important_notes/optimization_track_20260416_v100_hh_datapath_redesign_r4/tables"
        / "v100_hh_datapath_redesign_r4_matrix_20260416_v100_hh_datapath_redesign_r4.csv"
    )
    a100_matrix = (
        REPO
        / "data/important_notes/optimization_track_20260416_a100_hh_datapath_redesign_r4/tables"
        / "a100_hh_datapath_redesign_r4_matrix_20260416_a100_hh_datapath_redesign_r4.csv"
    )

    v100_core = load_row(v100_matrix, "v100r4_0001")
    v100_sidecar = load_row(v100_matrix, "v100r4_0015")
    a100_sidecar = load_row(a100_matrix, "a100r4_0015")

    core_env = build_v100_env(v100_core, "355055", "fal101", 29500, "/tmp/r4test")
    if core_env.get("LAUNCH_BACKEND") == "slurm_direct":
        raise AssertionError("core rows should not force slurm_direct")
    if core_env.get("SPLIT_EVAL_AFTER_TRAIN") == "1":
        raise AssertionError("core rows should not force split eval")

    core_cmd = build_v100_srun_cmd("355055", ["fal101"], v100_core)
    assert_flag(core_cmd, "--ntasks=1")
    assert_flag(core_cmd, "--ntasks-per-node=1")
    assert_flag(core_cmd, "--cpus-per-task=24")

    v100_env = build_v100_env(v100_sidecar, "355055", "fal101", 29500, "/tmp/r4test")
    if v100_env.get("LAUNCH_BACKEND") != "slurm_direct":
        raise AssertionError("v100 sidecar should force slurm_direct")
    if v100_env.get("SPLIT_EVAL_AFTER_TRAIN") != "1":
        raise AssertionError("v100 sidecar should force split eval")

    v100_cmd = build_v100_srun_cmd("355055", ["fal101", "fal103", "fal124", "fal130"], v100_sidecar)
    assert_flag(v100_cmd, "--ntasks=8")
    assert_flag(v100_cmd, "--ntasks-per-node=2")
    assert_flag(v100_cmd, "--cpus-per-task=12")

    a100_env = build_a100_env(a100_sidecar, "5014617", "tcnode001", 29500, "/tmp/r4test")
    if a100_env.get("LAUNCH_BACKEND") != "slurm_direct":
        raise AssertionError("a100 sidecar should force slurm_direct")
    if a100_env.get("SPLIT_EVAL_AFTER_TRAIN") != "1":
        raise AssertionError("a100 sidecar should force split eval")

    a100_cmd = build_a100_srun_cmd("5014617", ["tcnode001", "tcnode002", "tcnode003", "tcnode004"], a100_sidecar)
    assert_flag(a100_cmd, "--ntasks=8")
    assert_flag(a100_cmd, "--ntasks-per-node=2")
    assert_flag(a100_cmd, "--cpus-per-task=12")

    print("DATAPATH_R4_LAUNCH_TEST_OK")


if __name__ == "__main__":
    main()
