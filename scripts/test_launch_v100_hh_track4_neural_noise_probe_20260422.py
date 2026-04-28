#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from launch_v100_hh_track4_neural_noise_probe_20260422 import build_env, build_srun_cmd, slices_for_mode


def sample_row(**overrides: str) -> dict[str, str]:
    row = {
        "row_id": "v100t4nce_0001",
        "phase": "phaseCP4_track4_v100_neural_noise_clean_eval",
        "launch_mode": "concurrent_8x1n",
        "launch_group": "track4_v100_neural_noise_clean_eval_packed",
        "strategy_id": "effnet_beta005_invvar_e500_noise001_clean_eval",
        "track_key": "track4_hh_full",
        "policy": "strong",
        "nodes": "1",
        "gpus_per_node": "1",
        "cpus_per_node": "12",
        "seed": "1071",
        "params_file": "pytorch/configs/example.yaml",
        "tar_path": "/tmp/example.tar",
        "data_prefix": "concatenated_data",
        "curr": "0.1",
        "data_access_mode": "direct_tar",
        "shared_data_dir": "/tmp/shared",
        "save_predictions": "test",
        "eval_only_checkpoint": "/tmp/checkpoint.pt",
        "eval_only_use_gpu": "1",
    }
    row.update(overrides)
    return row


def assert_flag(args: list[str], expected: str) -> None:
    if expected not in args:
        raise AssertionError(f"Missing expected flag: {expected}\nargs={args}")


def main() -> None:
    row = sample_row()
    os.environ["FOLLOWUP_TMP_BASE"] = "/tmp/v100_hh_track4_neural_noise_probe_test"

    env = build_env(row, "368960", "fal117", 24567, Path("/tmp/t4nce"))
    if env.get("EVAL_ONLY_USE_GPU") != "1":
        raise AssertionError("expected eval-only GPU override in environment")

    cmd = build_srun_cmd("falcon", "368960", ["fal117"], row)
    assert_flag(cmd, "-M")
    assert_flag(cmd, "falcon")
    assert_flag(cmd, "--exact")
    assert_flag(cmd, "--cpus-per-task=12")
    assert_flag(cmd, "--gres=gpu:v100:1")

    slices = slices_for_mode("concurrent_8x1n", [sample_row(row_id=f"v100t4nce_{i:04d}") for i in range(1, 9)], ["fal117", "fal118", "fal119", "fal120"])
    if slices != [["fal117"], ["fal117"], ["fal118"], ["fal118"], ["fal119"], ["fal119"], ["fal120"], ["fal120"]]:
        raise AssertionError(f"unexpected packed slices: {slices}")

    print("V100_HH_TRACK4_NEURAL_NOISE_PROBE_LAUNCH_TEST_OK")


if __name__ == "__main__":
    main()
