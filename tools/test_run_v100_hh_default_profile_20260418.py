#!/usr/bin/env python3
from __future__ import annotations

from run_v100_hh_default_profile_20260418 import build_launch_cmd


def assert_in(cmd: list[str], token: str) -> None:
    if token not in cmd:
        raise AssertionError(f"Missing token {token!r} in {cmd!r}")


def assert_not_in(cmd: list[str], token: str) -> None:
    if token in cmd:
        raise AssertionError(f"Unexpected token {token!r} in {cmd!r}")


def main() -> None:
    cmd = build_launch_cmd(
        profile="track3_default",
        alloc_job_id="355055",
        alloc_nodelist="fal[101,103,124,130]",
        dry_run=True,
        force_rerun=False,
    )
    assert_in(cmd, "--alloc-job-id")
    assert_in(cmd, "355055")
    assert_in(cmd, "--alloc-nodelist")
    assert_in(cmd, "fal[101,103,124,130]")
    assert_in(cmd, "--strategy")
    assert_in(cmd, "track3_default_1n_s5")
    assert_in(cmd, "--dry-run")
    assert_not_in(cmd, "--force-rerun")

    cmd = build_launch_cmd(
        profile="broad_defaults",
        alloc_job_id=None,
        alloc_nodelist=None,
        dry_run=False,
        force_rerun=True,
    )
    assert cmd.count("--strategy") == 2, cmd
    assert_in(cmd, "track3_default_1n_s5")
    assert_in(cmd, "track4_default_1n_s5")
    assert_in(cmd, "--force-rerun")

    cmd = build_launch_cmd(
        profile="all",
        alloc_job_id=None,
        alloc_nodelist=None,
        dry_run=False,
        force_rerun=False,
    )
    assert cmd.count("--strategy") == 4, cmd
    assert_in(cmd, "track3_targeted_4n_b100")
    assert_in(cmd, "track4_targeted_4n_accum2")

    print("V100_HH_DEFAULT_PROFILE_WRAPPER_TEST_OK")


if __name__ == "__main__":
    main()
