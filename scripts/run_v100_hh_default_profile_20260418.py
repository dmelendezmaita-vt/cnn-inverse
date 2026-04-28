#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
BUILD_SCRIPT = REPO / "scripts" / "build_v100_hh_default_policy_20260418.py"
LAUNCH_SCRIPT = REPO / "scripts" / "launch_v100_hh_default_policy_20260418.py"

PROFILE_STRATEGIES = {
    "track3_default": ["track3_default_1n_s5"],
    "track4_default": ["track4_default_1n_s5"],
    "broad_defaults": ["track3_default_1n_s5", "track4_default_1n_s5"],
    "track3_targeted": ["track3_targeted_4n_b100"],
    "track4_targeted": ["track4_targeted_4n_accum2"],
    "all": [
        "track3_default_1n_s5",
        "track4_default_1n_s5",
        "track3_targeted_4n_b100",
        "track4_targeted_4n_accum2",
    ],
}


def build_launch_cmd(
    *,
    profile: str,
    alloc_job_id: str | None,
    alloc_nodelist: str | None,
    dry_run: bool,
    force_rerun: bool,
) -> list[str]:
    cmd = ["python", str(LAUNCH_SCRIPT)]
    if alloc_job_id:
        cmd.extend(["--alloc-job-id", alloc_job_id])
    if alloc_nodelist:
        cmd.extend(["--alloc-nodelist", alloc_nodelist])
    for strategy in PROFILE_STRATEGIES[profile]:
        cmd.extend(["--strategy", strategy])
    if dry_run:
        cmd.append("--dry-run")
    if force_rerun:
        cmd.append("--force-rerun")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch named HH default-policy profiles.")
    ap.add_argument("profile", choices=sorted(PROFILE_STRATEGIES))
    ap.add_argument("--alloc-job-id", default=None)
    ap.add_argument("--alloc-nodelist", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()

    subprocess.run(["python", str(BUILD_SCRIPT)], check=True, cwd=REPO)
    cmd = build_launch_cmd(
        profile=args.profile,
        alloc_job_id=args.alloc_job_id,
        alloc_nodelist=args.alloc_nodelist,
        dry_run=args.dry_run,
        force_rerun=args.force_rerun,
    )
    subprocess.run(cmd, check=True, cwd=REPO)


if __name__ == "__main__":
    main()
