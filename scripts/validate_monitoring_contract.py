#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(message)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the unified monitoring contract for one resource monitor directory.")
    ap.add_argument("--resource-dir", required=True)
    ap.add_argument("--gpu-expected", type=int, default=0)
    args = ap.parse_args()

    resource_dir = Path(args.resource_dir)
    require(resource_dir.exists(), f"resource_dir does not exist: {resource_dir}")

    raw_path = resource_dir / "resource_monitor.json"
    summary_path = resource_dir / "resource_summary.json"
    time_path = resource_dir / "resource_time.txt"
    sacct_path = resource_dir / "sacct_step.txt"
    env_path = resource_dir / "task_env.json"
    gpu_csv = resource_dir / "gpu_monitor.csv"

    for path in [raw_path, summary_path, time_path, env_path]:
        require(path.exists(), f"missing monitoring artifact: {path}")

    summary = json.loads(summary_path.read_text())
    raw = json.loads(raw_path.read_text())
    require(isinstance(raw.get("summary"), dict), f"missing raw summary in {raw_path}")
    require(summary.get("return_code") is not None, f"missing return_code in {summary_path}")
    require(isinstance(summary.get("time_metrics"), dict), f"missing time_metrics in {summary_path}")
    time_metrics = summary.get("time_metrics", {})
    require(
        any(str(key).startswith("Elapsed (wall clock) time") for key in time_metrics),
        f"missing elapsed wall metric in {summary_path}",
    )
    require(isinstance(summary.get("resource_monitor_summary"), dict), f"missing resource_monitor_summary in {summary_path}")

    if args.gpu_expected > 0:
        require(gpu_csv.exists(), f"missing gpu_monitor.csv in {resource_dir}")
        require(sacct_path.exists(), f"missing sacct_step.txt in {resource_dir}")
        require(bool(summary.get("sacct_present")), f"sacct not present in {summary_path}")
        gpu_metrics = summary.get("gpu_metrics", {})
        require(int(gpu_metrics.get("gpu_sample_count", 0)) > 0, f"missing GPU samples in {summary_path}")
        per_gpu = gpu_metrics.get("per_gpu", [])
        require(isinstance(per_gpu, list), f"invalid per_gpu payload in {summary_path}")
        require(
            len(per_gpu) >= int(args.gpu_expected),
            f"expected at least {args.gpu_expected} monitored GPUs, got {len(per_gpu)} in {summary_path}",
        )

    print(json.dumps({"resource_dir": str(resource_dir), "gpu_expected": args.gpu_expected, "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
