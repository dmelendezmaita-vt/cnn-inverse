#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require_summary_keys(payload: dict, *, path: Path, keys: list[str]) -> None:
    require(isinstance(payload, dict), f"invalid summary payload in {path}")
    for key in keys:
        require(key in payload, f"missing summary key {key!r} in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the unified monitoring contract for one resource monitor directory.")
    ap.add_argument("--resource-dir", required=True)
    ap.add_argument("--gpu-expected", type=int, default=None)
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

    summary = load_json(summary_path)
    raw = load_json(raw_path)
    task_env = load_json(env_path)
    raw_summary = raw.get("summary")
    merged_summary = summary.get("resource_monitor_summary")
    require(isinstance(raw_summary, dict), f"missing raw summary in {raw_path}")
    require(isinstance(merged_summary, dict), f"missing resource_monitor_summary in {summary_path}")
    require(isinstance(raw.get("samples"), list), f"missing raw samples in {raw_path}")
    require(len(raw.get("samples", [])) > 0, f"empty raw samples in {raw_path}")
    require_summary_keys(
        raw_summary,
        path=raw_path,
        keys=[
            "sample_count",
            "max_tree_cpu_percent",
            "max_tree_rss_mb",
            "max_tree_vms_mb",
            "max_node_cpu_percent_mean",
            "max_node_memory_used_mb",
            "max_node_memory_percent",
            "gpu_by_uuid",
        ],
    )
    require_summary_keys(
        merged_summary,
        path=summary_path,
        keys=[
            "sample_count",
            "max_tree_cpu_percent",
            "max_tree_rss_mb",
            "max_tree_vms_mb",
            "max_node_cpu_percent_mean",
            "max_node_memory_used_mb",
            "max_node_memory_percent",
            "gpu_by_uuid",
        ],
    )
    require(int(raw_summary.get("sample_count", 0)) > 0, f"nonpositive raw sample_count in {raw_path}")
    require(int(merged_summary.get("sample_count", 0)) > 0, f"nonpositive resource sample_count in {summary_path}")
    require(summary.get("return_code") is not None, f"missing return_code in {summary_path}")
    require(isinstance(summary.get("time_metrics"), dict), f"missing time_metrics in {summary_path}")
    time_metrics = summary.get("time_metrics", {})
    require(
        any(str(key).startswith("Elapsed (wall clock) time") for key in time_metrics),
        f"missing elapsed wall metric in {summary_path}",
    )
    gpu_expected = args.gpu_expected
    if gpu_expected is None:
        gpu_expected = int(task_env.get("gpu_expected") or summary.get("gpu_expected") or 0)
    is_slurm_context = any(
        bool(str(task_env.get(key, "")).strip())
        for key in ["alloc_job_id", "slurm_job_id", "slurm_step_id"]
    )
    if is_slurm_context:
        require(sacct_path.exists(), f"missing sacct_step.txt in {resource_dir}")
        require(bool(summary.get("sacct_present")), f"sacct not present in {summary_path}")

    if int(gpu_expected) > 0:
        require(gpu_csv.exists(), f"missing gpu_monitor.csv in {resource_dir}")
        gpu_metrics = summary.get("gpu_metrics", {})
        require(int(gpu_metrics.get("gpu_sample_count", 0)) > 0, f"missing GPU samples in {summary_path}")
        per_gpu = gpu_metrics.get("per_gpu", [])
        require(isinstance(per_gpu, list), f"invalid per_gpu payload in {summary_path}")
        require(
            len(per_gpu) >= int(gpu_expected),
            f"expected at least {gpu_expected} monitored GPUs, got {len(per_gpu)} in {summary_path}",
        )
        raw_gpus = raw_summary.get("gpu_by_uuid", {})
        require(isinstance(raw_gpus, dict), f"invalid raw gpu_by_uuid payload in {raw_path}")
        require(
            len(raw_gpus) >= int(gpu_expected),
            f"expected at least {gpu_expected} raw monitored GPUs, got {len(raw_gpus)} in {raw_path}",
        )
        require(
            any(float((item or {}).get("sample_count") or 0) > 0 for item in per_gpu),
            f"no per-GPU sample counts recorded in {summary_path}",
        )

    print(json.dumps({"resource_dir": str(resource_dir), "gpu_expected": int(gpu_expected), "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
