#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from features_scale_cache_utils import (
    summarize_feature_scale_cache_results,
    warm_feature_scale_cache_for_rows,
)
from shared_data_utils import ensure_shared_data
from split_array_cache_utils import (
    summarize_split_array_cache_results,
    warm_split_array_cache_for_rows,
)

DATE_TAG = "20260417_v100_hh_split_array_cache_canary"
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
RUNS = OUT_ROOT / "runs"
LOGS = OUT_ROOT / "logs"

MATRIX_CSV = TABLES / f"v100_hh_split_array_cache_canary_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_hh_split_array_cache_canary_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"v100_hh_split_array_cache_canary_run_log_{DATE_TAG}.json"
STEP_SCRIPT = REPO / "scripts" / "run_v100_interactive_step_20260411.sh"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def hostnames(nodelist: str) -> List[str]:
    proc = subprocess.run(["scontrol", "show", "hostnames", nodelist], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def active_step_nodes(alloc_job_id: str) -> set[str]:
    proc = subprocess.run(["squeue", "-s", "-h", "-j", str(alloc_job_id), "-o", "%i|%N"], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return set()
    used = set()
    for ln in (proc.stdout or "").splitlines():
        parts = ln.strip().split("|", 1)
        if len(parts) != 2:
            continue
        step_id, nodelist = parts
        if "." not in step_id:
            continue
        if not nodelist or nodelist in {"(null)", "None"}:
            continue
        for h in hostnames(nodelist):
            used.add(h)
    return used


def resolve_allocation(args):
    alloc_job_id = args.alloc_job_id or os.environ.get("SLURM_JOB_ID")
    alloc_nodelist = args.alloc_nodelist or os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not alloc_job_id or not alloc_nodelist:
        raise SystemExit("Need live allocation context or explicit --alloc-job-id/--alloc-nodelist")
    return alloc_job_id, alloc_nodelist, hostnames(alloc_nodelist)


def registry_fields():
    return [
        "row_id", "phase", "launch_mode", "launch_group", "strategy_id", "track_key", "policy", "nodes", "seed",
        "status", "return_code", "alloc_job_id", "alloc_nodelist", "assigned_nodes", "started_at", "ended_at",
        "elapsed_sec", "run_output_root", "stdout_log", "stderr_log", "params_file", "shared_data_dir", "notes",
    ]


def row_matches(row, args):
    if args.row_ids and row["row_id"] not in args.row_ids:
        return False
    if args.phases and row["phase"] not in args.phases:
        return False
    if args.strategies and row["strategy_id"] not in args.strategies:
        return False
    return True


def build_env(row, alloc_job_id, master_addr, master_port, run_output_root):
    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO),
            "ALLOC_JOB_ID": str(alloc_job_id),
            "RUN_ID": row["row_id"],
            "RUN_OUTPUT_ROOT": str(run_output_root),
            "PARAMS_FILE": row["params_file"],
            "TAR_PATH": row["tar_path"],
            "DATA_PREFIX": row["data_prefix"],
            "CURR": row["curr"],
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "STEP_NNODES": row["nodes"],
            "NPROC_PER_NODE": row["gpus_per_node"],
            "DATA_ACCESS_MODE": row["data_access_mode"],
            "SHARED_DATA_DIR": row["shared_data_dir"],
            "SAVE_PREDICTIONS": row["save_predictions"],
            "FOLLOWUP_TMP_BASE": "/projects/neuro-collab/other/v100if_tmp",
        }
    )
    return env


def build_srun_cmd(alloc_job_id, node_slice, row):
    n_nodes = int(row["nodes"])
    nodelist = ",".join(node_slice)
    return [
        "srun",
        f"--jobid={alloc_job_id}",
        "--exclusive",
        f"-N{n_nodes}",
        f"-w{nodelist}",
        f"--ntasks={n_nodes}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={row['cpus_per_node']}",
        f"--gres=gpu:v100:{row['gpus_per_node']}",
        "--kill-on-bad-exit=1",
        "bash",
        str(STEP_SCRIPT),
    ]


def make_run_output_root(row):
    return RUNS / row["phase"] / row["launch_mode"] / row["row_id"]


def make_logs(row):
    LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS / f"{row['row_id']}.out", LOGS / f"{row['row_id']}.err"


def launch_one(row, alloc_job_id, alloc_nodelist, hosts_for_launch, dry_run):
    n_nodes = int(row["nodes"])
    if len(hosts_for_launch) < n_nodes:
        raise RuntimeError(f"Need {n_nodes} free hosts, have {len(hosts_for_launch)}")
    node_slice = hosts_for_launch[:n_nodes]
    run_output_root = make_run_output_root(row)
    run_output_root.mkdir(parents=True, exist_ok=True)
    stdout_log, stderr_log = make_logs(row)
    master_addr = node_slice[0]
    master_port = 20000 + (abs(hash(row["row_id"])) % 20000)
    cmd = build_srun_cmd(alloc_job_id, node_slice, row)
    env = build_env(row, alloc_job_id, master_addr, master_port, run_output_root)
    started_at = now_iso()
    if dry_run:
        return {
            "row_id": row["row_id"], "phase": row["phase"], "launch_mode": row["launch_mode"], "launch_group": row["launch_group"],
            "strategy_id": row["strategy_id"], "track_key": row["track_key"], "policy": row["policy"], "nodes": row["nodes"], "seed": row["seed"],
            "status": "DRYRUN", "return_code": "", "alloc_job_id": alloc_job_id, "alloc_nodelist": alloc_nodelist,
            "assigned_nodes": ",".join(node_slice), "started_at": started_at, "ended_at": started_at, "elapsed_sec": "0.0",
            "run_output_root": str(run_output_root), "stdout_log": str(stdout_log), "stderr_log": str(stderr_log),
            "params_file": row["params_file"], "shared_data_dir": row["shared_data_dir"], "notes": "dry-run only",
        }
    start_ts = time.time()
    with stdout_log.open("w") as out_f, stderr_log.open("w") as err_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env)
        rc = proc.wait()
    ended_at = now_iso()
    return {
        "row_id": row["row_id"], "phase": row["phase"], "launch_mode": row["launch_mode"], "launch_group": row["launch_group"],
        "strategy_id": row["strategy_id"], "track_key": row["track_key"], "policy": row["policy"], "nodes": row["nodes"], "seed": row["seed"],
        "status": "COMPLETED" if rc == 0 else "FAILED", "return_code": str(rc), "alloc_job_id": alloc_job_id, "alloc_nodelist": alloc_nodelist,
        "assigned_nodes": ",".join(node_slice), "started_at": started_at, "ended_at": ended_at, "elapsed_sec": f"{max(0.0, time.time()-start_ts):.3f}",
        "run_output_root": str(run_output_root), "stdout_log": str(stdout_log), "stderr_log": str(stderr_log),
        "params_file": row["params_file"], "shared_data_dir": row["shared_data_dir"], "notes": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc-job-id", default=None)
    ap.add_argument("--alloc-nodelist", default=None)
    ap.add_argument("--row-id", dest="row_ids", action="append")
    ap.add_argument("--phase", dest="phases", action="append")
    ap.add_argument("--strategy", dest="strategies", action="append")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()

    alloc_job_id, alloc_nodelist, hosts = resolve_allocation(args)
    busy_hosts = active_step_nodes(alloc_job_id)
    hosts_for_launch = [h for h in hosts if h not in busy_hosts] or hosts

    matrix_rows = [row for row in load_csv(MATRIX_CSV) if row_matches(row, args)]
    existing = {row["row_id"]: row for row in load_csv(REGISTRY_CSV)}
    selected = []
    for row in matrix_rows:
        prev = existing.get(row["row_id"])
        if prev and (not args.force_rerun) and prev.get("status") == "COMPLETED":
            continue
        selected.append(row)
    if args.limit > 0:
        selected = selected[: args.limit]

    prewarm_scale_results = []
    prewarm_scale_summary = None
    prewarm_scale_force_refresh = os.environ.get("CLEAR_FEATURE_SCALE_CACHE", "0") == "1"
    prewarm_split_results = []
    prewarm_split_summary = None
    prewarm_split_force_refresh = os.environ.get("CLEAR_SPLIT_ARRAY_CACHE", "0") == "1"

    if not args.dry_run:
        for row in selected:
            if row["shared_data_dir"]:
                ensure_shared_data(row["tar_path"], row["shared_data_dir"])
        if os.environ.get("PREWARM_FEATURE_SCALE_CACHE", "1") != "0":
            prewarm_scale_results = warm_feature_scale_cache_for_rows(selected, force=prewarm_scale_force_refresh)
            prewarm_scale_summary = summarize_feature_scale_cache_results(prewarm_scale_results)
        if os.environ.get("PREWARM_SPLIT_ARRAY_CACHE", "1") != "0":
            prewarm_split_results = warm_split_array_cache_for_rows(selected, force=prewarm_split_force_refresh)
            prewarm_split_summary = summarize_split_array_cache_results(prewarm_split_results)

    results = []
    for row in selected:
        results.append(launch_one(row, alloc_job_id, alloc_nodelist, hosts_for_launch, args.dry_run))

    merged = {row["row_id"]: row for row in existing.values()}
    for row in results:
        merged[row["row_id"]] = row
    write_csv(REGISTRY_CSV, [merged[k] for k in sorted(merged)], registry_fields())

    run_log = {
        "date_tag": DATE_TAG,
        "timestamp": now_iso(),
        "alloc_job_id": alloc_job_id,
        "alloc_nodelist": alloc_nodelist,
        "hosts": hosts,
        "busy_hosts": sorted(busy_hosts),
        "hosts_for_launch": hosts_for_launch,
        "matrix_csv": str(MATRIX_CSV),
        "registry_csv": str(REGISTRY_CSV),
        "selected_rows": len(selected),
        "results_written": len(results),
        "prewarm_scale_cache_force_refresh": prewarm_scale_force_refresh,
        "prewarm_scale_cache_summary": prewarm_scale_summary,
        "prewarm_scale_cache_results": prewarm_scale_results,
        "prewarm_split_array_cache_force_refresh": prewarm_split_force_refresh,
        "prewarm_split_array_cache_summary": prewarm_split_summary,
        "prewarm_split_array_cache_results": prewarm_split_results,
        "dry_run": bool(args.dry_run),
        "phases": args.phases,
        "strategies": args.strategies,
        "row_ids": args.row_ids,
    }
    RUN_LOG_JSON.write_text(json.dumps(run_log, indent=2) + "\n")
    print(json.dumps(run_log, indent=2))


if __name__ == "__main__":
    main()
