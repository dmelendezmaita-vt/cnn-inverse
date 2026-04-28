#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from features_scale_cache_utils import (
    summarize_feature_scale_cache_results,
    warm_feature_scale_cache_for_rows,
)
from shared_data_utils import ensure_shared_data
from split_array_cache_utils import (
    prepare_rows_for_split_array_cache,
    summarize_split_array_cache_results,
    warm_split_array_cache_for_rows,
)

DATE_TAG = "20260419_v100_hh_track4_convresnet_retune"
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
RUNS = OUT_ROOT / "runs"
LOGS = OUT_ROOT / "logs"

MATRIX_CSV = TABLES / f"v100_hh_track4_convresnet_retune_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_hh_track4_convresnet_retune_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"v100_hh_track4_convresnet_retune_run_log_{DATE_TAG}.json"
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


def hostnames(nodelist: str, cluster: Optional[str] = None) -> List[str]:
    cmd = ["scontrol"]
    if cluster:
        cmd.extend(["-M", cluster])
    cmd.extend(["show", "hostnames", nodelist])
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to resolve hostnames for nodelist={nodelist}: {proc.stderr}")
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def active_step_nodes(alloc_job_id: str, cluster: Optional[str] = None) -> set[str]:
    cmd = ["squeue"]
    if cluster:
        cmd.extend(["-M", cluster])
    cmd.extend(["-s", "-h", "-j", str(alloc_job_id), "-o", "%i|%N"])
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return set()

    used = set()
    for ln in (proc.stdout or "").splitlines():
        parts = ln.strip().split("|", 1)
        if len(parts) != 2:
            continue
        step_id, nodelist = parts
        if not re.fullmatch(r"\d+\.\d+", step_id):
            continue
        if not nodelist or nodelist in {"(null)", "None"}:
            continue
        try:
            for host in hostnames(nodelist, cluster=cluster):
                used.add(host)
        except Exception:
            continue
    return used


def resolve_allocation(args: argparse.Namespace) -> Tuple[str, str, List[str]]:
    alloc_job_id = args.alloc_job_id or os.environ.get("SLURM_JOB_ID")
    alloc_nodelist = args.alloc_nodelist or os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not alloc_job_id or not alloc_nodelist:
        raise SystemExit(
            "This launcher must run inside the interactive allocation or be given "
            "--alloc-job-id and --alloc-nodelist explicitly."
        )
    hosts = hostnames(alloc_nodelist, cluster=args.cluster)
    return alloc_job_id, alloc_nodelist, hosts


def registry_fields() -> List[str]:
    return [
        "row_id",
        "phase",
        "launch_mode",
        "launch_group",
        "strategy_id",
        "track_key",
        "policy",
        "nodes",
        "seed",
        "status",
        "return_code",
        "alloc_job_id",
        "alloc_nodelist",
        "assigned_nodes",
        "started_at",
        "ended_at",
        "elapsed_sec",
        "run_output_root",
        "stdout_log",
        "stderr_log",
        "params_file",
        "shared_data_dir",
        "notes",
    ]


def row_matches(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if args.row_ids and row["row_id"] not in args.row_ids:
        return False
    if args.phases and row["phase"] not in args.phases:
        return False
    if args.strategies and row["strategy_id"] not in args.strategies:
        return False
    if args.launch_modes and row["launch_mode"] not in args.launch_modes:
        return False
    return True


def make_run_output_root(row: Dict[str, str]) -> Path:
    return RUNS / row["phase"] / row["launch_mode"] / row["row_id"]


def make_logs(row: Dict[str, str]) -> Tuple[Path, Path]:
    LOGS.mkdir(parents=True, exist_ok=True)
    return LOGS / f"{row['row_id']}.out", LOGS / f"{row['row_id']}.err"


def build_env(
    row: Dict[str, str],
    alloc_job_id: str,
    master_addr: str,
    master_port: int,
    run_output_root: Path,
) -> Dict[str, str]:
    env = os.environ.copy()
    followup_tmp_base = Path("/projects/neuro-collab/other/v100if_tmp")
    followup_tmp_base.mkdir(parents=True, exist_ok=True)
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
            "FOLLOWUP_TMP_BASE": str(followup_tmp_base),
        }
    )
    return env


def build_srun_cmd(
    cluster: Optional[str],
    alloc_job_id: str,
    node_slice: Sequence[str],
    row: Dict[str, str],
) -> List[str]:
    n_nodes = int(row["nodes"])
    nodelist = ",".join(node_slice)
    cmd = ["srun"]
    if cluster:
        cmd.extend(["-M", cluster])
    cmd.extend(
        [
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
    )
    return cmd


def launch_one(
    cluster: Optional[str],
    row: Dict[str, str],
    alloc_job_id: str,
    alloc_nodelist: str,
    node_slice: Sequence[str],
    dry_run: bool,
) -> Dict[str, str]:
    run_output_root = make_run_output_root(row)
    run_output_root.mkdir(parents=True, exist_ok=True)
    stdout_log, stderr_log = make_logs(row)
    master_addr = node_slice[0]
    master_port = 20000 + (abs(hash(row["row_id"])) % 20000)
    if row["shared_data_dir"] and not dry_run:
        ensure_shared_data(row["tar_path"], row["shared_data_dir"])

    cmd = build_srun_cmd(cluster, alloc_job_id, node_slice, row)
    env = build_env(row, alloc_job_id, master_addr, master_port, run_output_root)

    started_at = now_iso()
    start_ts = time.time()
    if dry_run:
        return {
            "row_id": row["row_id"],
            "phase": row["phase"],
            "launch_mode": row["launch_mode"],
            "launch_group": row["launch_group"],
            "strategy_id": row["strategy_id"],
            "track_key": row["track_key"],
            "policy": row["policy"],
            "nodes": row["nodes"],
            "seed": row["seed"],
            "status": "DRYRUN",
            "return_code": "",
            "alloc_job_id": alloc_job_id,
            "alloc_nodelist": alloc_nodelist,
            "assigned_nodes": ",".join(node_slice),
            "started_at": started_at,
            "ended_at": started_at,
            "elapsed_sec": "0.0",
            "run_output_root": str(run_output_root),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "params_file": row["params_file"],
            "shared_data_dir": row["shared_data_dir"],
            "notes": "",
        }

    with stdout_log.open("w") as out_f, stderr_log.open("w") as err_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env)
        rc = proc.wait()
    ended_at = now_iso()
    elapsed_sec = max(0.0, time.time() - start_ts)
    return {
        "row_id": row["row_id"],
        "phase": row["phase"],
        "launch_mode": row["launch_mode"],
        "launch_group": row["launch_group"],
        "strategy_id": row["strategy_id"],
        "track_key": row["track_key"],
        "policy": row["policy"],
        "nodes": row["nodes"],
        "seed": row["seed"],
        "status": "COMPLETED" if rc == 0 else "FAILED",
        "return_code": str(rc),
        "alloc_job_id": alloc_job_id,
        "alloc_nodelist": alloc_nodelist,
        "assigned_nodes": ",".join(node_slice),
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_sec": f"{elapsed_sec:.3f}",
        "run_output_root": str(run_output_root),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "params_file": row["params_file"],
        "shared_data_dir": row["shared_data_dir"],
        "notes": "",
    }


def launch_group(
    cluster: Optional[str],
    group_rows: List[Dict[str, str]],
    alloc_job_id: str,
    alloc_nodelist: str,
    hosts: Sequence[str],
    dry_run: bool,
) -> List[Dict[str, str]]:
    mode = group_rows[0]["launch_mode"]
    if mode == "concurrent_4x1n":
        if len(group_rows) != 4 or len(hosts) < 4:
            raise RuntimeError(f"{mode} requires 4 rows and 4 hosts")
        slices = [[hosts[i]] for i in range(4)]
    elif mode == "concurrent_3x1n":
        if len(group_rows) != 3 or len(hosts) < 3:
            raise RuntimeError(f"{mode} requires 3 rows and 3 hosts")
        slices = [[hosts[i]] for i in range(3)]
    elif mode == "concurrent_2x2n":
        if len(group_rows) != 2 or len(hosts) < 4:
            raise RuntimeError(f"{mode} requires 2 rows and 4 hosts")
        slices = [list(hosts[0:2]), list(hosts[2:4])]
    else:
        raise RuntimeError(f"Unsupported grouped launch_mode={mode}")

    if dry_run:
        return [
            launch_one(cluster, row, alloc_job_id, alloc_nodelist, node_slice, dry_run=True)
            for row, node_slice in zip(group_rows, slices)
        ]

    procs = []
    started_at = now_iso()
    results = []
    for row, node_slice in zip(group_rows, slices):
        run_output_root = make_run_output_root(row)
        run_output_root.mkdir(parents=True, exist_ok=True)
        stdout_log, stderr_log = make_logs(row)
        if row["shared_data_dir"]:
            ensure_shared_data(row["tar_path"], row["shared_data_dir"])
        master_addr = node_slice[0]
        master_port = 20000 + (abs(hash(row["row_id"])) % 20000)
        cmd = build_srun_cmd(cluster, alloc_job_id, node_slice, row)
        env = build_env(row, alloc_job_id, master_addr, master_port, run_output_root)
        out_f = stdout_log.open("w")
        err_f = stderr_log.open("w")
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env)
        procs.append((row, node_slice, proc, out_f, err_f, started_at, time.time(), run_output_root, stdout_log, stderr_log))

    for row, node_slice, proc, out_f, err_f, started, start_ts, run_output_root, stdout_log, stderr_log in procs:
        rc = proc.wait()
        out_f.close()
        err_f.close()
        ended = now_iso()
        elapsed_sec = max(0.0, time.time() - start_ts)
        results.append(
            {
                "row_id": row["row_id"],
                "phase": row["phase"],
                "launch_mode": row["launch_mode"],
                "launch_group": row["launch_group"],
                "strategy_id": row["strategy_id"],
                "track_key": row["track_key"],
                "policy": row["policy"],
                "nodes": row["nodes"],
                "seed": row["seed"],
                "status": "COMPLETED" if rc == 0 else "FAILED",
                "return_code": str(rc),
                "alloc_job_id": alloc_job_id,
                "alloc_nodelist": alloc_nodelist,
                "assigned_nodes": ",".join(node_slice),
                "started_at": started,
                "ended_at": ended,
                "elapsed_sec": f"{elapsed_sec:.3f}",
                "run_output_root": str(run_output_root),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "params_file": row["params_file"],
                "shared_data_dir": row["shared_data_dir"],
                "notes": "",
            }
        )
    return results


def dedicated_node_options(hosts: Sequence[str], node_count: int) -> List[List[str]]:
    if len(hosts) < node_count:
        raise RuntimeError(f"Need at least {node_count} hosts, got {len(hosts)}")
    return [list(hosts[i : i + node_count]) for i in range(0, len(hosts) - node_count + 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch V100 interactive follow-up tasks inside an existing interactive allocation.")
    ap.add_argument("--alloc-job-id", default=None)
    ap.add_argument("--alloc-nodelist", default=None)
    ap.add_argument("--row-id", dest="row_ids", action="append")
    ap.add_argument("--phase", dest="phases", action="append")
    ap.add_argument("--strategy", dest="strategies", action="append")
    ap.add_argument("--launch-mode", dest="launch_modes", action="append")
    ap.add_argument("--limit", type=int, default=0, help="Maximum number of matrix rows to consider (0 means all).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--cluster", default=os.environ.get("SLURM_CLUSTER_NAME") or os.environ.get("SLURM_CLUSTERS"))
    args = ap.parse_args()

    if not MATRIX_CSV.exists():
        raise SystemExit(f"Missing matrix CSV: {MATRIX_CSV}. Run the builder first.")
    if not STEP_SCRIPT.exists():
        raise SystemExit(f"Missing step script: {STEP_SCRIPT}")

    alloc_job_id, alloc_nodelist, hosts = resolve_allocation(args)
    busy_hosts = active_step_nodes(alloc_job_id, cluster=args.cluster)
    hosts_for_launch = [h for h in hosts if h not in busy_hosts]
    if not hosts_for_launch:
        hosts_for_launch = hosts
    matrix_rows = [row for row in load_csv(MATRIX_CSV) if row_matches(row, args)]

    existing = {row["row_id"]: row for row in load_csv(REGISTRY_CSV)}
    out_rows: List[Dict[str, str]] = list(existing.values())

    selected = []
    for row in matrix_rows:
        prev = existing.get(row["row_id"])
        if prev and (not args.force_rerun) and prev.get("status") == "COMPLETED":
            continue
        selected.append(row)

    if args.limit > 0:
        selected = selected[: args.limit]
    split_array_cache_enabled = os.environ.get("PREWARM_SPLIT_ARRAY_CACHE", "0") != "0"
    selected = prepare_rows_for_split_array_cache(
        selected,
        NOTES / "split_array_cache_param_overrides",
        enable=split_array_cache_enabled,
    )

    prewarm_results = []
    prewarm_summary = None
    prewarm_force_refresh = os.environ.get("CLEAR_FEATURE_SCALE_CACHE", "0") == "1"
    prewarm_split_results = []
    prewarm_split_summary = None
    prewarm_split_force_refresh = os.environ.get("CLEAR_SPLIT_ARRAY_CACHE", "0") == "1"
    if (not args.dry_run) and os.environ.get("PREWARM_FEATURE_SCALE_CACHE", "1") != "0":
        prewarm_results = warm_feature_scale_cache_for_rows(selected, force=prewarm_force_refresh)
        prewarm_summary = summarize_feature_scale_cache_results(prewarm_results)
    if (not args.dry_run) and split_array_cache_enabled:
        prewarm_split_results = warm_split_array_cache_for_rows(selected, force=prewarm_split_force_refresh)
        prewarm_split_summary = summarize_split_array_cache_results(prewarm_split_results)

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    dedicated: List[Dict[str, str]] = []
    for row in selected:
        if row["launch_group"]:
            grouped[(row["launch_mode"], row["launch_group"])].append(row)
        else:
            dedicated.append(row)

    results: List[Dict[str, str]] = []
    slot_index: Dict[int, int] = defaultdict(int)
    for row in dedicated:
        node_count = int(row["nodes"])
        options = dedicated_node_options(hosts_for_launch, node_count)
        node_slice = options[slot_index[node_count] % len(options)]
        slot_index[node_count] += 1
        results.append(launch_one(args.cluster, row, alloc_job_id, alloc_nodelist, node_slice, dry_run=args.dry_run))

    for (mode, group_id), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: r["row_id"])
        if not args.force_rerun:
            if any(existing.get(r["row_id"], {}).get("status") == "COMPLETED" for r in rows):
                continue
        results.extend(launch_group(args.cluster, rows, alloc_job_id, alloc_nodelist, hosts_for_launch, dry_run=args.dry_run))

    merged = {row["row_id"]: row for row in out_rows}
    for row in results:
        merged[row["row_id"]] = row
    write_csv(REGISTRY_CSV, [merged[k] for k in sorted(merged)], registry_fields())

    run_log = {
        "date_tag": DATE_TAG,
        "timestamp": now_iso(),
        "alloc_job_id": alloc_job_id,
        "alloc_nodelist": alloc_nodelist,
        "cluster": args.cluster,
        "hosts": hosts,
        "busy_hosts": sorted(busy_hosts),
        "hosts_for_launch": hosts_for_launch,
        "matrix_csv": str(MATRIX_CSV),
        "registry_csv": str(REGISTRY_CSV),
        "selected_rows": len(selected),
        "results_written": len(results),
        "prewarm_scale_cache_force_refresh": prewarm_force_refresh,
        "prewarm_scale_cache_summary": prewarm_summary,
        "prewarm_scale_cache_results": prewarm_results,
        "prewarm_split_array_cache_force_refresh": prewarm_split_force_refresh,
        "prewarm_split_array_cache_summary": prewarm_split_summary,
        "prewarm_split_array_cache_results": prewarm_split_results,
        "dry_run": bool(args.dry_run),
        "force_rerun": bool(args.force_rerun),
        "phases": args.phases,
        "strategies": args.strategies,
        "launch_modes": args.launch_modes,
    }
    RUN_LOG_JSON.write_text(json.dumps(run_log, indent=2) + "\n")
    print(json.dumps(run_log, indent=2))


if __name__ == "__main__":
    main()
