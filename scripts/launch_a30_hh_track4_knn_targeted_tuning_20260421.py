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

from shared_data_utils import ensure_shared_data

DATE_TAG = "20260421_a30_hh_track4_knn_targeted_tuning"
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
RUNS = OUT_ROOT / "runs"
LOGS = OUT_ROOT / "logs"

MATRIX_CSV = TABLES / f"a30_hh_track4_knn_targeted_tuning_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"a30_hh_track4_knn_targeted_tuning_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"a30_hh_track4_knn_targeted_tuning_run_log_{DATE_TAG}.json"
STEP_SCRIPT = REPO / "scripts" / "run_classical_baseline_step_20260420.sh"


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
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to resolve hostnames for nodelist={nodelist}: {proc.stderr}")
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def active_step_nodes(alloc_job_id: str, cluster: Optional[str] = None) -> set[str]:
    cmd = ["squeue"]
    if cluster:
        cmd.extend(["-M", cluster])
    cmd.extend(["-s", "-h", "-j", str(alloc_job_id), "-o", "%i|%N"])
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
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
        for host in hostnames(nodelist, cluster=cluster):
            used.add(host)
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


def build_env(row: Dict[str, str], run_output_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO),
            "RUN_ID": row["row_id"],
            "RUN_OUTPUT_ROOT": str(run_output_root),
            "PARAMS_FILE": row["params_file"],
            "BASELINE_NAME": row["baseline_name"],
            "DATA_DIR": row["shared_data_dir"],
            "DATA_PREFIX": row["data_prefix"],
            "CURR": row["curr"],
        }
    )
    return env


def build_srun_cmd(
    cluster: Optional[str],
    alloc_job_id: str,
    node_slice: Sequence[str],
    row: Dict[str, str],
) -> List[str]:
    nodelist = ",".join(node_slice)
    cmd = ["srun"]
    if cluster:
        cmd.extend(["-M", cluster])
    cmd.extend(
        [
            f"--jobid={alloc_job_id}",
            "--exclusive",
            "-N1",
            f"-w{nodelist}",
            "--ntasks=1",
            "--ntasks-per-node=1",
            f"--cpus-per-task={row['cpus_per_node']}",
            "--kill-on-bad-exit=1",
            "bash",
            str(STEP_SCRIPT),
        ]
    )
    return cmd


def launch_group(
    cluster: Optional[str],
    group_rows: List[Dict[str, str]],
    alloc_job_id: str,
    alloc_nodelist: str,
    hosts: Sequence[str],
    dry_run: bool,
) -> List[Dict[str, str]]:
    mode = group_rows[0]["launch_mode"]
    if mode != "concurrent_4x1n":
        raise RuntimeError(f"Unsupported launch_mode={mode}; baseline finalists package expects concurrent_4x1n")
    if len(group_rows) > 4 or len(hosts) < len(group_rows):
        raise RuntimeError(f"{mode} requires <=4 rows and enough hosts, got rows={len(group_rows)} hosts={len(hosts)}")

    slices = [[hosts[i]] for i in range(len(group_rows))]
    started_at = now_iso()
    procs = []
    results = []
    for row, node_slice in zip(group_rows, slices):
        run_output_root = make_run_output_root(row)
        run_output_root.mkdir(parents=True, exist_ok=True)
        stdout_log, stderr_log = make_logs(row)
        if row["shared_data_dir"] and not dry_run:
            ensure_shared_data(row["tar_path"], row["shared_data_dir"])
        cmd = build_srun_cmd(cluster, alloc_job_id, node_slice, row)
        env = build_env(row, run_output_root)
        if dry_run:
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
            )
            continue
        out_f = stdout_log.open("w")
        err_f = stderr_log.open("w")
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env)
        procs.append((row, node_slice, proc, out_f, err_f, time.time(), run_output_root, stdout_log, stderr_log))

    if dry_run:
        return results

    for row, node_slice, proc, out_f, err_f, start_ts, run_output_root, stdout_log, stderr_log in procs:
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
                "started_at": started_at,
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch Track4 A30 knn targeted tuning rows inside an existing A30 allocation.")
    ap.add_argument("--alloc-job-id", default=None)
    ap.add_argument("--alloc-nodelist", default=None)
    ap.add_argument("--row-id", dest="row_ids", action="append")
    ap.add_argument("--phase", dest="phases", action="append")
    ap.add_argument("--strategy", dest="strategies", action="append")
    ap.add_argument("--launch-mode", dest="launch_modes", action="append")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--cluster", default=os.environ.get("SLURM_CLUSTER_NAME") or os.environ.get("SLURM_CLUSTERS"))
    args = ap.parse_args()

    if not MATRIX_CSV.exists():
        raise SystemExit(f"Missing matrix CSV: {MATRIX_CSV}. Run the builder first.")
    if not STEP_SCRIPT.exists():
        raise SystemExit(f"Missing step script: {STEP_SCRIPT}")

    alloc_job_id, alloc_nodelist, hosts_all = resolve_allocation(args)
    used_nodes = active_step_nodes(alloc_job_id, cluster=args.cluster)
    hosts = [h for h in hosts_all if h not in used_nodes]
    if len(hosts) < 4:
        raise SystemExit(f"Need 4 free hosts in allocation {alloc_job_id}, found {len(hosts)} free: {hosts}")

    matrix_rows = [row for row in load_csv(MATRIX_CSV) if row_matches(row, args)]
    if args.limit > 0:
        matrix_rows = matrix_rows[: args.limit]

    existing = {row["row_id"]: row for row in load_csv(REGISTRY_CSV)}
    pending_rows = []
    for row in matrix_rows:
        prev = existing.get(row["row_id"])
        if prev and prev.get("status") == "COMPLETED" and not args.force_rerun:
            continue
        pending_rows.append(row)

    if not pending_rows:
        print("All requested baseline finalist rows already completed.")
        return

    registry_rows = list(existing.values())
    pending_by_group: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in pending_rows:
        pending_by_group[row["launch_group"]].append(row)

    for group_name in sorted(pending_by_group):
        group_rows = pending_by_group[group_name]
        group_results = launch_group(
            cluster=args.cluster,
            group_rows=group_rows,
            alloc_job_id=alloc_job_id,
            alloc_nodelist=alloc_nodelist,
            hosts=hosts[:4],
            dry_run=args.dry_run,
        )
        for result in group_results:
            existing[result["row_id"]] = result
        registry_rows = list(existing.values())
        write_csv(REGISTRY_CSV, registry_rows, registry_fields())

    summary = {
        "date_tag": DATE_TAG,
        "ran_at": now_iso(),
        "alloc_job_id": alloc_job_id,
        "alloc_nodelist": alloc_nodelist,
        "completed_rows": sum(1 for row in registry_rows if row["status"] == "COMPLETED"),
        "failed_rows": sum(1 for row in registry_rows if row["status"] == "FAILED"),
        "dryrun_rows": sum(1 for row in registry_rows if row["status"] == "DRYRUN"),
    }
    RUN_LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
