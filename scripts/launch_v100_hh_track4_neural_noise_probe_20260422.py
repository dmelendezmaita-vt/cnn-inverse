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

DATE_TAG = "20260422_v100_hh_track4_neural_noise_probe"
REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
DEFAULT_OUT_ROOT = REPO / f"data/important_notes/optimization_track_{DATE_TAG}"
TABLES = DEFAULT_OUT_ROOT / "tables"
NOTES = DEFAULT_OUT_ROOT / "notes"

MATRIX_CSV = TABLES / f"v100_hh_track4_neural_noise_probe_matrix_{DATE_TAG}.csv"
REGISTRY_CSV = TABLES / f"v100_hh_track4_neural_noise_probe_registry_{DATE_TAG}.csv"
RUN_LOG_JSON = NOTES / f"v100_hh_track4_neural_noise_probe_run_log_{DATE_TAG}.json"
STEP_SCRIPT = REPO / "scripts/run_interactive_step_with_gpu_monitor_20260420.sh"


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


def make_run_output_root(runs_root: Path, row: Dict[str, str]) -> Path:
    return runs_root / row["phase"] / row["launch_mode"] / row["row_id"]


def make_logs(logs_root: Path, row: Dict[str, str]) -> Tuple[Path, Path]:
    logs_root.mkdir(parents=True, exist_ok=True)
    return logs_root / f"{row['row_id']}.out", logs_root / f"{row['row_id']}.err"


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
            "EVAL_ONLY_CHECKPOINT": row["eval_only_checkpoint"],
            "EVAL_ONLY_USE_GPU": row.get("eval_only_use_gpu", os.environ.get("EVAL_ONLY_USE_GPU", "0")),
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
            "--exact",
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


def slices_for_mode(mode: str, group_rows: List[Dict[str, str]], hosts: Sequence[str]) -> List[List[str]]:
    if mode == "concurrent_4x1n":
        if len(group_rows) != 4 or len(hosts) < 4:
            raise RuntimeError("concurrent_4x1n requires 4 rows and 4 hosts")
        return [[hosts[i]] for i in range(4)]
    if mode == "concurrent_8x1n":
        if len(group_rows) != 8 or len(hosts) < 4:
            raise RuntimeError("concurrent_8x1n requires 8 rows and 4 hosts")
        return [[hosts[i // 2]] for i in range(8)]
    raise RuntimeError(f"Unsupported launch_mode={mode}; neural noise launcher expects concurrent_4x1n or concurrent_8x1n")


def launch_group(
    cluster: Optional[str],
    group_rows: List[Dict[str, str]],
    alloc_job_id: str,
    alloc_nodelist: str,
    hosts: Sequence[str],
    dry_run: bool,
    *,
    runs_root: Path,
    logs_root: Path,
) -> List[Dict[str, str]]:
    mode = group_rows[0]["launch_mode"]
    slices = slices_for_mode(mode, group_rows, hosts)

    started_at = now_iso()
    procs = []
    results = []
    for row, node_slice in zip(group_rows, slices):
        run_output_root = make_run_output_root(runs_root, row)
        run_output_root.mkdir(parents=True, exist_ok=True)
        stdout_log, stderr_log = make_logs(logs_root, row)
        if row["shared_data_dir"] and not dry_run:
            ensure_shared_data(row["tar_path"], row["shared_data_dir"])
        master_addr = node_slice[0]
        master_port = 20000 + (abs(hash(row["row_id"])) % 20000)
        cmd = build_srun_cmd(cluster, alloc_job_id, node_slice, row)
        env = build_env(row, alloc_job_id, master_addr, master_port, run_output_root)
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
    ap = argparse.ArgumentParser(description="Launch V100 Track4 neural noise probe rows inside an existing V100 allocation.")
    ap.add_argument("--alloc-job-id", default=None)
    ap.add_argument("--alloc-nodelist", default=None)
    ap.add_argument("--matrix-csv", default=str(MATRIX_CSV))
    ap.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    ap.add_argument("--run-log-json", default=str(RUN_LOG_JSON))
    ap.add_argument("--row-id", dest="row_ids", action="append")
    ap.add_argument("--phase", dest="phases", action="append")
    ap.add_argument("--strategy", dest="strategies", action="append")
    ap.add_argument("--launch-mode", dest="launch_modes", action="append")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--cluster", default=os.environ.get("SLURM_CLUSTER_NAME") or os.environ.get("SLURM_CLUSTERS"))
    args = ap.parse_args()

    matrix_csv = Path(args.matrix_csv)
    registry_csv = Path(args.registry_csv)
    run_log_json = Path(args.run_log_json)
    out_root = registry_csv.parent.parent
    notes_root = out_root / "notes"
    runs_root = out_root / "runs"
    logs_root = out_root / "logs"

    if not matrix_csv.exists():
        raise SystemExit(f"Missing matrix CSV: {matrix_csv}. Run the builder first.")
    if not STEP_SCRIPT.exists():
        raise SystemExit(f"Missing step script: {STEP_SCRIPT}")

    alloc_job_id, alloc_nodelist, hosts_all = resolve_allocation(args)
    used_nodes = active_step_nodes(alloc_job_id, cluster=args.cluster)
    hosts = [h for h in hosts_all if h not in used_nodes]
    if len(hosts) < 4:
        raise SystemExit(f"Need 4 free hosts in allocation {alloc_job_id}, found {len(hosts)} free: {hosts}")

    matrix_rows = [row for row in load_csv(matrix_csv) if row_matches(row, args)]
    if args.limit > 0:
        matrix_rows = matrix_rows[: args.limit]

    existing = {row["row_id"]: row for row in load_csv(registry_csv)}
    pending_rows = []
    for row in matrix_rows:
        prev = existing.get(row["row_id"])
        if prev and prev.get("status") == "COMPLETED" and not args.force_rerun:
            continue
        pending_rows.append(row)

    if args.limit > 0:
        pending_rows = pending_rows[: args.limit]

    split_array_cache_enabled = os.environ.get("PREWARM_SPLIT_ARRAY_CACHE", "1") != "0"
    pending_rows = prepare_rows_for_split_array_cache(
        pending_rows,
        notes_root / "split_array_cache_param_overrides",
        enable=split_array_cache_enabled,
    )

    if not pending_rows:
        print("All requested neural noise probe rows already completed.")
        return

    prewarm_results = warm_feature_scale_cache_for_rows(pending_rows)
    prewarm_split_results = warm_split_array_cache_for_rows(pending_rows)

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
            runs_root=runs_root,
            logs_root=logs_root,
        )
        for result in group_results:
            existing[result["row_id"]] = result
        registry_rows = list(existing.values())
        write_csv(registry_csv, registry_rows, registry_fields())

    summary = {
        "date_tag": DATE_TAG,
        "ran_at": now_iso(),
        "alloc_job_id": alloc_job_id,
        "alloc_nodelist": alloc_nodelist,
        "completed_rows": sum(1 for row in registry_rows if row["status"] == "COMPLETED"),
        "failed_rows": sum(1 for row in registry_rows if row["status"] == "FAILED"),
        "dryrun_rows": sum(1 for row in registry_rows if row["status"] == "DRYRUN"),
        "feature_scale_cache": summarize_feature_scale_cache_results(prewarm_results),
        "split_array_cache": summarize_split_array_cache_results(prewarm_split_results),
    }
    run_log_json.parent.mkdir(parents=True, exist_ok=True)
    run_log_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
