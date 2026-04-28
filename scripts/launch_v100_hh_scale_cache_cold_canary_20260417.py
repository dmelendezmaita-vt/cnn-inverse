#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from features_scale_cache_utils import warm_feature_scale_cache_for_rows
from shared_data_utils import ensure_shared_data

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
OUT_ROOT = REPO / "data/important_notes/optimization_track_20260417_v100_hh_scale_cache_cold_canary"
NOTES = OUT_ROOT / "notes"
RUNS = OUT_ROOT / "runs"
LOGS = OUT_ROOT / "logs"
STEP_SCRIPT = REPO / "scripts" / "run_v100_interactive_step_20260411.sh"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def hostnames(nodelist: str):
    proc = subprocess.run(["scontrol", "show", "hostnames", nodelist], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or f"failed to expand nodelist={nodelist}")
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
        if not re.fullmatch(r"\d+\.\d+", step_id):
            continue
        if not nodelist or nodelist in {"(null)", "None"}:
            continue
        for h in hostnames(nodelist):
            used.add(h)
    return used


def make_row(track_key: str, seed: int) -> dict[str, str]:
    if track_key == "track3_hh_reduced":
        return {
            "row_id": f"v100cachecold_{track_key}_s{seed}",
            "phase": "phaseR4_scale_cache_cold_canary",
            "launch_mode": "dedicated",
            "launch_group": "",
            "strategy_id": "R1_accum2_nosync",
            "track_key": track_key,
            "policy": "strong",
            "nodes": "4",
            "seed": str(seed),
            "gpus_per_node": "2",
            "cpus_per_node": "24",
            "params_file": "pytorch/configs/reruns_20260416_v100_hh_datapath_redesign_r4/track3_hh_reduced/params_t3_R1_accum2_nosync_n4_s971.yaml",
            "tar_path": "/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar",
            "data_prefix": "reduced_data",
            "curr": "0.1",
            "data_access_mode": "direct_tar",
            "shared_data_dir": "/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260411_v100_interactive/shared_data/track3_hh_reduced/reduced_data",
            "save_predictions": "none",
        }
    if track_key == "track4_hh_full":
        return {
            "row_id": f"v100cachecold_{track_key}_s{seed}",
            "phase": "phaseR4_scale_cache_cold_canary",
            "launch_mode": "dedicated",
            "launch_group": "",
            "strategy_id": "R1_accum2_nosync",
            "track_key": track_key,
            "policy": "strong",
            "nodes": "4",
            "seed": str(seed),
            "gpus_per_node": "2",
            "cpus_per_node": "24",
            "params_file": "pytorch/configs/reruns_20260416_v100_hh_datapath_redesign_r4/track4_hh_full/params_t4_R1_accum2_nosync_n4_s971.yaml",
            "tar_path": "/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar",
            "data_prefix": "concatenated_data",
            "curr": "0.1",
            "data_access_mode": "direct_tar",
            "shared_data_dir": "/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260411_v100_interactive/shared_data/track4_hh_full/concatenated_data",
            "save_predictions": "test",
        }
    raise ValueError(f"Unsupported track_key={track_key}")


def build_env(row, alloc_job_id: str, master_addr: str, master_port: int, run_output_root: Path):
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


def build_srun_cmd(alloc_job_id: str, node_slice, row):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc-job-id", required=True)
    ap.add_argument("--alloc-nodelist", required=True)
    ap.add_argument("--track-key", choices=["track3_hh_reduced", "track4_hh_full"], default="track3_hh_reduced")
    ap.add_argument("--seed", type=int, default=971)
    ap.add_argument("--force-clear-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    row = make_row(args.track_key, args.seed)
    hosts = hostnames(args.alloc_nodelist)
    busy_hosts = active_step_nodes(args.alloc_job_id)
    node_slice = [h for h in hosts if h not in busy_hosts] or hosts
    node_slice = node_slice[: int(row["nodes"])]
    if len(node_slice) < int(row["nodes"]):
        raise RuntimeError(f"Need {row['nodes']} hosts, have {len(node_slice)}")

    ensure_shared_data(row["tar_path"], row["shared_data_dir"])
    warm_results = warm_feature_scale_cache_for_rows([row], force=args.force_clear_cache)

    run_output_root = RUNS / row["track_key"] / row["row_id"]
    run_output_root.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    stdout_log = LOGS / f"{row['row_id']}.out"
    stderr_log = LOGS / f"{row['row_id']}.err"

    started_at = now_iso()
    master_addr = node_slice[0]
    master_port = 33650 + (abs(hash(row["row_id"])) % 100)
    cmd = build_srun_cmd(args.alloc_job_id, node_slice, row)
    env = build_env(row, args.alloc_job_id, master_addr, master_port, run_output_root)

    payload = {
        "timestamp": now_iso(),
        "row": row,
        "alloc_job_id": args.alloc_job_id,
        "alloc_nodelist": args.alloc_nodelist,
        "assigned_nodes": node_slice,
        "warm_results": warm_results,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "run_output_root": str(run_output_root),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        payload["status"] = "dry_run"
        NOTES.mkdir(parents=True, exist_ok=True)
        (NOTES / f"{row['row_id']}_dry_run.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    start_ts = time.time()
    with stdout_log.open("w") as out_f, stderr_log.open("w") as err_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env)
        rc = proc.wait()

    payload.update(
        {
            "status": "COMPLETED" if rc == 0 else "FAILED",
            "return_code": rc,
            "started_at": started_at,
            "ended_at": now_iso(),
            "elapsed_sec": round(time.time() - start_ts, 3),
        }
    )
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / f"{row['row_id']}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
