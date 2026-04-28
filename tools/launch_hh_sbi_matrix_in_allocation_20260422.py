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
from typing import Dict, List, Optional

from shared_data_utils import ensure_shared_data

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STEP_SCRIPT = REPO / "tools" / "run_sbi_baseline_step_20260422.sh"


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


def resolve_allocation(args: argparse.Namespace) -> tuple[str, str, str]:
    alloc_job_id = args.alloc_job_id or os.environ.get("SLURM_JOB_ID")
    alloc_nodelist = args.alloc_nodelist or os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not alloc_job_id or not alloc_nodelist:
        raise SystemExit(
            "This launcher must run inside the allocation or receive --alloc-job-id and --alloc-nodelist explicitly."
        )
    hosts = hostnames(alloc_nodelist, cluster=args.cluster)
    if not hosts:
        raise SystemExit(f"Could not resolve any hosts for allocation nodelist={alloc_nodelist}")
    return alloc_job_id, alloc_nodelist, hosts[0]


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
        "device",
        "notes",
    ]


def row_matches(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if args.row_ids and row["row_id"] not in args.row_ids:
        return False
    if args.phases and row["phase"] not in args.phases:
        return False
    if args.strategies and row["strategy_id"] not in args.strategies:
        return False
    return True


def make_run_output_root(out_root: Path, row: Dict[str, str]) -> Path:
    return out_root / "runs" / row["phase"] / row["launch_mode"] / row["row_id"]


def make_logs(out_root: Path, row: Dict[str, str]) -> tuple[Path, Path]:
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{row['row_id']}.out", logs_dir / f"{row['row_id']}.err"


def build_env(row: Dict[str, str], run_output_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO),
            "RUN_ID": row["row_id"],
            "RUN_OUTPUT_ROOT": str(run_output_root),
            "PARAMS_FILE": row["params_file"],
            "SBI_METHOD": row["method_name"],
            "DENSITY_ESTIMATOR": row["density_estimator"],
            "DATA_DIR": row["shared_data_dir"],
            "DATA_PREFIX": row["data_prefix"],
            "CURR": row["curr"],
            "N_TRAIN": row["n_train"],
            "EVAL_LIMIT": row["eval_limit"],
            "DEVICE": row["device"],
            "EMBEDDING_DIM": row["embedding_dim"],
            "EMBEDDING_HIDDEN": row["embedding_hidden"],
            "HIDDEN_FEATURES": row["hidden_features"],
            "NUM_TRANSFORMS": row["num_transforms"],
            "TRAINING_BATCH_SIZE": row["training_batch_size"],
            "LEARNING_RATE": row["learning_rate"],
            "STOP_AFTER_EPOCHS": row["stop_after_epochs"],
            "MAX_NUM_EPOCHS": row["max_num_epochs"],
            "POSTERIOR_SAMPLES": row["posterior_samples"],
            "SEED": row["seed"],
        }
    )
    return env


def build_srun_cmd(
    cluster: Optional[str],
    alloc_job_id: str,
    host: str,
    row: Dict[str, str],
    step_script: Path,
) -> List[str]:
    cmd = ["srun"]
    if cluster:
        cmd.extend(["-M", cluster])
    cmd.extend(
        [
            f"--jobid={alloc_job_id}",
            "--exclusive",
            "-N1",
            f"-w{host}",
            "--ntasks=1",
            "--ntasks-per-node=1",
            f"--cpus-per-task={row['cpus_per_node']}",
            "--kill-on-bad-exit=1",
            "bash",
            str(step_script),
        ]
    )
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch the Checkpoint 2 SBI matrix inside an existing allocation.")
    ap.add_argument("--matrix-csv", required=True)
    ap.add_argument("--registry-csv", required=True)
    ap.add_argument("--run-log-json", required=True)
    ap.add_argument("--alloc-job-id", default=None)
    ap.add_argument("--alloc-nodelist", default=None)
    ap.add_argument("--row-id", dest="row_ids", action="append")
    ap.add_argument("--phase", dest="phases", action="append")
    ap.add_argument("--strategy", dest="strategies", action="append")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--step-script", default=str(DEFAULT_STEP_SCRIPT))
    ap.add_argument("--cluster", default=os.environ.get("SLURM_CLUSTER_NAME") or os.environ.get("SLURM_CLUSTERS"))
    args = ap.parse_args()

    matrix_csv = Path(args.matrix_csv)
    registry_csv = Path(args.registry_csv)
    run_log_json = Path(args.run_log_json)
    out_root = registry_csv.parent.parent
    step_script = Path(args.step_script)

    if not matrix_csv.exists():
        raise SystemExit(f"Missing matrix CSV: {matrix_csv}")
    if not step_script.exists():
        raise SystemExit(f"Missing step script: {step_script}")

    alloc_job_id, alloc_nodelist, host = resolve_allocation(args)
    matrix_rows = [row for row in load_csv(matrix_csv) if row_matches(row, args)]
    if args.limit and 0 < args.limit:
        matrix_rows = matrix_rows[: args.limit]
    if not matrix_rows:
        raise SystemExit("No matching rows selected from the SBI matrix.")

    existing_rows = load_csv(registry_csv)
    existing_by_id = {row["row_id"]: row for row in existing_rows}

    selected_ids = {row["row_id"] for row in matrix_rows}
    launched = 0
    for row in matrix_rows:
        prior = existing_by_id.get(row["row_id"])
        if prior and prior.get("status") == "COMPLETED" and not args.force_rerun:
            continue

        run_output_root = make_run_output_root(out_root, row)
        run_output_root.mkdir(parents=True, exist_ok=True)
        stdout_log, stderr_log = make_logs(out_root, row)

        if row["shared_data_dir"] and not args.dry_run:
            ensure_shared_data(row["tar_path"], row["shared_data_dir"])

        started_at = now_iso()
        if args.dry_run:
            result = {
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
                "assigned_nodes": host,
                "started_at": started_at,
                "ended_at": started_at,
                "elapsed_sec": "0.0",
                "run_output_root": str(run_output_root),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "params_file": row["params_file"],
                "shared_data_dir": row["shared_data_dir"],
                "device": row["device"],
                "notes": "",
            }
        else:
            cmd = build_srun_cmd(args.cluster, alloc_job_id, host, row, step_script)
            env = build_env(row, run_output_root)
            start_ts = time.time()
            with stdout_log.open("w") as out_f, stderr_log.open("w") as err_f:
                proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env)
                rc = proc.wait()
            ended_at = now_iso()
            result = {
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
                "assigned_nodes": host,
                "started_at": started_at,
                "ended_at": ended_at,
                "elapsed_sec": f"{max(0.0, time.time() - start_ts):.3f}",
                "run_output_root": str(run_output_root),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "params_file": row["params_file"],
                "shared_data_dir": row["shared_data_dir"],
                "device": row["device"],
                "notes": "",
            }
        existing_by_id[row["row_id"]] = result
        launched += 1
        ordered_rows = []
        for matrix_row in load_csv(matrix_csv):
            if matrix_row["row_id"] in existing_by_id:
                ordered_rows.append(existing_by_id[matrix_row["row_id"]])
        write_csv(registry_csv, ordered_rows, registry_fields())

    final_rows = []
    for matrix_row in load_csv(matrix_csv):
        if matrix_row["row_id"] in existing_by_id:
            final_rows.append(existing_by_id[matrix_row["row_id"]])
    write_csv(registry_csv, final_rows, registry_fields())

    counts = {}
    for row in final_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    run_log = {
        "timestamp": now_iso(),
        "alloc_job_id": alloc_job_id,
        "alloc_nodelist": alloc_nodelist,
        "assigned_host": host,
        "selected_rows": len(selected_ids),
        "launched_rows": launched,
        "status_counts": counts,
        "registry_csv": str(registry_csv),
    }
    run_log_json.parent.mkdir(parents=True, exist_ok=True)
    run_log_json.write_text(json.dumps(run_log, indent=2) + "\n")
    print(json.dumps(run_log, indent=2))


if __name__ == "__main__":
    main()
