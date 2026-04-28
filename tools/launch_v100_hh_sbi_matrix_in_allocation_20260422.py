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

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO / "data/important_notes/optimization_track_20260422_v100_hh_track4_checkpoint2_sbi_family_block"
DEFAULT_TABLES = DEFAULT_OUT_ROOT / "tables"
DEFAULT_NOTES = DEFAULT_OUT_ROOT / "notes"
DEFAULT_MATRIX_CSV = DEFAULT_TABLES / "v100_hh_track4_checkpoint2_sbi_family_block_matrix_20260422_v100_hh_track4_checkpoint2_sbi_family_block.csv"
DEFAULT_REGISTRY_CSV = DEFAULT_TABLES / "v100_hh_track4_checkpoint2_sbi_family_block_registry_20260422_v100_hh_track4_checkpoint2_sbi_family_block.csv"
DEFAULT_RUN_LOG_JSON = DEFAULT_NOTES / "v100_hh_track4_checkpoint2_sbi_family_block_run_log_20260422_v100_hh_track4_checkpoint2_sbi_family_block.json"
DEFAULT_STEP_SCRIPT = REPO / "tools/run_v100_sbi_step_20260422.sh"


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
        raise RuntimeError(f"Failed to resolve hostnames for nodelist={nodelist}: {proc.stderr}")
    return [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]


def active_step_nodes(alloc_job_id: str) -> set[str]:
    proc = subprocess.run(["squeue", "-s", "-h", "-j", str(alloc_job_id), "-o", "%i|%N"], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return set()
    used = set()
    for ln in (proc.stdout or "").splitlines():
        step_id, _, nodelist = ln.partition("|")
        if not re.fullmatch(r"\d+\.\d+", step_id):
            continue
        if not nodelist or nodelist in {"(null)", "None"}:
            continue
        for host in hostnames(nodelist):
            used.add(host)
    return used


def resolve_allocation(args: argparse.Namespace) -> Tuple[str, str, List[str]]:
    alloc_job_id = args.alloc_job_id or os.environ.get("SLURM_JOB_ID")
    alloc_nodelist = args.alloc_nodelist or os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not alloc_job_id or not alloc_nodelist:
        raise SystemExit("Need live allocation context or explicit --alloc-job-id/--alloc-nodelist.")
    return alloc_job_id, alloc_nodelist, hostnames(alloc_nodelist)


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


def build_env(row: Dict[str, str], alloc_job_id: str, run_output_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO),
            "ALLOC_JOB_ID": str(alloc_job_id),
            "RUN_ID": row["row_id"],
            "RUN_OUTPUT_ROOT": str(run_output_root),
            "PARAMS_FILE": row["params_file"],
            "SBI_METHOD": row["method_name"],
            "MODEL_NAME": row["model_name"],
            "SAMPLE_WITH": row["sample_with"],
            "DATA_DIR": row["shared_data_dir"],
            "DATA_PREFIX": row["data_prefix"],
            "CURR": row["curr"],
            "N_TRAIN": row["n_train"],
            "EVAL_LIMIT": row["eval_limit"],
            "DEVICE": row["device"],
            "FEATURE_MODE": row.get("feature_mode", "raw"),
            "EMBEDDING_DIM": row["embedding_dim"],
            "EMBEDDING_HIDDEN": row["embedding_hidden"],
            "HIDDEN_FEATURES": row["hidden_features"],
            "NUM_TRANSFORMS": row["num_transforms"],
            "TRAINING_BATCH_SIZE": row["training_batch_size"],
            "LEARNING_RATE": row["learning_rate"],
            "STOP_AFTER_EPOCHS": row["stop_after_epochs"],
            "MAX_NUM_EPOCHS": row["max_num_epochs"],
            "TRAIN_ADDITIVE_NOISE_STD": row.get("train_additive_noise_std", "0.0"),
            "TRAIN_MULTIPLICATIVE_NOISE_STD": row.get("train_multiplicative_noise_std", "0.0"),
            "TRAIN_BASELINE_DRIFT_STD": row.get("train_baseline_drift_std", "0.0"),
            "TRAIN_MASK_FRACTION": row.get("train_mask_fraction", "0.0"),
            "EVAL_ADDITIVE_NOISE_STD": row.get("eval_additive_noise_std", "0.0"),
            "EVAL_MULTIPLICATIVE_NOISE_STD": row.get("eval_multiplicative_noise_std", "0.0"),
            "EVAL_BASELINE_DRIFT_STD": row.get("eval_baseline_drift_std", "0.0"),
            "EVAL_MASK_FRACTION": row.get("eval_mask_fraction", "0.0"),
            "POSTERIOR_SAMPLES": row["posterior_samples"],
            "SEED": row["seed"],
            "COMPUTE_MAP": row["compute_map"],
            "MAP_NUM_ITER": row.get("map_num_iter", ""),
            "MAP_NUM_TO_OPTIMIZE": row.get("map_num_to_optimize", ""),
            "MAP_LEARNING_RATE": row.get("map_learning_rate", ""),
            "MAP_NUM_INIT_SAMPLES": row.get("map_num_init_samples", ""),
            "DECISION_RULE": row.get("decision_rule", "mean"),
            "SELECTION_SPLIT": row.get("selection_split", "validate"),
            "SELECTION_COV_LAMBDA": row.get("selection_cov_lambda", "1.0"),
            "SELECTION_CANDIDATE_RULES": row.get("selection_candidate_rules", "mean,median"),
        }
    )
    return env


def build_srun_cmd(step_script: Path, alloc_job_id: str, node_slice: Sequence[str], row: Dict[str, str]) -> List[str]:
    nodelist = ",".join(node_slice)
    return [
        "srun",
        f"--jobid={alloc_job_id}",
        "--exact",
        "-N1",
        f"-w{nodelist}",
        "--ntasks=1",
        "--ntasks-per-node=1",
        f"--cpus-per-task={row['cpus_per_node']}",
        f"--gres=gpu:v100:{row['gpus_per_node']}",
        "--kill-on-bad-exit=1",
        "bash",
        str(step_script),
    ]


def slices_for_mode(mode: str, group_rows: List[Dict[str, str]], hosts: Sequence[str]) -> List[List[str]]:
    if mode == "concurrent_4x1n":
        if len(group_rows) != 4 or len(hosts) < 4:
            raise RuntimeError("concurrent_4x1n requires 4 rows and 4 hosts")
        return [[hosts[i]] for i in range(4)]
    if mode == "concurrent_8x1n":
        if len(group_rows) != 8 or len(hosts) < 4:
            raise RuntimeError("concurrent_8x1n requires 8 rows and 4 hosts")
        return [[hosts[i // 2]] for i in range(8)]
    raise RuntimeError(f"Unsupported grouped launch_mode={mode}")


def launch_group(
    group_rows: List[Dict[str, str]],
    alloc_job_id: str,
    alloc_nodelist: str,
    hosts: Sequence[str],
    dry_run: bool,
    *,
    runs_root: Path,
    logs_root: Path,
    step_script: Path,
) -> List[Dict[str, str]]:
    mode = group_rows[0]["launch_mode"]
    slices = slices_for_mode(mode, group_rows, hosts)

    if dry_run:
        return []

    procs = []
    started_at = now_iso()
    results = []
    for row, node_slice in zip(group_rows, slices):
        run_output_root = make_run_output_root(runs_root, row)
        run_output_root.mkdir(parents=True, exist_ok=True)
        stdout_log, stderr_log = make_logs(logs_root, row)
        if row["shared_data_dir"]:
            ensure_shared_data(row["tar_path"], row["shared_data_dir"])
        env = build_env(row, alloc_job_id, run_output_root)
        cmd = build_srun_cmd(step_script, alloc_job_id, node_slice, row)
        out_f = stdout_log.open("w")
        err_f = stderr_log.open("w")
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env)
        procs.append((row, node_slice, proc, out_f, err_f, time.time(), run_output_root, stdout_log, stderr_log, started_at))

    for row, node_slice, proc, out_f, err_f, start_ts, run_output_root, stdout_log, stderr_log, started in procs:
        rc = proc.wait()
        out_f.close()
        err_f.close()
        ended = now_iso()
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
                "elapsed_sec": f"{max(0.0, time.time() - start_ts):.3f}",
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
    ap = argparse.ArgumentParser(description="Launch the Falcon V100 Checkpoint 2 SBI family block inside an existing allocation.")
    ap.add_argument("--alloc-job-id", default=None)
    ap.add_argument("--alloc-nodelist", default=None)
    ap.add_argument("--matrix-csv", default=str(DEFAULT_MATRIX_CSV))
    ap.add_argument("--registry-csv", default=str(DEFAULT_REGISTRY_CSV))
    ap.add_argument("--run-log-json", default=str(DEFAULT_RUN_LOG_JSON))
    ap.add_argument("--step-script", default=str(DEFAULT_STEP_SCRIPT))
    ap.add_argument("--row-id", dest="row_ids", action="append")
    ap.add_argument("--phase", dest="phases", action="append")
    ap.add_argument("--strategy", dest="strategies", action="append")
    ap.add_argument("--launch-mode", dest="launch_modes", action="append")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()

    matrix_csv = Path(args.matrix_csv)
    registry_csv = Path(args.registry_csv)
    run_log_json = Path(args.run_log_json)
    step_script = Path(args.step_script)
    out_root = registry_csv.parent.parent
    runs_root = out_root / "runs"
    logs_root = out_root / "logs"

    if not matrix_csv.exists():
        raise SystemExit(f"Missing matrix CSV: {matrix_csv}")
    if not step_script.exists():
        raise SystemExit(f"Missing step script: {step_script}")

    alloc_job_id, alloc_nodelist, hosts = resolve_allocation(args)
    busy_hosts = active_step_nodes(alloc_job_id)
    hosts_for_launch = [h for h in hosts if h not in busy_hosts]
    if len(hosts_for_launch) < 4:
        hosts_for_launch = hosts

    matrix_rows = [row for row in load_csv(matrix_csv) if row_matches(row, args)]
    existing = {row["row_id"]: row for row in load_csv(registry_csv)}

    selected = []
    for row in matrix_rows:
        prev = existing.get(row["row_id"])
        if prev and (not args.force_rerun) and prev.get("status") == "COMPLETED":
            continue
        selected.append(row)
    if args.limit > 0:
        selected = selected[: args.limit]

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[(row["launch_mode"], row["launch_group"])].append(row)

    results: List[Dict[str, str]] = []
    for (_mode, _group), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: r["row_id"])
        results.extend(
            launch_group(
                rows,
                alloc_job_id,
                alloc_nodelist,
                hosts_for_launch,
                dry_run=args.dry_run,
                runs_root=runs_root,
                logs_root=logs_root,
                step_script=step_script,
            )
        )

    merged = dict(existing)
    for row in results:
        merged[row["row_id"]] = row
    ordered = [merged[r["row_id"]] for r in load_csv(matrix_csv) if r["row_id"] in merged]
    write_csv(registry_csv, ordered, registry_fields())

    run_log = {
        "timestamp": now_iso(),
        "alloc_job_id": alloc_job_id,
        "alloc_nodelist": alloc_nodelist,
        "hosts": hosts,
        "busy_hosts": sorted(busy_hosts),
        "hosts_for_launch": hosts_for_launch,
        "selected_rows": len(selected),
        "results_written": len(results),
        "dry_run": bool(args.dry_run),
        "force_rerun": bool(args.force_rerun),
    }
    run_log_json.write_text(json.dumps(run_log, indent=2) + "\n")
    print(json.dumps(run_log, indent=2))


if __name__ == "__main__":
    main()
