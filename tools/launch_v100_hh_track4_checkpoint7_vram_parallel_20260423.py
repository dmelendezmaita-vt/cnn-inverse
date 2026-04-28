#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
STEP_SCRIPT = REPO / "tools" / "run_v100_sbi_step_20260422.sh"
BASE_PARAMS = (
    "src/pytorch/configs/reruns_20260422_v100_hh_track4_checkpoint2_snpe_largebudget_followup/"
    "track4_hh_full/params_t4cp2large_snpe_maf_h192_t8_n12288_s1201.yaml"
)
SHARED_DATA = (
    REPO
    / "data"
    / "important_notes"
    / "optimization_track_20260411_v100_interactive"
    / "shared_data"
    / "track4_hh_full"
    / "concatenated_data"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def hostnames(nodelist: str) -> list[str]:
    proc = subprocess.run(["scontrol", "show", "hostnames", nodelist], text=True, capture_output=True, check=True)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_paths(date_tag: str) -> dict[str, Path]:
    root = REPO / "data" / "important_notes" / f"optimization_track_{date_tag}"
    tables = root / "tables"
    runs = root / "runs"
    logs = root / "logs"
    notes = root / "notes"
    return {
        "root": root,
        "tables": tables,
        "runs": runs,
        "logs": logs,
        "notes": notes,
        "matrix_csv": tables / f"v100_hh_track4_checkpoint7_vram_parallel_matrix_{date_tag}.csv",
        "registry_csv": tables / f"v100_hh_track4_checkpoint7_vram_parallel_registry_{date_tag}.csv",
        "run_log_jsonl": notes / f"v100_hh_track4_checkpoint7_vram_parallel_run_log_{date_tag}.jsonl",
    }


def base_row(row_id: str, representative: str, method: str, model_name: str, sample_with: str, seed: int) -> dict[str, str]:
    return {
        "row_id": row_id,
        "representative": representative,
        "method_name": method,
        "model_name": model_name,
        "sample_with": sample_with,
        "seed": str(seed),
        "params_file": BASE_PARAMS,
        "data_dir": str(SHARED_DATA),
        "data_prefix": "concatenated_data",
        "curr": "0.1",
        "n_train": "12288",
        "eval_limit": "256",
        "device": "cuda",
        "feature_mode": "raw_plus_fft256_summary12",
        "embedding_dim": "64",
        "embedding_hidden": "256",
        "hidden_features": "192",
        "num_transforms": "8",
        "training_batch_size": "128",
        "learning_rate": "5.0e-4",
        "stop_after_epochs": "20",
        "max_num_epochs": "100",
        "train_additive_noise_std": "0.0",
        "train_multiplicative_noise_std": "0.0",
        "train_baseline_drift_std": "0.0",
        "train_mask_fraction": "0.0",
        "eval_additive_noise_std": "0.0",
        "eval_multiplicative_noise_std": "0.0",
        "eval_baseline_drift_std": "0.0",
        "eval_mask_fraction": "0.0",
        "posterior_samples": "16",
        "decision_rule": "mean",
        "selection_split": "validate",
        "selection_cov_lambda": "1.0",
        "selection_candidate_rules": "mean,median",
        "status": "PENDING",
        "notes": "Checkpoint7 parallel V100 GPU telemetry pass for final representative coverage.",
    }


def build_matrix(seed: int, row_id_prefix: str) -> list[dict[str, str]]:
    rows = [
        base_row(f"{row_id_prefix}_0001", "sbi_feature_aware_snpe_clean", "snpe", "maf", "direct", seed),
        base_row(f"{row_id_prefix}_0002", "sbi_fmpe_clean", "fmpe", "mlp", "ode", seed),
        base_row(f"{row_id_prefix}_0003", "sbi_npse_clean_not_promoted", "npse", "mlp", "sde", seed),
        base_row(f"{row_id_prefix}_0004", "sbi_snpe_matched_drift010", "snpe", "maf", "direct", seed),
        base_row(f"{row_id_prefix}_0005", "sbi_snpe_matched_mask20", "snpe", "maf", "direct", seed),
    ]
    rows[3]["train_baseline_drift_std"] = "0.1"
    rows[3]["eval_baseline_drift_std"] = "0.1"
    rows[4]["train_mask_fraction"] = "0.2"
    rows[4]["eval_mask_fraction"] = "0.2"
    return rows


def env_for_row(row: dict[str, str], run_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO),
            "RUN_ID": row["row_id"],
            "RUN_OUTPUT_ROOT": str(run_root),
            "PARAMS_FILE": row["params_file"],
            "SBI_METHOD": row["method_name"],
            "MODEL_NAME": row["model_name"],
            "SAMPLE_WITH": row["sample_with"],
            "DATA_DIR": row["data_dir"],
            "DATA_PREFIX": row["data_prefix"],
            "CURR": row["curr"],
            "N_TRAIN": row["n_train"],
            "EVAL_LIMIT": row["eval_limit"],
            "DEVICE": row["device"],
            "FEATURE_MODE": row["feature_mode"],
            "EMBEDDING_DIM": row["embedding_dim"],
            "EMBEDDING_HIDDEN": row["embedding_hidden"],
            "HIDDEN_FEATURES": row["hidden_features"],
            "NUM_TRANSFORMS": row["num_transforms"],
            "TRAINING_BATCH_SIZE": row["training_batch_size"],
            "LEARNING_RATE": row["learning_rate"],
            "STOP_AFTER_EPOCHS": row["stop_after_epochs"],
            "MAX_NUM_EPOCHS": row["max_num_epochs"],
            "TRAIN_ADDITIVE_NOISE_STD": row["train_additive_noise_std"],
            "TRAIN_MULTIPLICATIVE_NOISE_STD": row["train_multiplicative_noise_std"],
            "TRAIN_BASELINE_DRIFT_STD": row["train_baseline_drift_std"],
            "TRAIN_MASK_FRACTION": row["train_mask_fraction"],
            "EVAL_ADDITIVE_NOISE_STD": row["eval_additive_noise_std"],
            "EVAL_MULTIPLICATIVE_NOISE_STD": row["eval_multiplicative_noise_std"],
            "EVAL_BASELINE_DRIFT_STD": row["eval_baseline_drift_std"],
            "EVAL_MASK_FRACTION": row["eval_mask_fraction"],
            "POSTERIOR_SAMPLES": row["posterior_samples"],
            "SEED": row["seed"],
            "COMPUTE_MAP": "0",
            "DECISION_RULE": row["decision_rule"],
            "SELECTION_SPLIT": row["selection_split"],
            "SELECTION_COV_LAMBDA": row["selection_cov_lambda"],
            "SELECTION_CANDIDATE_RULES": row["selection_candidate_rules"],
            "OMP_NUM_THREADS": "12",
            "MKL_NUM_THREADS": "12",
        }
    )
    return env


def launch_one(
    row: dict[str, str],
    alloc_job_id: str,
    host: str,
    runs_root: Path,
    logs_root: Path,
) -> dict[str, object]:
    run_root = runs_root / row["row_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_root / f"{row['row_id']}.out"
    stderr_log = logs_root / f"{row['row_id']}.err"
    cmd = [
        "srun",
        f"--jobid={alloc_job_id}",
        "--exact",
        "-N1",
        f"-w{host}",
        "--ntasks=1",
        "--ntasks-per-node=1",
        "--cpus-per-task=12",
        "--gres=gpu:v100:1",
        "--kill-on-bad-exit=1",
        "bash",
        str(STEP_SCRIPT),
    ]
    started_at = now_iso()
    start_ts = time.time()
    out_f = stdout_log.open("w")
    err_f = stderr_log.open("w")
    proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, env=env_for_row(row, run_root), text=True)
    return {
        "row": row,
        "host": host,
        "proc": proc,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
        "out_f": out_f,
        "err_f": err_f,
        "run_root": run_root,
        "started_at": started_at,
        "start_ts": start_ts,
    }


def finalize(entry: dict[str, object]) -> dict[str, str]:
    row = entry["row"]
    proc = entry["proc"]
    entry["out_f"].close()
    entry["err_f"].close()
    ended_at = now_iso()
    return {
        "row_id": row["row_id"],
        "representative": row["representative"],
        "status": "COMPLETED" if proc.returncode == 0 else "FAILED",
        "return_code": str(proc.returncode),
        "alloc_job_id": os.environ.get("ALLOC_JOB_ID", ""),
        "assigned_node": entry["host"],
        "started_at": entry["started_at"],
        "ended_at": ended_at,
        "elapsed_sec": f"{time.time() - entry['start_ts']:.3f}",
        "run_output_root": str(entry["run_root"]),
        "stdout_log": str(entry["stdout_log"]),
        "stderr_log": str(entry["stderr_log"]),
        "notes": row["notes"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run parallel V100 GPU telemetry tests for HH Track4 final SBI reps.")
    ap.add_argument("--alloc-job-id", required=True)
    ap.add_argument("--alloc-nodelist", required=True)
    ap.add_argument("--date-tag", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--row-id-prefix", required=True)
    ap.add_argument("--parallelism", type=int, default=5)
    ap.add_argument("--continue-on-failure", action="store_true")
    args = ap.parse_args()

    os.environ["ALLOC_JOB_ID"] = args.alloc_job_id
    paths = make_paths(args.date_tag)
    for key in ("tables", "runs", "logs", "notes"):
        paths[key].mkdir(parents=True, exist_ok=True)

    rows = build_matrix(args.seed, args.row_id_prefix)
    write_csv(paths["matrix_csv"], rows, list(rows[0].keys()))

    hosts = hostnames(args.alloc_nodelist)
    if not hosts:
        raise SystemExit("No hosts resolved from allocation nodelist.")
    max_parallelism = max(1, min(args.parallelism, len(hosts) * 2))
    host_cycle = deque([hosts[i % len(hosts)] for i in range(max_parallelism)])

    registry: list[dict[str, str]] = []
    queue = deque(rows)
    active: list[dict[str, object]] = []

    while queue or active:
        while queue and len(active) < max_parallelism:
            row = queue.popleft()
            host = host_cycle[0]
            host_cycle.rotate(-1)
            print(f"[{now_iso()}] starting {row['row_id']} {row['representative']} on {host}", flush=True)
            active.append(launch_one(row, args.alloc_job_id, host, paths["runs"], paths["logs"]))

        next_active: list[dict[str, object]] = []
        for entry in active:
            proc = entry["proc"]
            rc = proc.poll()
            if rc is None:
                next_active.append(entry)
                continue
            print(
                f"[{now_iso()}] finished {entry['row']['row_id']} status={'COMPLETED' if rc == 0 else 'FAILED'}",
                flush=True,
            )
            result = finalize(entry)
            registry.append(result)
            write_csv(paths["registry_csv"], registry, list(registry[0].keys()))
            with paths["run_log_jsonl"].open("a") as f:
                f.write(json.dumps(result, sort_keys=True) + "\n")
            if rc != 0 and not args.continue_on_failure:
                raise SystemExit(f"Stopping after failed row {entry['row']['row_id']}")
        active = next_active
        if active:
            time.sleep(2.0)

    print(f"Wrote matrix: {paths['matrix_csv']}")
    print(f"Wrote registry: {paths['registry_csv']}")
    print(f"Wrote run log: {paths['run_log_jsonl']}")


if __name__ == "__main__":
    main()
