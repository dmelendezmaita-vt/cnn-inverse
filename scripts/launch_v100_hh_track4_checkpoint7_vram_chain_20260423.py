#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import csv
from dataclasses import dataclass
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
DEFAULT_DATE_TAG = "20260423"
DEFAULT_OUTPUT_TAG = "v100_hh_track4_checkpoint7_vram_chain_r2"
DEFAULT_ROW_ID_PREFIX = "v100cp7vramr2"
DEFAULT_SEED = 3101
STEP_SCRIPT = REPO / "scripts" / "run_v100_sbi_step_20260422.sh"
BASE_PARAMS = (
    "pytorch/configs/reruns_20260422_v100_hh_track4_checkpoint2_snpe_largebudget_followup/"
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


@dataclass(frozen=True)
class RuntimePaths:
    combined_tag: str
    out_root: Path
    tables: Path
    runs: Path
    logs: Path
    notes: Path
    matrix_csv: Path
    registry_csv: Path
    run_log_jsonl: Path


@dataclass(frozen=True)
class Slot:
    host: str
    slot_label: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hostnames(nodelist: str) -> list[str]:
    proc = subprocess.run(["scontrol", "show", "hostnames", nodelist], text=True, capture_output=True, check=True)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def build_runtime_paths(date_tag: str, output_tag: str) -> RuntimePaths:
    combined_tag = f"{date_tag}_{output_tag}" if output_tag else date_tag
    out_root = REPO / "data" / "important_notes" / f"optimization_track_{combined_tag}"
    tables = out_root / "tables"
    runs = out_root / "runs"
    logs = out_root / "logs"
    notes = out_root / "notes"
    return RuntimePaths(
        combined_tag=combined_tag,
        out_root=out_root,
        tables=tables,
        runs=runs,
        logs=logs,
        notes=notes,
        matrix_csv=tables / f"v100_hh_track4_checkpoint7_vram_chain_matrix_{combined_tag}.csv",
        registry_csv=tables / f"v100_hh_track4_checkpoint7_vram_chain_registry_{combined_tag}.csv",
        run_log_jsonl=notes / f"v100_hh_track4_checkpoint7_vram_chain_run_log_{combined_tag}.jsonl",
    )


def base_row(
    row_id: str,
    representative: str,
    method: str,
    model_name: str,
    sample_with: str,
    seed: int,
    params_file: str,
    data_dir: str,
) -> dict[str, str]:
    return {
        "row_id": row_id,
        "representative": representative,
        "method_name": method,
        "model_name": model_name,
        "sample_with": sample_with,
        "seed": str(seed),
        "params_file": params_file,
        "data_dir": data_dir,
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
        "notes": "Checkpoint7 chained GPU VRAM telemetry pass for final representative coverage.",
    }


def build_matrix(row_id_prefix: str, seed: int, params_file: str, data_dir: str) -> list[dict[str, str]]:
    rows = [
        base_row(
            f"{row_id_prefix}_0001",
            "sbi_feature_aware_snpe_clean",
            "snpe",
            "maf",
            "direct",
            seed,
            params_file,
            data_dir,
        ),
        base_row(
            f"{row_id_prefix}_0002",
            "sbi_fmpe_clean",
            "fmpe",
            "mlp",
            "ode",
            seed,
            params_file,
            data_dir,
        ),
        base_row(
            f"{row_id_prefix}_0003",
            "sbi_npse_clean_not_promoted",
            "npse",
            "mlp",
            "sde",
            seed,
            params_file,
            data_dir,
        ),
        base_row(
            f"{row_id_prefix}_0004",
            "sbi_snpe_matched_drift010",
            "snpe",
            "maf",
            "direct",
            seed,
            params_file,
            data_dir,
        ),
        base_row(
            f"{row_id_prefix}_0005",
            "sbi_snpe_matched_mask20",
            "snpe",
            "maf",
            "direct",
            seed,
            params_file,
            data_dir,
        ),
    ]
    rows[3]["train_baseline_drift_std"] = "0.1"
    rows[3]["eval_baseline_drift_std"] = "0.1"
    rows[4]["train_mask_fraction"] = "0.2"
    rows[4]["eval_mask_fraction"] = "0.2"
    return rows


def env_for_row(row: dict[str, str], run_root: Path, omp_num_threads: int, mkl_num_threads: int) -> dict[str, str]:
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
            "OMP_NUM_THREADS": str(omp_num_threads),
            "MKL_NUM_THREADS": str(mkl_num_threads),
        }
    )
    return env


def build_slots(hosts: list[str], gpus_per_host: int, max_concurrent: int) -> list[Slot]:
    slots: list[Slot] = []
    for gpu_slot in range(gpus_per_host):
        for host in hosts:
            slots.append(Slot(host=host, slot_label=f"{host}:slot{gpu_slot + 1}"))
    return slots[:max_concurrent]


def sorted_registry_rows(registry_by_row_id: dict[str, dict[str, str]], row_order: dict[str, int]) -> list[dict[str, str]]:
    return sorted(registry_by_row_id.values(), key=lambda row: row_order[row["row_id"]])


def run_one(
    row: dict[str, str],
    alloc_job_id: str,
    slot: Slot,
    paths: RuntimePaths,
    cpus_per_task: int,
    omp_num_threads: int,
    mkl_num_threads: int,
) -> dict[str, str]:
    run_root = paths.runs / row["row_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    stdout_log = paths.logs / f"{row['row_id']}.out"
    stderr_log = paths.logs / f"{row['row_id']}.err"
    cmd = [
        "srun",
        f"--jobid={alloc_job_id}",
        "--exact",
        "-N1",
        f"-w{slot.host}",
        "--ntasks=1",
        "--ntasks-per-node=1",
        f"--cpus-per-task={cpus_per_task}",
        "--gres=gpu:v100:1",
        "--kill-on-bad-exit=1",
        "bash",
        str(STEP_SCRIPT),
    ]
    started_at = now_iso()
    start = time.time()
    with stdout_log.open("w") as out, stderr_log.open("w") as err:
        proc = subprocess.run(
            cmd,
            stdout=out,
            stderr=err,
            env=env_for_row(row, run_root, omp_num_threads, mkl_num_threads),
            text=True,
        )
    ended_at = now_iso()
    result = {
        "row_id": row["row_id"],
        "representative": row["representative"],
        "status": "COMPLETED" if proc.returncode == 0 else "FAILED",
        "return_code": str(proc.returncode),
        "alloc_job_id": alloc_job_id,
        "assigned_node": slot.host,
        "assigned_slot": slot.slot_label,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_sec": f"{time.time() - start:.3f}",
        "run_output_root": str(run_root),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "notes": row["notes"],
    }
    with paths.run_log_jsonl.open("a") as f:
        f.write(json.dumps(result, sort_keys=True) + "\n")
    return result


def submit_row(
    pool: ThreadPoolExecutor,
    row: dict[str, str],
    slot: Slot,
    alloc_job_id: str,
    paths: RuntimePaths,
    cpus_per_task: int,
    omp_num_threads: int,
    mkl_num_threads: int,
) -> Future[dict[str, str]]:
    print(
        f"[{now_iso()}] starting {row['row_id']} {row['representative']} on {slot.host} ({slot.slot_label})",
        flush=True,
    )
    return pool.submit(
        run_one,
        row,
        alloc_job_id,
        slot,
        paths,
        cpus_per_task,
        omp_num_threads,
        mkl_num_threads,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run chained V100 GPU VRAM telemetry tests for HH Track4 final SBI reps.")
    ap.add_argument("--alloc-job-id", required=True)
    ap.add_argument("--alloc-nodelist", required=True)
    ap.add_argument("--date-tag", default=DEFAULT_DATE_TAG)
    ap.add_argument("--output-tag", default=DEFAULT_OUTPUT_TAG)
    ap.add_argument("--row-id-prefix", default=DEFAULT_ROW_ID_PREFIX)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--params-file", default=BASE_PARAMS)
    ap.add_argument("--data-dir", default=str(SHARED_DATA))
    ap.add_argument("--row-limit", type=int, default=None)
    ap.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum number of rows to run at once. Default keeps the original serial launcher behavior.",
    )
    ap.add_argument(
        "--gpus-per-host",
        type=int,
        default=2,
        help="Concurrency slots per host to expose when max-concurrent > 1. Live V100 allocation currently has 2 GPUs per host.",
    )
    ap.add_argument("--cpus-per-task", type=int, default=12)
    ap.add_argument(
        "--omp-num-threads",
        type=int,
        default=None,
        help="Defaults to --cpus-per-task when omitted.",
    )
    ap.add_argument(
        "--mkl-num-threads",
        type=int,
        default=None,
        help="Defaults to --cpus-per-task when omitted.",
    )
    ap.add_argument("--continue-on-failure", action="store_true")
    args = ap.parse_args()

    if args.max_concurrent < 1:
        raise SystemExit("--max-concurrent must be >= 1")
    if args.gpus_per_host < 1:
        raise SystemExit("--gpus-per-host must be >= 1")
    if args.cpus_per_task < 1:
        raise SystemExit("--cpus-per-task must be >= 1")

    omp_num_threads = args.omp_num_threads if args.omp_num_threads is not None else args.cpus_per_task
    mkl_num_threads = args.mkl_num_threads if args.mkl_num_threads is not None else args.cpus_per_task

    paths = build_runtime_paths(args.date_tag, args.output_tag)
    for path in (paths.tables, paths.runs, paths.logs, paths.notes):
        path.mkdir(parents=True, exist_ok=True)

    rows = build_matrix(args.row_id_prefix, args.seed, args.params_file, args.data_dir)
    if args.row_limit is not None:
        rows = rows[: args.row_limit]
    if not rows:
        raise SystemExit("No rows selected after applying --row-limit")
    write_csv(paths.matrix_csv, rows, list(rows[0].keys()))

    hosts = hostnames(args.alloc_nodelist)
    registry_by_row_id: dict[str, dict[str, str]] = {}
    row_order = {row["row_id"]: idx for idx, row in enumerate(rows)}

    if args.max_concurrent == 1:
        for idx, row in enumerate(rows):
            slot = Slot(host=hosts[idx % len(hosts)], slot_label=f"{hosts[idx % len(hosts)]}:serial")
            print(f"[{now_iso()}] starting {row['row_id']} {row['representative']} on {slot.host}", flush=True)
            result = run_one(
                row,
                args.alloc_job_id,
                slot,
                paths,
                args.cpus_per_task,
                omp_num_threads,
                mkl_num_threads,
            )
            registry_by_row_id[row["row_id"]] = result
            registry = sorted_registry_rows(registry_by_row_id, row_order)
            write_csv(paths.registry_csv, registry, list(registry[0].keys()))
            print(
                f"[{now_iso()}] finished {row['row_id']} status={result['status']} elapsed_sec={result['elapsed_sec']}",
                flush=True,
            )
            if result["status"] != "COMPLETED" and not args.continue_on_failure:
                raise SystemExit(f"Stopping chain after failed row {row['row_id']}")
    else:
        slots = build_slots(hosts, args.gpus_per_host, min(args.max_concurrent, len(rows), len(hosts) * args.gpus_per_host))
        pending_rows = iter(rows)
        stop_scheduling = False
        futures: dict[Future[dict[str, str]], tuple[dict[str, str], Slot]] = {}

        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            for slot in slots:
                try:
                    row = next(pending_rows)
                except StopIteration:
                    break
                futures[
                    submit_row(
                        pool,
                        row,
                        slot,
                        args.alloc_job_id,
                        paths,
                        args.cpus_per_task,
                        omp_num_threads,
                        mkl_num_threads,
                    )
                ] = (row, slot)

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                freed_slots: list[Slot] = []
                for future in done:
                    row, slot = futures.pop(future)
                    result = future.result()
                    registry_by_row_id[row["row_id"]] = result
                    registry = sorted_registry_rows(registry_by_row_id, row_order)
                    write_csv(paths.registry_csv, registry, list(registry[0].keys()))
                    print(
                        f"[{now_iso()}] finished {row['row_id']} status={result['status']} elapsed_sec={result['elapsed_sec']}",
                        flush=True,
                    )
                    freed_slots.append(slot)
                    if result["status"] != "COMPLETED" and not args.continue_on_failure:
                        stop_scheduling = True

                while not stop_scheduling and freed_slots:
                    try:
                        row = next(pending_rows)
                    except StopIteration:
                        break
                    slot = freed_slots.pop(0)
                    futures[
                        submit_row(
                            pool,
                            row,
                            slot,
                            args.alloc_job_id,
                            paths,
                            args.cpus_per_task,
                            omp_num_threads,
                            mkl_num_threads,
                        )
                    ] = (row, slot)

            if stop_scheduling and not args.continue_on_failure:
                failed_row_ids = [
                    row_id for row_id, result in registry_by_row_id.items() if result["status"] != "COMPLETED"
                ]
                failed_desc = ", ".join(sorted(failed_row_ids)) or "unknown"
                raise SystemExit(
                    "Stopping chain after failure in concurrent mode. "
                    f"No new rows were launched after the first failure. Failed rows so far: {failed_desc}"
                )

    print(f"Wrote matrix: {paths.matrix_csv}")
    print(f"Wrote registry: {paths.registry_csv}")
    print(f"Wrote run log: {paths.run_log_jsonl}")


if __name__ == "__main__":
    main()
