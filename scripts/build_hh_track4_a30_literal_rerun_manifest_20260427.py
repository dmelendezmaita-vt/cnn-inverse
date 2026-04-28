#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
PYTHON_BIN = Path("/projects/neuro-collab/conda/neuro-collab-env/bin/python")
SHARED_DATA_SCRIPT = REPO / "scripts" / "shared_data_utils.py"
GPU_STEP_SCRIPT = REPO / "scripts" / "run_v100_interactive_step_20260411.sh"
CPU_STEP_SCRIPT = REPO / "scripts" / "run_classical_baseline_step_20260420.sh"

CAMPAIGN_NAME = "hh_track4_a30_literal_rerun_20260427"
DEFAULT_OUT_ROOT = IMPORTANT / "optimization_track_20260427_hh_track4_a30_literal_rerun"
DEFAULT_TABLES = DEFAULT_OUT_ROOT / "tables"
DEFAULT_NOTES = DEFAULT_OUT_ROOT / "notes"

MANIFEST_JSON = DEFAULT_TABLES / "hh_track4_a30_literal_rerun_manifest_20260427.json"
MANIFEST_CSV = DEFAULT_TABLES / "hh_track4_a30_literal_rerun_manifest_20260427.csv"
CATEGORY_REPORT = DEFAULT_NOTES / "hh_track4_a30_literal_rerun_manifest_categories_20260427.md"

CHECKPOINT6_SUMMARY = (
    IMPORTANT
    / "optimization_track_20260423_hh_track4_checkpoint6_shift_stability"
    / "notes"
    / "summary.json"
)
CHECKPOINT6_FAMILY_METRICS = (
    IMPORTANT
    / "optimization_track_20260423_hh_track4_checkpoint6_shift_stability"
    / "tables"
    / "family_shift_metrics.csv"
)
CHECKPOINT7_PARTIAL = (
    IMPORTANT
    / "optimization_track_20260423_hh_track4_checkpoint7_partial_saturation"
    / "notes"
    / "summary.json"
)
CHECKPOINT7_REPRESENTATIVES = (
    IMPORTANT
    / "optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency"
    / "tables"
    / "hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv"
)
CHECKPOINT7_FAMILY_AUDIT = (
    IMPORTANT
    / "optimization_track_20260423_hh_track4_checkpoint7_efficiency_audit"
    / "tables"
    / "family_efficiency_audit.csv"
)

PHASE_KEY_RE = re.compile(r"^phase[A-Z]+_track4_(?P<phase_key>.+)$")
MATRIX_NAME_RE = re.compile(r"^(?P<prefix>.+)_matrix_(?P<date>\d{8})_.+\.csv$")
SHIFT_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<suffix>clean|noise\d+|mult\d+|drift\d+|trunc\d+|downstep\d+|mask\d+)$")
SOURCE_RUN_RE = re.compile(r"/runs/(?P<phase>[^/]+)/(?P<launch_mode>[^/]+)/(?P<row_id>[^/]+)/[^/]+/?$")


@dataclass(frozen=True)
class PhaseSpec:
    phase: str
    matrix_csv: Path
    registry_csv: Path
    manifest_category: str
    analysis_status: str
    analysis_summary: str
    evidence_paths: tuple[Path, ...]


PHASE_SPECS: tuple[PhaseSpec, ...] = (
    PhaseSpec(
        phase="phaseAK_track4_a30_replication",
        matrix_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_replication"
        / "tables"
        / "a30_hh_track4_replication_matrix_20260420_a30_hh_track4_replication.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_replication"
        / "tables"
        / "a30_hh_track4_replication_registry_20260420_a30_hh_track4_replication.csv",
        manifest_category="literal_negative_analyzed_neural",
        analysis_status="included_negative_branch",
        analysis_summary=(
            "The A30 replication family is retained literally because the later checkpoint6/checkpoint7 "
            "analyses kept the A30 neural line as analyzed evidence, even though the representative neural "
            "branch was later dominated on clean accuracy and unstable under shift."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT7_PARTIAL, CHECKPOINT7_REPRESENTATIVES),
    ),
    PhaseSpec(
        phase="phaseAL_track4_a30_gpu_profile",
        matrix_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_gpu_profile"
        / "tables"
        / "a30_hh_track4_gpu_profile_matrix_20260420_a30_hh_track4_gpu_profile.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_gpu_profile"
        / "tables"
        / "a30_hh_track4_gpu_profile_registry_20260420_a30_hh_track4_gpu_profile.csv",
        manifest_category="support_profile_only",
        analysis_status="included_support_branch",
        analysis_summary=(
            "The GPU-profile phase is retained literally as instrumentation support, because the later family "
            "efficiency audit relies on A30-side timing and hardware context from this profiling path."
        ),
        evidence_paths=(CHECKPOINT7_FAMILY_AUDIT,),
    ),
    PhaseSpec(
        phase="phaseAM_track4_a30_effnet_schedule_retune",
        matrix_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_effnet_schedule_retune"
        / "tables"
        / "a30_hh_track4_effnet_schedule_retune_matrix_20260420_a30_hh_track4_effnet_schedule_retune.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_effnet_schedule_retune"
        / "tables"
        / "a30_hh_track4_effnet_schedule_retune_registry_20260420_a30_hh_track4_effnet_schedule_retune.csv",
        manifest_category="literal_negative_analyzed_neural",
        analysis_status="included_negative_branch",
        analysis_summary=(
            "The A30 EfficientNet schedule-retune rows are retained literally as analyzed-negative neural "
            "branch work, rather than being dropped just because the later consolidated analyses did not "
            "promote the neural line."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT7_PARTIAL, CHECKPOINT7_REPRESENTATIVES),
    ),
    PhaseSpec(
        phase="phaseAN_track4_a30_effnet_tail_schedule_retune",
        matrix_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_effnet_tail_schedule_retune"
        / "tables"
        / "a30_hh_track4_effnet_tail_schedule_retune_matrix_20260420_a30_hh_track4_effnet_tail_schedule_retune.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_effnet_tail_schedule_retune"
        / "tables"
        / "a30_hh_track4_effnet_tail_schedule_retune_registry_20260420_a30_hh_track4_effnet_tail_schedule_retune.csv",
        manifest_category="literal_negative_analyzed_neural",
        analysis_status="included_negative_branch",
        analysis_summary=(
            "The tail-schedule retune rows are retained literally because they belong to the analyzed A30 neural "
            "search branch, which later closed as dominated rather than being erased from the rerun set."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT7_PARTIAL, CHECKPOINT7_REPRESENTATIVES),
    ),
    PhaseSpec(
        phase="phaseAO_track4_a30_effnet_local_matrix",
        matrix_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_effnet_local_matrix"
        / "tables"
        / "a30_hh_track4_effnet_local_matrix_matrix_20260420_a30_hh_track4_effnet_local_matrix.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_effnet_local_matrix"
        / "tables"
        / "a30_hh_track4_effnet_local_matrix_registry_20260420_a30_hh_track4_effnet_local_matrix.csv",
        manifest_category="literal_negative_analyzed_neural",
        analysis_status="included_negative_branch",
        analysis_summary=(
            "The local A30 EfficientNet matrix is retained literally because the later neural closure is a "
            "negative analyzed result, not an excuse to omit the matrix rows that produced it."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT7_PARTIAL, CHECKPOINT7_REPRESENTATIVES),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_top3_confirmation",
        matrix_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_top3_confirmation"
        / "tables"
        / "a30_hh_track4_top3_confirmation_matrix_20260420_a30_hh_track4_top3_confirmation.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_top3_confirmation"
        / "tables"
        / "a30_hh_track4_top3_confirmation_registry_20260420_a30_hh_track4_top3_confirmation.csv",
        manifest_category="literal_negative_analyzed_neural",
        analysis_status="included_negative_branch",
        analysis_summary=(
            "The top-3 confirmation rows are retained literally because checkpoint7 explicitly keeps the "
            "A30 EfficientNet confirmation representative as a dominated neural reference."
        ),
        evidence_paths=(CHECKPOINT7_REPRESENTATIVES, CHECKPOINT7_FAMILY_AUDIT, CHECKPOINT7_PARTIAL),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_noise_robustness",
        matrix_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_noise_robustness"
        / "tables"
        / "a30_hh_track4_noise_robustness_matrix_20260420_a30_hh_track4_noise_robustness.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_noise_robustness"
        / "tables"
        / "a30_hh_track4_noise_robustness_registry_20260420_a30_hh_track4_noise_robustness.csv",
        manifest_category="literal_negative_analyzed_neural_eval_only",
        analysis_status="included_negative_branch",
        analysis_summary=(
            "The eval-only neural noise-robustness follow-up is retained literally because it is the explicit "
            "negative analyzed robustness branch anchored to the top-3 EfficientNet winner checkpoints."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT7_PARTIAL, CHECKPOINT7_REPRESENTATIVES),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_classical_baselines",
        matrix_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_classical_baselines"
        / "tables"
        / "a30_hh_track4_classical_baselines_matrix_20260420_a30_hh_track4_classical_baselines.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_classical_baselines"
        / "tables"
        / "a30_hh_track4_classical_baselines_registry_20260420_a30_hh_track4_classical_baselines.csv",
        manifest_category="promoted_classical_search",
        analysis_status="included_promoted_branch",
        analysis_summary=(
            "The classical baseline sweep is retained as promoted search work because the later checkpoint6/"
            "checkpoint7 analyses place the direct clean winner and the robustness winner inside the classical family."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT7_PARTIAL, CHECKPOINT7_REPRESENTATIVES),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_extra_trees_tuning",
        matrix_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_extra_trees_tuning"
        / "tables"
        / "a30_hh_track4_extra_trees_tuning_matrix_20260420_a30_hh_track4_extra_trees_tuning.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_extra_trees_tuning"
        / "tables"
        / "a30_hh_track4_extra_trees_tuning_registry_20260420_a30_hh_track4_extra_trees_tuning.csv",
        manifest_category="promoted_classical_search",
        analysis_status="included_promoted_branch",
        analysis_summary=(
            "The Extra Trees tuning sweep is retained as promoted classical search work because it feeds the later "
            "clean-data classical winners and confirmation phases."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT7_REPRESENTATIVES),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_classical_robustness",
        matrix_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_classical_robustness"
        / "tables"
        / "a30_hh_track4_classical_robustness_matrix_20260420_a30_hh_track4_classical_robustness.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_classical_robustness"
        / "tables"
        / "a30_hh_track4_classical_robustness_registry_20260420_a30_hh_track4_classical_robustness.csv",
        manifest_category="promoted_classical_robustness",
        analysis_status="included_promoted_branch",
        analysis_summary=(
            "The classical robustness sweep is retained as promoted robustness work because checkpoint6 keeps a "
            "classical robustness winner and the unstable-shift analysis stays centered on classical contenders."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT6_FAMILY_METRICS, CHECKPOINT7_PARTIAL),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_baseline_finalists_confirmation",
        matrix_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_baseline_finalists_confirmation"
        / "tables"
        / "a30_hh_track4_baseline_finalists_confirmation_matrix_20260420_a30_hh_track4_baseline_finalists_confirmation.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260420_a30_hh_track4_baseline_finalists_confirmation"
        / "tables"
        / "a30_hh_track4_baseline_finalists_confirmation_registry_20260420_a30_hh_track4_baseline_finalists_confirmation.csv",
        manifest_category="promoted_classical_confirmation",
        analysis_status="included_promoted_branch",
        analysis_summary=(
            "The baseline-finalists confirmation phase is retained as promoted classical confirmation work because "
            "the later direct Track4 clean and robustness references come from this non-neural confirmation path."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT7_REPRESENTATIVES),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_knn_targeted_tuning",
        matrix_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_knn_targeted_tuning"
        / "tables"
        / "a30_hh_track4_knn_targeted_tuning_matrix_20260421_a30_hh_track4_knn_targeted_tuning.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_knn_targeted_tuning"
        / "tables"
        / "a30_hh_track4_knn_targeted_tuning_registry_20260421_a30_hh_track4_knn_targeted_tuning.csv",
        manifest_category="promoted_classical_search",
        analysis_status="included_promoted_branch",
        analysis_summary=(
            "The targeted kNN tuning phase is retained as promoted search work because the later robustness head-to-head "
            "package still uses tuned kNN candidates as literal contenders."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT6_FAMILY_METRICS),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_extra_trees_targeted_tuning",
        matrix_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_extra_trees_targeted_tuning"
        / "tables"
        / "a30_hh_track4_extra_trees_targeted_tuning_matrix_20260421_a30_hh_track4_extra_trees_targeted_tuning.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_extra_trees_targeted_tuning"
        / "tables"
        / "a30_hh_track4_extra_trees_targeted_tuning_registry_20260421_a30_hh_track4_extra_trees_targeted_tuning.csv",
        manifest_category="promoted_classical_search",
        analysis_status="included_promoted_branch",
        analysis_summary=(
            "The targeted Extra Trees tuning phase is retained as promoted search work because it extends the clean-data "
            "classical frontier before the later confirmation and robustness sweeps."
        ),
        evidence_paths=(CHECKPOINT7_REPRESENTATIVES, CHECKPOINT7_PARTIAL),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_extra_trees_targeted_confirmation",
        matrix_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_extra_trees_targeted_confirmation"
        / "tables"
        / "a30_hh_track4_extra_trees_targeted_confirmation_matrix_20260421_a30_hh_track4_extra_trees_targeted_confirmation.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_extra_trees_targeted_confirmation"
        / "tables"
        / "a30_hh_track4_extra_trees_targeted_confirmation_registry_20260421_a30_hh_track4_extra_trees_targeted_confirmation.csv",
        manifest_category="promoted_classical_confirmation",
        analysis_status="included_promoted_branch",
        analysis_summary=(
            "The targeted Extra Trees confirmation phase is retained as promoted classical confirmation work because "
            "the later head-to-head package keeps these tree finalists as literal comparison rows."
        ),
        evidence_paths=(CHECKPOINT7_REPRESENTATIVES, CHECKPOINT7_PARTIAL),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_knn_robustness_tuning",
        matrix_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_knn_robustness_tuning"
        / "tables"
        / "a30_hh_track4_knn_robustness_tuning_matrix_20260421_a30_hh_track4_knn_robustness_tuning.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_knn_robustness_tuning"
        / "tables"
        / "a30_hh_track4_knn_robustness_tuning_registry_20260421_a30_hh_track4_knn_robustness_tuning.csv",
        manifest_category="promoted_classical_robustness",
        analysis_status="included_promoted_branch",
        analysis_summary=(
            "The kNN robustness sweep is retained as promoted robustness work because checkpoint6 explicitly keeps a "
            "kNN robustness winner inside the classical family."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT6_FAMILY_METRICS, CHECKPOINT7_PARTIAL),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_tuned_tree_robustness",
        matrix_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_tuned_tree_robustness"
        / "tables"
        / "a30_hh_track4_tuned_tree_robustness_matrix_20260421_a30_hh_track4_tuned_tree_robustness.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_tuned_tree_robustness"
        / "tables"
        / "a30_hh_track4_tuned_tree_robustness_registry_20260421_a30_hh_track4_tuned_tree_robustness.csv",
        manifest_category="promoted_classical_robustness",
        analysis_status="included_promoted_branch",
        analysis_summary=(
            "The tuned-tree robustness sweep is retained as promoted robustness work because it extends the classical "
            "robustness frontier that remains competitive in the later shift analyses."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT6_FAMILY_METRICS),
    ),
    PhaseSpec(
        phase="phaseAP_track4_a30_frontier_head_to_head",
        matrix_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_frontier_head_to_head"
        / "tables"
        / "a30_hh_track4_frontier_head_to_head_matrix_20260421_a30_hh_track4_frontier_head_to_head.csv",
        registry_csv=IMPORTANT
        / "optimization_track_20260421_a30_hh_track4_frontier_head_to_head"
        / "tables"
        / "a30_hh_track4_frontier_head_to_head_registry_20260421_a30_hh_track4_frontier_head_to_head.csv",
        manifest_category="promoted_classical_head_to_head",
        analysis_status="included_promoted_branch",
        analysis_summary=(
            "The classical frontier head-to-head package is retained literally because it is the direct comparison "
            "surface for the strongest tree and tuned-kNN contenders that survive the later analyses."
        ),
        evidence_paths=(CHECKPOINT6_SUMMARY, CHECKPOINT7_PARTIAL, CHECKPOINT7_REPRESENTATIVES),
    ),
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build a literal A30 HH Track4 rerun manifest that reuses the historical A30 matrices and "
            "keeps analyzed-negative neural branches in the campaign."
        )
    )
    ap.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help=f"Manifest output root. Default: {DEFAULT_OUT_ROOT}",
    )
    return ap.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def phase_key(phase: str) -> str:
    match = PHASE_KEY_RE.match(phase)
    if not match:
        raise ValueError(f"Could not derive phase key from {phase}")
    return match.group("phase_key")


def derive_launcher_script(matrix_csv: Path) -> Path:
    match = MATRIX_NAME_RE.match(matrix_csv.name)
    if not match:
        raise ValueError(f"Could not derive launcher script from matrix filename {matrix_csv.name}")
    prefix = match.group("prefix")
    date = match.group("date")
    launcher = REPO / "scripts" / f"launch_{prefix}_{date}.py"
    if not launcher.exists():
        raise FileNotFoundError(f"Missing launcher script for {matrix_csv}: {launcher}")
    return launcher


def strategy_base(strategy_id: str) -> str:
    match = SHIFT_SUFFIX_RE.match(strategy_id)
    if match:
        return match.group("base")
    return strategy_id


def normalized_baseline(row: dict[str, str]) -> str:
    baseline_name = row.get("baseline_name", "").strip()
    if baseline_name:
        return baseline_name
    return strategy_base(row["strategy_id"])


def resource_class_for(gpus_per_task: int) -> str:
    if gpus_per_task <= 0:
        return "cpu_only"
    if gpus_per_task == 1:
        return "gpu1"
    if gpus_per_task == 2:
        return "gpu2"
    raise ValueError(f"Unsupported gpus_per_task={gpus_per_task}")


def task_id_for(phase: str, row_id: str) -> str:
    return f"{phase_key(phase)}__{row_id}"


def sorted_unique(values: list[str]) -> list[str]:
    return sorted(set(v for v in values if v))


def expected_output_paths(
    campaign_root: Path,
    phase: str,
    launch_group: str,
    task_id: str,
    *,
    include_checkpoints: bool,
    is_gpu_task: bool,
) -> list[str]:
    group = launch_group or "ungrouped"
    run_root = campaign_root / "runs" / phase / group / task_id
    paths = [str(run_root), str(run_root / "metrics_summary.json"), str(run_root / "metrics_per_target.json"), str(run_root / "predictions.npz")]
    if is_gpu_task:
        paths.append(str(run_root / "params.yaml"))
    else:
        paths.append(str(run_root / "params_snapshot.yaml"))
    if include_checkpoints:
        paths.append(str(run_root / "checkpoints"))
    return paths


def stable_master_port(row_id: str) -> int:
    digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    return 20000 + (int(digest[:8], 16) % 20000)


def shell_assignments(env_map: dict[str, str]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in env_map.items() if value != "")


def build_cpu_direct_command(row: dict[str, str], campaign_run_root: Path) -> str:
    preamble = ""
    shared_data_dir = row.get("shared_data_dir", "").strip()
    tar_path = row.get("tar_path", "").strip()
    if shared_data_dir and tar_path:
        preamble = (
            f"{shlex.quote(str(PYTHON_BIN))} {shlex.quote(str(SHARED_DATA_SCRIPT.relative_to(REPO)))} "
            f"--tar-path {shlex.quote(tar_path)} --shared-dir {shlex.quote(shared_data_dir)} && "
        )

    env_map = {
        "REPO_ROOT": str(REPO),
        "RUN_ID": row["row_id"],
        "RUN_OUTPUT_ROOT": str(campaign_run_root),
        "PARAMS_FILE": row["params_file"],
        "BASELINE_NAME": row.get("baseline_name", ""),
        "FEATURE_MODE": row.get("feature_mode", "") or "raw",
        "DATA_DIR": shared_data_dir,
        "DATA_PREFIX": row.get("data_prefix", ""),
        "CURR": row.get("curr", "") or "0.1",
    }
    inner = (
        f"cd {shlex.quote(str(REPO))} && "
        f"{preamble}"
        f"env {shell_assignments(env_map)} bash {shlex.quote(str(CPU_STEP_SCRIPT.relative_to(REPO)))}"
    )
    return f"bash -lc {shlex.quote(inner)}"


def build_gpu_direct_command(
    row: dict[str, str],
    campaign_run_root: Path,
    *,
    eval_only_source_root: Path | None = None,
) -> str:
    eval_only_checkpoint = str(eval_only_source_root) if eval_only_source_root is not None else row.get("eval_only_checkpoint", "")
    env_map = {
        "REPO_ROOT": str(REPO),
        "RUN_ID": row["row_id"],
        "RUN_OUTPUT_ROOT": str(campaign_run_root),
        "PARAMS_FILE": row["params_file"],
        "TAR_PATH": row.get("tar_path", ""),
        "DATA_PREFIX": row.get("data_prefix", ""),
        "CURR": row.get("curr", "") or "0.1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(stable_master_port(row["row_id"])),
        "STEP_NNODES": row.get("nodes", "") or "1",
        "NPROC_PER_NODE": row.get("gpus_per_node", "") or "1",
        "DATA_ACCESS_MODE": row.get("data_access_mode", "") or "direct_tar",
        "SHARED_DATA_DIR": row.get("shared_data_dir", ""),
        "SAVE_PREDICTIONS": row.get("save_predictions", ""),
        "FOLLOWUP_TMP_BASE": "/projects/neuro-collab/other/v100if_tmp",
        "EVAL_ONLY_CHECKPOINT": eval_only_checkpoint,
        "SPLIT_EVAL_AFTER_TRAIN": row.get("split_eval_after_train", ""),
        "SKIP_INLINE_SPLIT_EVAL": row.get("skip_inline_split_eval", ""),
        "LAUNCH_BACKEND": row.get("launch_backend", ""),
    }
    inner = (
        f"cd {shlex.quote(str(REPO))} && "
        f'env ALLOC_JOB_ID="$SLURM_JOB_ID" {shell_assignments(env_map)} '
        f"bash {shlex.quote(str(GPU_STEP_SCRIPT.relative_to(REPO)))}"
    )
    return f"bash -lc {shlex.quote(inner)}"


def build_direct_command(
    row: dict[str, str],
    campaign_run_root: Path,
    gpus_per_task: int,
    *,
    eval_only_source_root: Path | None = None,
) -> str:
    if gpus_per_task > 0:
        return build_gpu_direct_command(row, campaign_run_root, eval_only_source_root=eval_only_source_root)
    return build_cpu_direct_command(row, campaign_run_root)


def parse_source_dependency(
    source_run_root: str,
    task_id_by_phase_row: dict[tuple[str, str], str],
) -> str | None:
    match = SOURCE_RUN_RE.search(source_run_root)
    if not match:
        return None
    return task_id_by_phase_row.get((match.group("phase"), match.group("row_id")))


def load_analysis_snapshot() -> dict[str, Any]:
    checkpoint6_summary = json.loads(CHECKPOINT6_SUMMARY.read_text())
    checkpoint7_partial = json.loads(CHECKPOINT7_PARTIAL.read_text())
    representatives = {row["representative"]: row for row in load_csv(CHECKPOINT7_REPRESENTATIVES)}
    family_audit = {row["representative"]: row for row in load_csv(CHECKPOINT7_FAMILY_AUDIT)}
    return {
        "checkpoint6_summary": checkpoint6_summary,
        "checkpoint7_partial": checkpoint7_partial,
        "representatives": representatives,
        "family_audit": family_audit,
    }


def choose_dependency(
    *,
    phase: str,
    by_strategy: dict[tuple[str, str], str],
    by_baseline: dict[tuple[str, str], str],
    by_phase: dict[str, str],
    strategy: str | None = None,
    baseline: str | None = None,
) -> list[str]:
    deps: list[str] = []
    if strategy:
        dep = by_strategy.get((phase, strategy))
        if dep:
            deps.append(dep)
    if baseline:
        dep = by_baseline.get((phase, baseline))
        if dep:
            deps.append(dep)
    if not deps:
        dep = by_phase.get(phase)
        if dep:
            deps.append(dep)
    return deps


def assign_dependencies(tasks: list[dict[str, Any]]) -> None:
    task_id_by_phase_row = {
        (task["phase"], task["source_row_id"]): task["task_id"]
        for task in tasks
    }
    first_task_by_phase: dict[str, str] = {}
    first_task_by_phase_strategy: dict[tuple[str, str], str] = {}
    first_task_by_phase_baseline: dict[tuple[str, str], str] = {}
    for task in tasks:
        phase = task["phase"]
        first_task_by_phase.setdefault(phase, task["task_id"])
        first_task_by_phase_strategy.setdefault((phase, task["strategy_base"]), task["task_id"])
        first_task_by_phase_baseline.setdefault((phase, task["baseline_name_norm"]), task["task_id"])

    for task in tasks:
        phase = task["phase"]
        strategy = task["strategy_base"]
        baseline = task["baseline_name_norm"]
        row = task["source_row"]
        deps: list[str] = []

        if phase == "phaseAN_track4_a30_effnet_tail_schedule_retune":
            deps.extend(
                choose_dependency(
                    phase="phaseAM_track4_a30_effnet_schedule_retune",
                    strategy=strategy,
                    by_strategy=first_task_by_phase_strategy,
                    by_baseline=first_task_by_phase_baseline,
                    by_phase=first_task_by_phase,
                )
            )
        elif phase == "phaseAP_track4_a30_top3_confirmation":
            deps.extend(
                choose_dependency(
                    phase="phaseAO_track4_a30_effnet_local_matrix",
                    strategy=strategy,
                    by_strategy=first_task_by_phase_strategy,
                    by_baseline=first_task_by_phase_baseline,
                    by_phase=first_task_by_phase,
                )
            )
        elif phase == "phaseAP_track4_a30_noise_robustness":
            explicit = parse_source_dependency(row.get("source_run_root", ""), task_id_by_phase_row)
            if explicit:
                deps.append(explicit)
            else:
                deps.extend(
                    choose_dependency(
                        phase="phaseAP_track4_a30_top3_confirmation",
                        strategy=strategy,
                        by_strategy=first_task_by_phase_strategy,
                        by_baseline=first_task_by_phase_baseline,
                        by_phase=first_task_by_phase,
                    )
                )
        elif phase == "phaseAP_track4_a30_classical_robustness":
            deps.extend(
                choose_dependency(
                    phase="phaseAP_track4_a30_classical_baselines",
                    baseline=baseline,
                    by_strategy=first_task_by_phase_strategy,
                    by_baseline=first_task_by_phase_baseline,
                    by_phase=first_task_by_phase,
                )
            )
        elif phase == "phaseAP_track4_a30_extra_trees_tuning":
            deps.extend(
                choose_dependency(
                    phase="phaseAP_track4_a30_classical_baselines",
                    baseline="extra_trees_200",
                    by_strategy=first_task_by_phase_strategy,
                    by_baseline=first_task_by_phase_baseline,
                    by_phase=first_task_by_phase,
                )
            )
        elif phase == "phaseAP_track4_a30_baseline_finalists_confirmation":
            if baseline.startswith("extra_trees"):
                deps.extend(
                    choose_dependency(
                        phase="phaseAP_track4_a30_extra_trees_tuning",
                        strategy=baseline,
                        baseline=baseline,
                        by_strategy=first_task_by_phase_strategy,
                        by_baseline=first_task_by_phase_baseline,
                        by_phase=first_task_by_phase,
                    )
                )
            elif baseline.startswith("knn"):
                deps.extend(
                    choose_dependency(
                        phase="phaseAP_track4_a30_classical_robustness",
                        strategy=f"{baseline}_clean",
                        baseline=baseline,
                        by_strategy=first_task_by_phase_strategy,
                        by_baseline=first_task_by_phase_baseline,
                        by_phase=first_task_by_phase,
                    )
                )
                deps.extend(
                    choose_dependency(
                        phase="phaseAP_track4_a30_classical_baselines",
                        baseline=baseline,
                        by_strategy=first_task_by_phase_strategy,
                        by_baseline=first_task_by_phase_baseline,
                        by_phase=first_task_by_phase,
                    )
                )
        elif phase == "phaseAP_track4_a30_knn_targeted_tuning":
            deps.extend(
                choose_dependency(
                    phase="phaseAP_track4_a30_baseline_finalists_confirmation",
                    baseline="knn_k5",
                    by_strategy=first_task_by_phase_strategy,
                    by_baseline=first_task_by_phase_baseline,
                    by_phase=first_task_by_phase,
                )
            )
        elif phase == "phaseAP_track4_a30_knn_robustness_tuning":
            deps.extend(
                choose_dependency(
                    phase="phaseAP_track4_a30_knn_targeted_tuning",
                    strategy=baseline,
                    baseline=baseline,
                    by_strategy=first_task_by_phase_strategy,
                    by_baseline=first_task_by_phase_baseline,
                    by_phase=first_task_by_phase,
                )
            )
        elif phase == "phaseAP_track4_a30_extra_trees_targeted_tuning":
            deps.extend(
                choose_dependency(
                    phase="phaseAP_track4_a30_baseline_finalists_confirmation",
                    strategy=baseline,
                    baseline=baseline,
                    by_strategy=first_task_by_phase_strategy,
                    by_baseline=first_task_by_phase_baseline,
                    by_phase=first_task_by_phase,
                )
            )
            if baseline == "extra_trees_500_leaf3":
                deps.extend(
                    choose_dependency(
                        phase="phaseAP_track4_a30_extra_trees_tuning",
                        baseline="extra_trees_500",
                        by_strategy=first_task_by_phase_strategy,
                        by_baseline=first_task_by_phase_baseline,
                        by_phase=first_task_by_phase,
                    )
                )
        elif phase == "phaseAP_track4_a30_extra_trees_targeted_confirmation":
            deps.extend(
                choose_dependency(
                    phase="phaseAP_track4_a30_extra_trees_targeted_tuning",
                    strategy=baseline,
                    baseline=baseline,
                    by_strategy=first_task_by_phase_strategy,
                    by_baseline=first_task_by_phase_baseline,
                    by_phase=first_task_by_phase,
                )
            )
        elif phase == "phaseAP_track4_a30_tuned_tree_robustness":
            deps.extend(
                choose_dependency(
                    phase="phaseAP_track4_a30_extra_trees_targeted_confirmation",
                    strategy=strategy,
                    baseline=strategy,
                    by_strategy=first_task_by_phase_strategy,
                    by_baseline=first_task_by_phase_baseline,
                    by_phase=first_task_by_phase,
                )
            )
            if not deps:
                deps.extend(
                    choose_dependency(
                        phase="phaseAP_track4_a30_extra_trees_targeted_tuning",
                        strategy=strategy,
                        baseline=strategy,
                        by_strategy=first_task_by_phase_strategy,
                        by_baseline=first_task_by_phase_baseline,
                        by_phase=first_task_by_phase,
                    )
                )
        elif phase == "phaseAP_track4_a30_frontier_head_to_head":
            if baseline.startswith("extra_trees"):
                deps.extend(
                    choose_dependency(
                        phase="phaseAP_track4_a30_extra_trees_targeted_confirmation",
                        strategy=baseline,
                        baseline=baseline,
                        by_strategy=first_task_by_phase_strategy,
                        by_baseline=first_task_by_phase_baseline,
                        by_phase=first_task_by_phase,
                    )
                )
            elif baseline.startswith("knn"):
                deps.extend(
                    choose_dependency(
                        phase="phaseAP_track4_a30_knn_robustness_tuning",
                        strategy=baseline,
                        baseline=baseline,
                        by_strategy=first_task_by_phase_strategy,
                        by_baseline=first_task_by_phase_baseline,
                        by_phase=first_task_by_phase,
                    )
                )

        task["depends_on"] = sorted_unique(deps)


def build_tasks(phase_specs: tuple[PhaseSpec, ...], campaign_root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    campaign_run_root_by_phase_row: dict[tuple[str, str], Path] = {}
    for spec in phase_specs:
        matrix_rows = load_csv(spec.matrix_csv)
        registry_rows = load_csv(spec.registry_csv)
        registry_by_row_id = {row["row_id"]: row for row in registry_rows}
        fallback_runtime = median(float(row["elapsed_sec"]) for row in registry_rows if row.get("elapsed_sec"))

        for row in matrix_rows:
            registry_row = registry_by_row_id.get(row["row_id"])
            if registry_row is None:
                raise KeyError(f"Missing registry row for {spec.phase} row_id={row['row_id']}")

            gpus_per_task = int(row.get("gpus_per_node") or "0")
            cpus_per_task = int(row["cpus_per_node"])
            resource_class = resource_class_for(gpus_per_task)
            task_id = task_id_for(spec.phase, row["row_id"])
            task_label = f"{row['phase_label']}: {row['strategy_id']} seed {row['seed']}"
            launch_group = row.get("launch_group", "")
            run_group = launch_group or "ungrouped"
            campaign_run_root = campaign_root / "runs" / spec.phase / run_group / task_id
            campaign_run_root_by_phase_row[(spec.phase, row["row_id"])] = campaign_run_root
            source_run_output_root = registry_row["run_output_root"]
            include_checkpoints = bool(gpus_per_task) and not row.get("eval_only_checkpoint")
            eval_only_source_root: Path | None = None
            match = SOURCE_RUN_RE.search(row.get("source_run_root", ""))
            if match is not None:
                eval_only_source_root = campaign_run_root_by_phase_row.get((match.group("phase"), match.group("row_id")))
            command = build_direct_command(
                row=row,
                campaign_run_root=campaign_run_root,
                gpus_per_task=gpus_per_task,
                eval_only_source_root=eval_only_source_root,
            )

            expected_runtime_sec = float(registry_row["elapsed_sec"]) if registry_row.get("elapsed_sec") else float(fallback_runtime)
            output_paths = expected_output_paths(
                campaign_root=campaign_root,
                phase=spec.phase,
                launch_group=launch_group,
                task_id=task_id,
                include_checkpoints=include_checkpoints,
                is_gpu_task=bool(gpus_per_task),
            )

            tasks.append(
                {
                    "task_id": task_id,
                    "phase": spec.phase,
                    "phase_label": row["phase_label"],
                    "launch_group": launch_group,
                    "label": task_label,
                    "command": command,
                    "resource_class": resource_class,
                    "cpus_per_task": cpus_per_task,
                    "gpus_per_task": gpus_per_task,
                    "expected_runtime_sec": round(expected_runtime_sec, 3),
                    "depends_on": [],
                    "output_paths": output_paths,
                    "manifest_category": spec.manifest_category,
                    "analysis_status": spec.analysis_status,
                    "analysis_summary": spec.analysis_summary,
                    "analysis_evidence_paths": [str(path) for path in spec.evidence_paths],
                    "family": row["family"],
                    "track_key": row["track_key"],
                    "launch_mode": row["launch_mode"],
                    "strategy_id": row["strategy_id"],
                    "strategy_base": strategy_base(row["strategy_id"]),
                    "strategy_description": row["strategy_description"],
                    "baseline_name": row.get("baseline_name", ""),
                    "baseline_name_norm": normalized_baseline(row),
                    "seed": int(row["seed"]),
                    "policy": row["policy"],
                    "nodes": int(row["nodes"]),
                    "source_matrix": str(spec.matrix_csv),
                    "source_registry": str(spec.registry_csv),
                    "source_row_id": row["row_id"],
                    "source_run_output_root": source_run_output_root,
                    "source_params_file": row["params_file"],
                    "source_notes": row.get("notes", ""),
                    "source_row": row,
                }
            )

    assign_dependencies(tasks)
    return tasks


def flatten_tasks_for_csv(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        rows.append(
            {
                "task_id": task["task_id"],
                "phase": task["phase"],
                "phase_label": task["phase_label"],
                "launch_group": task["launch_group"],
                "label": task["label"],
                "resource_class": task["resource_class"],
                "cpus_per_task": task["cpus_per_task"],
                "gpus_per_task": task["gpus_per_task"],
                "expected_runtime_sec": task["expected_runtime_sec"],
                "depends_on": json.dumps(task["depends_on"]),
                "output_paths": json.dumps(task["output_paths"]),
                "command": task["command"],
                "manifest_category": task["manifest_category"],
                "analysis_status": task["analysis_status"],
                "analysis_summary": task["analysis_summary"],
                "analysis_evidence_paths": json.dumps(task["analysis_evidence_paths"]),
                "family": task["family"],
                "track_key": task["track_key"],
                "launch_mode": task["launch_mode"],
                "strategy_id": task["strategy_id"],
                "strategy_description": task["strategy_description"],
                "baseline_name": task["baseline_name"],
                "seed": task["seed"],
                "source_matrix": task["source_matrix"],
                "source_registry": task["source_registry"],
                "source_row_id": task["source_row_id"],
                "source_run_output_root": task["source_run_output_root"],
                "source_params_file": task["source_params_file"],
                "source_notes": task["source_notes"],
            }
        )
    return rows


def build_manifest_json(campaign_root: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    public_tasks = []
    for task in tasks:
        public_tasks.append(
            {
                "task_id": task["task_id"],
                "phase": task["phase"],
                "label": task["label"],
                "command": task["command"],
                "resource_class": task["resource_class"],
                "cpus_per_task": task["cpus_per_task"],
                "gpus_per_task": task["gpus_per_task"],
                "expected_runtime_sec": task["expected_runtime_sec"],
                "depends_on": task["depends_on"],
                "output_paths": task["output_paths"],
                "phase_label": task["phase_label"],
                "launch_group": task["launch_group"],
                "launch_mode": task["launch_mode"],
                "manifest_category": task["manifest_category"],
                "analysis_status": task["analysis_status"],
                "analysis_summary": task["analysis_summary"],
                "analysis_evidence_paths": task["analysis_evidence_paths"],
                "strategy_id": task["strategy_id"],
                "strategy_description": task["strategy_description"],
                "baseline_name": task["baseline_name"],
                "seed": task["seed"],
                "source_matrix": task["source_matrix"],
                "source_registry": task["source_registry"],
                "source_row_id": task["source_row_id"],
                "source_run_output_root": task["source_run_output_root"],
                "source_params_file": task["source_params_file"],
            }
        )

    return {
        "campaign_name": CAMPAIGN_NAME,
        "campaign_root": str(campaign_root),
        "allocation_defaults": {
            "repo_root": str(REPO),
            "python_executable": str(PYTHON_BIN),
            "mirror_subdir_template": "runs/{phase}/{launch_group}/{task_id}",
            "command_policy": (
                "Each direct A30 task executes the underlying row step script inside the clean queue allocation, "
                "with row-specific environment variables derived from the historical matrix entry, so that the queue "
                "is the only active scheduler and the campaign run root is written directly."
            ),
            "allocation_note": (
                "The clean queue supplies the interactive allocation placement, while the direct task commands "
                "execute the row payload itself rather than calling the historical nested launchers."
            ),
            "literal_branch_policy": (
                "Analyzed-negative A30 neural branches are intentionally retained in tasks instead of being filtered "
                "out to only promoted classical rows."
            ),
            "resource_defaults": {
                "cpu_only": {"cpus_per_task": 64, "gpus_per_task": 0},
                "gpu1": {"cpus_per_task": 24, "gpus_per_task": 1},
                "gpu2": {"cpus_per_task": 24, "gpus_per_task": 2},
            },
            "step_scripts": {
                "cpu_only": str(CPU_STEP_SCRIPT),
                "gpu": str(GPU_STEP_SCRIPT),
            },
        },
        "tasks": public_tasks,
    }


def build_category_report(tasks: list[dict[str, Any]], analysis_snapshot: dict[str, Any]) -> str:
    category_to_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        category_to_tasks[task["manifest_category"]].append(task)

    checkpoint6_summary = analysis_snapshot["checkpoint6_summary"]
    checkpoint7_partial = analysis_snapshot["checkpoint7_partial"]
    representatives = analysis_snapshot["representatives"]
    family_audit = analysis_snapshot["family_audit"]

    effnet_rep = representatives.get("effnet_beta005_invvar_e500", {})
    rf_rep = representatives.get("random_forest_500", {})
    knn_rep = representatives.get("knn_k11", {})
    effnet_audit = family_audit.get("effnet_beta005_invvar_e500", {})
    rf_audit = family_audit.get("random_forest_500", {})

    category_meanings = {
        "literal_negative_analyzed_neural": (
            "Rows from the A30 neural search and confirmation line that were later analyzed and retained as "
            "dominated or non-promoted evidence."
        ),
        "literal_negative_analyzed_neural_eval_only": (
            "Eval-only neural follow-up rows that depend on confirmed top-3 checkpoints and remain part of the "
            "literal negative branch."
        ),
        "support_profile_only": (
            "Profiling and instrumentation rows that support the family-efficiency readout rather than a headline "
            "accuracy claim."
        ),
        "promoted_classical_search": (
            "Classical search rows that feed the later clean-data and robustness contenders."
        ),
        "promoted_classical_robustness": (
            "Classical robustness rows that feed the checkpoint6 shift-stability comparison."
        ),
        "promoted_classical_confirmation": (
            "Fresh-seed classical confirmation rows that stabilize finalist claims."
        ),
        "promoted_classical_head_to_head": (
            "Literal head-to-head comparison rows between the strongest tuned tree and kNN contenders."
        ),
    }

    lines = [
        "# HH Track4 A30 Literal Rerun Manifest Categories",
        "",
        "## Literal policy",
        (
            "This rerun manifest keeps the full literal A30 HH Track4 branch history that is still represented by "
            "existing matrices and registries, which means that analyzed-negative neural phases are preserved as "
            "tasks instead of being filtered away to only the promoted classical branches."
        ),
        "",
        "## Analysis anchors",
        f"- checkpoint6 reference rank signature: `{checkpoint6_summary['ranking_summary']['reference_rank_signature']}`",
        f"- checkpoint6 unstable shift labels: `{', '.join(checkpoint6_summary['ranking_summary']['unstable_shift_labels'])}`",
        f"- checkpoint7 best clean family by MAE: `{checkpoint7_partial['best_clean_family_by_mae']}`",
        f"- checkpoint7 neural representative role: `{effnet_rep.get('role', 'missing')}`",
        f"- checkpoint7 classical clean representative role: `{rf_rep.get('role', 'missing')}`",
        f"- checkpoint7 kNN representative role: `{knn_rep.get('role', 'missing')}`",
        f"- family audit neural runtime note: `{effnet_audit.get('notes', 'missing')}`",
        f"- family audit classical runtime note: `{rf_audit.get('notes', 'missing')}`",
        "",
        "## Category table",
        "",
        "| category | task_count | phase_count | meaning | evidence files |",
        "|---|---:|---:|---|---|",
    ]

    for category in sorted(category_to_tasks):
        category_tasks = category_to_tasks[category]
        phases = sorted({task["phase"] for task in category_tasks})
        evidence = sorted({path for task in category_tasks for path in task["analysis_evidence_paths"]})
        evidence_short = "<br>".join(
            f"`{Path(path).parents[1].name}/{Path(path).name}`" if len(Path(path).parents) >= 2 else f"`{Path(path).name}`"
            for path in evidence
        )
        lines.append(
            "| "
            f"`{category}` | {len(category_tasks)} | {len(phases)} | {category_meanings[category]} | {evidence_short} |"
        )

    lines.extend(
        [
            "",
            "## Phase mapping",
            "",
            "| phase | category | tasks | example_strategy |",
            "|---|---|---:|---|",
        ]
    )
    for phase in sorted({task["phase"] for task in tasks}):
        phase_tasks = [task for task in tasks if task["phase"] == phase]
        lines.append(
            "| "
            f"`{phase}` | `{phase_tasks[0]['manifest_category']}` | {len(phase_tasks)} | "
            f"`{phase_tasks[0]['strategy_id']}` |"
        )

    lines.extend(
        [
            "",
            "## Dependency policy",
            "",
            "- `phaseAP_track4_a30_top3_confirmation` depends on the earlier local-matrix anchor for the same EfficientNet strategy.",
            "- `phaseAP_track4_a30_noise_robustness` depends on the explicit top-3 checkpoint source recorded in each matrix row.",
            "- classical robustness and confirmation phases depend on the earlier baseline or tuning anchor that produced the same baseline family.",
            "- frontier head-to-head rows depend on the targeted tree confirmation anchor or the tuned kNN robustness anchor for the matching baseline family.",
            "",
            "## Output policy",
            "",
            "- each task declares deterministic mirrored outputs under the new campaign root using `runs/{phase}/{launch_group}/{task_id}`",
            "- direct A30 tasks execute the underlying row step scripts inside the clean queue allocation, which keeps monitoring one-to-one with the experiment row and avoids nesting another scheduler under the queue",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.out_root)
    tasks = build_tasks(PHASE_SPECS, campaign_root)
    manifest = build_manifest_json(campaign_root, tasks)
    csv_rows = flatten_tasks_for_csv(tasks)
    analysis_snapshot = load_analysis_snapshot()
    category_report = build_category_report(tasks, analysis_snapshot)

    tables_dir = campaign_root / "tables"
    notes_dir = campaign_root / "notes"
    write_json(tables_dir / MANIFEST_JSON.name, manifest)
    write_csv(tables_dir / MANIFEST_CSV.name, csv_rows, list(csv_rows[0].keys()))
    write_text(notes_dir / CATEGORY_REPORT.name, category_report)

    print(tables_dir / MANIFEST_JSON.name)
    print(tables_dir / MANIFEST_CSV.name)
    print(notes_dir / CATEGORY_REPORT.name)
    print(f"tasks={len(tasks)}")
    print(f"categories={dict(Counter(task['manifest_category'] for task in tasks))}")


if __name__ == "__main__":
    main()
