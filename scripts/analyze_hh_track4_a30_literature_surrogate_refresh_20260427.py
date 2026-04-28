#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
sys.path.append(str(REPO / "pytorch"))
from data import _apply_scale_inverse, _apply_targets_transform, _resolve_split_array_cache_dir, preprocess_targets  # type: ignore  # noqa: E402


OUR_OUTPUT_PREFIX = "hh_track4_a30_literature_"
ALIGNED_COMPARABILITY = "assumption_conditioned_aligned_multicurrent_a30"
POOLSEQ_COMPARABILITY = "assumption_conditioned_poolseq_sbi_a30"
ACTIVE_COMPARABILITY = "assumption_conditioned_surrogate_active_design_a30"
SIMFORMER_COMPARABILITY = "upstream_hh_benchmark_not_directly_comparable"
GENERALIZED_BAYES_COMPARABILITY = "assumption_conditioned_local_adaptation_not_exact_paper_code"
TASK_REGISTRY_REQUIRED_FIELDS = {"task_id", "status"}
PARAMS_FILE = REPO / "pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml"

RUN_INVENTORY_FIELDS = [
    "task_id",
    "run_name",
    "phase",
    "phase_label",
    "launch_group",
    "queue_status",
    "branch_family",
    "method_family",
    "artifact_kind",
    "comparability_class",
    "framework",
    "artifact_dir",
    "metrics_path",
    "source_run_name",
    "source_reference",
    "notes",
]

FRAMEWORK_SNAPSHOT_FIELDS = [
    "run_name",
    "task_id",
    "framework",
    "branch_family",
    "comparability_class",
    "mae",
    "mse",
    "r2",
    "runtime_sec",
    "artifact_dir",
    "source",
]

DECISION_RULE_FIELDS = [
    "run_name",
    "task_id",
    "family",
    "comparability_class",
    "decision_rule",
    "mae",
    "mse",
    "r2",
    "prediction_file",
    "source",
]

ACTIVE_POLICY_FIELDS = [
    "run_name",
    "task_id",
    "method_family",
    "comparability_class",
    "acquisition_policy",
    "mean_abs_log10_error_params_mean",
    "mean_relative_error_params_mean",
    "posterior_contraction_mean_mean",
    "ess_mean",
    "runtime_sec",
    "source",
]

METHOD_COMPARISON_FIELDS = [
    "result_space",
    "name",
    "branch_family",
    "comparability_class",
    "metric_name",
    "metric_value",
    "secondary_metric_name",
    "secondary_metric_value",
    "source",
    "note",
]

BRANCH_REGISTRY_FIELDS = [
    "branch",
    "status",
    "completed_runs",
    "running_tasks",
    "pending_tasks",
    "failed_tasks",
    "comparability_class",
    "evidence",
    "note",
]

KEY_RESULTS_FIELDS = [
    "section",
    "name",
    "branch_family",
    "comparability_class",
    "metric_name",
    "metric_value",
    "source",
    "note",
]

LIMITATION_FIELDS = [
    "limitation_id",
    "scope",
    "statement",
    "severity",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Regenerate A30-local literature and surrogate comparison tables from a single clean "
            "campaign root, while reading only manifest, registry, and run artifacts stored under that root."
        )
    )
    ap.add_argument("--campaign-root", required=True)
    return ap.parse_args()


def safe_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> object:
    return json.loads(path.read_text())


def is_within_root(path: Path, campaign_root: Path) -> bool:
    try:
        path.resolve().relative_to(campaign_root.resolve())
        return True
    except ValueError:
        return False


def relpath(path: Path | None, campaign_root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(campaign_root.resolve()))
    except ValueError:
        return str(path)


def campaign_tag(campaign_root: Path) -> str:
    name = campaign_root.name
    return name[len("optimization_track_") :] if name.startswith("optimization_track_") else name


def metric_value(obj: dict, key: str, split: str = "test") -> float | None:
    value = obj.get(key)
    if isinstance(value, dict):
        for candidate in (split, "test", "validate", "train", "value"):
            if candidate in value:
                parsed = safe_float(value[candidate])
                if parsed is not None:
                    return parsed
        return None
    return safe_float(value)


def metric_from_summary(metrics_obj: dict, key: str) -> float | None:
    direct = metric_value(metrics_obj, key)
    if direct is not None:
        return direct
    nested = metrics_obj.get("test_metrics", {})
    if isinstance(nested, dict):
        nested_value = metric_value(nested, key)
        if nested_value is not None:
            return nested_value
    return None


def numeric_sort_key(value: object, default: float = float("inf")) -> float:
    parsed = safe_float(value)
    return parsed if parsed is not None else default


def load_manifest(campaign_root: Path) -> tuple[Path, dict]:
    tables_dir = campaign_root / "tables"
    candidates = sorted(tables_dir.glob("*.json"))
    for path in candidates:
        obj = load_json(path)
        if isinstance(obj, dict) and isinstance(obj.get("tasks"), list):
            return path, obj
    raise FileNotFoundError(f"Could not find a campaign manifest JSON under {tables_dir}")


def load_task_registry(campaign_root: Path) -> tuple[Path | None, dict[str, dict[str, str]]]:
    tables_dir = campaign_root / "tables"
    candidates = sorted(tables_dir.glob("*.csv"))
    best_path: Path | None = None
    best_rows: dict[str, dict[str, str]] = {}
    for path in candidates:
        rows = read_csv_rows(path)
        if not rows:
            continue
        if not TASK_REGISTRY_REQUIRED_FIELDS.issubset(rows[0].keys()):
            continue
        if len(rows) > len(best_rows):
            best_path = path
            best_rows = {row["task_id"]: row for row in rows if row.get("task_id", "").strip()}
    return best_path, best_rows


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def infer_source_run_name(task: dict[str, object]) -> str:
    for key in ("source_run_name", "run_name"):
        value = str(task.get(key, "")).strip()
        if value:
            return value
    source_root = str(task.get("source_run_output_root", "")).strip()
    if source_root:
        return Path(source_root).name
    return str(task.get("task_id", "")).strip()


def task_expected_root(task: dict[str, object], campaign_root: Path) -> Path | None:
    raw_paths = task.get("output_paths", [])
    if not isinstance(raw_paths, list):
        return None
    candidates: list[Path] = []
    runs_root = (campaign_root / "runs").resolve()
    for raw in raw_paths:
        candidate = Path(str(raw))
        if not is_within_root(candidate, campaign_root):
            continue
        if candidate.suffix:
            candidate = candidate.parent
        try:
            candidate.resolve().relative_to(runs_root)
        except ValueError:
            continue
        candidates.append(candidate)
    if not candidates:
        return None
    candidates = dedupe_paths(candidates)
    return min(candidates, key=lambda path: (len(path.parts), str(path)))


def infer_branch_from_text(text: str) -> str:
    lowered = text.lower()
    if "generalized_bayes" in lowered:
        return "generalized_bayes_npe"
    if "simformer" in lowered:
        return "simformer"
    if "asnpe" in lowered:
        return "asnpe"
    if "active_sequential" in lowered:
        return "active_sequential"
    if "pool_sequential_sbi" in lowered or "fmpe_" in lowered or "snpe_" in lowered:
        return "poolseq_sbi"
    if "bayesflow" in lowered:
        return "bayesflow"
    if "tmnre" in lowered:
        return "tmnre"
    if "swyft" in lowered:
        return "swyft"
    if "temporal_cnn" in lowered:
        return "temporal_deterministic"
    return ""


def infer_branch_family_from_task(task: dict[str, object]) -> str:
    for key in ("source_run_name", "source_run_output_root", "label", "phase", "command"):
        branch = infer_branch_from_text(str(task.get(key, "")))
        if branch:
            return branch
    return ""


def infer_comparability_class(branch_family: str) -> str:
    if branch_family in {"bayesflow", "swyft", "tmnre", "temporal_deterministic"}:
        return ALIGNED_COMPARABILITY
    if branch_family == "poolseq_sbi":
        return POOLSEQ_COMPARABILITY
    if branch_family in {"active_sequential", "asnpe"}:
        return ACTIVE_COMPARABILITY
    if branch_family == "simformer":
        return SIMFORMER_COMPARABILITY
    if branch_family == "generalized_bayes_npe":
        return GENERALIZED_BAYES_COMPARABILITY
    return ""


def build_task_index(
    campaign_root: Path,
    manifest: dict,
    queue_registry: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, object]], dict[Path, str]]:
    tasks_by_id: dict[str, dict[str, object]] = {}
    roots_to_task_id: dict[Path, str] = {}
    for raw_task in manifest.get("tasks", []):
        if not isinstance(raw_task, dict):
            continue
        task_id = str(raw_task.get("task_id", "")).strip()
        if not task_id:
            continue
        task = dict(raw_task)
        task["task_id"] = task_id
        task["source_run_name"] = infer_source_run_name(task)
        task["branch_family_hint"] = infer_branch_family_from_task(task)
        task["comparability_hint"] = infer_comparability_class(str(task["branch_family_hint"]))
        task["queue_status"] = queue_registry.get(task_id, {}).get("status", "PENDING")
        expected_root = task_expected_root(task, campaign_root)
        if expected_root is not None:
            task["expected_root"] = str(expected_root)
            roots_to_task_id[expected_root.resolve()] = task_id
        tasks_by_id[task_id] = task
    return tasks_by_id, roots_to_task_id


def discover_candidate_roots(campaign_root: Path, roots_to_task_id: dict[Path, str]) -> list[Path]:
    runs_root = campaign_root / "runs"
    candidates: list[Path] = [Path(root) for root in roots_to_task_id]
    if runs_root.exists():
        for filename in (
            "metrics_summary.json",
            "active_sequential_manifest.json",
            "asnpe_manifest.json",
            "simformer_hh_metrics_summary.json",
            "generalized_bayes_npe_manifest.json",
        ):
            candidates.extend(path.parent for path in runs_root.rglob(filename))
    return [path for path in dedupe_paths(candidates) if is_within_root(path, campaign_root)]


def match_task_id(root: Path, roots_to_task_id: dict[Path, str]) -> str:
    root_resolved = root.resolve()
    for expected_root, task_id in roots_to_task_id.items():
        if root_resolved == expected_root:
            return task_id
    for expected_root, task_id in roots_to_task_id.items():
        try:
            root_resolved.relative_to(expected_root)
            return task_id
        except ValueError:
            pass
        try:
            expected_root.relative_to(root_resolved)
            return task_id
        except ValueError:
            pass
    return ""


def prediction_pair_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.load(path)
    key_pairs = [
        ("test_pred", "test_true"),
        ("posterior_mean", "theta_true"),
        ("posterior_mean", "targets"),
        ("predictions", "targets"),
    ]
    for pred_key, true_key in key_pairs:
        if pred_key in arr.files and true_key in arr.files:
            pred = np.asarray(arr[pred_key], dtype=np.float32)
            true = np.asarray(arr[true_key], dtype=np.float32)
            if pred.shape == true.shape and pred.ndim == 2:
                return pred, true
    return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_true - y_pred
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = 1.0 - float(np.sum(np.square(diff))) / denom if denom > 0.0 else 0.0
    return {"mae": mae, "mse": mse, "r2": r2}


def load_params(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def postprocess_targets_array(y_norm: np.ndarray, targets_scale: dict) -> np.ndarray:
    restored = _apply_scale_inverse(y_norm.astype(np.float32, copy=False), targets_scale)
    restored = _apply_targets_transform(restored, targets_scale.get("transform"), inverse=True, array_name="targets")
    return restored.astype(np.float32, copy=False)


def build_targets_scale() -> dict:
    params = load_params(PARAMS_FILE)
    params["data"]["data_dir"] = str(
        REPO
        / "data"
        / "important_notes"
        / "optimization_track_20260411_v100_interactive"
        / "shared_data"
        / "track4_hh_full"
        / "concatenated_data"
    )
    params["data"]["data_prefix"] = "concatenated_data"
    params["data"]["curr"] = "0.1"
    params["data"]["Ntrain"] = 4096
    params["data"]["Nvalidate"] = 1024
    params["data"]["Ntest"] = 128
    params["data"]["split_array_cache_enabled"] = True
    cache_dir = _resolve_split_array_cache_dir(params["data"])
    target_train = np.load(cache_dir / "targets_train.npy", mmap_mode="r")
    target_validate = np.load(cache_dir / "targets_validate.npy", mmap_mode="r")
    target_test = np.load(cache_dir / "targets_test.npy", mmap_mode="r")
    targets = {
        "train": np.array(target_train[:4096], copy=True).astype(np.float32, copy=False),
        "validate": np.array(target_validate[:1024], copy=True).astype(np.float32, copy=False),
        "test": np.array(target_test[:128], copy=True).astype(np.float32, copy=False),
    }
    logger = logging.getLogger("a30_literature_refresh_targets")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return preprocess_targets(targets, params, logger=logger)


TARGETS_SCALE = build_targets_scale()


def sample_medoid(samples: np.ndarray) -> np.ndarray:
    diffs = samples[:, :, None, :] - samples[:, None, :, :]
    dist = np.sum(np.square(diffs), axis=3)
    medoid_idx = np.argmin(np.sum(dist, axis=2), axis=1)
    return samples[np.arange(samples.shape[0]), medoid_idx]


def gaussian_pseudo_samples(mean: np.ndarray, std: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = np.maximum(std, 1.0e-8)
    noise = rng.normal(size=(mean.shape[0], n_samples, mean.shape[1])).astype(np.float32)
    return (mean[:, None, :] + std[:, None, :] * noise).astype(np.float32, copy=False)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.empty((values.shape[0], values.shape[2]), dtype=np.float32)
    for i in range(values.shape[0]):
        for j in range(values.shape[2]):
            idx = np.argsort(values[i, :, j])
            v = values[i, idx, j]
            w = weights[i, idx, j]
            w = w / np.sum(w)
            cdf = np.cumsum(w)
            out[i, j] = v[np.searchsorted(cdf, 0.5, side="left")]
    return out


def swyft_map_bank(theta_bank_norm: np.ndarray, weights_1d: np.ndarray) -> np.ndarray:
    score = np.sum(np.log(np.clip(weights_1d, 1.0e-12, None)), axis=2)
    idx = np.argmax(score, axis=1)
    return theta_bank_norm[idx]


def save_prediction_pair(path: Path, pred: np.ndarray, true: np.ndarray) -> Path:
    np.savez_compressed(
        path,
        test_pred=np.asarray(pred, dtype=np.float32),
        test_true=np.asarray(true, dtype=np.float32),
    )
    return path


def ensure_mean_prediction_file(run_root: Path) -> Path | None:
    out_path = run_root / "predictions_mean.npz"
    if out_path.exists():
        return out_path
    raw_path = run_root / "predictions_test.npz"
    if not raw_path.exists():
        return None
    pair = prediction_pair_from_npz(raw_path)
    if pair is None:
        return None
    pred, true = pair
    return save_prediction_pair(out_path, pred, true)


def ensure_bayesflow_rule_files(run_root: Path) -> list[Path]:
    raw_path = run_root / "predictions_test.npz"
    if not raw_path.exists():
        return []
    arr = np.load(raw_path)
    if "test_true" not in arr.files:
        return []
    y_true = np.asarray(arr["test_true"], dtype=np.float32)
    outputs: list[Path] = []
    if "posterior_samples_norm" in arr.files:
        samples_norm = np.asarray(arr["posterior_samples_norm"], dtype=np.float32)
        samples = postprocess_targets_array(
            samples_norm.reshape(-1, samples_norm.shape[-1]),
            TARGETS_SCALE,
        ).reshape(samples_norm.shape)
    else:
        mean = np.asarray(arr["test_pred"], dtype=np.float32)
        std = np.asarray(arr["posterior_std"], dtype=np.float32)
        samples = gaussian_pseudo_samples(mean, std, n_samples=128, seed=20260425)
    rules = {
        "mean": np.mean(samples, axis=1),
        "median": np.median(samples, axis=1),
        "medoid": sample_medoid(samples),
    }
    for rule_name, pred in rules.items():
        out_path = run_root / f"predictions_{rule_name}.npz"
        if not out_path.exists():
            save_prediction_pair(out_path, pred, y_true)
        outputs.append(out_path)
    return outputs


def ensure_swyft_rule_files(run_root: Path) -> list[Path]:
    raw_path = run_root / "predictions_test.npz"
    if not raw_path.exists():
        return []
    arr = np.load(raw_path)
    if "test_true" not in arr.files:
        return []
    y_true = np.asarray(arr["test_true"], dtype=np.float32)
    outputs: list[Path] = []
    if {"theta_bank_norm", "posterior_weights_1d"}.issubset(arr.files):
        theta_bank_norm = np.asarray(arr["theta_bank_norm"], dtype=np.float32)
        weights = np.asarray(arr["posterior_weights_1d"], dtype=np.float32)
        theta_bank = postprocess_targets_array(theta_bank_norm, TARGETS_SCALE)
        repeated_bank = np.broadcast_to(theta_bank[None, :, :], weights.shape)
        rules = {
            "mean": np.sum(theta_bank[None, :, :] * weights, axis=1),
            "median": weighted_median(repeated_bank, weights),
            "map_bank": postprocess_targets_array(swyft_map_bank(theta_bank_norm, weights), TARGETS_SCALE),
        }
    else:
        pair = prediction_pair_from_npz(raw_path)
        if pair is None:
            return []
        pred, true = pair
        rules = {"mean": pred}
        y_true = true
    for rule_name, pred in rules.items():
        out_path = run_root / f"predictions_{rule_name}.npz"
        if not out_path.exists():
            save_prediction_pair(out_path, pred, y_true)
        outputs.append(out_path)
    return outputs


def ensure_poolseq_rule_files(run_root: Path) -> list[Path]:
    raw_path = run_root / "predictions_test.npz"
    if not raw_path.exists():
        return []
    arr = np.load(raw_path)
    if not {"targets", "posterior_mean", "posterior_median", "posterior_samples", "posterior_selected"}.issubset(arr.files):
        return []
    y_true = np.asarray(arr["targets"], dtype=np.float32)
    outputs: list[Path] = []
    rules: dict[str, np.ndarray] = {
        "mean": np.asarray(arr["posterior_mean"], dtype=np.float32),
        "median": np.asarray(arr["posterior_median"], dtype=np.float32),
        "selected": np.asarray(arr["posterior_selected"], dtype=np.float32),
        "medoid": sample_medoid(np.asarray(arr["posterior_samples"], dtype=np.float32)),
    }
    for rule_name, pred in rules.items():
        out_path = run_root / f"predictions_{rule_name}.npz"
        if not out_path.exists():
            save_prediction_pair(out_path, pred, y_true)
        outputs.append(out_path)
    return outputs


def ensure_simformer_rule_files(run_root: Path) -> list[Path]:
    raw_path = run_root / "simformer_hh_predictions.npz"
    if not raw_path.exists():
        return []
    arr = np.load(raw_path)
    if not {"theta_true", "posterior_mean", "posterior_samples"}.issubset(arr.files):
        return []
    y_true = np.asarray(arr["theta_true"], dtype=np.float32)
    samples = np.asarray(arr["posterior_samples"], dtype=np.float32)
    rules: dict[str, np.ndarray] = {
        "mean": np.asarray(arr["posterior_mean"], dtype=np.float32),
        "median": np.median(samples, axis=1),
        "medoid": sample_medoid(samples),
    }
    outputs: list[Path] = []
    for rule_name, pred in rules.items():
        out_path = run_root / f"predictions_{rule_name}.npz"
        if not out_path.exists():
            save_prediction_pair(out_path, pred, y_true)
        outputs.append(out_path)
    return outputs


def explicit_prediction_rule_files(run_root: Path) -> list[Path]:
    files = []
    for path in sorted(run_root.glob("predictions_*.npz")):
        if path.name == "predictions_test.npz":
            continue
        files.append(path)
    return files


def classify_framework_branch(framework: str) -> str:
    lowered = framework.lower()
    if lowered.startswith("bayesflow"):
        return "bayesflow"
    if lowered.startswith("swyft_tmnre"):
        return "tmnre"
    if lowered.startswith("swyft"):
        return "swyft"
    if lowered.startswith("temporal_cnn"):
        return "temporal_deterministic"
    return lowered or "framework_other"


def artifact_kind_for_metrics_summary(metrics_obj: dict) -> str:
    framework = str(metrics_obj.get("framework", "")).strip()
    method = str(metrics_obj.get("method", "")).strip()
    if framework:
        return "aligned_framework"
    if method in {"fmpe", "snpe", "npse", "snle", "snre"}:
        return "poolseq_sbi"
    return ""


def build_run_inventory(
    campaign_root: Path,
    tasks_by_id: dict[str, dict[str, object]],
    roots_to_task_id: dict[Path, str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    inventory_rows: list[dict[str, object]] = []
    discovered_runs: dict[str, dict[str, object]] = {}

    for root in discover_candidate_roots(campaign_root, roots_to_task_id):
        task_id = match_task_id(root, roots_to_task_id)
        task = tasks_by_id.get(task_id, {})
        source_run_name = str(task.get("source_run_name", root.name))
        branch_family = str(task.get("branch_family_hint", ""))
        comparability_class = str(task.get("comparability_hint", ""))
        phase = str(task.get("phase", ""))
        phase_label = str(task.get("phase_label", phase))
        launch_group = str(task.get("launch_group", ""))
        queue_status = str(task.get("queue_status", ""))
        notes = ""
        framework = ""
        method_family = ""
        metrics_path: Path | None = None
        artifact_kind = ""

        if (root / "active_sequential_manifest.json").exists():
            artifact_kind = "active_sequential"
            branch_family = branch_family or "active_sequential"
            comparability_class = comparability_class or ACTIVE_COMPARABILITY
            manifest = load_json(root / "active_sequential_manifest.json")
            if isinstance(manifest, dict):
                method_family = str(manifest.get("method_family", "active_sequential"))
                notes = str(manifest.get("acquisition_rule", ""))
        elif (root / "asnpe_manifest.json").exists():
            artifact_kind = "asnpe"
            branch_family = branch_family or "asnpe"
            comparability_class = comparability_class or ACTIVE_COMPARABILITY
            manifest = load_json(root / "asnpe_manifest.json")
            if isinstance(manifest, dict):
                method_family = str(manifest.get("method_family", "asnpe"))
                notes = str(manifest.get("acquisition_definition", ""))
        elif (root / "simformer_hh_metrics_summary.json").exists():
            artifact_kind = "simformer"
            branch_family = branch_family or "simformer"
            comparability_class = comparability_class or SIMFORMER_COMPARABILITY
            metrics_path = root / "simformer_hh_metrics_summary.json"
            metrics_obj = load_json(metrics_path)
            if isinstance(metrics_obj, dict):
                method_family = str(metrics_obj.get("method_family", "simformer"))
                notes = str(metrics_obj.get("limitation_note", ""))
        elif (root / "generalized_bayes_npe_manifest.json").exists():
            artifact_kind = "generalized_bayes_npe"
            branch_family = branch_family or "generalized_bayes_npe"
            comparability_class = comparability_class or GENERALIZED_BAYES_COMPARABILITY
            metrics_path = root / "generalized_bayes_npe_manifest.json"
            metrics_obj = load_json(metrics_path)
            if isinstance(metrics_obj, dict):
                method_family = str(metrics_obj.get("method_family", "generalized_bayes_npe"))
                notes = str(metrics_obj.get("adaptation_note", ""))
        elif (root / "metrics_summary.json").exists():
            metrics_path = root / "metrics_summary.json"
            metrics_obj = load_json(metrics_path)
            if isinstance(metrics_obj, dict):
                artifact_kind = artifact_kind_for_metrics_summary(metrics_obj)
                framework = str(metrics_obj.get("framework", "")).strip()
                if artifact_kind == "aligned_framework":
                    branch_family = branch_family or classify_framework_branch(framework)
                    comparability_class = comparability_class or infer_comparability_class(branch_family)
                    method_family = framework
                elif artifact_kind == "poolseq_sbi":
                    branch_family = branch_family or "poolseq_sbi"
                    comparability_class = comparability_class or POOLSEQ_COMPARABILITY
                    method_family = str(metrics_obj.get("method", "poolseq_sbi"))
                else:
                    continue
        else:
            continue

        run_name = source_run_name or root.name
        inventory_row = {
            "task_id": task_id,
            "run_name": run_name,
            "phase": phase,
            "phase_label": phase_label,
            "launch_group": launch_group,
            "queue_status": queue_status,
            "branch_family": branch_family,
            "method_family": method_family,
            "artifact_kind": artifact_kind,
            "comparability_class": comparability_class,
            "framework": framework,
            "artifact_dir": relpath(root, campaign_root),
            "metrics_path": relpath(metrics_path, campaign_root) if metrics_path else "",
            "source_run_name": source_run_name,
            "source_reference": str(task.get("source_run_output_root", "")) or str(task.get("source_matrix", "")),
            "notes": notes,
        }
        inventory_rows.append(inventory_row)
        discovered_runs[run_name] = {
            "task_id": task_id,
            "run_name": run_name,
            "root": root,
            "artifact_kind": artifact_kind,
            "branch_family": branch_family,
            "comparability_class": comparability_class,
            "framework": framework,
            "method_family": method_family,
            "metrics_path": metrics_path,
            "phase": phase,
            "phase_label": phase_label,
            "launch_group": launch_group,
            "queue_status": queue_status,
            "notes": notes,
        }
    inventory_rows.sort(key=lambda row: (row["branch_family"], row["run_name"]))
    return inventory_rows, discovered_runs


def build_framework_snapshot_rows(
    campaign_root: Path,
    discovered_runs: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_name, meta in discovered_runs.items():
        if meta["artifact_kind"] != "aligned_framework":
            continue
        metrics_path = meta.get("metrics_path")
        if not isinstance(metrics_path, Path) or not metrics_path.exists():
            continue
        metrics_obj = load_json(metrics_path)
        if not isinstance(metrics_obj, dict):
            continue
        rows.append(
            {
                "run_name": run_name,
                "task_id": meta["task_id"],
                "framework": meta["framework"],
                "branch_family": meta["branch_family"],
                "comparability_class": meta["comparability_class"],
                "mae": metric_from_summary(metrics_obj, "mae"),
                "mse": metric_from_summary(metrics_obj, "mse"),
                "r2": metric_from_summary(metrics_obj, "r2"),
                "runtime_sec": safe_float(metrics_obj.get("total_runtime_sec")) or safe_float(metrics_obj.get("training_runtime_sec")),
                "artifact_dir": relpath(meta["root"], campaign_root),
                "source": relpath(metrics_path, campaign_root),
            }
        )
    rows = [row for row in rows if safe_float(row["mae"]) is not None]
    rows.sort(key=lambda row: (float(row["mae"]), -float(row["r2"] or 0.0), str(row["run_name"])))
    return rows


def build_decision_rule_rows(
    campaign_root: Path,
    discovered_runs: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_name, meta in discovered_runs.items():
        root = meta["root"]
        if meta["artifact_kind"] == "poolseq_sbi":
            ensure_poolseq_rule_files(root)
        elif meta["artifact_kind"] == "simformer":
            ensure_simformer_rule_files(root)
        elif meta["artifact_kind"] == "aligned_framework":
            family = str(meta["branch_family"])
            if family == "bayesflow":
                ensure_bayesflow_rule_files(root)
            elif family in {"swyft", "tmnre"}:
                ensure_swyft_rule_files(root)
            else:
                ensure_mean_prediction_file(root)

        for pred_path in explicit_prediction_rule_files(root):
            pair = prediction_pair_from_npz(pred_path)
            if pair is None:
                continue
            pred, true = pair
            metrics = compute_metrics(true, pred)
            rows.append(
                {
                    "run_name": run_name,
                    "task_id": meta["task_id"],
                    "family": meta["branch_family"],
                    "comparability_class": meta["comparability_class"],
                    "decision_rule": pred_path.stem.replace("predictions_", "", 1),
                    **metrics,
                    "prediction_file": relpath(pred_path, campaign_root),
                    "source": relpath(root, campaign_root),
                }
            )
    rows.sort(key=lambda row: (float(row["mae"]), -float(row["r2"]), str(row["run_name"]), str(row["decision_rule"])))
    return rows


def build_active_policy_rows(
    campaign_root: Path,
    discovered_runs: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_name, meta in discovered_runs.items():
        root = meta["root"]
        if meta["artifact_kind"] == "active_sequential":
            manifest_path = root / "active_sequential_manifest.json"
            manifest = load_json(manifest_path)
            if not isinstance(manifest, dict):
                continue
            aggregate = manifest.get("aggregate_metrics", {})
            if not isinstance(aggregate, dict):
                aggregate = {}
            rows.append(
                {
                    "run_name": run_name,
                    "task_id": meta["task_id"],
                    "method_family": str(manifest.get("method_family", "active_sequential")),
                    "comparability_class": ACTIVE_COMPARABILITY,
                    "acquisition_policy": str(manifest.get("acquisition_policy", "")),
                    "mean_abs_log10_error_params_mean": safe_float(aggregate.get("mean_abs_log10_error_params_mean")),
                    "mean_relative_error_params_mean": safe_float(aggregate.get("mean_relative_error_params_mean")),
                    "posterior_contraction_mean_mean": safe_float(aggregate.get("posterior_contraction_mean_mean")),
                    "ess_mean": safe_float(aggregate.get("ess_mean")),
                    "runtime_sec": safe_float(manifest.get("runtime_sec")),
                    "source": relpath(manifest_path, campaign_root),
                }
            )
        elif meta["artifact_kind"] == "asnpe":
            manifest_path = root / "asnpe_manifest.json"
            manifest = load_json(manifest_path)
            if not isinstance(manifest, dict):
                continue
            aggregate = manifest.get("aggregate_metrics", {})
            if not isinstance(aggregate, dict):
                aggregate = {}
            mapped_true = manifest.get("mapped_true", {})
            mapped_pred = manifest.get("mapped_posterior_mean", {})
            mean_relative_error = None
            if isinstance(mapped_true, dict) and isinstance(mapped_pred, dict):
                rel_errors = []
                for key, true_val in mapped_true.items():
                    pred_val = mapped_pred.get(key)
                    true_num = safe_float(true_val)
                    pred_num = safe_float(pred_val)
                    if true_num is None or pred_num is None:
                        continue
                    rel_errors.append(abs(pred_num - true_num) / max(abs(true_num), 1.0e-12))
                if rel_errors:
                    mean_relative_error = float(np.mean(np.asarray(rel_errors, dtype=np.float64)))
            rows.append(
                {
                    "run_name": run_name,
                    "task_id": meta["task_id"],
                    "method_family": str(manifest.get("method_family", "asnpe")),
                    "comparability_class": ACTIVE_COMPARABILITY,
                    "acquisition_policy": "asnpe_style",
                    "mean_abs_log10_error_params_mean": (
                        safe_float(aggregate.get("mean_abs_log10_error_params_mean"))
                        or safe_float(manifest.get("posterior_mean_abs_log10_error"))
                    ),
                    "mean_relative_error_params_mean": (
                        safe_float(aggregate.get("mean_relative_error_params_mean"))
                        or mean_relative_error
                    ),
                    "posterior_contraction_mean_mean": (
                        safe_float(aggregate.get("posterior_std_norm_mean_mean"))
                        or safe_float(aggregate.get("posterior_contraction_mean_mean"))
                    ),
                    "ess_mean": "",
                    "runtime_sec": safe_float(manifest.get("runtime_sec")),
                    "source": relpath(manifest_path, campaign_root),
                }
            )
    rows = [row for row in rows if safe_float(row["mean_abs_log10_error_params_mean"]) is not None]
    rows.sort(key=lambda row: (float(row["mean_abs_log10_error_params_mean"]), numeric_sort_key(row["runtime_sec"]), str(row["run_name"])))
    return rows


def build_method_comparison_rows(
    campaign_root: Path,
    framework_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    active_rows: list[dict[str, object]],
    discovered_runs: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in framework_rows:
        rows.append(
            {
                "result_space": "aligned_framework_snapshot",
                "name": row["run_name"],
                "branch_family": row["branch_family"],
                "comparability_class": row["comparability_class"],
                "metric_name": "mae",
                "metric_value": row["mae"],
                "secondary_metric_name": "r2",
                "secondary_metric_value": row["r2"],
                "source": row["source"],
                "note": row["framework"],
            }
        )
    for row in decision_rows:
        rows.append(
            {
                "result_space": "posterior_decision_rules",
                "name": f"{row['run_name']}::{row['decision_rule']}",
                "branch_family": row["family"],
                "comparability_class": row["comparability_class"],
                "metric_name": "mae",
                "metric_value": row["mae"],
                "secondary_metric_name": "r2",
                "secondary_metric_value": row["r2"],
                "source": row["prediction_file"],
                "note": row["decision_rule"],
            }
        )
    for row in active_rows:
        rows.append(
            {
                "result_space": "active_policy_comparison",
                "name": row["run_name"],
                "branch_family": "asnpe" if row["acquisition_policy"] == "asnpe_style" else "active_sequential",
                "comparability_class": row["comparability_class"],
                "metric_name": "mean_abs_log10_error_params_mean",
                "metric_value": row["mean_abs_log10_error_params_mean"],
                "secondary_metric_name": "runtime_sec",
                "secondary_metric_value": row["runtime_sec"],
                "source": row["source"],
                "note": row["acquisition_policy"] or "legacy_default",
            }
        )
    for run_name, meta in discovered_runs.items():
        if meta["artifact_kind"] == "simformer":
            metrics_obj = load_json(meta["root"] / "simformer_hh_metrics_summary.json")
            if not isinstance(metrics_obj, dict):
                continue
            test_metrics = metrics_obj.get("test_metrics", {})
            if not isinstance(test_metrics, dict):
                continue
            rows.append(
                {
                    "result_space": "simformer_reference",
                    "name": run_name,
                    "branch_family": "simformer",
                    "comparability_class": SIMFORMER_COMPARABILITY,
                    "metric_name": "mae",
                    "metric_value": safe_float(test_metrics.get("mae")),
                    "secondary_metric_name": "r2",
                    "secondary_metric_value": safe_float(test_metrics.get("r2")),
                    "source": relpath(meta["root"] / "simformer_hh_metrics_summary.json", campaign_root),
                    "note": "official upstream HH benchmark",
                }
            )
        elif meta["artifact_kind"] == "generalized_bayes_npe":
            manifest = load_json(meta["root"] / "generalized_bayes_npe_manifest.json")
            if not isinstance(manifest, dict):
                continue
            beta_summary = manifest.get("beta_summary", [])
            if not isinstance(beta_summary, list):
                continue
            for beta_row in beta_summary:
                if not isinstance(beta_row, dict):
                    continue
                beta = safe_float(beta_row.get("beta"))
                rows.append(
                    {
                        "result_space": "generalized_bayes_beta_sweep",
                        "name": f"{run_name}::beta_{beta:.2f}" if beta is not None else run_name,
                        "branch_family": "generalized_bayes_npe",
                        "comparability_class": GENERALIZED_BAYES_COMPARABILITY,
                        "metric_name": "mean_abs_log10_error_params_mean",
                        "metric_value": safe_float(beta_row.get("mean_abs_log10_error_params_mean")),
                        "secondary_metric_name": "posterior_std_norm_mean_mean",
                        "secondary_metric_value": safe_float(beta_row.get("posterior_std_norm_mean_mean")),
                        "source": relpath(meta["root"] / "generalized_bayes_npe_manifest.json", campaign_root),
                        "note": f"beta={beta_row.get('beta')}",
                    }
                )
    rows = [row for row in rows if safe_float(row["metric_value"]) is not None]
    rows.sort(key=lambda row: (row["result_space"], numeric_sort_key(row["metric_value"]), str(row["name"])))
    return rows


def build_branch_registry_rows(
    tasks_by_id: dict[str, dict[str, object]],
    discovered_runs: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    branch_order = [
        "bayesflow",
        "swyft",
        "tmnre",
        "poolseq_sbi",
        "active_sequential",
        "asnpe",
        "simformer",
        "generalized_bayes_npe",
        "temporal_deterministic",
    ]
    task_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for task in tasks_by_id.values():
        branch = str(task.get("branch_family_hint", "")).strip()
        if not branch:
            continue
        task_counts[branch][str(task.get("queue_status", "PENDING"))] += 1

    completed_runs: dict[str, list[str]] = defaultdict(list)
    for run_name, meta in discovered_runs.items():
        branch = str(meta["branch_family"])
        if branch:
            completed_runs[branch].append(run_name)

    rows: list[dict[str, object]] = []
    for branch in branch_order:
        status_counts = task_counts.get(branch, {})
        completed = len(completed_runs.get(branch, []))
        running = sum(status_counts.get(state, 0) for state in ("RUNNING", "LAUNCHING"))
        pending = sum(status_counts.get(state, 0) for state in ("PENDING", "BLOCKED"))
        failed = status_counts.get("FAILED", 0)
        if completed > 0 and running == 0 and pending == 0 and failed == 0:
            status = "completed"
        elif completed > 0 and (running > 0 or pending > 0 or failed > 0):
            status = "partial"
        elif running > 0:
            status = "running"
        elif pending > 0:
            status = "pending"
        elif failed > 0:
            status = "failed"
        else:
            status = "not_present"
        evidence = ", ".join(completed_runs.get(branch, [])[:8])
        note = ""
        if branch == "simformer":
            note = "official upstream HH benchmark, not directly comparable to local aligned or surrogate rows"
        elif branch == "generalized_bayes_npe":
            note = "bounded local beta-conditioned adaptation, which should not be reported as the exact paper implementation"
        elif branch in {"bayesflow", "swyft", "tmnre"}:
            note = "decision-rule coverage is faithful only for explicit local rule files plus mean rows derivable from raw predictions"
        elif branch == "poolseq_sbi":
            note = "mean, median, selected, and medoid rules are locally derivable from campaign-stored posterior outputs"
        rows.append(
            {
                "branch": branch,
                "status": status,
                "completed_runs": completed,
                "running_tasks": running,
                "pending_tasks": pending,
                "failed_tasks": failed,
                "comparability_class": infer_comparability_class(branch),
                "evidence": evidence,
                "note": note,
            }
        )
    return rows


def build_key_results_rows(
    framework_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    active_rows: list[dict[str, object]],
    method_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if framework_rows:
        best = framework_rows[0]
        rows.append(
            {
                "section": "aligned_framework_snapshot",
                "name": best["run_name"],
                "branch_family": best["branch_family"],
                "comparability_class": best["comparability_class"],
                "metric_name": "mae",
                "metric_value": best["mae"],
                "source": best["source"],
                "note": best["framework"],
            }
        )
    if decision_rows:
        comparable_rows = [row for row in decision_rows if row["comparability_class"] != SIMFORMER_COMPARABILITY]
        best = comparable_rows[0] if comparable_rows else decision_rows[0]
        rows.append(
            {
                "section": "posterior_decision_rules",
                "name": f"{best['run_name']}::{best['decision_rule']}",
                "branch_family": best["family"],
                "comparability_class": best["comparability_class"],
                "metric_name": "mae",
                "metric_value": best["mae"],
                "source": best["prediction_file"],
                "note": best["decision_rule"],
            }
        )
    heuristic_rows = [row for row in active_rows if row["acquisition_policy"] != "asnpe_style"]
    if heuristic_rows:
        best = heuristic_rows[0]
        rows.append(
            {
                "section": "active_policy_heuristic",
                "name": best["run_name"],
                "branch_family": "active_sequential",
                "comparability_class": best["comparability_class"],
                "metric_name": "mean_abs_log10_error_params_mean",
                "metric_value": best["mean_abs_log10_error_params_mean"],
                "source": best["source"],
                "note": best["acquisition_policy"] or "legacy_default",
            }
        )
    asnpe_rows = [row for row in active_rows if row["acquisition_policy"] == "asnpe_style"]
    if asnpe_rows:
        best = asnpe_rows[0]
        rows.append(
            {
                "section": "asnpe",
                "name": best["run_name"],
                "branch_family": "asnpe",
                "comparability_class": best["comparability_class"],
                "metric_name": "mean_abs_log10_error_params_mean",
                "metric_value": best["mean_abs_log10_error_params_mean"],
                "source": best["source"],
                "note": "asnpe_style",
            }
        )
    simformer_rows = [row for row in method_rows if row["result_space"] == "simformer_reference"]
    if simformer_rows:
        best = min(simformer_rows, key=lambda row: (-float(row["secondary_metric_value"]), float(row["metric_value"])))
        rows.append(
            {
                "section": "simformer_reference",
                "name": best["name"],
                "branch_family": "simformer",
                "comparability_class": best["comparability_class"],
                "metric_name": best["metric_name"],
                "metric_value": best["metric_value"],
                "source": best["source"],
                "note": "official upstream HH benchmark",
            }
        )
    gb_rows = [row for row in method_rows if row["result_space"] == "generalized_bayes_beta_sweep"]
    if gb_rows:
        best = gb_rows[0]
        rows.append(
            {
                "section": "generalized_bayes_beta_sweep",
                "name": best["name"],
                "branch_family": "generalized_bayes_npe",
                "comparability_class": best["comparability_class"],
                "metric_name": best["metric_name"],
                "metric_value": best["metric_value"],
                "source": best["source"],
                "note": best["note"],
            }
        )
    return rows


def build_limitations_rows() -> list[dict[str, object]]:
    return [
        {
            "limitation_id": "L1",
            "scope": "comparability",
            "statement": "Do not rank aligned BayesFlow, Swyft, TMNRE, pool-sequential SBI, surrogate active-design, Simformer upstream HH, and generalized-Bayes adaptation rows in one undifferentiated table, because they target different observation contracts and posterior objectives.",
            "severity": "high",
        },
        {
            "limitation_id": "L2",
            "scope": "decision_rules",
            "statement": "For normalized BayesFlow and Swyft posterior runs, this refresh can reconstruct mean rows directly from local predictions_test artifacts, while median, medoid, and map-bank rows are regenerated only when explicit predictions_* files are present under the campaign root.",
            "severity": "high",
        },
        {
            "limitation_id": "L3",
            "scope": "active_design",
            "statement": "Active sequential and ASNPE rows remain assumption-conditioned surrogate branches, so they should be reported as active-design evidence rather than as direct Track4 frontier claims.",
            "severity": "high",
        },
        {
            "limitation_id": "L4",
            "scope": "simformer",
            "statement": "Simformer rows remain upstream HH benchmark references, which are methodologically useful but not numerically comparable to the local aligned or surrogate Track4-style workloads.",
            "severity": "high",
        },
        {
            "limitation_id": "L5",
            "scope": "generalized_bayes",
            "statement": "The generalized-Bayes NPE branch is a bounded local beta-conditioned adaptation, which should be documented as such, because it is not the authors' exact released workflow.",
            "severity": "medium",
        },
    ]


def build_branch_registry_report(rows: list[dict[str, object]], csv_path: Path) -> str:
    lines = [
        "# HH Track4 A30 Literature and Surrogate Branch Registry",
        "",
        f"- source csv: `{csv_path.name}`",
        "",
        "| branch | status | completed_runs | running_tasks | pending_tasks | failed_tasks | comparability_class | note |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['branch']}` | `{row['status']}` | {int(row['completed_runs'])} | {int(row['running_tasks'])} | {int(row['pending_tasks'])} | {int(row['failed_tasks'])} | `{row['comparability_class']}` | {row['note']} |"
        )
    return "\n".join(lines) + "\n"


def build_reporting_synthesis(
    campaign_root: Path,
    inventory_rows: list[dict[str, object]],
    framework_rows: list[dict[str, object]],
    decision_rows: list[dict[str, object]],
    active_rows: list[dict[str, object]],
    method_rows: list[dict[str, object]],
    branch_rows: list[dict[str, object]],
    limitation_rows: list[dict[str, object]],
) -> str:
    lines = [
        "# HH Track4 A30 Literature and Surrogate Refresh",
        "",
        f"- campaign root: `{campaign_root}`",
        f"- discovered later-branch runs: `{len(inventory_rows)}`",
        "",
        "## Scope",
        "",
        "This refresh is restricted to the later aligned literature and surrogate branches under the supplied A30 campaign root, namely the BayesFlow, Swyft, TMNRE, pool-sequential SBI, surrogate active-sequential, ASNPE, Simformer, and generalized-Bayes branches, while the earlier direct A30 classical refresh remains out of scope for this script.",
        "",
        "## Coverage",
        "",
        "| branch | status | completed_runs | note |",
        "|---|---|---:|---|",
    ]
    for row in branch_rows:
        lines.append(f"| `{row['branch']}` | `{row['status']}` | {int(row['completed_runs'])} | {row['note']} |")

    lines += [
        "",
        "## Key Tables",
        "",
        "| table | rows | interpretation |",
        "|---|---:|---|",
        f"| framework snapshot | {len(framework_rows)} | aligned framework-level snapshot from campaign-local metrics_summary artifacts |",
        f"| posterior decision rules | {len(decision_rows)} | campaign-local rule rows, which include explicit rule files plus mean rows that can be derived directly from raw predictions |",
        f"| active policy comparison | {len(active_rows)} | surrogate active-sequential and ASNPE policy rows |",
        f"| normalized method comparison | {len(method_rows)} | direct comparison table with explicit comparability classes |",
        "",
        "## Best Current Rows",
        "",
        "| result space | best row | primary metric | note |",
        "|---|---|---:|---|",
    ]
    if framework_rows:
        best = framework_rows[0]
        lines.append(f"| aligned framework snapshot | `{best['run_name']}` | {float(best['mae']):.6f} | `{best['framework']}` |")
    if decision_rows:
        best = decision_rows[0]
        lines.append(f"| posterior decision rules | `{best['run_name']}::{best['decision_rule']}` | {float(best['mae']):.6f} | `{best['family']}` |")
    heuristic_rows = [row for row in active_rows if row["acquisition_policy"] != "asnpe_style"]
    if heuristic_rows:
        best = heuristic_rows[0]
        lines.append(f"| heuristic active policy | `{best['run_name']}` | {float(best['mean_abs_log10_error_params_mean']):.6f} | `{best['acquisition_policy'] or 'legacy_default'}` |")
    asnpe_rows = [row for row in active_rows if row["acquisition_policy"] == "asnpe_style"]
    if asnpe_rows:
        best = asnpe_rows[0]
        lines.append(f"| ASNPE | `{best['run_name']}` | {float(best['mean_abs_log10_error_params_mean']):.6f} | `asnpe_style` |")
    simformer_rows = [row for row in method_rows if row["result_space"] == "simformer_reference"]
    if simformer_rows:
        best = min(simformer_rows, key=lambda row: (-float(row["secondary_metric_value"]), float(row["metric_value"])))
        lines.append(f"| Simformer upstream HH | `{best['name']}` | {float(best['secondary_metric_value']):.6f} | `secondary metric is r2` |")
    gb_rows = [row for row in method_rows if row["result_space"] == "generalized_bayes_beta_sweep"]
    if gb_rows:
        best = gb_rows[0]
        lines.append(f"| generalized-Bayes beta sweep | `{best['name']}` | {float(best['metric_value']):.6f} | `{best['note']}` |")

    lines += [
        "",
        "## Limitations",
        "",
        "| id | scope | severity | statement |",
        "|---|---|---|---|",
    ]
    for row in limitation_rows:
        lines.append(f"| `{row['limitation_id']}` | `{row['scope']}` | `{row['severity']}` | {row['statement']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).resolve()
    if not campaign_root.exists():
        raise SystemExit(f"Missing campaign root: {campaign_root}")

    tag = campaign_tag(campaign_root)
    manifest_path, manifest = load_manifest(campaign_root)
    registry_path, queue_registry = load_task_registry(campaign_root)
    tasks_by_id, roots_to_task_id = build_task_index(campaign_root, manifest, queue_registry)
    inventory_rows, discovered_runs = build_run_inventory(campaign_root, tasks_by_id, roots_to_task_id)
    framework_rows = build_framework_snapshot_rows(campaign_root, discovered_runs)
    decision_rows = build_decision_rule_rows(campaign_root, discovered_runs)
    active_rows = build_active_policy_rows(campaign_root, discovered_runs)
    method_rows = build_method_comparison_rows(campaign_root, framework_rows, decision_rows, active_rows, discovered_runs)
    branch_rows = build_branch_registry_rows(tasks_by_id, discovered_runs)
    key_rows = build_key_results_rows(framework_rows, decision_rows, active_rows, method_rows)
    limitation_rows = build_limitations_rows()

    tables_dir = campaign_root / "tables"
    reports_dir = campaign_root / "reports"
    inventory_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}run_inventory_20260427_{tag}.csv"
    framework_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}framework_snapshot_20260427_{tag}.csv"
    decision_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}posterior_decision_rules_20260427_{tag}.csv"
    active_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}active_policy_comparison_20260427_{tag}.csv"
    method_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}method_comparison_20260427_{tag}.csv"
    branch_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}branch_registry_20260427_{tag}.csv"
    key_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}reporting_key_results_20260427_{tag}.csv"
    limitations_csv = tables_dir / f"{OUR_OUTPUT_PREFIX}reporting_limitations_20260427_{tag}.csv"
    branch_md = reports_dir / f"{OUR_OUTPUT_PREFIX}branch_registry_20260427_{tag}.md"
    synthesis_md = reports_dir / f"{OUR_OUTPUT_PREFIX}reporting_synthesis_20260427_{tag}.md"

    write_csv_rows(inventory_csv, RUN_INVENTORY_FIELDS, inventory_rows)
    write_csv_rows(framework_csv, FRAMEWORK_SNAPSHOT_FIELDS, framework_rows)
    write_csv_rows(decision_csv, DECISION_RULE_FIELDS, decision_rows)
    write_csv_rows(active_csv, ACTIVE_POLICY_FIELDS, active_rows)
    write_csv_rows(method_csv, METHOD_COMPARISON_FIELDS, method_rows)
    write_csv_rows(branch_csv, BRANCH_REGISTRY_FIELDS, branch_rows)
    write_csv_rows(key_csv, KEY_RESULTS_FIELDS, key_rows)
    write_csv_rows(limitations_csv, LIMITATION_FIELDS, limitation_rows)
    write_text(branch_md, build_branch_registry_report(branch_rows, branch_csv))
    write_text(
        synthesis_md,
        build_reporting_synthesis(
            campaign_root=campaign_root,
            inventory_rows=inventory_rows,
            framework_rows=framework_rows,
            decision_rows=decision_rows,
            active_rows=active_rows,
            method_rows=method_rows,
            branch_rows=branch_rows,
            limitation_rows=limitation_rows,
        ),
    )

    print(manifest_path)
    if registry_path is not None:
        print(registry_path)
    print(inventory_csv)
    print(framework_csv)
    print(decision_csv)
    print(active_csv)
    print(method_csv)
    print(branch_csv)
    print(key_csv)
    print(limitations_csv)
    print(branch_md)
    print(synthesis_md)


if __name__ == "__main__":
    main()
