#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
PYTHON_BIN = "/projects/neuro-collab/conda/neuro-collab-env/bin/python"
PYTHONPATH_VALUE = f"{REPO}:/projects/neuro-collab/code/dl-kit-main:${{PYTHONPATH:-}}"
PARAMS_TRACK4 = "pytorch/configs/fourth_track_hh_full/params_dnn_tar_hh_full.yaml"
IMPORTANT = REPO / "data" / "important_notes"
SHARED_DATA_DIR = (
    IMPORTANT / "optimization_track_20260411_v100_interactive" / "shared_data" / "track4_hh_full" / "concatenated_data"
)

PHASE_DIRECT_FITTING = "phaseSURR_direct_fitting"
PHASE_RUNALL = "phaseLIT_aligned_runall"
HYBRID_WRAPPER = "scripts/run_hh_track4_assumption_conditioned_hybrid_refinement_surrogate_bundle_integration_20260425.py"
WABC_WRAPPER = "scripts/run_hh_track4_assumption_conditioned_wasserstein_abc_surrogate_bundle_integration_20260425.py"
HYBRID_REFRESH = "scripts/analyze_hh_track4_a30_hybrid_initializer_comparison_refresh_20260427.py"

HYBRID_RUNTIME_SEC = 45.0
MIDPOINT_RUNTIME_SEC = 20.745
WABC_SMOKE_RUNTIME_SEC = 59.099
WABC_FULL_RUNTIME_SEC = 21.314
DEFAULT_HYBRID_CPUS = 8
DEFAULT_HYBRID_GPUS = 0
DEFAULT_HYBRID_RESOURCE_CLASS = "cpu_only"
DIRECT_FITTING_SEED = "20260427"

TASK_FAMILY_HYBRID_RAW = "surrogate_direct_fitting_hybrid_prediction_test"
TASK_FAMILY_HYBRID_RULE = "surrogate_direct_fitting_hybrid_decision_rule"
TASK_FAMILY_HYBRID_MIDPOINT = "surrogate_direct_fitting_hybrid_midpoint"
TASK_FAMILY_WABC = "surrogate_direct_fitting_wasserstein_abc"

RULE_STEM_RE = re.compile(r"^predictions_(?P<rule>[A-Za-z0-9_]+)$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Append the clean A30 hybrid-refinement tail and direct-fitting bundle rows to the live "
            "HH Track4 literal rerun manifest, using campaign-local prediction outputs."
        )
    )
    ap.add_argument("--campaign-root", required=True)
    ap.add_argument("--manifest-json", required=True)
    return ap.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def normalize_csv_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def write_manifest_csv(path: Path, tasks: list[dict[str, Any]]) -> None:
    existing_header: list[str] = []
    if path.exists():
        with path.open(newline="") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []

    discovered = set(existing_header)
    for task in tasks:
        discovered.update(task.keys())

    header = list(existing_header)
    for key in sorted(discovered):
        if key not in header:
            header.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        for task in tasks:
            row = {key: normalize_csv_value(task.get(key, "")) for key in header}
            writer.writerow(row)


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "task"


def shell_join(parts: list[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def python_command(script_rel: str, *args: object) -> str:
    cmd = [PYTHON_BIN, str(REPO / script_rel), *map(str, args)]
    return f'cd {shlex.quote(str(REPO))} && env PYTHONPATH="{PYTHONPATH_VALUE}" {shell_join(cmd)}'


def resolve_campaign_path(raw_path: str, campaign_root: Path) -> Path | None:
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = campaign_root / path
    try:
        resolved = path.expanduser().resolve()
    except FileNotFoundError:
        resolved = path.expanduser()
    try:
        resolved.relative_to(campaign_root.resolve())
    except ValueError:
        return None
    return resolved


def hybrid_run_root(campaign_root: Path, label: str) -> Path:
    return campaign_root / "runs" / PHASE_DIRECT_FITTING / label


def base_task(
    *,
    task_id: str,
    label: str,
    command: str,
    expected_runtime_sec: float,
    output_paths: list[Path],
    depends_on: list[str],
    task_family: str,
    source_run_name: str,
    source_prediction_file: str,
    source_decision_rule: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "phase": PHASE_DIRECT_FITTING,
        "phase_label": "assumption_conditioned_surrogate_direct_fitting",
        "launch_group": task_family,
        "launch_mode": "appended_live_tail",
        "label": label,
        "command": command,
        "resource_class": DEFAULT_HYBRID_RESOURCE_CLASS,
        "cpus_per_task": DEFAULT_HYBRID_CPUS,
        "gpus_per_task": DEFAULT_HYBRID_GPUS,
        "expected_runtime_sec": round(expected_runtime_sec, 3),
        "depends_on": sorted(set(dep for dep in depends_on if dep)),
        "output_paths": [str(path) for path in output_paths],
        "manifest_category": "surrogate_direct_fitting_literal_tail",
        "analysis_status": "included_literal_tail",
        "analysis_summary": (
            "The clean A30 manifest retains the literal surrogate direct-fitting tail, which includes raw "
            "prediction seeds, posterior decision-rule seeds, and baseline bundle rows, because later "
            "writing uses negative and pathological direct-fitting evidence explicitly."
        ),
        "analysis_evidence_paths": [source_prediction_file] if source_prediction_file else [],
        "family": "hybrid_refinement" if task_family != TASK_FAMILY_WABC else "wasserstein_abc",
        "track_key": "surrogate_direct_fitting",
        "strategy_id": label,
        "strategy_description": notes,
        "baseline_name": "",
        "seed": DIRECT_FITTING_SEED,
        "source_matrix": "",
        "source_registry": "",
        "source_row_id": "",
        "source_run_output_root": str(output_paths[0].parent),
        "source_params_file": PARAMS_TRACK4,
        "source_notes": notes,
        "task_family": task_family,
        "source_run_name": source_run_name,
        "source_prediction_file": source_prediction_file,
        "source_decision_rule": source_decision_rule,
    }


def existing_task_ids(tasks: list[dict[str, Any]]) -> set[str]:
    return {str(task.get("task_id", "")).strip() for task in tasks if str(task.get("task_id", "")).strip()}


def task_lookup_by_label(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task in tasks:
        label = str(task.get("label", "")).strip()
        if label:
            out[label] = task
    return out


def run_output_root_from_task(task: dict[str, Any], campaign_root: Path) -> Path | None:
    for raw_path in task.get("output_paths", []):
        path = resolve_campaign_path(str(raw_path), campaign_root)
        if path is None:
            continue
        if path.suffix:
            return path.parent
        return path
    return None


def runall_source_tasks(tasks: list[dict[str, Any]], campaign_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in tasks:
        if str(task.get("phase", "")) != PHASE_RUNALL:
            continue
        run_root = run_output_root_from_task(task, campaign_root)
        if run_root is None:
            continue
        task = dict(task)
        task["_run_root"] = run_root
        out.append(task)
    out.sort(key=lambda item: str(item.get("label", "")))
    return out


def local_decision_rule_predictions(source_task: dict[str, Any]) -> list[tuple[str, Path]]:
    run_root = Path(source_task["_run_root"])
    found: list[tuple[str, Path]] = []
    for path in sorted(run_root.glob("predictions_*.npz")):
        if path.name == "predictions_test.npz":
            continue
        match = RULE_STEM_RE.match(path.stem)
        if not match:
            continue
        found.append((match.group("rule"), path))
    return found


def find_campaign_local_decision_tables(campaign_root: Path) -> list[Path]:
    tables_dir = campaign_root / "tables"
    if not tables_dir.exists():
        return []
    candidates: list[tuple[int, Path]] = []
    for path in sorted(tables_dir.glob("*.csv")):
        lowered = path.name.lower()
        if "decision" not in lowered or "rule" not in lowered:
            continue
        try:
            rows = read_csv_rows(path)
        except Exception:
            continue
        if not rows:
            continue
        header = set(rows[0].keys())
        if {"run_name", "decision_rule", "prediction_file"} <= header:
            candidates.append((len(rows), path))
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    return [path for _, path in candidates]


def decision_table_lookup(campaign_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for path in find_campaign_local_decision_tables(campaign_root):
        for row in read_csv_rows(path):
            run_name = row.get("run_name", "").strip()
            rule = row.get("decision_rule", "").strip()
            pred = resolve_campaign_path(row.get("prediction_file", ""), campaign_root)
            if not run_name or not rule or pred is None:
                continue
            if pred.name == "predictions_test.npz":
                continue
            lookup.setdefault((run_name, rule), row)
    return lookup


def decision_rule_sources(
    source_task: dict[str, Any],
    local_decision_lookup: dict[tuple[str, str], dict[str, str]],
    campaign_root: Path,
) -> list[tuple[str, Path]]:
    label_name = str(source_task["label"])
    task_name = str(source_task["task_id"])
    sources: dict[str, Path] = {}
    for decision_rule, prediction_path in local_decision_rule_predictions(source_task):
        sources[decision_rule] = prediction_path
    for (lookup_run_name, decision_rule), row in local_decision_lookup.items():
        if lookup_run_name not in {label_name, task_name}:
            continue
        prediction_path = resolve_campaign_path(row.get("prediction_file", ""), campaign_root)
        if prediction_path is not None:
            sources[decision_rule] = prediction_path
    return sorted(sources.items())


def append_task(tasks: list[dict[str, Any]], task: dict[str, Any]) -> bool:
    current = existing_task_ids(tasks)
    task_id = str(task["task_id"])
    if task_id in current:
        return False
    tasks.append(task)
    return True


def upsert_task(tasks: list[dict[str, Any]], task: dict[str, Any]) -> None:
    task_id = str(task["task_id"])
    for idx, existing in enumerate(tasks):
        if str(existing.get("task_id", "")) == task_id:
            tasks[idx] = task
            return
    tasks.append(task)


def prediction_field_for_file(prediction_path: Path) -> str | None:
    with np.load(prediction_path) as arr:
        if "test_pred" in arr.files:
            return "test_pred"
        if "posterior_mean" in arr.files:
            return "posterior_mean"
    return None


def hybrid_prediction_task(
    *,
    campaign_root: Path,
    source_task: dict[str, Any],
    prediction_path: Path,
    label_suffix: str,
    task_family: str,
    decision_rule: str,
    runtime_sec: float,
) -> dict[str, Any]:
    source_label = str(source_task["label"])
    source_task_id = str(source_task["task_id"])
    bundle_label = f"{source_label}_{label_suffix}"
    task_id = f"a30hyb__{slugify(source_label)}__{slugify(label_suffix)}"
    label = f"hybrid_refinement_surrogate_bundle_prediction_{bundle_label}_20260427"
    run_root = hybrid_run_root(campaign_root, label)
    prediction_field = prediction_field_for_file(prediction_path)
    if prediction_field is None:
        raise ValueError(f"Unsupported prediction file schema for {prediction_path}")
    command = python_command(
        HYBRID_WRAPPER,
        "--save-dir",
        str(run_root),
        "--bundle-label",
        bundle_label,
        "--init-mode",
        "prediction",
        "--prediction-file",
        str(prediction_path),
        "--prediction-field",
        prediction_field,
        "--prediction-row-index",
        "0",
        "--n-exemplars",
        "3",
        "--trace-length",
        "2000",
        "--maxiter",
        "120",
        "--seed",
        DIRECT_FITTING_SEED,
    )
    notes = (
        f"Hybrid refinement seeded from campaign-local prediction output `{prediction_path.name}` for "
        f"`{source_label}`, retained literally without score filtering."
    )
    return base_task(
        task_id=task_id,
        label=label,
        command=command,
        expected_runtime_sec=runtime_sec,
        depends_on=[source_task_id],
        output_paths=[
            run_root / "surrogate_bundle_integration_manifest.json",
            run_root / "hybrid_refinement_manifest.json",
        ],
        task_family=task_family,
        source_run_name=source_label,
        source_prediction_file=str(prediction_path),
        source_decision_rule=decision_rule,
        notes=notes,
    )


def midpoint_task(campaign_root: Path) -> dict[str, Any]:
    label = "hybrid_refinement_surrogate_bundle_midpoint_full_20260427"
    run_root = hybrid_run_root(campaign_root, label)
    command = python_command(
        HYBRID_WRAPPER,
        "--save-dir",
        str(run_root),
        "--bundle-label",
        "midpoint_full_a30",
        "--init-mode",
        "midpoint",
        "--n-exemplars",
        "3",
        "--trace-length",
        "2000",
        "--maxiter",
        "120",
        "--seed",
        DIRECT_FITTING_SEED,
    )
    return base_task(
        task_id="a30dfit__midpoint_full",
        label=label,
        command=command,
        expected_runtime_sec=MIDPOINT_RUNTIME_SEC,
        depends_on=[],
        output_paths=[
            run_root / "surrogate_bundle_integration_manifest.json",
            run_root / "hybrid_refinement_manifest.json",
        ],
        task_family=TASK_FAMILY_HYBRID_MIDPOINT,
        source_run_name="",
        source_prediction_file="",
        source_decision_rule="midpoint",
        notes="Midpoint-seeded baseline hybrid refinement row for the clean A30 surrogate direct-fitting package.",
    )


def wasserstein_task(campaign_root: Path, mode: str) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported Wasserstein mode {mode}")
    if mode == "smoke":
        runtime_sec = WABC_SMOKE_RUNTIME_SEC
        extra = ["--n-particles", "16", "--keep-top", "4", "--n-projections", "8", "--n-exemplars", "2", "--trace-length", "1000"]
    else:
        runtime_sec = WABC_FULL_RUNTIME_SEC
        extra = ["--n-particles", "256", "--keep-top", "24", "--n-projections", "64", "--n-exemplars", "3", "--trace-length", "2000"]
    label = f"wasserstein_abc_surrogate_bundle_{mode}_20260427"
    run_root = hybrid_run_root(campaign_root, label)
    command = python_command(
        WABC_WRAPPER,
        "--save-dir",
        str(run_root),
        "--bundle-label",
        f"{mode}_a30",
        *extra,
        "--seed",
        DIRECT_FITTING_SEED,
    )
    return base_task(
        task_id=f"a30dfit__wasserstein_abc_{mode}",
        label=label,
        command=command,
        expected_runtime_sec=runtime_sec,
        depends_on=[],
        output_paths=[
            run_root / "surrogate_bundle_integration_manifest.json",
            run_root / "wasserstein_abc_manifest.json",
        ],
        task_family=TASK_FAMILY_WABC,
        source_run_name="",
        source_prediction_file="",
        source_decision_rule="",
        notes=f"Wasserstein-ABC {mode} bundle row retained in the clean A30 surrogate direct-fitting package.",
    )


def summarize_families(tasks: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for task in tasks:
        family = str(task.get("task_family", "")).strip()
        if not family:
            continue
        out[family] = out.get(family, 0) + 1
    return dict(sorted(out.items()))


def main() -> None:
    args = parse_args()
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    manifest_path = Path(args.manifest_json).expanduser().resolve()
    manifest = load_json(manifest_path)

    manifest_root = Path(str(manifest.get("campaign_root", ""))).expanduser().resolve()
    if manifest_root != campaign_root:
        raise SystemExit(
            f"--campaign-root {campaign_root} does not match manifest campaign_root {manifest_root}"
        )

    tasks: list[dict[str, Any]] = list(manifest["tasks"])
    before_task_count = len(tasks)
    before_ids = existing_task_ids(tasks)

    source_tasks = runall_source_tasks(tasks, campaign_root)
    if not source_tasks:
        raise SystemExit(
            f"No {PHASE_RUNALL} source tasks with predictions_test outputs were found in {manifest_path}"
        )

    local_decision_lookup = decision_table_lookup(campaign_root)
    appended: list[dict[str, Any]] = []

    midpoint = midpoint_task(campaign_root)
    if append_task(tasks, midpoint):
        appended.append(midpoint)

    for mode in ("smoke", "full"):
        task = wasserstein_task(campaign_root, mode)
        if append_task(tasks, task):
            appended.append(task)

    for source_task in source_tasks:
        run_root = Path(source_task["_run_root"])
        raw_prediction = run_root / "predictions_test.npz"
        if raw_prediction.exists():
            raw_task = hybrid_prediction_task(
                campaign_root=campaign_root,
                source_task=source_task,
                prediction_path=raw_prediction,
                label_suffix="test",
                task_family=TASK_FAMILY_HYBRID_RAW,
                decision_rule="test",
                runtime_sec=HYBRID_RUNTIME_SEC,
            )
            if append_task(tasks, raw_task):
                appended.append(raw_task)

        for decision_rule, prediction_path in decision_rule_sources(source_task, local_decision_lookup, campaign_root):
            if not prediction_path.exists():
                continue
            rule_task = hybrid_prediction_task(
                campaign_root=campaign_root,
                source_task=source_task,
                prediction_path=prediction_path,
                label_suffix=decision_rule,
                task_family=TASK_FAMILY_HYBRID_RULE,
                decision_rule=decision_rule,
                runtime_sec=HYBRID_RUNTIME_SEC,
            )
            if append_task(tasks, rule_task):
                appended.append(rule_task)

    tag = campaign_root.name[len("optimization_track_") :] if campaign_root.name.startswith("optimization_track_") else campaign_root.name
    hybrid_task_ids = sorted(
        str(task["task_id"])
        for task in tasks
        if str(task.get("phase", "")) == PHASE_DIRECT_FITTING
    )
    hybrid_csv = campaign_root / "tables" / f"hh_track4_a30_hybrid_initializer_comparison_20260427_{tag}.csv"
    hybrid_md = campaign_root / "reports" / f"hh_track4_a30_hybrid_initializer_comparison_20260427_{tag}.md"
    upsert_task(
        tasks,
        {
            "task_id": "a30analysis__hybrid_refresh",
            "phase": "phaseANALYSIS_hybrid",
            "phase_label": "assumption_conditioned_surrogate_direct_fitting_analysis",
            "launch_group": "hybrid_refresh",
            "launch_mode": "appended_live_tail",
            "label": "a30_hybrid_refresh",
            "command": python_command(HYBRID_REFRESH, "--campaign-root", str(campaign_root)),
            "resource_class": "cpu_only",
            "cpus_per_task": 4,
            "gpus_per_task": 0,
            "expected_runtime_sec": 20.0,
            "depends_on": hybrid_task_ids,
            "output_paths": [str(hybrid_csv), str(hybrid_md)],
            "manifest_category": "surrogate_direct_fitting_literal_tail",
            "analysis_status": "included_literal_tail",
            "analysis_summary": "Refresh the A30-local hybrid initializer comparison after the appended direct-fitting hybrid tasks complete.",
            "analysis_evidence_paths": [str(hybrid_csv)],
            "family": "hybrid_refinement",
            "track_key": "surrogate_direct_fitting",
            "strategy_id": "a30_hybrid_refresh",
            "strategy_description": "A30-local hybrid initializer comparison refresh",
            "baseline_name": "",
            "seed": DIRECT_FITTING_SEED,
            "source_matrix": "",
            "source_registry": "",
            "source_row_id": "",
            "source_run_output_root": str(campaign_root / "runs" / PHASE_DIRECT_FITTING),
            "source_params_file": PARAMS_TRACK4,
            "source_notes": "auto-appended after hybrid task expansion",
            "task_family": "surrogate_direct_fitting_analysis",
            "source_run_name": "",
            "source_prediction_file": "",
            "source_decision_rule": "",
        },
    )

    manifest["tasks"] = tasks
    write_json(manifest_path, manifest)

    csv_path = manifest_path.with_suffix(".csv")
    if csv_path.exists():
        write_manifest_csv(csv_path, tasks)

    after_ids = existing_task_ids(tasks)
    summary_path = campaign_root / "notes" / "hh_track4_a30_hybrid_append_summary_20260427.json"
    summary = {
        "campaign_root": str(campaign_root),
        "manifest_json": str(manifest_path),
        "task_count_before": before_task_count,
        "task_count_after": len(tasks),
        "tasks_added": len(after_ids - before_ids),
        "appended_families": summarize_families(appended),
        "decision_rule_seed_count": sum(1 for task in appended if task.get("task_family") == TASK_FAMILY_HYBRID_RULE),
        "raw_prediction_seed_count": sum(1 for task in appended if task.get("task_family") == TASK_FAMILY_HYBRID_RAW),
        "decision_tables_seen": [str(path) for path in find_campaign_local_decision_tables(campaign_root)],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
