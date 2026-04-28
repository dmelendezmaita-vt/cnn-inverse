#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON_BIN = "/projects/neuro-collab/conda/neuro-collab-env/bin/python"
PYTHONPATH_VALUE = f"{REPO}:/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/vendor/dlkit:${{PYTHONPATH:-}}"
CAMPAIGN_ROOT = REPO / "data" / "important_notes" / "optimization_track_20260427_hh_track4_a30_literal_rerun"
RUNS = CAMPAIGN_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Append active-design budget-ladder tasks into the clean A30 manifest.")
    ap.add_argument("--manifest-json", default=str(CAMPAIGN_ROOT / "tables" / "hh_track4_a30_literal_rerun_manifest_20260427.json"))
    return ap.parse_args()


def shell_join(parts: list[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def python_command(script_rel: str, *args: object) -> str:
    cmd = [PYTHON_BIN, str(REPO / script_rel), *map(str, args)]
    return f'cd {shlex.quote(str(REPO))} && env PYTHONPATH="{PYTHONPATH_VALUE}" {shell_join(cmd)}'


def save_dir(label: str) -> Path:
    return RUNS / "phaseSURR_active" / label


def add_task(tasks: list[dict[str, object]], task: dict[str, object]) -> None:
    known = {str(item["task_id"]) for item in tasks}
    if str(task["task_id"]) in known:
        return
    tasks.append(task)


def base_task(
    *,
    task_id: str,
    label: str,
    command: str,
    expected_runtime_sec: float,
    output_paths: list[str],
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "phase": "phaseSURR_active",
        "label": label,
        "command": command,
        "resource_class": "cpu_only",
        "cpus_per_task": 8,
        "gpus_per_task": 0,
        "expected_runtime_sec": expected_runtime_sec,
        "depends_on": [],
        "output_paths": output_paths,
        "notes": "active-design budget-ladder follow-up appended after clean A30 base campaign completion",
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_json).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    tasks: list[dict[str, object]] = list(manifest["tasks"])

    active_rows = [
        (
            "a30surr__active_sequential_uncertainty_smoke_20260428",
            "active_sequential_uncertainty_smoke_20260428",
            [
                "--save-dir", str(save_dir("active_sequential_uncertainty_smoke_20260428")),
                "--acquisition-policy", "uncertainty",
                "--n-episodes", "1",
                "--n-particles", "64",
                "--n-rounds", "2",
                "--seed", "20260428",
            ],
            10.0,
        ),
        (
            "a30surr__active_sequential_disagreement_extended_20260428",
            "active_sequential_disagreement_extended_20260428",
            [
                "--save-dir", str(save_dir("active_sequential_disagreement_extended_20260428")),
                "--acquisition-policy", "disagreement",
                "--n-episodes", "8",
                "--n-particles", "256",
                "--n-rounds", "6",
                "--seed", "20260428",
            ],
            180.0,
        ),
        (
            "a30surr__active_sequential_uncertainty_extended_20260428",
            "active_sequential_uncertainty_extended_20260428",
            [
                "--save-dir", str(save_dir("active_sequential_uncertainty_extended_20260428")),
                "--acquisition-policy", "uncertainty",
                "--n-episodes", "8",
                "--n-particles", "256",
                "--n-rounds", "6",
                "--seed", "20260428",
            ],
            180.0,
        ),
        (
            "a30surr__active_sequential_wasserstein_extended_20260428",
            "active_sequential_wasserstein_extended_20260428",
            [
                "--save-dir", str(save_dir("active_sequential_wasserstein_extended_20260428")),
                "--acquisition-policy", "wasserstein",
                "--n-episodes", "8",
                "--n-particles", "256",
                "--n-rounds", "6",
                "--seed", "20260428",
            ],
            240.0,
        ),
    ]
    for task_id, label, extra, runtime in active_rows:
        run_root = Path(extra[1])
        add_task(
            tasks,
            base_task(
                task_id=task_id,
                label=label,
                command=python_command("tools/run_hh_track4_assumption_conditioned_active_sequential_design_20260425.py", *extra),
                expected_runtime_sec=runtime,
                output_paths=[str(run_root / "active_sequential_manifest.json")],
            ),
        )

    asnpe_root = save_dir("asnpe_extended_v100_20260428")
    add_task(
        tasks,
        base_task(
            task_id="a30surr__asnpe_extended_v100_20260428",
            label="asnpe_extended_v100_20260428",
            command=python_command(
                "tools/run_hh_track4_assumption_conditioned_asnpe_20260426.py",
                "--save-dir", str(asnpe_root),
                "--bundle-label", "extended_a30_20260428",
                "--n-rounds", "6",
                "--initial-sims", "192",
                "--round-sims", "48",
                "--candidate-pool", "128",
                "--epochs", "28",
                "--batch-size", "32",
                "--posterior-samples", "1536",
                "--seed", "20260428",
            ),
            expected_runtime_sec=900.0,
            output_paths=[str(asnpe_root / "asnpe_manifest.json")],
        ),
    )

    for task in tasks:
        if str(task.get("task_id", "")) == "a30analysis__literature_surrogate_refresh":
            deps = list(task.get("depends_on", []))
            for tid in [
                "a30surr__active_sequential_uncertainty_smoke_20260428",
                "a30surr__active_sequential_disagreement_extended_20260428",
                "a30surr__active_sequential_uncertainty_extended_20260428",
                "a30surr__active_sequential_wasserstein_extended_20260428",
                "a30surr__asnpe_extended_v100_20260428",
            ]:
                if tid not in deps:
                    deps.append(tid)
            task["depends_on"] = deps

    manifest["tasks"] = tasks
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"task_count": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()
