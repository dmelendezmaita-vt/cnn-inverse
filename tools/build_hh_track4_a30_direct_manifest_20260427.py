#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
IMPORTANT = REPO / "data" / "important_notes"
CAMPAIGN_ROOT = IMPORTANT / "optimization_track_20260427_hh_track4_a30_literal_rerun"
TABLES = CAMPAIGN_ROOT / "tables"
PYTHON_BIN = "/projects/neuro-collab/conda/neuro-collab-env/bin/python"
LITERAL_BUILDER = REPO / "tools" / "build_hh_track4_a30_literal_rerun_manifest_20260427.py"
DIRECT_REFRESH = REPO / "tools" / "analyze_hh_track4_a30_campaign_refresh_20260427.py"
PYTHONPATH_VALUE = f"{REPO}:/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/vendor/dlkit:${{PYTHONPATH:-}}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build the clean A30 direct-task base manifest by reusing the broader literal A30 family set, "
            "then inject clean-campaign allocation defaults and direct-refresh analysis tasks."
        )
    )
    ap.add_argument("--alloc-job-id", required=True)
    ap.add_argument("--out-json", default=str(TABLES / "hh_track4_a30_literal_rerun_manifest_20260427.json"))
    return ap.parse_args()


def run_builder(out_root: Path) -> None:
    subprocess.run(
        [PYTHON_BIN, str(LITERAL_BUILDER), "--out-root", str(out_root)],
        check=True,
        text=True,
    )


def direct_refresh_task(campaign_root: Path, depends_on: list[str]) -> dict[str, object]:
    tables_dir = campaign_root / "tables"
    reports_dir = campaign_root / "reports"
    tag = campaign_root.name[len("optimization_track_") :] if campaign_root.name.startswith("optimization_track_") else campaign_root.name
    return {
        "task_id": "a30analysis__direct_campaign_refresh",
        "phase": "phaseANALYSIS_direct",
        "label": "direct_campaign_refresh",
        "command": (
            f'cd {REPO} && '
            f'env PYTHONPATH="{PYTHONPATH_VALUE}" {PYTHON_BIN} {DIRECT_REFRESH} --campaign-root {campaign_root}'
        ),
        "resource_class": "cpu_only",
        "cpus_per_task": 4,
        "gpus_per_task": 0,
        "expected_runtime_sec": 45.0,
        "depends_on": sorted(depends_on),
        "output_paths": [
            str(tables_dir / f"hh_track4_a30_run_comparison_{tag}.csv"),
            str(tables_dir / f"hh_track4_a30_strategy_comparison_{tag}.csv"),
            str(tables_dir / f"hh_track4_a30_noise_robustness_{tag}.csv"),
            str(tables_dir / f"hh_track4_a30_prediction_diagnostics_{tag}.csv"),
            str(tables_dir / f"hh_track4_a30_target_diagnostics_{tag}.csv"),
            str(tables_dir / f"hh_track4_a30_literature_branch_registry_{tag}.csv"),
            str(tables_dir / f"hh_track4_a30_reporting_key_results_{tag}.csv"),
            str(tables_dir / f"hh_track4_a30_reporting_limitations_{tag}.csv"),
            str(reports_dir / f"hh_track4_a30_literature_branch_registry_{tag}.md"),
            str(reports_dir / f"hh_track4_a30_reporting_synthesis_{tag}.md"),
        ],
        "notes": "Refresh direct-track A30-only comparison outputs after the direct family set finishes.",
    }


def main() -> None:
    args = parse_args()
    out_json = Path(args.out_json).expanduser().resolve()
    out_root = out_json.parent.parent if out_json.parent.name == "tables" else out_json.parent
    run_builder(out_root)

    built_manifest_path = out_root / "tables" / "hh_track4_a30_literal_rerun_manifest_20260427.json"
    manifest = json.loads(built_manifest_path.read_text())
    tasks: list[dict[str, object]] = list(manifest["tasks"])

    manifest["campaign_name"] = "hh_track4_a30_literal_rerun_20260427"
    manifest["campaign_root"] = str(out_root)
    alloc_defaults = dict(manifest.get("allocation_defaults", {}))
    alloc_defaults["alloc_job_id"] = args.alloc_job_id
    alloc_defaults["gpus_per_node"] = 4
    alloc_defaults["cpus_per_node"] = 64
    manifest["allocation_defaults"] = alloc_defaults

    direct_depends = sorted(str(task["task_id"]) for task in tasks)
    refresh = direct_refresh_task(out_root, direct_depends)
    existing = {str(task["task_id"]) for task in tasks}
    if refresh["task_id"] not in existing:
        tasks.append(refresh)

    manifest["tasks"] = tasks
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(out_json)


if __name__ == "__main__":
    main()
