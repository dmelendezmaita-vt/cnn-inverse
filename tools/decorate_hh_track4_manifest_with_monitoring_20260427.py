#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON_BIN = "/projects/neuro-collab/conda/neuro-collab-env/bin/python"
WRAPPER = REPO / "tools/run_hh_track4_monitored_command_20260427.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Decorate a campaign manifest so every task emits per-experiment resource sidecars.")
    ap.add_argument("--manifest-json", required=True)
    return ap.parse_args()


def shell_join(parts: list[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def infer_resource_dir(task: dict[str, object], campaign_root: Path) -> Path:
    output_paths = task.get("output_paths", [])
    if isinstance(output_paths, list) and output_paths:
        raw = Path(str(output_paths[0]))
        task_id = str(task["task_id"])
        if raw.suffix:
            return raw.parent / "resource_monitor" / task_id
        return raw / "resource_monitor" / task_id
    return campaign_root / "resources" / str(task["task_id"])


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_json).resolve()
    manifest = json.loads(manifest_path.read_text())
    campaign_root = Path(str(manifest["campaign_root"])).resolve()
    changed = 0
    for task in manifest["tasks"]:
        if task.get("monitoring_wrapped"):
            continue
        resource_dir = infer_resource_dir(task, campaign_root)
        wrapped = (
            f"cd {shlex.quote(str(REPO))} && "
            f"{shell_join([PYTHON_BIN, WRAPPER, '--resource-dir', str(resource_dir), '--gpu-expected', str(task.get('gpus_per_task', 0)), '--command', str(task['command'])])}"
        )
        task["command"] = wrapped
        task["resource_dir"] = str(resource_dir)
        task["resource_summary_json"] = str(resource_dir / "resource_summary.json")
        task["resource_time_txt"] = str(resource_dir / "resource_time.txt")
        task["gpu_monitor_csv"] = str(resource_dir / "gpu_monitor.csv")
        task["task_env_json"] = str(resource_dir / "task_env.json")
        task["monitoring_wrapped"] = True
        changed += 1
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"wrapped_tasks": changed}, indent=2))


if __name__ == "__main__":
    main()
