from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import reset_hh_track4_a30_invalid_tasks_20260427 as reset_tool


def write_registry(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class ResetInvalidTasksTest(unittest.TestCase):
    def test_apply_resets_only_failed_blocked_and_pre_boundary_completed_gpu_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            tables = root / "tables"
            tables.mkdir(parents=True)
            manifest_path = tables / "manifest.json"
            registry_path = tables / "registry.csv"
            report_json = tables / "report.json"
            report_csv = tables / "report.csv"

            def task_paths(task_id: str) -> tuple[Path, Path, Path, Path]:
                run_dir = root / "runs" / task_id
                resource_dir = run_dir / "resource_monitor" / task_id
                stdout_log = root / "logs" / f"{task_id}.out"
                stderr_log = root / "logs" / f"{task_id}.err"
                return run_dir, resource_dir, stdout_log, stderr_log

            task_specs = [
                ("failed_cpu", "FAILED", 0, "2026-04-27T19:40:00-04:00"),
                ("blocked_cpu", "BLOCKED", 0, ""),
                ("old_gpu", "COMPLETED", 1, "2026-04-27T19:20:00-04:00"),
                ("new_gpu", "COMPLETED", 1, "2026-04-27T19:40:00-04:00"),
                ("done_cpu", "COMPLETED", 0, "2026-04-27T19:20:00-04:00"),
            ]

            manifest_tasks = []
            registry_rows = []
            for task_id, status, gpus, started_at in task_specs:
                run_dir, resource_dir, stdout_log, stderr_log = task_paths(task_id)
                resource_dir.mkdir(parents=True, exist_ok=True)
                run_dir.mkdir(parents=True, exist_ok=True)
                stdout_log.parent.mkdir(parents=True, exist_ok=True)
                (run_dir / "artifact.txt").write_text(task_id)
                (resource_dir / "resource_summary.json").write_text("{}\n")
                stdout_log.write_text("stdout\n")
                stderr_log.write_text("stderr\n")
                manifest_tasks.append(
                    {
                        "task_id": task_id,
                        "phase": "phaseX",
                        "label": task_id,
                        "command": "echo noop",
                        "resource_class": "gpu1" if gpus else "cpu_only",
                        "cpus_per_task": 4,
                        "gpus_per_task": gpus,
                        "expected_runtime_sec": 1.0,
                        "depends_on": [],
                        "output_paths": [str(run_dir), str(run_dir / "artifact.txt")],
                        "monitoring_wrapped": True,
                        "resource_dir": str(resource_dir),
                    }
                )
                registry_rows.append(
                    {
                        "task_id": task_id,
                        "phase": "phaseX",
                        "label": task_id,
                        "status": status,
                        "resource_class": "gpu1" if gpus else "cpu_only",
                        "cpus_per_task": "4",
                        "gpus_per_task": str(gpus),
                        "expected_runtime_sec": "1.0",
                        "depends_on": "[]",
                        "alloc_job_id": "368759" if status != "BLOCKED" else "",
                        "assigned_node": "fal003" if status != "BLOCKED" else "",
                        "started_at": started_at,
                        "ended_at": "2026-04-27T19:41:00-04:00" if started_at else "",
                        "elapsed_sec": "60" if started_at else "",
                        "return_code": "0" if status == "COMPLETED" else "1",
                        "command": "echo noop",
                        "stdout_log": str(stdout_log),
                        "stderr_log": str(stderr_log),
                        "notes": "note",
                    }
                )

            manifest_path.write_text(
                json.dumps(
                    {
                        "campaign_name": "test",
                        "campaign_root": str(root),
                        "allocation_defaults": {},
                        "tasks": manifest_tasks,
                    },
                    indent=2,
                )
                + "\n"
            )
            write_registry(registry_path, registry_rows)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "reset",
                    "--manifest-json",
                    str(manifest_path),
                    "--registry-csv",
                    str(registry_path),
                    "--campaign-root",
                    str(root),
                    "--gpu-watcher-fix-iso",
                    "2026-04-27T19:26:34-04:00",
                    "--report-json",
                    str(report_json),
                    "--report-csv",
                    str(report_csv),
                    "--apply",
                ]
                rc = reset_tool.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)

            with registry_path.open(newline="") as f:
                rows = {row["task_id"]: row for row in csv.DictReader(f)}

            for task_id in ("failed_cpu", "blocked_cpu", "old_gpu"):
                self.assertEqual(rows[task_id]["status"], "PENDING")
                self.assertEqual(rows[task_id]["started_at"], "")
                self.assertEqual(rows[task_id]["stdout_log"], "")
                run_dir, resource_dir, stdout_log, stderr_log = task_paths(task_id)
                self.assertFalse(run_dir.exists())
                self.assertFalse(resource_dir.exists())
                self.assertFalse(stdout_log.exists())
                self.assertFalse(stderr_log.exists())

            for task_id in ("new_gpu", "done_cpu"):
                self.assertEqual(rows[task_id]["status"], "COMPLETED")
                run_dir, resource_dir, stdout_log, stderr_log = task_paths(task_id)
                self.assertTrue(run_dir.exists())
                self.assertTrue(resource_dir.exists())
                self.assertTrue(stdout_log.exists())
                self.assertTrue(stderr_log.exists())

            report = json.loads(report_json.read_text())
            self.assertEqual(report["summary"]["reset_task_count"], 3)


if __name__ == "__main__":
    unittest.main()
