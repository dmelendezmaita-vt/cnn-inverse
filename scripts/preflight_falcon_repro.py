#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Preflight checks for the Falcon public reproduction contract.")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--data-prefix", required=False, default="")
    return ap.parse_args()


def run_command(command: List[str]) -> Dict[str, object]:
    proc = subprocess.run(command, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def run_alloc_command(allocation_job_id: str, command: List[str], *, nodes: int = 1, tasks: int = 1) -> Dict[str, object]:
    srun_command = [
        "srun",
        "--overlap",
        "--jobid",
        str(allocation_job_id),
        "--nodes",
        str(nodes),
        "--ntasks",
        str(tasks),
        "--ntasks-per-node",
        "1",
        "--kill-on-bad-exit=1",
    ] + command
    return run_command(srun_command)


def require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    errors: List[str] = []
    summary: Dict[str, object] = {
        "repo_root": str(REPO_ROOT),
        "save_dir": str(save_dir),
        "data_dir": str(Path(args.data_dir).resolve()),
        "data_prefix": args.data_prefix,
        "allocation_job_id": os.environ.get("ALLOC_JOB_ID"),
        "python": sys.executable,
        "checks": {},
    }

    alloc_job_id = os.environ.get("ALLOC_JOB_ID")
    require(bool(alloc_job_id), "ALLOC_JOB_ID is not set", errors)
    require(shutil.which("scontrol") is not None, "scontrol is not available", errors)
    require(shutil.which("srun") is not None, "srun is not available", errors)

    hh_tar_path = Path(args.data_dir).resolve()
    require(hh_tar_path.is_file(), f"HH tarball does not exist: {hh_tar_path}", errors)
    if hh_tar_path.is_file():
        summary["checks"]["hh_tar_size_bytes"] = hh_tar_path.stat().st_size
        require(tarfile.is_tarfile(hh_tar_path), f"HH tarball is not recognized as a tar archive: {hh_tar_path}", errors)

    job_info = ""
    if alloc_job_id and shutil.which("scontrol") is not None:
        proc = subprocess.run(
            ["scontrol", "show", "job", str(alloc_job_id)],
            text=True,
            capture_output=True,
            check=False,
        )
        summary["checks"]["scontrol_show_job"] = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode == 0:
            job_info = proc.stdout
        else:
            errors.append(f"scontrol show job {alloc_job_id} failed with return code {proc.returncode}")

    require("Partition=a30_normal_q" in job_info, "allocation is not on the expected a30_normal_q partition", errors)
    require("NumNodes=4" in job_info, "allocation does not report NumNodes=4", errors)
    require("gres/gpu:a30=16" in job_info or "gres/gpu:a30:4" in job_info, "allocation does not report the expected A30 GPU resources", errors)

    nvidia = run_alloc_command(str(alloc_job_id), ["nvidia-smi", "-L"])
    summary["checks"]["nvidia_smi_l"] = nvidia
    require(nvidia["returncode"] == 0, "nvidia-smi -L failed", errors)

    core_import = run_alloc_command(
        str(alloc_job_id),
        [
            sys.executable,
            "-c",
            "import torch, sbi, swyft, pytorch_lightning; "
            "print({'torch_cuda': torch.cuda.is_available(), 'cuda_devices': torch.cuda.device_count()})",
        ],
    )
    summary["checks"]["core_import"] = core_import
    require(core_import["returncode"] == 0, "core environment imports failed", errors)

    bayesflow_python = REPO_ROOT / ".venv-bayesflow" / "bin" / "python"
    require(bayesflow_python.is_file(), f"BayesFlow interpreter does not exist: {bayesflow_python}", errors)
    if bayesflow_python.is_file():
        bayesflow_import = run_alloc_command(
            str(alloc_job_id),
            [
                str(bayesflow_python),
                "-c",
                "import bayesflow, keras, numpy; "
                "print({'bayesflow': bayesflow.__version__, 'keras': keras.__version__, 'numpy': numpy.__version__})",
            ],
        )
        summary["checks"]["bayesflow_import"] = bayesflow_import
        require(bayesflow_import["returncode"] == 0, "BayesFlow environment imports failed", errors)

    if alloc_job_id and shutil.which("srun") is not None:
        launch_checks = {
            "step_1node": (1, 1, ["hostname"]),
            "step_2node": (2, 2, ["hostname"]),
            "step_4node": (4, 4, ["hostname"]),
        }
        for key, (nodes, tasks, command) in launch_checks.items():
            result = run_alloc_command(str(alloc_job_id), command, nodes=nodes, tasks=tasks)
            summary["checks"][key] = result
            require(result["returncode"] == 0, f"{key} launch check failed", errors)

    summary["errors"] = errors
    (save_dir / "preflight_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if errors:
        raise SystemExit("Falcon preflight failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
