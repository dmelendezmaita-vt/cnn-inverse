#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hh_track4_assumption_conditioned_surrogate_contract_20260425 import (  # noqa: E402
    DATE_TAG,
    RUNS_DIRECT_FITTING,
    write_json,
)


def sanitize_path_fragment(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "default"


def forwarded_flag_value(args: list[str], flag: str) -> str:
    for idx, value in enumerate(args):
        if value == flag and idx + 1 < len(args):
            return args[idx + 1]
        if value.startswith(f"{flag}="):
            return value.split("=", 1)[1]
    return ""


def resolve_bundle_save_dir(
    *,
    method_stem: str,
    cli_save_dir: str,
    forwarded_args: list[str],
    bundle_label: str,
) -> tuple[Path, bool]:
    explicit = cli_save_dir or forwarded_flag_value(forwarded_args, "--save-dir")
    if explicit:
        return Path(explicit), False
    label = sanitize_path_fragment(bundle_label)
    return RUNS_DIRECT_FITTING / f"{method_stem}_{label}_{DATE_TAG}", True


def delegated_args_with_save_dir(forwarded_args: list[str], save_dir: Path) -> list[str]:
    if forwarded_flag_value(forwarded_args, "--save-dir"):
        return list(forwarded_args)
    return ["--save-dir", str(save_dir), *forwarded_args]


def write_bundle_integration_sidecar(
    *,
    save_dir: Path,
    wrapper_name: str,
    delegated_script: Path,
    delegated_args: list[str],
    method_family: str,
    bundle_label: str,
    used_default_save_dir: bool,
    runtime_sec: float,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    child_manifest_files = sorted(path.name for path in save_dir.glob("*manifest.json"))
    child_csv_files = sorted(path.name for path in save_dir.glob("*.csv"))
    manifest = {
        "assumption_conditioned": True,
        "provenance_recovered": False,
        "bundle_integrated": True,
        "date_tag": DATE_TAG,
        "wrapper_script": wrapper_name,
        "delegated_script": str(delegated_script),
        "delegated_args": delegated_args,
        "method_family": method_family,
        "bundle_label": bundle_label,
        "save_dir": str(save_dir),
        "used_default_save_dir": bool(used_default_save_dir),
        "runtime_sec": float(runtime_sec),
        "child_manifest_files": child_manifest_files,
        "child_csv_files": child_csv_files,
    }
    if extra_metadata:
        manifest["extra_metadata"] = extra_metadata
    write_json(save_dir / "surrogate_bundle_integration_manifest.json", manifest)

    report_lines = [
        "# HH Track4 Surrogate Bundle Integration",
        "",
        f"Workspace-clock generated: `{DATE_TAG}`",
        "",
        f"- wrapper script: `{wrapper_name}`",
        f"- delegated script: `{delegated_script.name}`",
        f"- method family: `{method_family}`",
        f"- bundle label: `{bundle_label}`",
        f"- save dir: `{save_dir}`",
        f"- used default save dir: `{used_default_save_dir}`",
        f"- runtime sec: `{runtime_sec:.6f}`",
        f"- child manifest files: `{', '.join(child_manifest_files) if child_manifest_files else 'none'}`",
        f"- child csv files: `{', '.join(child_csv_files) if child_csv_files else 'none'}`",
        "",
        "This sidecar records that the run was launched through the surrogate bundle integration wrapper, while the delegated runner remains the source of the underlying method outputs.",
        "",
    ]
    (save_dir / "surrogate_bundle_integration_report.md").write_text("\n".join(report_lines))
    return manifest


def run_bundle_delegate(
    *,
    target_script: Path,
    wrapper_name: str,
    method_family: str,
    bundle_label: str,
    save_dir: Path,
    forwarded_args: list[str],
    used_default_save_dir: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    delegated_args = delegated_args_with_save_dir(forwarded_args, save_dir)
    command = [sys.executable, str(target_script), *delegated_args]
    t0 = time.time()
    subprocess.run(command, check=True)
    runtime_sec = time.time() - t0
    manifest = write_bundle_integration_sidecar(
        save_dir=save_dir,
        wrapper_name=wrapper_name,
        delegated_script=target_script,
        delegated_args=delegated_args,
        method_family=method_family,
        bundle_label=bundle_label,
        used_default_save_dir=used_default_save_dir,
        runtime_sec=runtime_sec,
        extra_metadata=extra_metadata,
    )
    print(json.dumps(manifest, indent=2))
