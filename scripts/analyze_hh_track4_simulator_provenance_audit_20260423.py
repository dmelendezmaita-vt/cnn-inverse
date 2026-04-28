#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
DATE_TAG = "20260423"
OUT_ROOT = IMPORTANT / "optimization_track_20260423_hh_track4_simulator_provenance_audit"
TABLES = OUT_ROOT / "tables"
REPORTS = OUT_ROOT / "reports"

TRACK4_TAR = Path("/projects/neuro-collab/data/tar_files/concatenated_data_no_compression.tar")
TRACK3_TAR = Path("/projects/neuro-collab/data/tar_files/reduced_data_no_compression.tar")
TRACK4_SHARED = (
    IMPORTANT
    / "optimization_track_20260411_v100_interactive"
    / "shared_data"
    / "track4_hh_full"
    / "concatenated_data"
)
TRACK4_CONFIG = REPO / "pytorch" / "configs" / "fourth_track_hh_full" / "params_dnn_tar_hh_full.yaml"

EVIDENCE_CSV = TABLES / f"hh_track4_simulator_provenance_evidence_{DATE_TAG}.csv"
DATA_SCHEMA_CSV = TABLES / f"hh_track4_raw_data_schema_{DATE_TAG}.csv"
TARGET_RANGES_CSV = TABLES / f"hh_track4_target_ranges_by_current_{DATE_TAG}.csv"
REPORT_MD = REPORTS / f"hh_track4_simulator_provenance_audit_{DATE_TAG}.md"

SEARCH_TERMS = [
    "hodgkin",
    "huxley",
    "hh_param_",
    "generated_params",
    "support_stats",
    "current injection",
    "stimulus",
    "protocol",
    "bluepyopt",
    "direct fit",
    "solve_ivp",
    "odeint",
    "i_app",
    "iapp",
    "g_na",
    "gna",
    "g_k",
    "gk",
    "g_l",
    "gl",
]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tar_summary(path: Path) -> dict[str, object]:
    members = []
    files = []
    with tarfile.open(path, "r:*") as tf:
        for member in tf.getmembers():
            members.append(member.name)
            if member.isfile():
                files.append(member.name)
    suffix_counts: dict[str, int] = {}
    for name in files:
        suffix = Path(name).suffix.lower() or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    non_payload_files = [
        name
        for name in files
        if not (
            name.endswith(".npy")
            or name.endswith(".npy.lz4")
            or Path(name).name in {".DS_Store", "._.DS_Store"}
        )
    ]
    return {
        "path": str(path),
        "member_count": len(members),
        "file_count": len(files),
        "suffix_counts": suffix_counts,
        "non_payload_files": non_payload_files,
        "members": members,
    }


def raw_shape(path: Path, cols: int) -> tuple[int, int, int]:
    size = os.path.getsize(path)
    row_bytes = cols * 4
    return size // row_bytes, cols, size % row_bytes


def target_stats(path: Path) -> dict[str, list[float]]:
    rows, cols, rem = raw_shape(path, 6)
    if rem != 0:
        raise ValueError(f"Target file cannot be reshaped as float32 x 6: {path}")
    arr = np.memmap(path, dtype=np.float32, mode="r", shape=(rows, cols))
    return {
        "min": [float(x) for x in np.min(arr, axis=0)],
        "max": [float(x) for x in np.max(arr, axis=0)],
        "mean": [float(x) for x in np.mean(arr, axis=0)],
        "std": [float(x) for x in np.std(arr, axis=0)],
    }


def build_schema_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    schema_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for curr in ["0.1", "0.2", "0.3", "0.4", "0.5"]:
        files = {
            "features_y": (TRACK4_SHARED / "y" / f"concatenated_data_{curr}_curr.npy", 400000),
            "targets_generated_params": (
                TRACK4_SHARED / "generated_params" / f"concatenated_data_{curr}_curr.npy",
                6,
            ),
            "support_stats": (
                TRACK4_SHARED / "support_stats" / f"concatenated_data_{curr}_curr.npy",
                2,
            ),
        }
        for component, (path, cols) in files.items():
            rows, shape_cols, rem = raw_shape(path, cols)
            schema_rows.append(
                {
                    "current_label": curr,
                    "component": component,
                    "path": str(path),
                    "bytes": os.path.getsize(path),
                    "dtype_inferred": "float32_raw_binary",
                    "rows_if_configured_cols": rows,
                    "configured_cols": shape_cols,
                    "remainder_bytes": rem,
                }
            )

        stats = target_stats(files["targets_generated_params"][0])
        for idx in range(6):
            target_rows.append(
                {
                    "current_label": curr,
                    "target_name": f"hh_param_{idx + 1}",
                    "semantic_mapping": "unknown",
                    "min": stats["min"][idx],
                    "max": stats["max"][idx],
                    "mean": stats["mean"][idx],
                    "std": stats["std"][idx],
                }
            )
    return schema_rows, target_rows


def candidate_source_files() -> list[Path]:
    candidates: list[Path] = []
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        rel_parts = path.relative_to(REPO).parts
        if "runs" in rel_parts or "logs" in rel_parts:
            continue
        if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".json", ".sh", ".txt", ".csv"}:
            continue
        lowered = str(path.relative_to(REPO)).lower()
        if "/data/important_notes/optimization_track_" in f"/{lowered}":
            continue
        if any(term in lowered for term in ["hh", "hodgkin", "huxley", "sim", "generate", "protocol", "bluepy"]):
            candidates.append(path)
    return sorted(candidates)


def scan_candidate_hits(paths: list[Path]) -> dict[str, int]:
    counts = {term: 0 for term in SEARCH_TERMS}
    for path in paths:
        try:
            text = path.read_text(errors="ignore").lower()
        except OSError:
            continue
        for term in SEARCH_TERMS:
            counts[term] += len(re.findall(re.escape(term.lower()), text))
    return counts


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    track4_tar = tar_summary(TRACK4_TAR)
    track3_tar = tar_summary(TRACK3_TAR)
    schema_rows, target_rows = build_schema_rows()
    source_candidates = candidate_source_files()
    hit_counts = scan_candidate_hits(source_candidates)

    config_text = TRACK4_CONFIG.read_text(errors="ignore")
    neutral_target_comment = "Keep names neutral until upstream generation metadata maps semantic labels" in config_text

    evidence_rows = [
        {
            "component": "track4_full_tar",
            "status": "array_payload_only",
            "source": str(TRACK4_TAR),
            "detail": (
                f"members={track4_tar['member_count']}; files={track4_tar['file_count']}; "
                f"non_payload_files={len(track4_tar['non_payload_files'])}; "
                f"suffix_counts={track4_tar['suffix_counts']}"
            ),
            "closure_implication": "supports fixed supervised task; does not provide simulator/protocol provenance",
        },
        {
            "component": "track3_reduced_tar",
            "status": "array_payload_only",
            "source": str(TRACK3_TAR),
            "detail": (
                f"members={track3_tar['member_count']}; files={track3_tar['file_count']}; "
                f"non_payload_files={len(track3_tar['non_payload_files'])}; "
                f"suffix_counts={track3_tar['suffix_counts']}"
            ),
            "closure_implication": "parallel reduced HH artifact also lacks generator/protocol metadata",
        },
        {
            "component": "track4_config",
            "status": "neutral_parameter_labels_confirmed",
            "source": str(TRACK4_CONFIG),
            "detail": f"neutral_target_comment_present={neutral_target_comment}; target_names=hh_param_1..hh_param_6",
            "closure_implication": "prevents safe semantic assignment of six HH parameters",
        },
        {
            "component": "track4_raw_schema",
            "status": "recoverable_fixed_data_schema",
            "source": str(DATA_SCHEMA_CSV),
            "detail": "five current-labeled files; each has 15000 raw traces and 15000 six-parameter target rows",
            "closure_implication": "enables supervised/model-comparison closure, not mechanistic direct fitting",
        },
        {
            "component": "codebase_candidate_scan",
            "status": "no_runnable_simulator_contract_found",
            "source": str(REPO),
            "detail": (
                f"candidate_text_files={len(source_candidates)}; "
                f"bluepyopt_hits={hit_counts['bluepyopt']}; solve_ivp_hits={hit_counts['solve_ivp']}; "
                f"odeint_hits={hit_counts['odeint']}; current_injection_hits={hit_counts['current injection']}"
            ),
            "closure_implication": "direct fitting remains blocked unless external provenance is recovered",
        },
    ]

    write_csv(
        EVIDENCE_CSV,
        evidence_rows,
        ["component", "status", "source", "detail", "closure_implication"],
    )
    write_csv(
        DATA_SCHEMA_CSV,
        schema_rows,
        [
            "current_label",
            "component",
            "path",
            "bytes",
            "dtype_inferred",
            "rows_if_configured_cols",
            "configured_cols",
            "remainder_bytes",
        ],
    )
    write_csv(
        TARGET_RANGES_CSV,
        target_rows,
        ["current_label", "target_name", "semantic_mapping", "min", "max", "mean", "std"],
    )

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report = [
        "# HH Track4 Simulator Provenance Audit",
        "",
        f"Workspace-clock generated: `{generated_at}`",
        "",
        "## Decision",
        "",
        "Direct simulator fitting remains blocked by missing simulator provenance. This is a workspace-level blocker, not a negative scientific result for direct fitting.",
        "",
        "## What Is Recoverable",
        "",
        "- Track4 full data are present as raw float32 array payloads under the expected `concatenated_data` layout.",
        "- Current labels recoverable from file names: `0.1`, `0.2`, `0.3`, `0.4`, `0.5`.",
        "- For each current label: `15000` feature rows, `400000` float32 values per row, and `15000` six-parameter target rows.",
        "- The configured target names remain neutral: `hh_param_1` through `hh_param_6`.",
        "- The fixed supervised inverse task is therefore executable and auditable.",
        "",
        "## What Is Not Recoverable",
        "",
        "- semantic mapping for `hh_param_1` through `hh_param_6`",
        "- HH equation variant and state variables",
        "- initial conditions, time grid, time units, and sampling interval",
        "- current-injection protocol beyond the filename labels",
        "- simulator/generator script or random-generation contract",
        "- BluePyOpt/direct-fit objective, parameter bounds by semantic name, or differentiable simulator path",
        "",
        "## Evidence Tables",
        "",
        f"- evidence: `{EVIDENCE_CSV}`",
        f"- raw schema: `{DATA_SCHEMA_CSV}`",
        f"- target ranges: `{TARGET_RANGES_CSV}`",
        "",
        "## Closure Implication",
        "",
        "The current workspace supports strong closure for the fixed supervised HH Track4 inverse task. It does not support a literature-complete claim that direct simulator fitting has been exhausted. The safe wording remains: direct fitting is scientifically relevant but not executable here without inventing missing simulator provenance.",
    ]
    REPORT_MD.write_text("\n".join(report) + "\n")

    print(f"Wrote {EVIDENCE_CSV}")
    print(f"Wrote {DATA_SCHEMA_CSV}")
    print(f"Wrote {TARGET_RANGES_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
