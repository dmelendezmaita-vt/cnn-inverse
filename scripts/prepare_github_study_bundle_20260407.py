#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
NOTES = REPO / "data/important_notes"
CFG = REPO / "pytorch/configs"
SCRIPTS = REPO / "scripts"
AP = NOTES / "advisor_packet_20260402"
SITE_SRC = AP / "site"

DATE_TAG = "20260407"
BUNDLE_ROOT = NOTES / f"github_upload_bundle_{DATE_TAG}"

# Strict upload allowlist (tight controls):
# - No markdown or PDF unless explicitly requested in a future revision.
# - Only extensions we actively use in this bundle generation.
GRAPH_EXT = {".png"}
DATA_EXT = {".csv", ".json", ".jsonl", ".npz", ".npy"}
CODE_EXT = {".py", ".sh", ".sbatch", ".yaml", ".yml", ".html", ".css", ".js"}

EXCLUDE_EXT = {".log", ".out", ".pt", ".pyc", ".mode", ".ready", ".gz", ".zip", ".sha256"}
# Exclude from generic sweeps; static site is copied via a dedicated step below.
EXCLUDE_DIR_PARTS = {"__pycache__", ".git", "run_dnn", "stage_status", "site"}

# Ignore testing runs and testing-job trackers by request.
EXCLUDE_NAME_SUBSTR = {"testing"}


def categorize(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    name = path.name.lower()

    if any(tok in name for tok in EXCLUDE_NAME_SUBSTR):
        return None
    if ext in EXCLUDE_EXT:
        return None
    if ext in GRAPH_EXT:
        return "graphs"
    if ext in DATA_EXT:
        return "data"
    if ext in CODE_EXT:
        return "code"
    return None


def blocked_path(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return any(x in parts for x in EXCLUDE_DIR_PARTS)


@dataclass
class PartSpec:
    name: str
    source_dirs: List[Path] = field(default_factory=list)
    extra_files: List[Path] = field(default_factory=list)
    extra_dirs: List[Path] = field(default_factory=list)


PARTS: List[PartSpec] = [
    PartSpec(
        name="01_first_track_paper_parity",
        source_dirs=[NOTES / "first_track_paper_parity"],
        extra_dirs=[CFG / "paper_equivalence_v100"],
        extra_files=[
            SCRIPTS / "submit_falcon_seeded_training_matrix_20260306.sh",
            AP / "figures" / "first_track_reproducibility_boxplots.png",
            AP / "tables" / "first_track_reproducibility_stats.csv",
            AP / "web_exports" / "first_track_meta_stats.json",
            AP / "notes" / "advisor_report_20260402.md",
        ],
    ),
    PartSpec(
        name="02_second_track_scalability",
        source_dirs=[NOTES / "second_track_scalability"],
        extra_files=[
            SCRIPTS / "submit_l40s_scaling_smoke.sh",
            SCRIPTS / "build_phase23_unified_submission_matrix_20260329.py",
            SCRIPTS / "submit_phase23_unified_20260329.py",
            AP / "figures" / "track2_strong_scaling.png",
            AP / "figures" / "track2_weak_scaling_proxy.png",
            AP / "figures" / "rerun_true_weak_scaling_20260402.png",
            AP / "tables" / "track2_strong_scaling_summary.csv",
            AP / "tables" / "track2_weak_scaling_assessment.csv",
            AP / "tables" / "track2_weak_scaling_proxy.csv",
            AP / "tables" / "rerun_true_weak_scaling_summary_20260402.csv",
            AP / "web_exports" / "track2_strong_scaling_summary.json",
            AP / "web_exports" / "track2_weak_scaling_proxy.json",
            AP / "web_exports" / "rerun_true_weak_scaling_summary_effective.json",
        ],
    ),
    PartSpec(
        name="03_third_track_hh",
        source_dirs=[NOTES / "third_track_hh"],
        extra_files=[
            SCRIPTS / "submit_hh_matrix_20260307.sh",
            SCRIPTS / "submit_hh_unseeded_and_falcon_20260307.sh",
            AP / "figures" / "track3_hh_reduced_scalability_accuracy.png",
            AP / "figures" / "track3_hh_reduced_per_dimension_scatter_fit.png",
            AP / "figures" / "hh_loss_curves_with_spike_markers.png",
            AP / "figures" / "loss_spike_incidence.png",
            AP / "figures" / "training_time_and_gpu_utilization.png",
            AP / "tables" / "track3_track4_scalability_training_only.csv",
            AP / "tables" / "track3_track4_per_dimension_fit_summary.csv",
            AP / "tables" / "loss_spike_summary_training_only.csv",
            AP / "tables" / "loss_spike_run_details_hh_training_only.csv",
            AP / "tables" / "timing_decomposition_hh_training_only.csv",
            AP / "tables" / "timing_decomposition_hh_run_level_training_only.csv",
            AP / "tables" / "gpu_utilization_time_profile_hh_training_only.csv",
            AP / "web_exports" / "per_dimension_fit_summary_training_only.json",
            AP / "web_exports" / "loss_spike_run_details_hh_training_only.json",
            AP / "web_exports" / "timing_decomposition_hh_run_level_training_only.json",
            AP / "web_exports" / "gpu_utilization_time_profile_hh_training_only.json",
        ],
        extra_dirs=[CFG / "reruns_20260329_phase23"],
    ),
    PartSpec(
        name="04_fourth_track_hh_full",
        source_dirs=[NOTES / "fourth_track_hh_full"],
        extra_files=[
            SCRIPTS / "submit_hh_full_unseeded_and_falcon_20260308.sh",
            SCRIPTS / "submit_hh_full_v2_unseeded_and_falcon_20260317.sh",
            AP / "figures" / "track4_hh_full_scalability_accuracy.png",
            AP / "figures" / "track4_hh_full_per_dimension_scatter_fit.png",
            AP / "figures" / "hh_loss_curves_with_spike_markers.png",
            AP / "figures" / "loss_spike_incidence.png",
            AP / "figures" / "training_time_and_gpu_utilization.png",
            AP / "tables" / "track3_track4_scalability_training_only.csv",
            AP / "tables" / "track3_track4_per_dimension_fit_summary.csv",
            AP / "tables" / "loss_spike_summary_training_only.csv",
            AP / "tables" / "loss_spike_run_details_hh_training_only.csv",
            AP / "tables" / "timing_decomposition_hh_training_only.csv",
            AP / "tables" / "timing_decomposition_hh_run_level_training_only.csv",
            AP / "tables" / "gpu_utilization_time_profile_hh_training_only.csv",
            AP / "web_exports" / "per_dimension_fit_summary_training_only.json",
            AP / "web_exports" / "loss_spike_run_details_hh_training_only.json",
            AP / "web_exports" / "timing_decomposition_hh_run_level_training_only.json",
            AP / "web_exports" / "gpu_utilization_time_profile_hh_training_only.json",
        ],
        extra_dirs=[CFG / "reruns_20260402_phase24"],
    ),
    PartSpec(
        name="05_fifth_track_hh_full_refined",
        source_dirs=[NOTES / "fifth_track_hh_full_refined"],
        extra_dirs=[CFG / "reruns_20260402_opttrack"],
    ),
    PartSpec(
        name="06_optimization_track_20260402",
        source_dirs=[NOTES / "optimization_track_20260402"],
        extra_files=[
            SCRIPTS / "build_optimization_submission_matrix_20260402.py",
            SCRIPTS / "submit_optimization_matrix_20260402.py",
        ],
    ),
    PartSpec(
        name="07_optimization_track_20260405",
        source_dirs=[NOTES / "optimization_track_20260405"],
        extra_files=[
            SCRIPTS / "build_optimization_submission_matrix_20260405.py",
            SCRIPTS / "submit_optimization_matrix_20260405.py",
            SCRIPTS / "monitor_optimization_matrix_20260405.py",
        ],
        extra_dirs=[CFG / "reruns_20260405_optfix"],
    ),
    PartSpec(
        name="08_optimization_track_20260406_v100",
        source_dirs=[NOTES / "optimization_track_20260406_v100"],
        extra_files=[
            SCRIPTS / "build_optimization_submission_matrix_20260406_v100.py",
            SCRIPTS / "submit_optimization_matrix_20260406_v100.py",
            SCRIPTS / "monitor_optimization_matrix_20260406_v100.py",
        ],
        extra_dirs=[CFG / "reruns_20260406_v100_optfix"],
    ),
    PartSpec(
        name="09_optimization_track_20260406_v100_large",
        source_dirs=[NOTES / "optimization_track_20260406_v100_large"],
        extra_files=[
            SCRIPTS / "build_v100_large_matrix_20260406.py",
            SCRIPTS / "submit_v100_large_matrix_20260406.py",
            SCRIPTS / "monitor_v100_large_matrix_20260406.py",
            SCRIPTS / "run_v100_large_bucket.py",
            SCRIPTS / "slurm_v100_large_bucket.sbatch",
        ],
        extra_dirs=[CFG / "reruns_20260406_v100_large_optfix"],
    ),
    PartSpec(
        name="10_advisor_packet_20260402",
        source_dirs=[NOTES / "advisor_packet_20260402"],
        extra_files=[
            NOTES / "advisor_packet_20260402" / "scripts" / "generate_advisor_packet_20260402.py",
        ],
    ),
]


def copy_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def collect_from_dir(part_dir: Path, source_root: Path, label: str, manifest_rows: List[Dict[str, str]]) -> None:
    if not source_root.exists():
        return

    for src in source_root.rglob("*"):
        if not src.is_file():
            continue
        if blocked_path(src):
            continue

        cat = categorize(src)
        if cat is None:
            continue

        rel = src.relative_to(source_root)
        dst = part_dir / cat / label / rel
        size = copy_file(src, dst)
        manifest_rows.append(
            {
                "part": part_dir.name,
                "category": cat,
                "source": str(src),
                "destination": str(dst),
                "size_bytes": str(size),
            }
        )


def collect_extra_file(part_dir: Path, src: Path, label: str, manifest_rows: List[Dict[str, str]]) -> None:
    if not src.exists() or not src.is_file():
        return
    if blocked_path(src):
        return

    cat = categorize(src)
    if cat is None:
        return

    dst = part_dir / cat / label / src.name
    size = copy_file(src, dst)
    manifest_rows.append(
        {
            "part": part_dir.name,
            "category": cat,
            "source": str(src),
            "destination": str(dst),
            "size_bytes": str(size),
        }
    )


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def copy_static_site_for_pages(manifest_rows: List[Dict[str, str]]) -> None:
    """
    Create a deploy-ready static website folder with index.html at folder root.
    """
    site_part = BUNDLE_ROOT / "11_static_website_github_pages"
    site_part.mkdir(parents=True, exist_ok=True)

    site_allow = {".html", ".css", ".js", ".json", ".png", ".sh"}

    if SITE_SRC.exists():
        for src in SITE_SRC.rglob("*"):
            if not src.is_file():
                continue
            ext = src.suffix.lower()
            if ext not in site_allow:
                continue

            rel = src.relative_to(SITE_SRC)
            dst = site_part / rel
            size = copy_file(src, dst)

            if ext == ".png":
                cat = "graphs"
            elif ext == ".json":
                cat = "data"
            else:
                cat = "code"

            manifest_rows.append(
                {
                    "part": site_part.name,
                    "category": cat,
                    "source": str(src),
                    "destination": str(dst),
                    "size_bytes": str(size),
                }
            )

        # Keep site self-consistent without requiring markdown upload.
        index_path = site_part / "index.html"
        if index_path.exists():
            text = index_path.read_text()
            text = text.replace(
                '<a class="btn btn--ghost" href="reports/advisor_report_20260402.md" target="_blank" rel="noopener">Open Markdown</a>',
                "",
            )
            index_path.write_text(text)


def main() -> None:
    if BUNDLE_ROOT.exists():
        shutil.rmtree(BUNDLE_ROOT)
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []

    for part in PARTS:
        part_dir = BUNDLE_ROOT / part.name
        (part_dir / "graphs").mkdir(parents=True, exist_ok=True)
        (part_dir / "data").mkdir(parents=True, exist_ok=True)
        (part_dir / "code").mkdir(parents=True, exist_ok=True)

        for src_dir in part.source_dirs:
            collect_from_dir(part_dir, src_dir, src_dir.name, manifest_rows)

        for extra_dir in part.extra_dirs:
            collect_from_dir(part_dir, extra_dir, extra_dir.name, manifest_rows)

        for extra_file in part.extra_files:
            collect_extra_file(part_dir, extra_file, "extra", manifest_rows)

    # Dedicated deploy-ready website folder.
    copy_static_site_for_pages(manifest_rows)

    # Manifest and per-part summaries.
    manifest_path = BUNDLE_ROOT / "bundle_manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        ["part", "category", "source", "destination", "size_bytes"],
    )

    # Compute summary.
    summary_rows: List[Dict[str, str]] = []
    totals_by_part: Dict[str, Dict[str, int]] = {}
    for row in manifest_rows:
        part = row["part"]
        cat = row["category"]
        sz = int(row["size_bytes"])
        if part not in totals_by_part:
            totals_by_part[part] = {"graphs": 0, "data": 0, "code": 0, "total": 0, "files": 0}
        totals_by_part[part][cat] += sz
        totals_by_part[part]["total"] += sz
        totals_by_part[part]["files"] += 1

    for part in sorted(totals_by_part):
        item = totals_by_part[part]
        summary_rows.append(
            {
                "part": part,
                "files": str(item["files"]),
                "graphs_bytes": str(item["graphs"]),
                "data_bytes": str(item["data"]),
                "code_bytes": str(item["code"]),
                "total_bytes": str(item["total"]),
            }
        )

    summary_path = BUNDLE_ROOT / "bundle_size_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        ["part", "files", "graphs_bytes", "data_bytes", "code_bytes", "total_bytes"],
    )

    readme = BUNDLE_ROOT / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# GitHub Upload Bundle ({DATE_TAG})",
                "",
                "This folder is prepared for GitHub upload with one folder per study part.",
                "",
                "Rules applied:",
                "- Included categories only: graphs, data, code.",
                "- Strict allowlist by extension only (unknown extensions are excluded).",
                "- Graphs: .png",
                "- Data: .csv, .json, .jsonl, .npz, .npy",
                "- Code: .py, .sh, .sbatch, .yaml, .yml, .html, .css, .js",
                "- Excluded testing-run files (filename contains 'testing').",
                "- Excluded markdown and PDF by default (only include if explicitly requested).",
                "- Excluded heavy/log artifacts (.log, .out, .pt, .pyc, .mode, .ready, .gz, .zip, .sha256).",
                "- Excluded cache/runtime directories (__pycache__, .git, run_dnn, stage_status).",
                "",
                "Generated files:",
                "- bundle_manifest.csv: full source -> destination manifest",
                "- bundle_size_summary.csv: per-part size breakdown",
                "",
                "Safe upload practice:",
                "- From your Git repository root, add only this folder path.",
                f"- Example: git add data/important_notes/{BUNDLE_ROOT.name}",
                "",
            ]
        )
    )

    print(f"bundle_root={BUNDLE_ROOT}")
    print(f"manifest={manifest_path}")
    print(f"summary={summary_path}")
    print(f"parts={len({r['part'] for r in manifest_rows})}")
    print(f"files={len(manifest_rows)}")


if __name__ == "__main__":
    main()
