#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / "optimization_track_20260423_hh_track4_checkpoint7_partial_saturation"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
FIGS = OUT_ROOT / "figures"

CLASSICAL_REGISTRY = (
    IMPORTANT
    / "optimization_track_20260423_hh_track4_checkpoint6_shift_stability_classical"
    / "tables"
    / "hh_track4_checkpoint6_shift_stability_classical_registry_20260423_hh_track4_checkpoint6_shift_stability_classical.csv"
)
NEURAL_CONFIRM_REGISTRY = (
    IMPORTANT
    / "optimization_track_20260420_a30_hh_track4_top3_confirmation"
    / "tables"
    / "a30_hh_track4_top3_confirmation_registry_20260420_a30_hh_track4_top3_confirmation.csv"
)
NEURAL_SHIFT_REGISTRY = (
    IMPORTANT
    / "optimization_track_20260423_v100_hh_track4_checkpoint6_shift_stability_neural"
    / "tables"
    / "v100_hh_track4_checkpoint6_shift_stability_neural_registry_20260423_v100_hh_track4_checkpoint6_shift_stability_neural.csv"
)
SBI_REGISTRY = (
    IMPORTANT
    / "optimization_track_20260422_v100_hh_track4_checkpoint2_snpe_seed_stability"
    / "tables"
    / "v100_hh_track4_checkpoint2_snpe_seed_stability_registry_20260422_v100_hh_track4_checkpoint2_snpe_seed_stability.csv"
)
CHECKPOINT6_SUMMARY = (
    IMPORTANT
    / "optimization_track_20260423_hh_track4_checkpoint6_shift_stability"
    / "notes"
    / "summary.json"
)
DIRECT_FIT_BLOCKED_NOTE = (
    IMPORTANT
    / "optimization_track_20260422_hh_track4_checkpoint2_sbi_analysis"
    / "notes"
    / "hh_track4_checkpoint2_sbi_analysis_notes_20260422.md"
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_run_dir(run_output_root: str) -> Path:
    root = Path(run_output_root)
    if not root.is_absolute():
        root = REPO / root
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories under {root}")
    return candidates[-1]


def classical_row(strategy_id: str) -> dict:
    row = [r for r in load_rows(CLASSICAL_REGISTRY) if r["strategy_id"] == strategy_id and r["status"] == "COMPLETED"][0]
    run_dir = resolve_run_dir(row["run_output_root"])
    metrics = json.loads((run_dir / "metrics_summary.json").read_text())
    return {
        "family": "classical",
        "representative": strategy_id.replace("__clean", ""),
        "clean_mae": float(metrics["mae"]["test"]),
        "clean_mse": float(metrics["mse"]["test"]),
        "clean_r2": float(metrics["r2"]["test"]),
        "runtime_sec": float(metrics["metadata"]["runtime_sec"]),
        "n_train": int(metrics["metadata"]["n_train"]),
        "source_registry": str(CLASSICAL_REGISTRY),
    }


def neural_row() -> dict:
    perf_rows = [r for r in load_rows(NEURAL_SHIFT_REGISTRY) if r["strategy_id"] == "effnet_beta005_invvar_e500_clean" and r["status"] == "COMPLETED"]
    perfs = []
    for r in perf_rows:
        run_dir = resolve_run_dir(r["run_output_root"])
        metrics = json.loads((run_dir / "metrics_summary.json").read_text())
        perfs.append(metrics)

    runtime_rows = [r for r in load_rows(NEURAL_CONFIRM_REGISTRY) if r["strategy_id"] == "effnet_beta005_invvar_e500" and r["status"] == "COMPLETED"]
    runtimes = [float(r["elapsed_sec"]) for r in runtime_rows]
    return {
        "family": "neural",
        "representative": "effnet_beta005_invvar_e500",
        "clean_mae": mean(float(m["mae"]["test"]) for m in perfs),
        "clean_mse": mean(float(m["mse"]["test"]) for m in perfs),
        "clean_r2": mean(float(m["r2"]["test"]) for m in perfs),
        "runtime_sec": mean(runtimes),
        "n_train": 4096,
        "source_registry": str(NEURAL_CONFIRM_REGISTRY),
    }


def sbi_row() -> dict:
    rows = [r for r in load_rows(SBI_REGISTRY) if r["strategy_id"].startswith("snpe_maf_h192_t8_n8192_") and r["status"] == "COMPLETED"]
    metrics = []
    for r in rows:
        run_dir = resolve_run_dir(r["run_output_root"])
        payload = json.loads((run_dir / "metrics_summary.json").read_text())
        metrics.append(payload)
    return {
        "family": "sbi",
        "representative": "snpe_maf_h192_t8_n8192",
        "clean_mae": mean(float(m["posterior_mean_metrics"]["mae"]) for m in metrics),
        "clean_mse": mean(float(m["posterior_mean_metrics"]["mse"]) for m in metrics),
        "clean_r2": mean(float(m["posterior_mean_metrics"]["r2"]) for m in metrics),
        "runtime_sec": mean(float(m["total_runtime_sec"]) for m in metrics),
        "n_train": 8192,
        "source_registry": str(SBI_REGISTRY),
    }


def blocked_direct_fit_row() -> dict:
    return {
        "family": "direct_fitting",
        "representative": "blocked_missing_simulator_metadata",
        "clean_mae": "",
        "clean_mse": "",
        "clean_r2": "",
        "runtime_sec": "",
        "n_train": "",
        "source_registry": str(DIRECT_FIT_BLOCKED_NOTE),
    }


def pareto_front(rows: list[dict]) -> list[dict]:
    candidates = [r for r in rows if isinstance(r["runtime_sec"], (int, float)) and isinstance(r["clean_mae"], (int, float))]
    front = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            if other["runtime_sec"] <= row["runtime_sec"] and other["clean_mae"] <= row["clean_mae"]:
                if other["runtime_sec"] < row["runtime_sec"] or other["clean_mae"] < row["clean_mae"]:
                    dominated = True
                    break
        if not dominated:
            front.append(row)
    front.sort(key=lambda r: (r["runtime_sec"], r["clean_mae"]))
    return front


def plot_clean_mae_vs_runtime(rows: list[dict]) -> None:
    candidates = [r for r in rows if isinstance(r["runtime_sec"], (int, float)) and isinstance(r["clean_mae"], (int, float))]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for row in candidates:
        ax.scatter(row["runtime_sec"], row["clean_mae"], s=60)
        ax.text(row["runtime_sec"], row["clean_mae"], row["family"], fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("runtime_sec")
    ax.set_ylabel("clean test MAE")
    ax.set_title("Checkpoint7 partial cost-aware frontier")
    fig.savefig(FIGS / "clean_mae_vs_runtime.pdf")
    plt.close(fig)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    rows = [
        classical_row("random_forest_500__clean"),
        classical_row("knn_k11__clean"),
        neural_row(),
        sbi_row(),
        blocked_direct_fit_row(),
    ]

    frontier_rows = pareto_front(rows)
    checkpoint6_summary = json.loads(CHECKPOINT6_SUMMARY.read_text())

    write_csv(TABLES / "family_representatives.csv", rows, list(rows[0].keys()))
    write_csv(TABLES / "pareto_front_clean.csv", frontier_rows, list(frontier_rows[0].keys()))
    plot_clean_mae_vs_runtime(rows)

    summary = {
        "checkpoint6_unstable_shift_labels": checkpoint6_summary["ranking_summary"]["unstable_shift_labels"],
        "best_clean_family_by_mae": min(
            [r for r in rows if isinstance(r["clean_mae"], (int, float))],
            key=lambda r: r["clean_mae"],
        )["family"],
        "direct_fitting_status": "blocked_missing_simulator_metadata",
        "saturation_claim_ready": False,
    }
    (NOTES / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    note_lines = [
        "# HH Track4 Checkpoint7 Partial Saturation (2026-04-23)",
        "",
        "## Purpose",
        "- summarize the current best available classical, neural, and SBI representatives in a partial cost-aware frontier package",
        "- explicitly record that a strong saturation claim is not yet available because direct fitting remains blocked",
        "",
        "## Outputs",
        f"- family representatives: `{TABLES / 'family_representatives.csv'}`",
        f"- Pareto front: `{TABLES / 'pareto_front_clean.csv'}`",
        f"- clean MAE vs runtime: `{FIGS / 'clean_mae_vs_runtime.pdf'}`",
        f"- summary json: `{NOTES / 'summary.json'}`",
        "",
        "## Current Read",
        f"- best clean MAE family: `{summary['best_clean_family_by_mae']}`",
        f"- checkpoint6 unstable shifts still active: `{summary['checkpoint6_unstable_shift_labels']}`",
        "- direct fitting / simulator-based family: still blocked by missing simulator metadata",
        "- strongest saturation claim is therefore not available yet; this is a partial checkpoint7 package only",
    ]
    (NOTES / "notes.md").write_text("\n".join(note_lines) + "\n")

    print(f"Wrote {TABLES / 'family_representatives.csv'}")
    print(f"Wrote {TABLES / 'pareto_front_clean.csv'}")


if __name__ == "__main__":
    main()
