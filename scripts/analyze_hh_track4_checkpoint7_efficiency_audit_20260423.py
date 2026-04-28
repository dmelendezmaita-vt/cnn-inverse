#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / "optimization_track_20260423_hh_track4_checkpoint7_efficiency_audit"
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


def average_completed_rows(registry: Path, strategy_id: str) -> list[dict[str, str]]:
    return [r for r in load_rows(registry) if r["strategy_id"] == strategy_id and r["status"] == "COMPLETED"]


def classical_row(strategy_id: str, representative: str) -> dict:
    row = average_completed_rows(CLASSICAL_REGISTRY, strategy_id)[0]
    run_dir = resolve_run_dir(row["run_output_root"])
    metrics = json.loads((run_dir / "metrics_summary.json").read_text())
    runtime_sec = float(metrics["metadata"]["runtime_sec"])
    n_train = int(metrics["metadata"]["n_train"])
    n_test = int(metrics["metadata"]["n_test"])
    return {
        "family": "classical",
        "representative": representative,
        "clean_mae": float(metrics["mae"]["test"]),
        "clean_mse": float(metrics["mse"]["test"]),
        "clean_r2": float(metrics["r2"]["test"]),
        "train_runtime_sec": "",
        "eval_runtime_sec": "",
        "total_runtime_sec": runtime_sec,
        "runtime_breakdown": "total_only",
        "runtime_source": "metrics_summary.metadata.runtime_sec",
        "n_train": n_train,
        "n_test_eval": n_test,
        "sec_per_1k_train": float(runtime_sec / max(1.0, n_train / 1000.0)),
        "sec_per_eval_example": float(runtime_sec / max(1, n_test)),
        "hardware": "CPU(tc047)",
        "launch_mode": row["launch_mode"],
        "source_registry": str(CLASSICAL_REGISTRY),
        "notes": "Combined runtime only; train/eval not logged separately for this family.",
    }


def neural_row() -> dict:
    perf_rows = average_completed_rows(NEURAL_SHIFT_REGISTRY, "effnet_beta005_invvar_e500_clean")
    perf_metrics = []
    eval_elapsed = []
    n_train = None
    n_test = None
    for row in perf_rows:
        run_dir = resolve_run_dir(row["run_output_root"])
        perf_metrics.append(json.loads((run_dir / "metrics_summary.json").read_text()))
        params = json.loads(json.dumps({}))  # placeholder for scope
        del params
        eval_elapsed.append(float(row["elapsed_sec"]))
        if n_train is None or n_test is None:
            import yaml
            params_yaml = yaml.safe_load((run_dir / "params.yaml").read_text())
            n_train = int(params_yaml["data"]["Ntrain"])
            n_test = int(params_yaml["data"]["Ntest"])

    train_rows = average_completed_rows(NEURAL_CONFIRM_REGISTRY, "effnet_beta005_invvar_e500")
    train_elapsed = [float(row["elapsed_sec"]) for row in train_rows]

    train_runtime = float(mean(train_elapsed))
    eval_runtime = float(mean(eval_elapsed))
    total_runtime = train_runtime + eval_runtime
    return {
        "family": "neural",
        "representative": "effnet_beta005_invvar_e500",
        "clean_mae": float(mean(float(m["mae"]["test"]) for m in perf_metrics)),
        "clean_mse": float(mean(float(m["mse"]["test"]) for m in perf_metrics)),
        "clean_r2": float(mean(float(m["r2"]["test"]) for m in perf_metrics)),
        "train_runtime_sec": train_runtime,
        "eval_runtime_sec": eval_runtime,
        "total_runtime_sec": total_runtime,
        "runtime_breakdown": "train_plus_eval_proxy",
        "runtime_source": "confirmation_registry.elapsed_sec + shift_registry.elapsed_sec",
        "n_train": int(n_train),
        "n_test_eval": int(n_test),
        "sec_per_1k_train": float(total_runtime / max(1.0, n_train / 1000.0)),
        "sec_per_eval_example": float(eval_runtime / max(1, n_test)),
        "hardware": "A30(train)+V100(eval)",
        "launch_mode": "concurrent_3x1n + concurrent_8x1n",
        "source_registry": f"{NEURAL_CONFIRM_REGISTRY} | {NEURAL_SHIFT_REGISTRY}",
        "notes": "Training and evaluation proxies come from separate representative packages.",
    }


def sbi_row() -> dict:
    seed_rows = [
        row
        for row in load_rows(SBI_REGISTRY)
        if row["strategy_id"].startswith("snpe_maf_h192_t8_n8192_") and row["status"] == "COMPLETED"
    ]
    metrics = []
    for row in seed_rows:
        run_dir = resolve_run_dir(row["run_output_root"])
        metrics.append(json.loads((run_dir / "metrics_summary.json").read_text()))

    train_runtime = float(mean(float(m["training_runtime_sec"]) for m in metrics))
    eval_runtime = float(mean(float(m["evaluation_runtime_sec"]) for m in metrics))
    total_runtime = float(mean(float(m["total_runtime_sec"]) for m in metrics))
    n_train = int(mean(int(m["n_train"]) for m in metrics))
    n_test = int(mean(int(m["n_test"]) for m in metrics))
    return {
        "family": "sbi",
        "representative": "snpe_maf_h192_t8_n8192",
        "clean_mae": float(mean(float(m["posterior_mean_metrics"]["mae"]) for m in metrics)),
        "clean_mse": float(mean(float(m["posterior_mean_metrics"]["mse"]) for m in metrics)),
        "clean_r2": float(mean(float(m["posterior_mean_metrics"]["r2"]) for m in metrics)),
        "train_runtime_sec": train_runtime,
        "eval_runtime_sec": eval_runtime,
        "total_runtime_sec": total_runtime,
        "runtime_breakdown": "train_plus_eval",
        "runtime_source": "metrics_summary runtime fields",
        "n_train": n_train,
        "n_test_eval": n_test,
        "sec_per_1k_train": float(total_runtime / max(1.0, n_train / 1000.0)),
        "sec_per_eval_example": float(eval_runtime / max(1, n_test)),
        "hardware": "V100",
        "launch_mode": seed_rows[0]["launch_mode"],
        "source_registry": str(SBI_REGISTRY),
        "notes": "Posterior-mean metrics shown; this is the promoted stable SNPE representative.",
    }


def blocked_direct_fit_row() -> dict:
    return {
        "family": "direct_fitting",
        "representative": "blocked_missing_simulator_metadata",
        "clean_mae": "",
        "clean_mse": "",
        "clean_r2": "",
        "train_runtime_sec": "",
        "eval_runtime_sec": "",
        "total_runtime_sec": "",
        "runtime_breakdown": "blocked",
        "runtime_source": "",
        "n_train": "",
        "n_test_eval": "",
        "sec_per_1k_train": "",
        "sec_per_eval_example": "",
        "hardware": "",
        "launch_mode": "",
        "source_registry": str(DIRECT_FIT_BLOCKED_NOTE),
        "notes": "No simulator contract in the workspace; direct-fitting efficiency cannot be benchmarked yet.",
    }


def pareto_front(rows: list[dict]) -> list[dict]:
    numeric = [r for r in rows if isinstance(r["clean_mae"], float) and isinstance(r["total_runtime_sec"], float)]
    front = []
    for row in numeric:
        dominated = False
        for other in numeric:
            if other is row:
                continue
            if other["total_runtime_sec"] <= row["total_runtime_sec"] and other["clean_mae"] <= row["clean_mae"]:
                if other["total_runtime_sec"] < row["total_runtime_sec"] or other["clean_mae"] < row["clean_mae"]:
                    dominated = True
                    break
        if not dominated:
            front.append(row)
    front.sort(key=lambda r: (r["total_runtime_sec"], r["clean_mae"]))
    return front


def plot_runtime_tradeoff(rows: list[dict], metric_key: str, filename: str, ylabel: str) -> None:
    numeric = [r for r in rows if isinstance(r[metric_key], float) and isinstance(r["total_runtime_sec"], float)]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for row in numeric:
        ax.scatter(row["total_runtime_sec"], row[metric_key], s=60)
        ax.text(row["total_runtime_sec"], row[metric_key], row["family"], fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("total_runtime_sec")
    ax.set_ylabel(ylabel)
    ax.set_title("Checkpoint7 efficiency audit")
    fig.savefig(FIGS / filename)
    plt.close(fig)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    rows = [
        classical_row("random_forest_500__clean", "random_forest_500"),
        classical_row("knn_k11__clean", "knn_k11"),
        neural_row(),
        sbi_row(),
        blocked_direct_fit_row(),
    ]
    front = pareto_front(rows)

    write_csv(TABLES / "family_efficiency_audit.csv", rows, list(rows[0].keys()))
    if front:
        write_csv(TABLES / "pareto_front_clean_mae_vs_total_runtime.csv", front, list(front[0].keys()))
    plot_runtime_tradeoff(rows, "clean_mae", "clean_mae_vs_total_runtime.pdf", "clean test MAE")
    plot_runtime_tradeoff(rows, "clean_mse", "clean_mse_vs_total_runtime.pdf", "clean test MSE")

    best_mae = min([r for r in rows if isinstance(r["clean_mae"], float)], key=lambda r: r["clean_mae"])
    best_mse = min([r for r in rows if isinstance(r["clean_mse"], float)], key=lambda r: r["clean_mse"])
    fastest = min([r for r in rows if isinstance(r["total_runtime_sec"], float)], key=lambda r: r["total_runtime_sec"])
    summary = {
        "best_clean_mae_family": best_mae["family"],
        "best_clean_mae_representative": best_mae["representative"],
        "best_clean_mse_family": best_mse["family"],
        "best_clean_mse_representative": best_mse["representative"],
        "fastest_family": fastest["family"],
        "fastest_representative": fastest["representative"],
        "efficiency_claim_ready": False,
        "direct_fitting_status": "blocked_missing_simulator_metadata",
        "standardized_microbenchmark_status": "missing_for_some_families",
    }
    (NOTES / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    note_lines = [
        "# HH Track4 Checkpoint7 Efficiency Audit (2026-04-23)",
        "",
        "## Purpose",
        "- move the checkpoint7 cost side beyond a single Pareto dot plot",
        "- compare the best classical, neural, and SBI representatives with honest runtime proxies from existing artifacts",
        "",
        "## Outputs",
        f"- family efficiency audit: `{TABLES / 'family_efficiency_audit.csv'}`",
        f"- Pareto front: `{TABLES / 'pareto_front_clean_mae_vs_total_runtime.csv'}`",
        f"- clean MAE vs total runtime: `{FIGS / 'clean_mae_vs_total_runtime.pdf'}`",
        f"- clean MSE vs total runtime: `{FIGS / 'clean_mse_vs_total_runtime.pdf'}`",
        f"- summary json: `{NOTES / 'summary.json'}`",
        "",
        "## Current Read",
        f"- best clean MAE representative: `{best_mae['representative']}` ({best_mae['family']})",
        f"- best clean MSE representative: `{best_mse['representative']}` ({best_mse['family']})",
        f"- fastest completed representative: `{fastest['representative']}` ({fastest['family']})",
        "- neural is clearly dominated on clean accuracy and is not the efficiency winner either",
        "- direct fitting is still blocked, so the strongest systems-efficiency closure claim is not yet available",
        "- runtime logging is not perfectly standardized across all families, so this audit is decision-grade but not a final microbenchmark package",
    ]
    (NOTES / "notes.md").write_text("\n".join(note_lines) + "\n")

    print(f"Wrote {TABLES / 'family_efficiency_audit.csv'}")
    print(f"Wrote {NOTES / 'notes.md'}")


if __name__ == "__main__":
    main()
