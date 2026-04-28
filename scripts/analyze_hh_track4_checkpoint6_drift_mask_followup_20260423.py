#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
DEFAULT_OUT_ROOT = IMPORTANT / "optimization_track_20260423_hh_track4_checkpoint6_drift_mask_followup_analysis"
DEFAULT_CLASSICAL_REGISTRIES = [
    IMPORTANT
    / "optimization_track_20260423_hh_track4_checkpoint6_drift_mask_followup_classical"
    / "tables"
    / "hh_track4_checkpoint6_drift_mask_followup_classical_registry_20260423_hh_track4_checkpoint6_drift_mask_followup_classical.csv"
]
DEFAULT_NEURAL_REGISTRIES = [
    IMPORTANT
    / "optimization_track_20260423_v100_hh_track4_checkpoint6_drift_mask_followup_neural"
    / "tables"
    / "v100_hh_track4_checkpoint6_drift_mask_followup_neural_registry_20260423_v100_hh_track4_checkpoint6_drift_mask_followup_neural.csv"
]
CP5_PARAMETER_STATUS = (
    IMPORTANT
    / "optimization_track_20260422_hh_track4_checkpoint5_ambiguity_map"
    / "tables"
    / "parameter_status.csv"
)
CP5_LOCAL_AMBIGUITY = (
    IMPORTANT
    / "optimization_track_20260422_hh_track4_checkpoint5_ambiguity_map"
    / "tables"
    / "local_ambiguity.csv"
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


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO / resolved
    return resolved


def resolve_run_dir(run_output_root: str) -> Path:
    root = resolve_repo_path(run_output_root)
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories under {root}")
    return candidates[-1]


def parse_shift(strategy_id: str) -> tuple[str, str, int]:
    if "__" in strategy_id:
        _base, shift = strategy_id.split("__", 1)
    else:
        prefixes = [
            "effnet_beta005_invvar_e500_",
            "random_forest_500_",
            "knn_k11_",
        ]
        shift = strategy_id
        for prefix in prefixes:
            if strategy_id.startswith(prefix):
                shift = strategy_id[len(prefix) :]
                break
    if shift == "clean":
        return "clean", "clean", 0
    if shift.startswith("drift"):
        return "drift", shift, int(shift[5:])
    if shift.startswith("mask"):
        return "mask", shift, int(shift[4:])
    raise ValueError(f"Unexpected follow-up shift token: {strategy_id}")


def parse_family(strategy_id: str) -> str:
    if strategy_id.startswith("random_forest_500"):
        return "random_forest"
    if strategy_id.startswith("knn_k11"):
        return "knn"
    if strategy_id.startswith("effnet_beta005_invvar_e500"):
        return "effnet"
    raise ValueError(f"Unexpected family in strategy_id={strategy_id}")


def load_classical_per_target(run_dir: Path) -> tuple[list[str], np.ndarray]:
    rows = json.loads((run_dir / "metrics_per_target.json").read_text())
    maes = np.array([float(r["mae"]) for r in rows], dtype=np.float64)
    target_names = [f"hh_param_{int(r['target_index']) + 1}" for r in rows]
    return target_names, maes


def load_neural_per_target(run_dir: Path) -> tuple[list[str], np.ndarray]:
    rows = load_rows(run_dir / "metrics_per_target.csv")
    test_rows = [r for r in rows if r["split"] == "test"]
    test_rows.sort(key=lambda r: int(r["target_index"]))
    target_names = [r["target_name"] for r in test_rows]
    maes = np.array([float(r["mae"]) for r in test_rows], dtype=np.float64)
    return target_names, maes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze HH track4 checkpoint6 drift/mask follow-up registries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--classical-registry",
        action="append",
        type=Path,
        help="Repeatable classical follow-up registry CSV. Defaults to the current focused follow-up classical registry.",
    )
    parser.add_argument(
        "--neural-registry",
        action="append",
        type=Path,
        help="Repeatable neural follow-up registry CSV. Defaults to the current focused follow-up neural registry.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Root directory for generated tables, notes, and figures.",
    )
    return parser.parse_args()


def collect_metrics(classical_registries: list[Path], neural_registries: list[Path]) -> tuple[list[dict], list[str]]:
    records = []
    target_names = None
    registry_specs = [("classical", path) for path in classical_registries] + [
        ("neural", path) for path in neural_registries
    ]
    for registry_type, registry_path in registry_specs:
        for row in load_rows(registry_path):
            if row["status"] != "COMPLETED":
                continue
            run_dir = resolve_run_dir(row["run_output_root"])
            metrics = json.loads((run_dir / "metrics_summary.json").read_text())
            family = parse_family(row["strategy_id"])
            shift_family, shift_label, severity = parse_shift(row["strategy_id"])
            if registry_type == "neural":
                names, per_target_mae = load_neural_per_target(run_dir)
            else:
                names, per_target_mae = load_classical_per_target(run_dir)
            if target_names is None:
                target_names = names
            records.append(
                {
                    "family": family,
                    "strategy_id": row["strategy_id"],
                    "shift_family": shift_family,
                    "shift_label": shift_label,
                    "severity": severity,
                    "mae": float(metrics["mae"]["test"]),
                    "mse": float(metrics["mse"]["test"]),
                    "r2": float(metrics["r2"]["test"]),
                    "per_target_mae": per_target_mae,
                }
            )
    assert target_names is not None
    return records, target_names


def mean_rows(records: list[dict]) -> tuple[list[dict], dict[tuple[str, str], np.ndarray]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        grouped[(rec["family"], rec["shift_label"])].append(rec)

    aggregate_rows = []
    per_target_lookup: dict[tuple[str, str], np.ndarray] = {}
    for (family, shift_label), items in sorted(grouped.items()):
        shift_family = items[0]["shift_family"]
        severity = items[0]["severity"]
        aggregate_rows.append(
            {
                "family": family,
                "shift_family": shift_family,
                "shift_label": shift_label,
                "severity": severity,
                "n_runs": len(items),
                "mae_mean": float(np.mean([r["mae"] for r in items])),
                "mae_std": float(np.std([r["mae"] for r in items])),
                "mse_mean": float(np.mean([r["mse"] for r in items])),
                "mse_std": float(np.std([r["mse"] for r in items])),
                "r2_mean": float(np.mean([r["r2"] for r in items])),
                "r2_std": float(np.std([r["r2"] for r in items])),
            }
        )
        per_target_lookup[(family, shift_label)] = np.mean(
            np.stack([r["per_target_mae"] for r in items], axis=0),
            axis=0,
        )
    return aggregate_rows, per_target_lookup


def load_cp5_labels() -> tuple[dict[str, str], dict[str, tuple[str, float]]]:
    status_rows = load_rows(CP5_PARAMETER_STATUS)
    status_map = {r["target_name"]: r["heuristic_status"] for r in status_rows}

    amb_rows = load_rows(CP5_LOCAL_AMBIGUITY)
    best_map: dict[str, tuple[str, float]] = {}
    for row in amb_rows:
        name = row["target_name"]
        corr = abs(float(row["corr_promoted_rf_abs_error"]))
        summary = row["summary_name"]
        prev = best_map.get(name)
        if prev is None or corr > prev[1]:
            best_map[name] = (summary, corr)
    return status_map, best_map


def build_ranking_rows(aggregate_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in aggregate_rows:
        grouped[row["shift_label"]].append(row)

    order = ["random_forest", "knn", "effnet"]
    role_map = {
        "random_forest": "random_forest_clean_winner",
        "knn": "knn_robustness_winner",
        "effnet": "efficientnet_neural_baseline",
    }
    reference_order = ["random_forest", "knn", "effnet"]

    rows = []
    for shift_label, items in sorted(grouped.items(), key=lambda kv: (kv[1][0]["shift_family"], kv[1][0]["severity"])):
        items = sorted(items, key=lambda r: (r["mae_mean"], order.index(r["family"])))
        ranked = [r["family"] for r in items]
        flips = 0
        ref_pos = {f: i for i, f in enumerate(reference_order)}
        cur_pos = {f: i for i, f in enumerate(ranked)}
        for i, left in enumerate(reference_order):
            for right in reference_order[i + 1 :]:
                if (ref_pos[left] - ref_pos[right]) * (cur_pos[left] - cur_pos[right]) < 0:
                    flips += 1
        row = {
            "shift_family": items[0]["shift_family"],
            "shift_label": shift_label,
            "severity": items[0]["severity"],
            "rank_signature": " > ".join(role_map[f] for f in ranked),
            "winner_family_role": role_map[ranked[0]],
            "winner_margin_vs_runner_up": float(items[1]["mae_mean"] - items[0]["mae_mean"]),
            "pairwise_flips_vs_reference": flips,
            "stable_vs_reference": flips == 0,
        }
        for rank, item in enumerate(items, start=1):
            row[f"rank_{rank}_family_role"] = role_map[item["family"]]
            row[f"rank_{rank}_mae_mean"] = item["mae_mean"]
        rows.append(row)
    return rows


def build_flip_driver_rows(
    target_names: list[str],
    per_target_lookup: dict[tuple[str, str], np.ndarray],
    cp5_status: dict[str, str],
    cp5_summary: dict[str, tuple[str, float]],
) -> list[dict]:
    families = ["random_forest", "knn", "effnet"]
    shift_labels = sorted({label for (_family, label) in per_target_lookup if label != "clean"}, key=lambda x: (0 if x.startswith("drift") else 1, int(x[5:] if x.startswith("drift") else x[4:])))
    clean_lookup = {family: per_target_lookup[(family, "clean")] for family in families}
    rows = []
    for shift_label in shift_labels:
        shift_family = "drift" if shift_label.startswith("drift") else "mask"
        severity = int(shift_label[5:] if shift_family == "drift" else shift_label[4:])
        fam_arrays = {family: per_target_lookup[(family, shift_label)] for family in families}
        for idx, target_name in enumerate(target_names):
            maes = {family: float(fam_arrays[family][idx]) for family in families}
            ranked = sorted(families, key=lambda family: maes[family])
            winner = ranked[0]
            runner_up = ranked[1]
            best_summary = cp5_summary.get(target_name, ("", float("nan")))
            rows.append(
                {
                    "shift_family": shift_family,
                    "shift_label": shift_label,
                    "severity": severity,
                    "target_index": idx,
                    "target_name": target_name,
                    "heuristic_status": cp5_status.get(target_name, ""),
                    "strongest_summary_name": best_summary[0],
                    "strongest_summary_corr": best_summary[1],
                    "rf_mae": maes["random_forest"],
                    "knn_mae": maes["knn"],
                    "effnet_mae": maes["effnet"],
                    "rf_delta_vs_clean": maes["random_forest"] - float(clean_lookup["random_forest"][idx]),
                    "knn_delta_vs_clean": maes["knn"] - float(clean_lookup["knn"][idx]),
                    "effnet_delta_vs_clean": maes["effnet"] - float(clean_lookup["effnet"][idx]),
                    "winner_family_role": winner,
                    "runner_up_family_role": runner_up,
                    "winner_margin_component": maes[runner_up] - maes[winner],
                }
            )
    return rows


def plot_family_mae_vs_severity(aggregate_rows: list[dict], figs_dir: Path) -> None:
    family_label = {
        "random_forest": "Random Forest",
        "knn": "kNN",
        "effnet": "EfficientNet",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, shift_family in zip(axes, ["drift", "mask"]):
        rows = [r for r in aggregate_rows if r["shift_family"] in {"clean", shift_family}]
        for family in ["random_forest", "knn", "effnet"]:
            fam_rows = [r for r in rows if r["family"] == family]
            fam_rows.sort(key=lambda r: r["severity"])
            xs = [r["severity"] for r in fam_rows]
            ys = [r["mae_mean"] for r in fam_rows]
            ax.plot(xs, ys, marker="o", label=family_label[family])
        ax.set_title(shift_family)
        ax.set_xlabel("severity")
        ax.set_ylabel("test MAE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.savefig(figs_dir / "family_mae_vs_severity.pdf")
    plt.close(fig)


def plot_flip_driver_heatmap(rows: list[dict], target_names: list[str], figs_dir: Path) -> None:
    status_map = {r["target_name"]: r["heuristic_status"] for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, shift_family in zip(axes, ["drift", "mask"]):
        subset = [r for r in rows if r["shift_family"] == shift_family]
        severities = sorted({int(r["severity"]) for r in subset})
        matrix = np.zeros((len(target_names), len(severities)), dtype=np.float64)
        label_matrix = [["" for _ in severities] for _ in target_names]
        for r in subset:
            i = target_names.index(r["target_name"])
            j = severities.index(int(r["severity"]))
            matrix[i, j] = float(r["winner_margin_component"])
            label_matrix[i][j] = str(r["winner_family_role"])[:3]
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(shift_family)
        ax.set_xticks(range(len(severities)), labels=severities)
        ylabels = [f"{name} ({status_map.get(name,'')})" for name in target_names]
        ax.set_yticks(range(len(target_names)), labels=ylabels)
        ax.set_xlabel("severity")
        for i in range(len(target_names)):
            for j in range(len(severities)):
                ax.text(j, i, label_matrix[i][j], ha="center", va="center", color="white", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(figs_dir / "flip_driver_contribution_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    classical_registries = [resolve_repo_path(path) for path in (args.classical_registry or DEFAULT_CLASSICAL_REGISTRIES)]
    neural_registries = [resolve_repo_path(path) for path in (args.neural_registry or DEFAULT_NEURAL_REGISTRIES)]
    out_root = resolve_repo_path(args.out_root)
    tables = out_root / "tables"
    notes = out_root / "notes"
    figs = out_root / "figures"

    out_root.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    records, target_names = collect_metrics(classical_registries, neural_registries)
    aggregate_rows, per_target_lookup = mean_rows(records)
    ranking_rows = build_ranking_rows(aggregate_rows)
    cp5_status, cp5_summary = load_cp5_labels()
    flip_rows = build_flip_driver_rows(target_names, per_target_lookup, cp5_status, cp5_summary)

    write_csv(
        tables / "family_shift_metrics.csv",
        aggregate_rows,
        list(aggregate_rows[0].keys()),
    )
    write_csv(
        tables / "ranking_stability.csv",
        ranking_rows,
        list(ranking_rows[0].keys()),
    )
    write_csv(
        tables / "flip_driver_decomposition.csv",
        flip_rows,
        list(flip_rows[0].keys()),
    )

    plot_family_mae_vs_severity(aggregate_rows, figs)
    plot_flip_driver_heatmap(flip_rows, target_names, figs)

    note_lines = [
        "# HH Track4 Checkpoint6 Drift/Mask Follow-Up Analysis",
        "",
        *[f"- classical registry: `{path}`" for path in classical_registries],
        *[f"- neural registry: `{path}`" for path in neural_registries],
        "",
        "## Outputs",
        f"- ranking stability: `{tables / 'ranking_stability.csv'}`",
        f"- family shift metrics: `{tables / 'family_shift_metrics.csv'}`",
        f"- flip driver decomposition: `{tables / 'flip_driver_decomposition.csv'}`",
        f"- severity curves: `{figs / 'family_mae_vs_severity.pdf'}`",
        f"- driver heatmap: `{figs / 'flip_driver_contribution_heatmap.pdf'}`",
    ]
    (notes / "notes.md").write_text("\n".join(note_lines) + "\n")

    print(f"Wrote {tables / 'ranking_stability.csv'}")
    print(f"Wrote {tables / 'family_shift_metrics.csv'}")
    print(f"Wrote {tables / 'flip_driver_decomposition.csv'}")


if __name__ == "__main__":
    main()
