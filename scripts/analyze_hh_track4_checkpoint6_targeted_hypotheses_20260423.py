#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
SOURCE_ROOT = IMPORTANT / "optimization_track_20260423_hh_track4_checkpoint6_drift_mask_boundary_analysis"
OUT_ROOT = IMPORTANT / "optimization_track_20260423_hh_track4_checkpoint6_targeted_hypotheses"
TABLES = OUT_ROOT / "tables"
NOTES = OUT_ROOT / "notes"
FIGS = OUT_ROOT / "figures"

DRIVER_CSV = SOURCE_ROOT / "tables" / "flip_driver_decomposition.csv"
RANKING_CSV = SOURCE_ROOT / "tables" / "ranking_stability.csv"


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


def plot_series(rows: list[dict], *, target_name: str, shift_family: str, fields: list[tuple[str, str]], out_name: str, title: str) -> None:
    subset = [r for r in rows if r["target_name"] == target_name and r["shift_family"] == shift_family]
    subset.sort(key=lambda r: int(r["severity"]))
    xs = [int(r["severity"]) for r in subset]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for field, label in fields:
        ys = [float(r[field]) for r in subset]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("severity")
    ax.set_ylabel("per-target MAE")
    ax.set_title(title)
    ax.legend()
    fig.savefig(FIGS / out_name)
    plt.close(fig)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    driver_rows = load_rows(DRIVER_CSV)
    ranking_rows = load_rows(RANKING_CSV)

    drift_h1 = [r for r in driver_rows if r["shift_family"] == "drift" and r["target_name"] == "hh_param_1"]
    mask_targets = [r for r in driver_rows if r["shift_family"] == "mask" and r["target_name"] in {"hh_param_1", "hh_param_4"}]

    drift_h1.sort(key=lambda r: int(r["severity"]))
    mask_targets.sort(key=lambda r: (r["target_name"], int(r["severity"])))

    write_csv(TABLES / "hh_param1_drift_hypothesis.csv", drift_h1, list(drift_h1[0].keys()))
    write_csv(TABLES / "hh_param1_hh_param4_mask_hypothesis.csv", mask_targets, list(mask_targets[0].keys()))

    plot_series(
        driver_rows,
        target_name="hh_param_1",
        shift_family="drift",
        fields=[("rf_mae", "Random Forest"), ("knn_mae", "kNN"), ("effnet_mae", "EfficientNet")],
        out_name="hh_param1_drift_mae_vs_severity.pdf",
        title="hh_param_1 under drift",
    )
    plot_series(
        driver_rows,
        target_name="hh_param_1",
        shift_family="mask",
        fields=[("rf_mae", "Random Forest"), ("knn_mae", "kNN"), ("effnet_mae", "EfficientNet")],
        out_name="hh_param1_mask_mae_vs_severity.pdf",
        title="hh_param_1 under masking",
    )
    plot_series(
        driver_rows,
        target_name="hh_param_4",
        shift_family="mask",
        fields=[("rf_mae", "Random Forest"), ("knn_mae", "kNN"), ("effnet_mae", "EfficientNet")],
        out_name="hh_param4_mask_mae_vs_severity.pdf",
        title="hh_param_4 under masking",
    )

    drift_flip = [r for r in ranking_rows if r["shift_family"] == "drift" and r["stable_vs_reference"] == "False"]
    mask_flip = [r for r in ranking_rows if r["shift_family"] == "mask" and r["stable_vs_reference"] == "False"]

    note_lines = [
        "# HH Track4 Checkpoint6 Targeted Hypotheses (2026-04-23)",
        "",
        "## Purpose",
        "- isolate the localized flip mechanisms after the checkpoint6 boundary search",
        "- focus specifically on `hh_param_1` under drift and `hh_param_1` / `hh_param_4` under masking",
        "",
        "## Outputs",
        f"- hh_param_1 drift table: `{TABLES / 'hh_param1_drift_hypothesis.csv'}`",
        f"- hh_param_1/hh_param_4 mask table: `{TABLES / 'hh_param1_hh_param4_mask_hypothesis.csv'}`",
        f"- hh_param_1 drift figure: `{FIGS / 'hh_param1_drift_mae_vs_severity.pdf'}`",
        f"- hh_param_1 mask figure: `{FIGS / 'hh_param1_mask_mae_vs_severity.pdf'}`",
        f"- hh_param_4 mask figure: `{FIGS / 'hh_param4_mask_mae_vs_severity.pdf'}`",
        "",
        "## Hypothesis Read",
        f"- drift becomes unstable by severity `{drift_flip[0]['severity'] if drift_flip else 'none'}` and is primarily driven by `hh_param_1`",
        f"- mask becomes unstable by severity `{mask_flip[0]['severity'] if mask_flip else 'none'}` and is dominated by `hh_param_1`, with `hh_param_4` acting as the strongest counter-driver",
        "- the neural family does not degrade monotonically under masking; it catastrophically fails at low masking, recovers at mask20, and fails again at stronger masking",
    ]
    (NOTES / "notes.md").write_text("\n".join(note_lines) + "\n")

    print(f"Wrote {TABLES / 'hh_param1_drift_hypothesis.csv'}")
    print(f"Wrote {TABLES / 'hh_param1_hh_param4_mask_hypothesis.csv'}")


if __name__ == "__main__":
    main()
