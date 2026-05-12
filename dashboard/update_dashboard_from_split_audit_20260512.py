#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path("/projects/neuro-collab/code/github_prep/fhn_dnn-1-implementation-in-pytorch-public")
DASHBOARD_JSON = REPO_ROOT / "dashboard" / "data" / "canonical_results.json"
SPLIT_AUDIT_JSON = REPO_ROOT / "dashboard" / "data" / "split_audit_summary_20260512.json"
SPLIT_SUMMARY_CSV = Path("/home/dmm96/manuscript_notes/ms/evidence_bundle_20260512/tables/split_audit_progress_summary.csv")
SPLIT_RUN_ROOT = REPO_ROOT / "runs" / "thesis_rewrite_20260512"


def load_split_rows() -> list[dict[str, str]]:
    return list(csv.DictReader(SPLIT_SUMMARY_CSV.open(newline="", encoding="utf-8")))


def split_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["current"], row["model"]): row for row in rows}


def float_or_zero(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def package_stats(root: Path) -> tuple[int, float]:
    files = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            files += 1
            total_bytes += path.stat().st_size
    return files, total_bytes / (1024 ** 3)


def update_inventory(summary: dict, rows: list[dict[str, str]]) -> None:
    completed_rows = 0
    for run_dir in sorted([p for p in SPLIT_RUN_ROOT.iterdir() if p.is_dir()]):
        if any(run_dir.glob("**/metrics_summary.json")):
            completed_rows += 1
    expected_rows = 30
    files, gib = package_stats(SPLIT_RUN_ROOT)

    summary["generated_on"] = "2026-05-12"
    summary["scope_note"] = (
        "The dashboard now foregrounds the single-current split-contract audit on the `0.1` and `0.3` HH slices. "
        "Older throughput, posterior, and robustness sections remain as supporting or archival context, while the direct HH interpretation is updated to the new audit surface."
    )

    kpis = summary["inventory"]["kpis"]
    if not any(item["label"] == "Split-audit rows complete" for item in kpis):
        kpis.append(
            {
                "label": "Split-audit rows complete",
                "value": f"{completed_rows} / {expected_rows}",
                "note": "Three classical `0.1` random retry rows are still missing, but the direct-neural and current `0.3` surfaces are complete."
            }
        )

    packages = summary["inventory"]["packages"]
    split_pkg = {
        "label": "Single-current split audit",
        "files": files,
        "gib": round(gib, 2),
        "role": "Current HH contract audit across sequential and random splits on the `0.1` and `0.3` current slices."
    }
    if not any(item["label"] == split_pkg["label"] for item in packages):
        packages.insert(0, split_pkg)

    rows_by_space = summary["inventory"]["rows_by_space"]
    split_space = {
        "space": "Single-current split audit",
        "rows": completed_rows,
        "numeric_cells": completed_rows * 4
    }
    if not any(item["space"] == split_space["space"] for item in rows_by_space):
        rows_by_space.insert(0, split_space)

    summary["inventory"]["interpretation_note"] = (
        "The dashboard now carries the live single-current split audit as the front door for HH direct interpretation. "
        "The audit already shows that the direct-neural row remains stable and noncompetitive under split changes, "
        "while the classical lane remains stronger and is now clearly metric-dependent. "
        "Older sections remain available for context, but they no longer define the leading interpretation surface."
    )


def update_reading_guide(summary: dict) -> None:
    guide = summary["reading_guide"]
    if not any(item["title"] == "Contract audit first" for item in guide):
        guide.insert(
            0,
            {
                "title": "Contract audit first",
                "badge": "Interpretation rule",
                "kind": "boundary",
                "body": "The single-current split audit is now the primary HH interpretation surface. Older clean closure, throughput, and aligned-posterior sections should be read as supporting or archival context unless they are revalidated on the same contract."
            },
        )


def build_primary_direct_thread(rows: list[dict[str, str]]) -> dict:
    lut = split_lookup(rows)
    order = [
        ("0.1", "direct_dnn", "DNN @0.1"),
        ("0.1", "extra_trees_500", "ET @0.1"),
        ("0.1", "random_forest_500", "RF @0.1"),
        ("0.3", "direct_dnn", "DNN @0.3"),
        ("0.3", "extra_trees_500", "ET @0.3"),
        ("0.3", "random_forest_500", "RF @0.3"),
    ]

    seq_mae = [float(lut[(curr, model)]["seq_mae"]) for curr, model, _ in order]
    rnd_mae = [float(lut[(curr, model)]["rand_mae_mean"]) for curr, model, _ in order]
    seq_w1 = [float(lut[(curr, model)]["seq_w1"]) for curr, model, _ in order]
    rnd_w1 = [float(lut[(curr, model)]["rand_w1_mean"]) for curr, model, _ in order]
    mae_delta_pct = [
        100.0 * (float(lut[(curr, model)]["rand_mae_mean"]) - float(lut[(curr, model)]["seq_mae"])) / float(lut[(curr, model)]["seq_mae"])
        for curr, model, _ in order
    ]

    return {
        "title": "Single-Current Split Contract Audit",
        "badge": "Primary evidence",
        "intro": "These runs compare sequential and random split behavior on the two audited HH current slices while keeping the single-current direct task fixed.",
        "how_to_read": "Read the MAE chart as paired point-accuracy sensitivity to split policy. Read the Wasserstein chart separately, because marginal fidelity can prefer a different classical family than paired point accuracy. Read the delta chart only as a split-sensitivity summary, not as a family ranking by itself.",
        "main_conclusion": "Split choice does not rescue the direct neural row or overturn the classical advantage. The direct DNN row remains stable and much worse than the classical rows on both current slices. Within the classical lane, random forest is the paired point-accuracy leader, while Extra Trees retains the lower empirical marginal Wasserstein.",
        "cards": [
            {
                "title": "Current direct audit snapshot",
                "body": "At current `0.1`, the direct DNN row stays near MAE `915.4-916.3`, while the classical rows stay near `497-509`. At current `0.3`, the direct DNN row stays near `918.0-920.7`, while the classical rows stay near `492-509`."
            },
            {
                "title": "Provisional caveat",
                "body": "The split audit has completed 27 of 30 rows. The remaining gap is only the last three classical `0.1` random retry rows, so the DNN-versus-classical separation is already stable, but the exact `0.1` classical random means are still slightly provisional."
            }
        ],
        "charts": [
            {
                "title": "Sequential versus random clean MAE by family and current slice",
                "subtitle": "Lower is better. Random bars report the mean across completed random-split rows.",
                "type": "grouped_bar",
                "unit": "MAE",
                "lower_is_better": True,
                "categories": [label for _, _, label in order],
                "series": [
                    {"label": "Sequential", "values": seq_mae, "color": "#6a4fb3"},
                    {"label": "Random mean", "values": rnd_mae, "color": "#1f6f78"}
                ],
                "notes": [
                    "The direct DNN row is nearly unchanged under split choice on both currents.",
                    "The classical rows remain much stronger than the direct DNN row under both split policies."
                ]
            },
            {
                "title": "Sequential versus random empirical marginal Wasserstein",
                "subtitle": "Lower is better. This chart answers a marginal-fidelity question rather than a paired point-accuracy question.",
                "type": "grouped_bar",
                "unit": "W1",
                "lower_is_better": True,
                "categories": [label for _, _, label in order],
                "series": [
                    {"label": "Sequential", "values": seq_w1, "color": "#b04a2f"},
                    {"label": "Random mean", "values": rnd_w1, "color": "#2a8a50"}
                ],
                "notes": [
                    "Extra Trees retains the lower empirical marginal Wasserstein in the classical lane on both currents.",
                    "This is why the direct frontier now needs two statements, namely paired point accuracy and marginal fidelity."
                ]
            },
            {
                "title": "Random-minus-sequential MAE change",
                "subtitle": "Negative values mean the random mean improved over the sequential row.",
                "type": "bar",
                "unit": "%",
                "lower_is_better": False,
                "categories": [label for _, _, label in order],
                "series": [
                    {"label": "MAE change", "values": mae_delta_pct, "color": "#8a6f1f"}
                ],
                "notes": [
                    "The direct DNN deltas are effectively zero on both audited currents.",
                    "The largest observed split effect is the `+2.20%` MAE penalty for Extra Trees on current `0.3`, which still does not threaten the DNN-versus-classical separation."
                ]
            }
        ]
    }


def update_execution_thread(summary: dict) -> None:
    for thread in summary["threads"]:
        if thread["title"] == "Execution Configuration and Run Policy":
            thread["badge"] = "Supporting evidence"
            thread["intro"] = "These earlier throughput studies motivated the HH sweep-setting question, but the current HH contract decision is still under audit."
            thread["how_to_read"] = "Read these charts as historical support for the execution question, not as a settled answer for the current single-current HH audit surface."
            thread["main_conclusion"] = "Earlier throughput packages favored grouped single-node execution, but the dashboard now treats that as supporting context until the single-current split and training audits close."
            thread["cards"] = [
                {
                    "title": "Why this thread moved down",
                    "body": "The current bundle now foregrounds the split-contract audit. The older throughput packages still matter, because they define plausible candidate sweep settings, but they no longer outrank the contract audit as the main HH interpretation surface."
                }
            ]
            break


def update_posterior_thread(summary: dict) -> None:
    for thread in summary["threads"]:
        if thread["title"] == "Posterior-Oriented and Representation Follow-Up":
            thread["badge"] = "Boundary and extension"
            thread["intro"] = "These older posterior-oriented rows are retained as descriptive extension context, but they are pending re-tabulation on the same canonical single-current contract."
            thread["how_to_read"] = "Read this section as archived posterior context rather than as a chapter-level winner table. The direct HH interpretation currently comes from the split audit above."
            thread["main_conclusion"] = "Posterior and aligned follow-up rows still broaden method coverage, but the dashboard no longer treats them as current decisive evidence until they are regenerated on the canonical single-current contract."
            if thread.get("cards"):
                thread["cards"].insert(
                    0,
                    {
                        "title": "Current status",
                        "body": "The posterior lane remains informative, but it is descriptive pending reprocessing on the canonical contract. These charts should therefore be read as coverage context, not as the current front-door HH claim."
                    },
                )
            break


def write_support_json(rows: list[dict[str, str]]) -> None:
    payload = {
        "generated_on": "2026-05-12",
        "source_csv": str(SPLIT_SUMMARY_CSV),
        "note": "Derived from the current single-current split-audit summary. Three classical `0.1` random retry rows were still missing when this artifact was generated.",
        "rows": rows,
    }
    SPLIT_AUDIT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    summary = json.loads(DASHBOARD_JSON.read_text(encoding="utf-8"))
    split_rows = load_split_rows()

    update_inventory(summary, split_rows)
    update_reading_guide(summary)
    update_execution_thread(summary)
    update_posterior_thread(summary)
    write_support_json(split_rows)

    new_thread = build_primary_direct_thread(split_rows)
    replaced = False
    for idx, thread in enumerate(summary["threads"]):
        if thread["title"] == "Primary Direct Fixed-Task Comparison":
            summary["threads"][idx] = new_thread
            replaced = True
            break
    if not replaced:
        summary["threads"].insert(2, new_thread)

    DASHBOARD_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(DASHBOARD_JSON)


if __name__ == "__main__":
    main()
