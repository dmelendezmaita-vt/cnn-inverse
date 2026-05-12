#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path("/projects/neuro-collab/code/github_prep/fhn_dnn-1-implementation-in-pytorch-public")
DASHBOARD_JSON = REPO_ROOT / "dashboard" / "data" / "canonical_results.json"
SPLIT_AUDIT_JSON = REPO_ROOT / "dashboard" / "data" / "split_audit_summary_20260512.json"
METRIC_ALIGNMENT_JSON = REPO_ROOT / "dashboard" / "data" / "metric_alignment_summary_20260512.json"
SPLIT_SUMMARY_CSV = Path("/home/dmm96/manuscript_notes/ms/evidence_bundle_20260512/tables/split_audit_progress_summary.csv")
EXPANDED_METRICS_CSV = Path("/home/dmm96/manuscript_notes/ms/metrics_milestone_modified_20260511/expanded_metrics_summary.csv")
ACTIVE_POLICY_CSV = REPO_ROOT / "runs" / "milestone_modified_noncan08_20260511" / "can10_assumption_conditioned_surrogates" / "active_sequential_policy_suite" / "active_sequential_policy_comparison.csv"
ASNPE_MANIFEST_JSON = REPO_ROOT / "runs" / "milestone_modified_noncan08_20260511" / "can10_assumption_conditioned_surrogates" / "asnpe_surrogate" / "asnpe_manifest.json"
SPLIT_RUN_ROOT = REPO_ROOT / "runs" / "thesis_rewrite_20260512"


def load_split_rows() -> list[dict[str, str]]:
    return list(csv.DictReader(SPLIT_SUMMARY_CSV.open(newline="", encoding="utf-8")))


def load_expanded_metric_rows() -> list[dict[str, str]]:
    return list(csv.DictReader(EXPANDED_METRICS_CSV.open(newline="", encoding="utf-8")))


def load_active_policy_rows() -> list[dict[str, str]]:
    rows = list(csv.DictReader(ACTIVE_POLICY_CSV.open(newline="", encoding="utf-8")))
    for row in rows:
        manifest = Path(row["run_dir"]) / "active_sequential_manifest.json"
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            row["runtime_sec"] = str(payload.get("runtime_sec", ""))
    return rows


def split_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["current"], row["model"]): row for row in rows}


def float_or_zero(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def package_stats(root: Path) -> tuple[int, float]:
    files = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            files += 1
            total_bytes += path.stat().st_size
    return files, total_bytes / (1024 ** 3)


def mean_test_nmae(metrics_per_target_csv: Path) -> float | None:
    rows = list(csv.DictReader(metrics_per_target_csv.open(newline="", encoding="utf-8")))
    vals = [
        float(row["nmae_range"])
        for row in rows
        if (row.get("split") == "test" or row.get("group") == "test") and row.get("nmae_range") not in ("", None)
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)


def collect_split_nmae_lookup() -> dict[tuple[str, str, str], float | None]:
    patterns = {
        ("0.1", "direct_dnn", "seq"): "A1_direct_seq_0p1",
        ("0.1", "direct_dnn", "rand"): "A2_direct_rand_0p1_seed_*",
        ("0.1", "extra_trees_500", "seq"): "A1_extra_trees_500_0p1",
        ("0.1", "extra_trees_500", "rand"): "A2_extra_trees_500_0p1_seed_*",
        ("0.1", "random_forest_500", "seq"): "A1_random_forest_500_0p1",
        ("0.1", "random_forest_500", "rand"): "A2_random_forest_500_0p1_seed_*",
        ("0.3", "direct_dnn", "seq"): "A3_direct_seq_0p3",
        ("0.3", "direct_dnn", "rand"): "A4_direct_rand_0p3_seed_*",
        ("0.3", "extra_trees_500", "seq"): "A3_extra_trees_500_0p3",
        ("0.3", "extra_trees_500", "rand"): "A4_extra_trees_500_0p3_seed_*",
        ("0.3", "random_forest_500", "seq"): "A3_random_forest_500_0p3",
        ("0.3", "random_forest_500", "rand"): "A4_random_forest_500_0p3_seed_*",
    }
    lookup: dict[tuple[str, str, str], float | None] = {}
    for key, pattern in patterns.items():
        vals = []
        for run_dir in sorted(SPLIT_RUN_ROOT.glob(pattern)):
            csv_matches = sorted(run_dir.rglob("metrics_per_target.csv"))
            if not csv_matches:
                continue
            value = mean_test_nmae(csv_matches[0])
            if value is not None:
                vals.append(value)
        lookup[key] = (sum(vals) / len(vals)) if vals else None
    return lookup


def metrics_row(
    rows: list[dict[str, str]],
    *,
    batch_id: str,
    run_name: str,
    rule: str,
) -> dict[str, str]:
    for row in rows:
        if row["batch_id"] == batch_id and row["run_name"] == run_name and row["rule"] == rule:
            return row
    raise KeyError((batch_id, run_name, rule))


def best_point_summary_mae(rows: list[dict[str, str]], *, batch_id: str, run_name: str) -> float:
    candidates = [
        row for row in rows
        if row["batch_id"] == batch_id and row["run_name"] == run_name and row["rule"] in {"posterior_mean", "posterior_median"}
    ]
    if not candidates:
        raise KeyError((batch_id, run_name, "posterior_mean|posterior_median"))
    return min(float(row["mae"]) for row in candidates)


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


def build_primary_direct_thread(
    rows: list[dict[str, str]],
    nmae_lookup: dict[tuple[str, str, str], float | None],
) -> dict:
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
    seq_nmae = [float(nmae_lookup[(curr, model, "seq")]) for curr, model, _ in order]
    rnd_nmae = [float(nmae_lookup[(curr, model, "rand")]) for curr, model, _ in order]
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
        "how_to_read": "Read the first three charts as the direct HH metric hierarchy that the manuscript now uses, namely paired MAE, mean range-normalized MAE, and empirical marginal Wasserstein. Read the delta chart only as a split-sensitivity summary, not as a family ranking by itself.",
        "main_conclusion": "Split choice does not rescue the direct neural row or overturn the classical advantage. The direct DNN row remains stable and much worse than the classical rows on both current slices. Within the classical lane, random forest remains the paired point-accuracy leader, while Extra Trees retains the lower empirical marginal Wasserstein.",
        "cards": [
            {
                "title": "Current direct audit snapshot",
                "body": "At current `0.1`, the direct DNN row stays near MAE `915.4-916.3`, while the classical rows stay near `497-509`. At current `0.3`, the direct DNN row stays near `918.0-920.7`, while the classical rows stay near `492-509`."
            },
            {
                "title": "Direct metric rule",
                "body": "The direct HH frontier is no longer read through MAE alone. Paired MAE and mean normalized MAE answer the paired recovery question, while empirical marginal Wasserstein answers whether the predicted population reproduces the held-out parameter marginals in log10 space."
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
                "title": "Sequential versus random mean normalized MAE by family and current slice",
                "subtitle": "Lower is better. This chart asks whether the paired error is large relative to each target's held-out spread.",
                "type": "grouped_bar",
                "unit": "mean nmae range",
                "lower_is_better": True,
                "categories": [label for _, _, label in order],
                "series": [
                    {"label": "Sequential", "values": seq_nmae, "color": "#4457a7"},
                    {"label": "Random mean", "values": rnd_nmae, "color": "#1d8f6e"}
                ],
                "notes": [
                    "The normalized view preserves the same separation, which means the direct DNN row is not only worse in absolute units but also worse relative to target scale.",
                    "The classical rows stay clustered in the lower normalized-error band on both audited currents."
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


def build_posterior_thread(expanded_rows: list[dict[str, str]]) -> dict:
    posterior_representatives = [
        ("snpe_raw", "SNPE raw"),
        ("snpe_median_rule", "SNPE feature-aware"),
        ("snpe_validate_calibrated", "SNPE calibrated"),
        ("fmpe_extension", "FMPE"),
        ("npse_extension", "NPSE"),
    ]
    point_mae = [
        best_point_summary_mae(expanded_rows, batch_id=("sci06_public_extensions" if name in {"fmpe_extension", "npse_extension", "snpe_validate_calibrated"} else "sci04_posterior_followup"), run_name=name)
        for name, _ in posterior_representatives
    ]
    selected_rows = [
        metrics_row(
            expanded_rows,
            batch_id=("sci06_public_extensions" if name in {"fmpe_extension", "npse_extension", "snpe_validate_calibrated"} else "sci04_posterior_followup"),
            run_name=name,
            rule="posterior_selected",
        )
        for name, _ in posterior_representatives
    ]
    coverage_gap = [float(row["cov90_gap"]) for row in selected_rows]
    interval_width = [float(row["width90_mean"]) for row in selected_rows]
    marginal_w1 = [float(row["posterior_truth_marginal_w1_log10"]) for row in selected_rows]
    sliced_w1 = [float(row["posterior_truth_sliced_w1_log10"]) for row in selected_rows]

    meanstd_test = metrics_row(expanded_rows, batch_id="can08_aligned_frameworks", run_name="bayesflow_meanstd", rule="test")
    structured_test = metrics_row(expanded_rows, batch_id="can08_aligned_frameworks", run_name="bayesflow_structured", rule="test")

    return {
        "title": "Posterior-Oriented and Representation Follow-Up",
        "badge": "Boundary and extension",
        "intro": "This posterior lane remains descriptive pending re-tabulation on the canonical single-current contract, but the dashboard now reads it through posterior-specific metrics instead of a generic MAE-only summary.",
        "how_to_read": "Read the first chart only as point-summary context. Read the later charts as the posterior metric family that the manuscript uses, namely coverage gap, interval width, and truth-centered Wasserstein. These rows are still not the front-door HH claim, because they are not yet regenerated on the current canonical contract.",
        "main_conclusion": "Posterior interpretation remains a calibration-and-sharpness tradeoff rather than a single leaderboard. NPSE has the lowest descriptive point-summary MAE in this retained single-current posterior block, but it pays for that with a much wider `90%` interval, while the other posterior families stay undercovered and only modestly separated on truth-centered Wasserstein.",
        "charts": [
            {
                "title": "Posterior point-summary MAE by representative family",
                "subtitle": "Lower is better. This chart is context only, because it collapses each posterior to one point summary.",
                "type": "bar",
                "unit": "MAE",
                "lower_is_better": True,
                "categories": [label for _, label in posterior_representatives],
                "series": [
                    {"label": "Best point-summary MAE", "values": point_mae, "color": "#1f6f78"}
                ],
                "notes": [
                    "NPSE is the strongest descriptive point-summary row by MAE in this retained single-current posterior block.",
                    "Point-summary MAE does not establish posterior quality by itself, because it ignores calibration and posterior concentration."
                ]
            },
            {
                "title": "Posterior 90% coverage gap by selected rule",
                "subtitle": "Lower is better. This chart measures the absolute gap between observed and nominal `90%` coverage.",
                "type": "bar",
                "unit": "coverage gap",
                "lower_is_better": True,
                "categories": [label for _, label in posterior_representatives],
                "series": [
                    {"label": "Coverage gap", "values": coverage_gap, "color": "#6a4fb3"}
                ],
                "notes": [
                    "NPSE has the smallest retained coverage gap in this descriptive block.",
                    "The other retained posterior families remain materially undercovered, which is why the manuscript does not treat point-summary MAE as a sufficient posterior metric."
                ]
            },
            {
                "title": "Posterior 90% interval width by selected rule",
                "subtitle": "Lower is sharper, but width has to be read together with coverage gap.",
                "type": "bar",
                "unit": "interval width",
                "lower_is_better": True,
                "categories": [label for _, label in posterior_representatives],
                "series": [
                    {"label": "Mean interval width", "values": interval_width, "color": "#b04a2f"}
                ],
                "notes": [
                    "NPSE improves the coverage gap only by accepting a much broader posterior interval.",
                    "This is the reason the manuscript treats coverage and width as a coupled posterior metric pair."
                ]
            },
            {
                "title": "Truth-centered posterior Wasserstein by selected rule",
                "subtitle": "Lower is better. Marginal and sliced values answer related but not identical posterior-concentration questions.",
                "type": "grouped_bar",
                "unit": "log10 W1",
                "lower_is_better": True,
                "categories": [label for _, label in posterior_representatives],
                "series": [
                    {"label": "Marginal W1", "values": marginal_w1, "color": "#2a8a50"},
                    {"label": "Sliced W1", "values": sliced_w1, "color": "#8a6f1f"}
                ],
                "notes": [
                    "The retained single-current posterior families are only modestly separated on truth-centered Wasserstein, which is why the main posterior reading remains a tradeoff rather than a winner-take-all ranking.",
                    "The aligned BayesFlow archive remains the cleaner calibration reference, but it belongs to a different, now non-front-door contract."
                ]
            },
        ],
        "cards": [
            {
                "title": "Posterior metric rule",
                "body": "This dashboard section now follows the manuscript's posterior metric logic. Point-summary MAE is kept only as context, while coverage gap, interval width, and truth-centered posterior Wasserstein carry the substantive interpretation."
            },
            {
                "title": "Archived aligned BayesFlow context",
                "body": "In the archived aligned BayesFlow surface, `bayesflow_meanstd` reaches test MAE `597.7`, coverage gap `0.0198`, interval width `1757.8`, and truth-centered marginal and sliced Wasserstein `0.886` and `0.951`, while `bayesflow_structured` raises MAE to `617.3` but narrows the coverage gap to `0.0107` and lowers empirical marginal Wasserstein to `0.568`. That calibration-sensitive tradeoff still matters interpretively, but it does not define the current admissible HH front door."
            },
            {
                "title": "Aligned archive values",
                "body": f"The archived aligned reference rows currently retained in the manuscript are `bayesflow_meanstd` test MAE `{float(meanstd_test['mae']):.1f}`, coverage gap `{float(meanstd_test['cov90_gap']):.4f}`, width `{float(meanstd_test['width90_mean']):.1f}`, and truth-centered marginal and sliced Wasserstein `{float(meanstd_test['posterior_truth_marginal_w1_log10']):.3f}` and `{float(meanstd_test['posterior_truth_sliced_w1_log10']):.3f}`. The structured aligned row reaches MAE `{float(structured_test['mae']):.1f}`, coverage gap `{float(structured_test['cov90_gap']):.4f}`, width `{float(structured_test['width90_mean']):.1f}`, and empirical marginal Wasserstein `{float(structured_test['empirical_marginal_w1_log10']):.3f}`."
            }
        ],
    }


def update_posterior_thread(summary: dict, expanded_rows: list[dict[str, str]]) -> None:
    replacement = build_posterior_thread(expanded_rows)
    for thread in summary["threads"]:
        if thread["title"] == "Posterior-Oriented and Representation Follow-Up":
            thread.clear()
            thread.update(replacement)
            break


def update_extensions_thread(summary: dict, active_rows: list[dict[str, str]], asnpe_manifest: dict) -> None:
    lut = {row["policy"]: row for row in active_rows}
    for thread in summary["threads"]:
        if thread["title"] == "Later Executable Extensions":
            thread["badge"] = "Coverage expansion"
            thread["intro"] = "These later executable extensions stay outside the primary direct HH frontier, but the dashboard now reads the active-sequential branch through the same metrics that the manuscript uses."
            thread["how_to_read"] = "Read mean absolute log10 error as the primary active-sequential score. Read contraction and ESS together, because sharper concentration without effective posterior support is not a sufficient gain. Read runtime only as operational support."
            thread["main_conclusion"] = "Inside the active-sequential branch, disagreement remains the primary error row and also has the strongest posterior contraction, while ESS stays nearly unchanged across the three policies. Wasserstein remains effectively tied on the primary error metric and retains the clearest distribution-aware acquisition interpretation."
            thread["charts"] = [
                {
                    "title": "Active-sequential error metric by representative policy",
                    "subtitle": "Lower is better. This remains the primary active-sequential score.",
                    "type": "bar",
                    "unit": "mean abs log10 error",
                    "lower_is_better": True,
                    "categories": ["Disagreement", "Wasserstein", "Uncertainty", "ASNPE extended"],
                    "series": [
                        {
                            "label": "Error metric",
                            "values": [
                                float(lut["disagreement"]["mean_abs_log10_error_params_mean"]),
                                float(lut["wasserstein"]["mean_abs_log10_error_params_mean"]),
                                float(lut["uncertainty"]["mean_abs_log10_error_params_mean"]),
                                float(asnpe_manifest["posterior_mean_abs_log10_error"]),
                            ],
                            "color": "#1f6f78",
                        }
                    ],
                    "notes": [
                        "Disagreement is the best active-sequential row on the primary error metric.",
                        "Wasserstein is only marginally worse on error, which is why the branch remains an interpretive rather than a decisive operational split."
                    ]
                },
                {
                    "title": "Active-sequential posterior contraction",
                    "subtitle": "Higher indicates tighter concentration under the sequential posterior summary used in this branch.",
                    "type": "bar",
                    "unit": "contraction",
                    "lower_is_better": False,
                    "categories": ["Disagreement", "Wasserstein", "Uncertainty"],
                    "series": [
                        {
                            "label": "Contraction",
                            "values": [
                                float(lut["disagreement"]["posterior_contraction_mean_mean"]),
                                float(lut["wasserstein"]["posterior_contraction_mean_mean"]),
                                float(lut["uncertainty"]["posterior_contraction_mean_mean"]),
                            ],
                            "color": "#6a4fb3",
                        }
                    ],
                    "notes": [
                        "Disagreement has the strongest contraction in the current active-sequential suite.",
                        "The Wasserstein and uncertainty rows remain close to each other on this concentration measure."
                    ]
                },
                {
                    "title": "Active-sequential effective sample size",
                    "subtitle": "Higher indicates slightly better retained posterior support, but the three policies are nearly tied here.",
                    "type": "bar",
                    "unit": "ESS",
                    "lower_is_better": False,
                    "categories": ["Disagreement", "Wasserstein", "Uncertainty"],
                    "series": [
                        {
                            "label": "ESS",
                            "values": [
                                float(lut["disagreement"]["ess_mean"]),
                                float(lut["wasserstein"]["ess_mean"]),
                                float(lut["uncertainty"]["ess_mean"]),
                            ],
                            "color": "#2a8a50",
                        }
                    ],
                    "notes": [
                        "ESS is nearly unchanged across the three active-sequential policies.",
                        "This is why the active-sequential interpretation does not reduce to an ESS-driven distinction."
                    ]
                },
                {
                    "title": "Active-sequential runtime by representative policy",
                    "subtitle": "Lower is better. Runtime remains secondary to the active-sequential metric logic.",
                    "type": "bar",
                    "unit": "seconds",
                    "lower_is_better": True,
                    "categories": ["Disagreement", "Wasserstein", "Uncertainty", "ASNPE extended"],
                    "series": [
                        {
                            "label": "Runtime",
                            "values": [
                                float(lut["disagreement"]["runtime_sec"]),
                                float(lut["wasserstein"]["runtime_sec"]),
                                float(lut["uncertainty"]["runtime_sec"]),
                                float(asnpe_manifest["runtime_sec"]),
                            ],
                            "color": "#b04a2f",
                        }
                    ],
                    "notes": [
                        "The three active-sequential policies stay in the same runtime band.",
                        "ASNPE is no longer a runtime outlier, but it remains materially weaker on the primary active-sequential error metric."
                    ]
                }
            ]
            thread["cards"] = [
                {
                    "title": "Metric hierarchy inside the active-sequential branch",
                    "body": "The manuscript treats this branch as a separate inferential object. Mean absolute log10 parameter error remains the primary score, while contraction and ESS clarify whether similar error levels conceal different posterior concentration behavior."
                },
                {
                    "title": "Wasserstein versus disagreement",
                    "body": "The Wasserstein policy remains within `0.31%` of disagreement on the primary error metric, while ESS stays essentially unchanged and disagreement retains the stronger contraction. The practical choice is therefore mostly interpretive, because Wasserstein keeps the clearest distance-aware acquisition meaning."
                }
            ]
            break


def write_metric_alignment_json(
    *,
    nmae_lookup: dict[tuple[str, str, str], float | None],
    expanded_rows: list[dict[str, str]],
    active_rows: list[dict[str, str]],
    asnpe_manifest: dict,
) -> None:
    payload = {
        "generated_on": "2026-05-12",
        "direct_metric_hierarchy": {
            "primary_metrics": ["mae", "mean_nmae_range", "empirical_marginal_wasserstein_log10"],
            "split_audit_mean_nmae_range": {
                f"{curr}:{model}:{split_kind}": value
                for (curr, model, split_kind), value in sorted(nmae_lookup.items())
            },
        },
        "posterior_metric_hierarchy": {
            "context_point_summary": ["mae"],
            "primary_posterior_metrics": ["cov90_gap", "width90_mean", "posterior_truth_marginal_w1_log10", "posterior_truth_sliced_w1_log10"],
            "aligned_archive_reference": {
                "bayesflow_meanstd_test": metrics_row(expanded_rows, batch_id="can08_aligned_frameworks", run_name="bayesflow_meanstd", rule="test"),
                "bayesflow_structured_test": metrics_row(expanded_rows, batch_id="can08_aligned_frameworks", run_name="bayesflow_structured", rule="test"),
            },
        },
        "active_metric_hierarchy": {
            "primary_metric": "mean_abs_log10_error_params_mean",
            "support_metrics": ["posterior_contraction_mean_mean", "ess_mean", "runtime_sec"],
            "active_policy_rows": active_rows,
            "asnpe_manifest": {
                "posterior_mean_abs_log10_error": asnpe_manifest["posterior_mean_abs_log10_error"],
                "runtime_sec": asnpe_manifest["runtime_sec"],
            },
        },
    }
    METRIC_ALIGNMENT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_support_json(rows: list[dict[str, str]]) -> None:
    payload = {
        "generated_on": "2026-05-12",
        "source_csv": str(SPLIT_SUMMARY_CSV),
        "note": "Derived from the current single-current split-audit summary. Three classical `0.1` random retry rows were still missing when this artifact was generated.",
        "rows": rows,
    }
    SPLIT_AUDIT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def dedupe_threads(summary: dict) -> None:
    seen_titles: set[str] = set()
    unique_threads = []
    for thread in summary["threads"]:
        title = thread.get("title", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        unique_threads.append(thread)
    summary["threads"] = unique_threads


def main() -> None:
    summary = json.loads(DASHBOARD_JSON.read_text(encoding="utf-8"))
    split_rows = load_split_rows()
    expanded_rows = load_expanded_metric_rows()
    active_rows = load_active_policy_rows()
    asnpe_manifest = json.loads(ASNPE_MANIFEST_JSON.read_text(encoding="utf-8"))
    nmae_lookup = collect_split_nmae_lookup()

    update_inventory(summary, split_rows)
    update_reading_guide(summary)
    update_execution_thread(summary)
    update_posterior_thread(summary, expanded_rows)
    update_extensions_thread(summary, active_rows, asnpe_manifest)
    write_support_json(split_rows)
    write_metric_alignment_json(
        nmae_lookup=nmae_lookup,
        expanded_rows=expanded_rows,
        active_rows=active_rows,
        asnpe_manifest=asnpe_manifest,
    )

    new_thread = build_primary_direct_thread(split_rows, nmae_lookup)
    replaced = False
    for idx, thread in enumerate(summary["threads"]):
        if thread["title"] in {"Primary Direct Fixed-Task Comparison", "Single-Current Split Contract Audit"}:
            summary["threads"][idx] = new_thread
            replaced = True
            break
    if not replaced:
        summary["threads"].insert(2, new_thread)

    dedupe_threads(summary)

    DASHBOARD_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(DASHBOARD_JSON)


if __name__ == "__main__":
    main()
