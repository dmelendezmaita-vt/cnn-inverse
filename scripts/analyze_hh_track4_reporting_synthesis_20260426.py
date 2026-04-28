#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
IMPORTANT = REPO / "data" / "important_notes"
OUT_ROOT = IMPORTANT / "optimization_track_20260426_hh_track4_reporting_synthesis"
TABLES = OUT_ROOT / "tables"
REPORTS = OUT_ROOT / "reports"

EFF = IMPORTANT / "optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency" / "tables" / "hh_track4_checkpoint7_standardized_efficiency_final_representatives_20260423.csv"
RUNALL = IMPORTANT / "optimization_track_20260425_hh_track4_run_all_literature_plan"
SURR = IMPORTANT / "optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing row {key}={value}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def build_key_results() -> list[dict[str, object]]:
    eff_rows = read_csv(EFF)
    framework_rows = sorted(read_csv(RUNALL / "tables" / "hh_track4_framework_snapshot_comparison_20260425.csv"), key=lambda r: metric(r, "mae"))
    decision_rows = sorted(read_csv(RUNALL / "tables" / "hh_track4_posterior_decision_rule_comparison_20260425.csv"), key=lambda r: metric(r, "mae"))
    hybrid_rows = sorted(read_csv(SURR / "tables" / "hh_track4_hybrid_initializer_comparison_20260425.csv"), key=lambda r: metric(r, "objective"))
    active_rows = read_csv(SURR / "tables" / "hh_track4_active_sequential_policy_comparison_20260425.csv")
    lc2st_rows = sorted(read_csv(RUNALL / "tables" / "hh_track4_posterior_lc2st_nf_20260426.csv"), key=lambda r: metric(r, "local_abs_dev_mean"))
    classifier_rows = sorted(read_csv(RUNALL / "tables" / "hh_track4_posterior_conditioned_classifier_20260426.csv"), key=lambda r: metric(r, "classifier_auc"))
    rank_rows = sorted(read_csv(RUNALL / "tables" / "hh_track4_posterior_rank_diagnostics_20260426.csv"), key=lambda r: metric(r, "rank_ks_mean"))

    simformer_primary = load_json(SURR / "runs" / "simformer" / "simformer_official_hh_primary_20260426" / "simformer_hh_metrics_summary.json")
    simformer_secondary = load_json(SURR / "runs" / "simformer" / "simformer_official_hh_secondary_20260426" / "simformer_hh_metrics_summary.json")
    gb_manifest = load_json(SURR / "runs" / "generalized_bayes_npe" / "generalized_bayes_npe_main_20260426" / "generalized_bayes_npe_manifest.json")
    asnpe_primary = load_json(SURR / "runs" / "active_sequential" / "asnpe_primary_v100_20260426" / "asnpe_manifest.json")

    rows: list[dict[str, object]] = []
    def add(section: str, name: str, comparability: str, metric_name: str, metric_value: object, source: str, note: str) -> None:
        rows.append(
            {
                "section": section,
                "name": name,
                "comparability_class": comparability,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "source": source,
                "note": note,
            }
        )

    # Fixed Track4 frontier
    for representative in [
        "random_forest_500",
        "knn_k11",
        "sbi_fmpe_clean",
        "sbi_feature_aware_snpe_clean",
        "sbi_snpe_matched_drift010",
        "sbi_snpe_matched_mask20",
        "sbi_npse_clean_not_promoted",
        "effnet_beta005_invvar_e500",
    ]:
        row = find_row(eff_rows, "representative", representative)
        add("fixed_track4_frontier", representative, "direct_track4_fixed_array", "mae_mean", row["mae_mean"], str(EFF), row["role"])
        add("fixed_track4_frontier", representative, "direct_track4_fixed_array", "r2_mean", row["r2_mean"], str(EFF), row["role"])

    # Literature posterior snapshot
    for row in framework_rows[:10]:
        add("aligned_framework_snapshot", row["run_name"], "assumption_conditioned_aligned_multicurrent", "mae", row["mae"], str(RUNALL / "tables" / "hh_track4_framework_snapshot_comparison_20260425.csv"), row.get("framework", ""))
        add("aligned_framework_snapshot", row["run_name"], "assumption_conditioned_aligned_multicurrent", "r2", row["r2"], str(RUNALL / "tables" / "hh_track4_framework_snapshot_comparison_20260425.csv"), row.get("framework", ""))

    # Best decision-rule rows
    for row in decision_rows[:12]:
        add("posterior_decision_rules", f"{row['run_name']}::{row['decision_rule']}", "assumption_conditioned_aligned_multicurrent", "mae", row["mae"], str(RUNALL / "tables" / "hh_track4_posterior_decision_rule_comparison_20260425.csv"), row["family"])
        add("posterior_decision_rules", f"{row['run_name']}::{row['decision_rule']}", "assumption_conditioned_aligned_multicurrent", "r2", row["r2"], str(RUNALL / "tables" / "hh_track4_posterior_decision_rule_comparison_20260425.csv"), row["family"])

    # Hybrid top rows and explicit pathological tail
    for row in hybrid_rows[:20]:
        add("hybrid_refinement", row["run_name"], "assumption_conditioned_surrogate_direct_fitting", "objective", row["objective"], str(SURR / "tables" / "hh_track4_hybrid_initializer_comparison_20260425.csv"), row["initializer"])
        add("hybrid_refinement", row["run_name"], "assumption_conditioned_surrogate_direct_fitting", "mean_trace_rmse_z", row["mean_trace_rmse_z"], str(SURR / "tables" / "hh_track4_hybrid_initializer_comparison_20260425.csv"), row["initializer"])
    for row in hybrid_rows[-4:]:
        add("hybrid_refinement_pathology", row["run_name"], "assumption_conditioned_surrogate_direct_fitting", "objective", row["objective"], str(SURR / "tables" / "hh_track4_hybrid_initializer_comparison_20260425.csv"), "pathological tail")

    # Sequential design
    for row in active_rows:
        label = row["run_name"]
        add("active_design", label, "assumption_conditioned_active_design", "mean_abs_log10_error_params_mean", row["mean_abs_log10_error_params_mean"], str(SURR / "tables" / "hh_track4_active_sequential_policy_comparison_20260425.csv"), row["acquisition_policy"] or "legacy_default")
        add("active_design", label, "assumption_conditioned_active_design", "runtime_sec", row["runtime_sec"], str(SURR / "tables" / "hh_track4_active_sequential_policy_comparison_20260425.csv"), row["acquisition_policy"] or "legacy_default")
    add("active_design", "asnpe_primary_v100_20260426", "assumption_conditioned_active_design_exact_asnpe", "mean_abs_log10_error_params_mean", asnpe_primary["aggregate_metrics"]["mean_abs_log10_error_params_mean"], str(SURR / "runs" / "active_sequential" / "asnpe_primary_v100_20260426" / "asnpe_manifest.json"), "exact-source ASNPE branch")

    # Diagnostics
    for row in lc2st_rows:
        add("diagnostics_lc2st_nf", row["run_name"], "exact_local_flow_diagnostic", "local_abs_dev_mean", row["local_abs_dev_mean"], str(RUNALL / "tables" / "hh_track4_posterior_lc2st_nf_20260426.csv"), "lower is better")
        add("diagnostics_lc2st_nf", row["run_name"], "exact_local_flow_diagnostic", "classifier_auc", row["classifier_auc"], str(RUNALL / "tables" / "hh_track4_posterior_lc2st_nf_20260426.csv"), "lower is better")
    for row in classifier_rows:
        add("diagnostics_conditioned_classifier", row["run_name"], "approximate_conditioned_classifier", "classifier_auc", row["classifier_auc"], str(RUNALL / "tables" / "hh_track4_posterior_conditioned_classifier_20260426.csv"), "lower is better")
    for row in rank_rows:
        add("diagnostics_rank", row["run_name"], "posterior_rank_diagnostic", "rank_ks_mean", row["rank_ks_mean"], str(RUNALL / "tables" / "hh_track4_posterior_rank_diagnostics_20260426.csv"), "lower is better")
        add("diagnostics_rank", row["run_name"], "posterior_rank_diagnostic", "coverage_90_mean", row["coverage_90_mean"], str(RUNALL / "tables" / "hh_track4_posterior_rank_diagnostics_20260426.csv"), "closer to nominal is better")

    # New literature branches
    for label, obj in [("simformer_primary", simformer_primary), ("simformer_secondary", simformer_secondary)]:
        add("new_literature_branch", label, "upstream_hh_benchmark_not_directly_comparable", "mae", obj["test_metrics"]["mae"], str(SURR / "reports" / "hh_track4_simformer_branch_20260426.md"), "official upstream HH benchmark")
        add("new_literature_branch", label, "upstream_hh_benchmark_not_directly_comparable", "r2", obj["test_metrics"]["r2"], str(SURR / "reports" / "hh_track4_simformer_branch_20260426.md"), "official upstream HH benchmark")
    for beta_row in gb_manifest["beta_summary"]:
        add("new_literature_branch", f"generalized_bayes_beta_{beta_row['beta']:.2f}", "assumption_conditioned_local_adaptation_not_exact_paper_code", "mean_abs_log10_error_params_mean", beta_row["mean_abs_log10_error_params_mean"], str(SURR / "reports" / "hh_track4_generalized_bayes_npe_branch_20260426.md"), "bounded local adaptation")
    return rows


def build_limitations() -> list[dict[str, object]]:
    rows = [
        {
            "limitation_id": "L1",
            "scope": "comparability",
            "statement": "Do not mix direct Track4 fixed-array results, assumption-conditioned aligned multi-current results, surrogate direct-fitting results, active-design surrogate results, and upstream Simformer HH benchmark results into a single ranked table without explicit comparability labels.",
            "severity": "high",
        },
        {
            "limitation_id": "L2",
            "scope": "provenance",
            "statement": "Direct fitting to the real Track4 generator remains blocked by missing simulator provenance, parameter semantics, and protocol metadata. Surrogate direct-fitting results are exploratory and assumption-conditioned.",
            "severity": "high",
        },
        {
            "limitation_id": "L3",
            "scope": "alignment_assumption",
            "statement": "The aligned multi-current BayesFlow and Swyft branches rely on the approved row-alignment assumption across current-specific files, which is not recovered provenance.",
            "severity": "high",
        },
        {
            "limitation_id": "L4",
            "scope": "diagnostics",
            "statement": "Exact L-C2ST-NF was implemented only for BayesFlow flow posteriors. Swyft diagnostics remain represented by conditioned classifiers and posterior-rank checks, not the exact flow-specific L-C2ST construction.",
            "severity": "medium",
        },
        {
            "limitation_id": "L5",
            "scope": "asnpe",
            "statement": "The exact ASNPE branch approximates p(phi|D) with a bootstrap deep ensemble of NDEs rather than a closed-form Bayesian posterior over network parameters.",
            "severity": "medium",
        },
        {
            "limitation_id": "L6",
            "scope": "generalized_bayes",
            "statement": "The generalized-Bayes NPE test is a bounded local adaptation based on SNIS-style tempered Gibbs weights over a surrogate simulation bank, not the authors' exact released implementation.",
            "severity": "medium",
        },
        {
            "limitation_id": "L7",
            "scope": "simformer",
            "statement": "The Simformer runs use the official upstream HH task and codepath, but they are not directly comparable to the local Track4 fixed-array workflow because the simulator, observation contract, and benchmark target differ.",
            "severity": "high",
        },
        {
            "limitation_id": "L8",
            "scope": "tmnre",
            "statement": "Rectangular TMNRE truncation was a demonstrated dead-end in this workspace, because it repeatedly retained the full bank. The score-pruned variant did run and was informative.",
            "severity": "medium",
        },
        {
            "limitation_id": "L9",
            "scope": "decision_rules",
            "statement": "Several Swyft mean-decoder rows are pathological and should be reported explicitly as unstable or degenerate, not silently pooled into average performance summaries.",
            "severity": "high",
        },
        {
            "limitation_id": "L10",
            "scope": "systems",
            "statement": "Systems-efficiency numbers are standardized enough for reporting, but hardware, training-set size, evaluation-set size, and posterior-sample counts differ across method families, so cross-family runtime claims must state those contracts.",
            "severity": "medium",
        },
    ]
    return rows


def build_report(key_rows: list[dict[str, object]], limitation_rows: list[dict[str, object]]) -> str:
    def section_rows(section: str) -> list[dict[str, object]]:
        return [row for row in key_rows if row["section"] == section]

    lines = [
        "# HH Track4 Consolidated Reporting Analysis",
        "",
        "## Scope",
        "",
        "This synthesis consolidates the fixed Track4 closure package, the later literature-backed native framework sweep, the assumption-conditioned surrogate direct-fitting and active-design branches, the exact BayesFlow flow diagnostics, and the later Simformer and generalized-Bayes additions.",
        "",
        "## Comparability Classes",
        "",
        "| class | meaning | reporting rule |",
        "|---|---|---|",
        "| `direct_track4_fixed_array` | original Track4 fixed supervised inverse workload | safe for headline task claims |",
        "| `assumption_conditioned_aligned_multicurrent` | aligned multi-current inference using the approved row-alignment assumption | report separately from the direct fixed-array frontier |",
        "| `assumption_conditioned_surrogate_direct_fitting` | surrogate compact-HH direct-fitting and hybrid refinement | report as exploratory surrogate evidence only |",
        "| `assumption_conditioned_active_design` | surrogate active acquisition policies and ASNPE-style loops | report as active-design evidence, not direct Track4 winner evidence |",
        "| `upstream_hh_benchmark_not_directly_comparable` | official upstream HH benchmark from another repo | do not compare numerically to local Track4 winners without a large disclaimer |",
        "",
        "## Core Findings",
        "",
        "| result space | best current row | key metric | note |",
        "|---|---|---:|---|",
        "| direct Track4 clean overall | `random_forest_500` | `MAE 441.033` | best clean fixed-array point estimator |",
        "| direct Track4 clean SBI | `sbi_fmpe_clean` | `MAE 582.303` | best clean fixed-array SBI representative |",
        "| direct Track4 robust drift | `sbi_snpe_matched_drift010` | `MAE 676.306` | best localized robust SBI path for `drift010` |",
        "| direct Track4 robust mask | `sbi_snpe_matched_mask20` | `MAE 618.976` | best localized robust SBI path for `mask20` |",
        "| aligned posterior decision rule | `bayesflow_set_posterior_v100_20260426::median` | `MAE 565.827` | best local aligned posterior decision-rule row by MAE |",
        "| hybrid surrogate initializer | `swyft_tmnre_score_prune_structured_v100_20260426_map_bank` | `objective 2.626042` | best surrogate direct-fitting initializer |",
        "| exact local BayesFlow diagnostic | `bayesflow_native_structured_currfusion_v100_20260425` | `L-C2ST-NF local_abs_dev_mean 0.062621` | strongest exact local flow-diagnostic row |",
        "| official upstream Simformer HH | `simformer_official_hh_secondary_20260426` | `R2 0.967805` | official upstream benchmark, not directly comparable |",
        "",
        "## Direct Track4 Fixed-Array Frontier",
        "",
        "| representative | role | MAE | MSE | R2 | source |",
        "|---|---|---:|---:|---:|---|",
    ]

    direct_rows = [row for row in key_rows if row["section"] == "fixed_track4_frontier" and row["metric_name"] == "mae_mean"]
    direct_r2 = {row["name"]: row["metric_value"] for row in key_rows if row["section"] == "fixed_track4_frontier" and row["metric_name"] == "r2_mean"}
    for row in direct_rows:
        lines.append(
            f"| `{row['name']}` | `{row['note']}` | {float(row['metric_value']):.6f} |  | {float(direct_r2[row['name']]):.6f} | `{row['source']}` |"
        )

    lines += [
        "",
        "## Aligned Literature Sweep",
        "",
        "| run | MAE | R2 | interpretation |",
        "|---|---:|---:|---|",
    ]
    snapshot_rows = [row for row in key_rows if row["section"] == "aligned_framework_snapshot" and row["metric_name"] == "mae"]
    snapshot_r2 = {row["name"]: row["metric_value"] for row in key_rows if row["section"] == "aligned_framework_snapshot" and row["metric_name"] == "r2"}
    for row in snapshot_rows[:10]:
        lines.append(f"| `{row['name']}` | {float(row['metric_value']):.6f} | {float(snapshot_r2[row['name']]):.6f} | `{row['note']}` |")

    lines += [
        "",
        "## Decision Rules And Hybrid Utility",
        "",
        "| top hybrid initializer | objective | mean_trace_rmse_z | note |",
        "|---|---:|---:|---|",
    ]
    hybrid_rows = [row for row in key_rows if row["section"] == "hybrid_refinement" and row["metric_name"] == "objective"]
    hybrid_rmse = {row["name"]: row["metric_value"] for row in key_rows if row["section"] == "hybrid_refinement" and row["metric_name"] == "mean_trace_rmse_z"}
    for row in hybrid_rows[:15]:
        lines.append(f"| `{row['name']}` | {float(row['metric_value']):.6f} | {float(hybrid_rmse[row['name']]):.6f} | `{row['note']}` |")

    lines += [
        "",
        "Notable pathological or near-degenerate hybrid seeds are present at the bottom of the table and should be reported as such, rather than averaged into family summaries.",
        "",
        "## Sequential And Active Design",
        "",
        "| run | policy | mean_abs_log10_error_params_mean | runtime_sec | note |",
        "|---|---|---:|---:|---|",
    ]
    active_rows = [row for row in key_rows if row["section"] == "active_design" and row["metric_name"] == "mean_abs_log10_error_params_mean"]
    active_runtime = {row["name"]: row["metric_value"] for row in key_rows if row["section"] == "active_design" and row["metric_name"] == "runtime_sec"}
    for row in active_rows:
        lines.append(f"| `{row['name']}` | `{row['note']}` | {float(row['metric_value']):.6f} | {float(active_runtime[row['name']]):.6f} | `{row['comparability_class']}` |")

    lines += [
        "",
        "Interpretation: the heuristic active-design family is tightly clustered around `0.61` mean absolute log10 parameter error, while the exact ASNPE-style branch completed but did not improve on that surrogate metric under the tested budget.",
        "",
        "## Diagnostics",
        "",
        "| diagnostic family | strongest row | key metric | note |",
        "|---|---|---:|---|",
    ]
    lc_rows = [row for row in key_rows if row["section"] == "diagnostics_lc2st_nf" and row["metric_name"] == "local_abs_dev_mean"]
    class_rows = [row for row in key_rows if row["section"] == "diagnostics_conditioned_classifier" and row["metric_name"] == "classifier_auc"]
    rank_rows = [row for row in key_rows if row["section"] == "diagnostics_rank" and row["metric_name"] == "rank_ks_mean"]
    if lc_rows:
        lines.append(f"| `L-C2ST-NF` | `{lc_rows[0]['name']}` | {float(lc_rows[0]['metric_value']):.6f} | exact BayesFlow flow-local diagnostic, lower is better |")
    if class_rows:
        lines.append(f"| conditioned classifier | `{class_rows[0]['name']}` | {float(class_rows[0]['metric_value']):.6f} | broader approximate separability check, lower is better |")
    if rank_rows:
        lines.append(f"| posterior rank KS | `{rank_rows[0]['name']}` | {float(rank_rows[0]['metric_value']):.6f} | lower is better |")

    lines += [
        "",
        "## New Literature Branches",
        "",
        "| method | status | reading |",
        "|---|---|---|",
        "| `Simformer` | completed | official upstream HH benchmark succeeded under reduced budget, but is not directly comparable to local Track4 claims |",
        "| generalized-Bayes NPE | completed | bounded local adaptation completed, but the tested branch is strongly negative and not competitive |",
        "",
        "## Limitations And Assumptions",
        "",
        "| id | scope | severity | statement |",
        "|---|---|---|---|",
    ]
    for row in limitation_rows:
        lines.append(f"| `{row['limitation_id']}` | `{row['scope']}` | `{row['severity']}` | {row['statement']} |")

    lines += [
        "",
        "## Reporting Guidance",
        "",
        "1. Use the direct fixed-array frontier for headline Track4 claims.",
        "2. Keep the aligned multi-current literature sweep in its own section, clearly labeled as assumption-conditioned.",
        "3. Keep surrogate direct-fitting and active-design results separate from the direct Track4 leaderboard.",
        "4. Treat exact `L-C2ST-NF` and posterior-rank diagnostics as calibration evidence, not as stand-alone accuracy winners.",
        "5. Report `Simformer` only as an official upstream HH benchmark reference, not as a directly comparable Track4 replacement.",
        "6. Report generalized-Bayes NPE as a negative or inconclusive bounded adaptation, unless a stronger source-faithful implementation changes that result.",
        "",
        "## Canonical Sources",
        "",
        f"- direct frontier systems table: `{EFF}`",
        f"- framework snapshot: `{RUNALL / 'tables' / 'hh_track4_framework_snapshot_comparison_20260425.csv'}`",
        f"- decision rules: `{RUNALL / 'tables' / 'hh_track4_posterior_decision_rule_comparison_20260425.csv'}`",
        f"- hybrid comparison: `{SURR / 'tables' / 'hh_track4_hybrid_initializer_comparison_20260425.csv'}`",
        f"- active design comparison: `{SURR / 'tables' / 'hh_track4_active_sequential_policy_comparison_20260425.csv'}`",
        f"- exact L-C2ST-NF: `{RUNALL / 'tables' / 'hh_track4_posterior_lc2st_nf_20260426.csv'}`",
        f"- branch registry: `{RUNALL / 'tables' / 'hh_track4_literature_branch_registry_20260426.csv'}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    key_rows = build_key_results()
    limitation_rows = build_limitations()
    write_csv(TABLES / "hh_track4_reporting_key_results_20260426.csv", key_rows)
    write_csv(TABLES / "hh_track4_reporting_limitations_20260426.csv", limitation_rows)
    report = build_report(key_rows, limitation_rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "hh_track4_reporting_analysis_20260426.md").write_text(report)
    print(REPORTS / "hh_track4_reporting_analysis_20260426.md")


if __name__ == "__main__":
    main()
