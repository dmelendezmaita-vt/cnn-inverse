#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


REPO = Path("/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch")
RUNALL = REPO / "data/important_notes/optimization_track_20260425_hh_track4_run_all_literature_plan"
SURR = REPO / "data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle"


def status_for(path: Path, *, completed: str = "completed", missing: str = "pending") -> str:
    return completed if path.exists() else missing


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = [
        {
            "branch": "rectangular_tmnre_truncation",
            "source": "TMNRE NeurIPS 2021",
            "url": "https://papers.nips.cc/paper_files/paper/2021/file/01632f7b7a127233fa1188bd6c2e42e1-Paper.pdf",
            "status": "exhausted_dead_end",
            "evidence": "schedule_matrix_and_structured_stage1_retained_full_bank",
        },
        {
            "branch": "score_pruned_tmnre",
            "source": "TMNRE alternative pruning implementation",
            "url": "https://papers.nips.cc/paper_files/paper/2021/file/01632f7b7a127233fa1188bd6c2e42e1-Paper.pdf",
            "status": status_for(RUNALL / "runs/swyft_tmnre_score_prune_structured_v100_20260426/metrics_summary.json"),
            "evidence": "swyft_tmnre_score_prune_structured_v100_20260426",
        },
        {
            "branch": "bayesflow_time_series_network",
            "source": "BayesFlow TimeSeriesNetwork docs",
            "url": "https://bayesflow.org/main/api/bayesflow.networks.TimeSeriesNetwork.html",
            "status": status_for(RUNALL / "runs/bayesflow_temporal_v100_20260426/metrics_summary.json"),
            "evidence": "bayesflow_temporal_v100_20260426",
        },
        {
            "branch": "bayesflow_time_series_transformer",
            "source": "BayesFlow TimeSeriesTransformer docs",
            "url": "https://bayesflow.org/main/api/bayesflow.networks.TimeSeriesTransformer.html",
            "status": status_for(RUNALL / "runs/bayesflow_temporal_transformer_v100_20260426/metrics_summary.json"),
            "evidence": "bayesflow_temporal_transformer_v100_20260426",
        },
        {
            "branch": "bayesflow_fusion_transformer",
            "source": "BayesFlow FusionTransformer docs",
            "url": "https://bayesflow.org/main/api/bayesflow.networks.FusionTransformer.html",
            "status": status_for(RUNALL / "runs/bayesflow_fusion_transformer_v100_20260426/metrics_summary.json"),
            "evidence": "bayesflow_fusion_transformer_v100_20260426",
        },
        {
            "branch": "bayesflow_set_transformer",
            "source": "BayesFlow SetTransformer docs",
            "url": "https://bayesflow.org/main/api/bayesflow.networks.SetTransformer.html",
            "status": status_for(RUNALL / "runs/bayesflow_set_posterior_v100_20260426/metrics_summary.json"),
            "evidence": "bayesflow_set_posterior_v100_20260426",
        },
        {
            "branch": "bayesflow_deep_set",
            "source": "BayesFlow DeepSet docs and Deep Sets paper",
            "url": "https://bayesflow.org/main/api/bayesflow.networks.DeepSet.html",
            "status": status_for(RUNALL / "runs/bayesflow_deepset_posterior_v100_20260426/metrics_summary.json"),
            "evidence": "bayesflow_deepset_posterior_v100_20260426",
        },
        {
            "branch": "hierarchical_set_posterior_family",
            "source": "HNPE arXiv 2021",
            "url": "https://arxiv.org/abs/2102.06477",
            "status": "covered_by_set_models"
            if (RUNALL / "runs/bayesflow_set_posterior_v100_20260426/metrics_summary.json").exists()
            and (RUNALL / "runs/bayesflow_deepset_posterior_v100_20260426/metrics_summary.json").exists()
            else "pending",
            "evidence": "set_transformer_and_deepset",
        },
        {
            "branch": "conditioned_classifier_diagnostics",
            "source": "L-C2ST NeurIPS 2023",
            "url": "https://papers.nips.cc/paper_files/paper/2023/file/b0313c2f4501a81d0e0d4a1e8fbf4995-Paper-Conference.pdf",
            "status": status_for(RUNALL / "tables/hh_track4_posterior_conditioned_classifier_20260426.csv", completed="expanded_completed"),
            "evidence": "hh_track4_posterior_conditioned_classifier_20260426.csv",
        },
        {
            "branch": "posterior_rank_diagnostics",
            "source": "SBC JRSSB 2020 and posterior rank checks",
            "url": "https://arxiv.org/abs/1804.06788",
            "status": status_for(RUNALL / "tables/hh_track4_posterior_rank_diagnostics_20260426.csv", completed="expanded_completed"),
            "evidence": "hh_track4_posterior_rank_diagnostics_20260426.csv",
        },
        {
            "branch": "lc2st_nf_exact",
            "source": "L-C2ST NeurIPS 2023",
            "url": "https://papers.nips.cc/paper_files/paper/2023/file/b0313c2f4501a81d0e0d4a1e8fbf4995-Paper-Conference.pdf",
            "status": status_for(RUNALL / "tables/hh_track4_posterior_lc2st_nf_20260426.csv"),
            "evidence": "hh_track4_posterior_lc2st_nf_20260426.csv",
        },
        {
            "branch": "asnpe_exact_loop",
            "source": "ASNPE NeurIPS 2024",
            "url": "https://proceedings.neurips.cc/paper_files/paper/2024/file/e6da278cdd692077b7e4a99d55573d9c-Paper-Conference.pdf",
            "status": status_for(SURR / "tables/hh_track4_asnpe_registry_20260426.csv"),
            "evidence": "hh_track4_asnpe_registry_20260426.csv",
        },
        {
            "branch": "generalized_bayes_npe_2026",
            "source": "Amortized Simulation-Based Inference in Generalized Bayes via Neural Posterior Estimation",
            "url": "https://arxiv.org/abs/2601.22367",
            "status": status_for(SURR / "tables/hh_track4_generalized_bayes_npe_registry_20260426.csv"),
            "evidence": "hh_track4_generalized_bayes_npe_registry_20260426.csv",
        },
        {
            "branch": "simformer_all_in_one_sbi_2024",
            "source": "All-in-one simulation-based inference",
            "url": "https://arxiv.org/abs/2404.09636",
            "status": status_for(SURR / "tables/hh_track4_simformer_registry_20260426.csv"),
            "evidence": "hh_track4_simformer_registry_20260426.csv",
        },
        {
            "branch": "decision_rule_to_hybrid_refresh",
            "source": "local downstream utility sweep",
            "url": "local",
            "status": status_for(SURR / "tables/hh_track4_hybrid_initializer_comparison_20260425.csv", completed="refreshed"),
            "evidence": "hh_track4_hybrid_initializer_comparison_20260425.csv",
        },
    ]
    out_csv = RUNALL / "tables/hh_track4_literature_branch_registry_20260426.csv"
    write_csv(out_csv, rows)
    report = [
        "# HH Track4 Literature Branch Registry",
        "",
        "| branch | source | status | evidence |",
        "|---|---|---|---|",
    ]
    for row in rows:
        report.append(f"| `{row['branch']}` | `{row['source']}` | `{row['status']}` | `{row['evidence']}` |")
    (RUNALL / "reports/hh_track4_literature_exhaustion_fix_20260426.md").write_text("\n".join(report) + "\n")
    print(out_csv)


if __name__ == "__main__":
    main()
