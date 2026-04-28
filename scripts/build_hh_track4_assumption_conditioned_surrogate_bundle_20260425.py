#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from hh_track4_assumption_conditioned_surrogate_contract_20260425 import (
    CONTRACTS,
    DATE_TAG,
    OBSERVED_CURRENT_LABELS,
    OUT_ROOT,
    REPORTS,
    RUNS_ACTIVE_SEQUENTIAL,
    RUNS_DIRECT_FITTING,
    TABLES,
    ACTIVE_CURRENT_GRID,
    claim_guardrails,
    combined_contract_manifest,
    ensure_bundle_dirs,
    observation_contract,
    parameter_schema_rows,
    protocol_contract,
    write_json,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parameter_rows_dict() -> list[dict[str, object]]:
    return [row.__dict__.copy() for row in parameter_schema_rows()]


def protocol_grid_rows() -> list[dict[str, object]]:
    rows = []
    observed = set(OBSERVED_CURRENT_LABELS)
    for current in ACTIVE_CURRENT_GRID:
        rows.append(
            {
                "current_label": current,
                "role": "observed_and_candidate" if current in observed else "candidate_only",
                "selection_scope": "surrogate_active_sequential_design",
                "interpretation": "relative_current_multiplier",
            }
        )
    return rows


def observation_rows_dict() -> list[dict[str, object]]:
    obs = observation_contract().__dict__.copy()
    return [{k: v for k, v in obs.items()}]


def guardrail_rows() -> list[dict[str, object]]:
    payload = claim_guardrails()
    rows = []
    for value in payload["allowed_uses"]:
        rows.append({"category": "allowed_use", "text": value})
    for value in payload["disallowed_claims"]:
        rows.append({"category": "disallowed_claim", "text": value})
    return rows


def build_report() -> str:
    return "\n".join(
        [
            "# HH Track4 Assumption-Conditioned Surrogate Bundle",
            "",
            f"Workspace-clock generated: `{DATE_TAG}`",
            "",
            "## Purpose",
            "",
            "This bundle creates the explicit folder structure and contract files needed to run assumption-conditioned direct-fitting and simulator-active methods, while preserving the distinction between surrogate workflow enablement and recovered Track4 provenance.",
            "",
            "## Literature-Informed Assumptions",
            "",
            "1. A compact single-compartment HH surrogate is acceptable for exploratory workflow development, because the target vector has six anonymous entries and the workspace has not recovered morphology or multicompartment metadata.",
            "2. Square-pulse current clamp is the default surrogate protocol family, because step-current stimulation is standard in compact HH fitting and BluePyOpt-style optimization workflows.",
            "3. The six anonymous targets are treated as repeated conductance and kinetics proxy slots plus a leak proxy, because the existing workspace prior already uses a repeated three-value support pattern and the current sandbox maps those slots consistently.",
            "4. The observed `0.1` to `0.5` labels are treated as relative current multipliers, not recovered physical units.",
            "5. Active sequential design uses the same surrogate contract, but it must be reported as assumption-conditioned and not provenance-recovered.",
            "",
            "## Created Structure",
            "",
            f"- bundle root: `{OUT_ROOT}`",
            f"- contracts: `{CONTRACTS}`",
            f"- tables: `{TABLES}`",
            f"- reports: `{REPORTS}`",
            f"- direct-fitting runs: `{RUNS_DIRECT_FITTING}`",
            f"- active-sequential runs: `{RUNS_ACTIVE_SEQUENTIAL}`",
            "",
            "## Artifacts",
            "",
            f"- parameter schema JSON: `{CONTRACTS / f'hh_track4_surrogate_parameter_schema_{DATE_TAG}.json'}`",
            f"- protocol contract JSON: `{CONTRACTS / f'hh_track4_surrogate_protocol_contract_{DATE_TAG}.json'}`",
            f"- observation contract JSON: `{CONTRACTS / f'hh_track4_surrogate_observation_contract_{DATE_TAG}.json'}`",
            f"- guardrails JSON: `{CONTRACTS / f'hh_track4_surrogate_claim_guardrails_{DATE_TAG}.json'}`",
            f"- combined manifest JSON: `{CONTRACTS / f'hh_track4_surrogate_bundle_manifest_{DATE_TAG}.json'}`",
            "",
            "## Existing Direct-Fitting Hooks",
            "",
            "- `run_hh_track4_assumption_conditioned_hybrid_refinement_20260425.py`",
            "- `run_hh_track4_assumption_conditioned_wasserstein_abc_20260425.py`",
            "",
            "## New Simulator-Active Hook",
            "",
            "- `run_hh_track4_assumption_conditioned_active_sequential_design_20260425.py`",
            "",
            "## Guardrails",
            "",
            "Every artifact derived from this bundle must remain labeled as assumption-conditioned and not provenance-recovered. The bundle enables workflow execution, but it does not create the authentic Track4 simulator contract.",
            "",
            "## Literature Trail",
            "",
            "- Van Geit et al., BluePyOpt 2016, current-clamp optimization context",
            "- Oesterle et al., eLife 2020, Bayesian neuron-model inference with stimulus optimization",
            "- Griesemer et al., ASNPE NeurIPS 2024, active sequential SBI logic",
            "",
            "## Operational Consequence",
            "",
            "This bundle removes the infrastructure excuse for the assumption-conditioned branches. It does not remove the scientific distinction between surrogate execution and recovered provenance.",
            "",
        ]
    )


def main() -> None:
    ensure_bundle_dirs()

    write_json(CONTRACTS / f"hh_track4_surrogate_parameter_schema_{DATE_TAG}.json", parameter_rows_dict())
    write_json(CONTRACTS / f"hh_track4_surrogate_protocol_contract_{DATE_TAG}.json", protocol_contract().__dict__)
    write_json(CONTRACTS / f"hh_track4_surrogate_observation_contract_{DATE_TAG}.json", observation_contract().__dict__)
    write_json(CONTRACTS / f"hh_track4_surrogate_claim_guardrails_{DATE_TAG}.json", claim_guardrails())
    write_json(CONTRACTS / f"hh_track4_surrogate_bundle_manifest_{DATE_TAG}.json", combined_contract_manifest())

    write_csv(TABLES / f"hh_track4_surrogate_parameter_schema_{DATE_TAG}.csv", parameter_rows_dict())
    write_csv(TABLES / f"hh_track4_surrogate_protocol_grid_{DATE_TAG}.csv", protocol_grid_rows())
    write_csv(TABLES / f"hh_track4_surrogate_observation_contract_{DATE_TAG}.csv", observation_rows_dict())
    write_csv(TABLES / f"hh_track4_surrogate_guardrails_{DATE_TAG}.csv", guardrail_rows())

    (REPORTS / f"hh_track4_assumption_conditioned_surrogate_bundle_{DATE_TAG}.md").write_text(build_report())

    print(OUT_ROOT)


if __name__ == "__main__":
    main()

