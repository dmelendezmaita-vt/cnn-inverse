# HH Track4 Assumption-Conditioned Surrogate Bundle

Workspace-clock generated: `20260425`

## Purpose

This bundle creates the explicit folder structure and contract files needed to run assumption-conditioned direct-fitting and simulator-active methods, while preserving the distinction between surrogate workflow enablement and recovered Track4 provenance.

## Literature-Informed Assumptions

1. A compact single-compartment HH surrogate is acceptable for exploratory workflow development, because the target vector has six anonymous entries and the workspace has not recovered morphology or multicompartment metadata.
2. Square-pulse current clamp is the default surrogate protocol family, because step-current stimulation is standard in compact HH fitting and BluePyOpt-style optimization workflows.
3. The six anonymous targets are treated as repeated conductance and kinetics proxy slots plus a leak proxy, because the existing workspace prior already uses a repeated three-value support pattern and the current sandbox maps those slots consistently.
4. The observed `0.1` to `0.5` labels are treated as relative current multipliers, not recovered physical units.
5. Active sequential design uses the same surrogate contract, but it must be reported as assumption-conditioned and not provenance-recovered.

## Created Structure

- bundle root: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle`
- contracts: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/contracts`
- tables: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/tables`
- reports: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/reports`
- direct-fitting runs: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/runs/direct_fitting`
- active-sequential runs: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/runs/active_sequential`

## Artifacts

- parameter schema JSON: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/contracts/hh_track4_surrogate_parameter_schema_20260425.json`
- protocol contract JSON: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/contracts/hh_track4_surrogate_protocol_contract_20260425.json`
- observation contract JSON: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/contracts/hh_track4_surrogate_observation_contract_20260425.json`
- guardrails JSON: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/contracts/hh_track4_surrogate_claim_guardrails_20260425.json`
- combined manifest JSON: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/contracts/hh_track4_surrogate_bundle_manifest_20260425.json`

## Existing Direct-Fitting Hooks

- `run_hh_track4_assumption_conditioned_hybrid_refinement_20260425.py`
- `run_hh_track4_assumption_conditioned_wasserstein_abc_20260425.py`

## New Simulator-Active Hook

- `run_hh_track4_assumption_conditioned_active_sequential_design_20260425.py`

## Guardrails

Every artifact derived from this bundle must remain labeled as assumption-conditioned and not provenance-recovered. The bundle enables workflow execution, but it does not create the authentic Track4 simulator contract.

## Literature Trail

- Van Geit et al., BluePyOpt 2016, current-clamp optimization context
- Oesterle et al., eLife 2020, Bayesian neuron-model inference with stimulus optimization
- Griesemer et al., ASNPE NeurIPS 2024, active sequential SBI logic

## Operational Consequence

This bundle removes the infrastructure excuse for the assumption-conditioned branches. It does not remove the scientific distinction between surrogate execution and recovered provenance.
