# HH Track4 Simulator Provenance Audit

Workspace-clock generated: `2026-04-23T23:23:53-04:00`

## Decision

Direct simulator fitting remains blocked by missing simulator provenance. This is a workspace-level blocker, not a negative scientific result for direct fitting.

## What Is Recoverable

- Track4 full data are present as raw float32 array payloads under the expected `concatenated_data` layout.
- Current labels recoverable from file names: `0.1`, `0.2`, `0.3`, `0.4`, `0.5`.
- For each current label: `15000` feature rows, `400000` float32 values per row, and `15000` six-parameter target rows.
- The configured target names remain neutral: `hh_param_1` through `hh_param_6`.
- The fixed supervised inverse task is therefore executable and auditable.

## What Is Not Recoverable

- semantic mapping for `hh_param_1` through `hh_param_6`
- HH equation variant and state variables
- initial conditions, time grid, time units, and sampling interval
- current-injection protocol beyond the filename labels
- simulator/generator script or random-generation contract
- BluePyOpt/direct-fit objective, parameter bounds by semantic name, or differentiable simulator path

## Evidence Tables

- evidence: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_simulator_provenance_audit/tables/hh_track4_simulator_provenance_evidence_20260423.csv`
- raw schema: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_simulator_provenance_audit/tables/hh_track4_raw_data_schema_20260423.csv`
- target ranges: `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260423_hh_track4_simulator_provenance_audit/tables/hh_track4_target_ranges_by_current_20260423.csv`

## Closure Implication

The current workspace supports strong closure for the fixed supervised HH Track4 inverse task. It does not support a literature-complete claim that direct simulator fitting has been exhausted. The safe wording remains: direct fitting is scientifically relevant but not executable here without inventing missing simulator provenance.
