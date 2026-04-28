# Data Scope

This public staging repo carries only the data that can be published cleanly and replayed without shipping cluster-local runtime state.

## Included Data

| Path | Included | Technical purpose |
| --- | --- | --- |
| `data/2020-12-09/` | Yes | Baseline FitzHugh-Nagumo arrays and notebooks used by the original inverse-map workflow |
| `data/important_notes/first_track_paper_parity/` | Yes, curated | Benchmark reconstruction notes and parity evidence |
| `data/important_notes/optimization_track_20260423_hh_track4_closure_plan/` | Yes, curated | Main bounded closure report, notes, and tables |
| `data/important_notes/optimization_track_20260423_hh_track4_checkpoint7_standardized_efficiency/` | Yes, curated | Executed efficiency evidence used in the final interpretation |
| `data/important_notes/optimization_track_20260423_hh_track4_simulator_provenance_audit/` | Yes, curated | Simulator provenance blocker analysis |
| `data/important_notes/optimization_track_20260424_hh_track4_next_testing_plan/` | Yes, curated | Post-closure planning logic |
| `data/important_notes/optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_fit_search/` | Yes, curated | Later compact-Hodgkin-Huxley direct-fitting screening |
| `data/important_notes/optimization_track_20260425_hh_track4_aligned_literature_plan/` | Yes, curated | Literature-aligned comparison planning |
| `data/important_notes/optimization_track_20260425_hh_track4_assumption_conditioned_surrogate_bundle/` | Yes, curated | Exploratory follow-up package, kept separate from the main frontier |
| `data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/` | Yes, curated | Canonical clean A30 reporting package |

## Excluded Data

| Excluded class | Why it is excluded |
| --- | --- |
| Large Hodgkin-Huxley tar archives under cluster-local storage | They are not suitable for GitHub distribution and are not recoverable from the current public tree alone |
| Runtime logs, queue stdout, and scheduler stderr | They are operational artifacts rather than stable scientific inputs |
| Shared scratch mirrors and extracted working copies | They are host-specific and inflate the public repo without improving reproducibility |
| Model checkpoints, tensor dumps, and binary intermediate artifacts | They are large, unstable, and should be regenerated from documented manifests when possible |

## Reproduction Boundary

The included FitzHugh-Nagumo data is enough for the baseline public run path described in the repository README. The included Hodgkin-Huxley evidence folders are sufficient to inspect the canonical notes, tables, and reports, while reproducing the full Hodgkin-Huxley executions still requires the external tar-backed dataset roots that the live cluster workflow used.
