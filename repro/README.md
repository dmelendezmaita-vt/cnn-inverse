# Reproduction Index

This directory is the canonical human entrypoint for the public repository. The order follows the current evidence chain rather than the older internal track chronology.

| Step | Directory | Purpose |
| --- | --- | --- |
| `00` | `00_source_snapshot/` | Recover or verify the untouched upstream source archive |
| `10` | `10_fhn_benchmark_reconstruction/` | Inspect the inherited FitzHugh-Nagumo benchmark reconstruction basis |
| `20` | `20_hh_canonical_clean_a30/` | Reproduce the canonical clean A30 Hodgkin-Huxley campaign description and outputs |
| `30` | `30_hh_closure_boundary/` | Read the final closure statement, efficiency evidence, and simulator-provenance blocker |
| `40` | `40_state_of_the_art_followup/` | Read the later executable state-of-the-art follow-up and remaining testing boundary |
| `90` | `90_legacy_upload_bundle/` | Read the note about the earlier upload bundle, which is no longer shipped in the current tree |

The machine-readable artifact map is `repro/canonical_artifact_index.csv`.
