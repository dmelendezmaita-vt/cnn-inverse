# Repository Re-Sort Evaluation (2026-04-28)

## Current Structural Problems

| Current pattern | Problem for reproducibility | Required correction |
| --- | --- | --- |
| `data/important_notes/...` as the main narrative surface | It preserves workspace chronology rather than the canonical reproduction chain | Add a numbered reproduction layer that orders artifacts by what a human must understand and execute |
| legacy `track`, `phase`, and `optimization_track_YYYYmmdd_*` names exposed directly | They are useful for provenance, but they are poor human entrypoints | Keep them as source paths, then wrap them in stable reproduction labels |
| very large undifferentiated `tools/` surface | A reproducer cannot infer which commands are canonical and which are exploratory or superseded | Publish a short canonical command index with explicit status labels |
| original untouched upstream zip absent from the public tree | Reproducing the inherited baseline is incomplete without an exact external reference to that archive | Ship a documented archive reference, its zip comment, its size, and an extraction command |
| earlier `github_upload_bundle_20260407` exposed as if it were the current public structure | It reflects an older track-based packaging logic | Keep it as a legacy bundle, not as the canonical organization |

## Recommended Public Organization

| Order | Directory | Role |
| --- | --- | --- |
| `00` | `repro/00_source_snapshot/` | Exact reference and extraction path for the untouched upstream archive |
| `10` | `repro/10_fhn_benchmark_reconstruction/` | Benchmark-parity and inherited-baseline reconstruction references |
| `20` | `repro/20_hh_canonical_clean_a30/` | Canonical clean homogeneous A30 manifest and analysis path |
| `30` | `repro/30_hh_closure_boundary/` | Final closure, efficiency, and simulator-provenance boundary |
| `40` | `repro/40_state_of_the_art_followup/` | Later executable literature-aligned and assumption-conditioned follow-up |
| `90` | `repro/90_legacy_upload_bundle/` | Earlier packaging bundle, retained as legacy context only |

## Immediate Conclusion

The current GitHub tree should remain provenance-rich, but its public reading order should be driven by `repro/`, because the canonical workflow is now:

1. recover or verify the original upstream source snapshot
2. understand the inherited FitzHugh-Nagumo benchmark reconstruction
3. reproduce the canonical clean A30 Hodgkin-Huxley campaign description
4. read the closure and provenance boundary
5. inspect the later follow-up branches as boundary-strengthening evidence
