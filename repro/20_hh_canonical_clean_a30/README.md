# Canonical Clean A30 Campaign

This is the current canonical clean Hodgkin-Huxley campaign in the public repository.

## Regeneration Commands

Build the literal A30 manifest:

```bash
python scripts/build_hh_track4_a30_literal_rerun_manifest_20260427.py \
  --out-root data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun
```

Build the clean A30 analysis package:

```bash
python scripts/analyze_hh_track4_a30_clean_results_20260428.py \
  --campaign-root data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun
```

## Canonical Outputs

| Output | Path |
| --- | --- |
| manifest report | `data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/notes/hh_track4_a30_literal_rerun_manifest_categories_20260427.md` |
| manifest table | `data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/tables/hh_track4_a30_literal_rerun_manifest_20260427.csv` |
| clean results analysis | `data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/reports/hh_track4_a30_clean_results_analysis_20260428.md` |
| clean direct summary | `data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/tables/hh_track4_a30_clean_direct_summary_20260428.csv` |
| clean key findings | `data/important_notes/optimization_track_20260427_hh_track4_a30_literal_rerun/tables/hh_track4_a30_clean_key_findings_20260428.csv` |

## Canonical Drivers

| Driver | Current statement |
| --- | --- |
| direct clean winner | `extra_trees_500_depth20` by clean mean MAE |
| aligned framework winner | `bayesflow_native_meanstd_v100_20260425` |
| comparable posterior decision-rule winner by MAE | `bayesflow_set_posterior_v100_20260426::median` |
| hybrid initializer winner | `hybrid_refinement_surrogate_bundle_prediction_snpe_maf_r3_pool4096_final2048_medoid_20260427` |
