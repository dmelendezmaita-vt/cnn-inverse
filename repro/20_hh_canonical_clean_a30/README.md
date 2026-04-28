# Canonical Clean A30 Campaign

This is the current canonical clean Hodgkin-Huxley campaign in the public repository.

## Regeneration Commands

Build the literal A30 manifest:

```bash
python tools/build_hh_track4_a30_literal_rerun_manifest_20260427.py \
  --out-root evidence/canonical_chain/20_hh_canonical_clean_a30/optimization_track_20260427_hh_track4_a30_literal_rerun
```

Build the clean A30 analysis package:

```bash
python tools/analyze_hh_track4_a30_clean_results_20260428.py \
  --campaign-root evidence/canonical_chain/20_hh_canonical_clean_a30/optimization_track_20260427_hh_track4_a30_literal_rerun
```

## Execution Boundary

The shipped clean A30 output package is reproducible as an inspected artifact set inside this public repository. Rebuilding the manifest from scratch still depends on historical A30 phase matrices and some earlier checkpoint analysis tables that remain in the original workspace rather than in the slim public tree.

## Canonical Outputs

| Output | Path |
| --- | --- |
| manifest report | `evidence/canonical_chain/20_hh_canonical_clean_a30/optimization_track_20260427_hh_track4_a30_literal_rerun/notes/hh_track4_a30_literal_rerun_manifest_categories_20260427.md` |
| manifest table | `evidence/canonical_chain/20_hh_canonical_clean_a30/optimization_track_20260427_hh_track4_a30_literal_rerun/tables/hh_track4_a30_literal_rerun_manifest_20260427.csv` |
| clean results analysis | `evidence/canonical_chain/20_hh_canonical_clean_a30/optimization_track_20260427_hh_track4_a30_literal_rerun/reports/hh_track4_a30_clean_results_analysis_20260428.md` |
| clean direct summary | `evidence/canonical_chain/20_hh_canonical_clean_a30/optimization_track_20260427_hh_track4_a30_literal_rerun/tables/hh_track4_a30_clean_direct_summary_20260428.csv` |
| clean key findings | `evidence/canonical_chain/20_hh_canonical_clean_a30/optimization_track_20260427_hh_track4_a30_literal_rerun/tables/hh_track4_a30_clean_key_findings_20260428.csv` |

## Canonical Drivers

| Driver | Current statement |
| --- | --- |
| direct clean winner | `extra_trees_500_depth20` by clean mean MAE |
| aligned framework winner | `bayesflow_native_meanstd_v100_20260425` |
| comparable posterior decision-rule winner by MAE | `bayesflow_set_posterior_v100_20260426::median` |
| hybrid initializer winner | `hybrid_refinement_surrogate_bundle_prediction_snpe_maf_r3_pool4096_final2048_medoid_20260427` |
