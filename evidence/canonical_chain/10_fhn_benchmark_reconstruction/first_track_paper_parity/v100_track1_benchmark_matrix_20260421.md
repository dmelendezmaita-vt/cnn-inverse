# Track 1 Benchmark Matrix (V100, Paper-Behavior, Seeded) — 2026-04-21

## Purpose

Record the canonical seeded Track 1 benchmark matrix after the benchmark lane
was separated from the corrected-split lane.

This matrix is the one to use for Track 1 benchmark references going forward.

## Canonical definition

- execution shape: `1 node / 1 GPU`
- data path: legacy local filesystem path
- dataset layout: `legacy_2020_hardcoded`
- split semantics: original paper-behavior overlap restored via
  `legacy_validate_test_overlap: true`
- seeds: `301-305`

Config root:

- `pytorch/configs/track1_benchmark_v100/`

Tracking CSV:

- `data/important_notes/first_track_paper_parity/v100_track1_benchmark_training_jobs.csv`

Output root:

- `/projects/neuro-collab/data/runs/track1_benchmark_v100/`

## Completed rows

| Seed | Run dir | Test MSE | Test MAE | Test R2 |
|---:|---|---:|---:|---:|
| 301 | `/projects/neuro-collab/data/runs/track1_benchmark_v100/bench_V100_1g_s301/368801` | 0.0025711060 | 0.0339366123 | 0.9745665789 |
| 302 | `/projects/neuro-collab/data/runs/track1_benchmark_v100/bench_V100_1g_s302/368801` | 0.0023862710 | 0.0327602103 | 0.9763079882 |
| 303 | `/projects/neuro-collab/data/runs/track1_benchmark_v100/bench_V100_1g_s303/368801` | 0.0025044824 | 0.0341999158 | 0.9752004147 |
| 304 | `/projects/neuro-collab/data/runs/track1_benchmark_v100/bench_V100_1g_s304/368801` | 0.0026720953 | 0.0344303884 | 0.9740264416 |
| 305 | `/projects/neuro-collab/data/runs/track1_benchmark_v100/bench_V100_1g_s305/368801` | 0.0027456242 | 0.0359236635 | 0.9726144671 |

## Aggregate summary

Test-set means across seeds `301-305`:

- mean `MSE = 0.0025759158`
- mean `MAE = 0.0342501581`
- mean `R2  = 0.9745431781`

Test-set population standard deviations across seeds `301-305`:

- std `MSE = 0.0001258113`
- std `MAE = 0.0010152717`
- std `R2  = 0.0012275182`

## Comparison to archived benchmark `242036`

Archived benchmark test metrics:

- `MSE = 0.0027437592`
- `MAE = 0.0348857418`
- `R2  = 0.9732103348`

Seeded benchmark-matrix mean vs archived benchmark:

- `MSE`: `-6.12%`
- `MAE`: `-1.82%`
- `R2`: `+0.137%`

## Interpretation

1. The canonical benchmark lane is now seeded and fully separated from the
   corrected-split lane.
2. Every row in this matrix uses the original paper-behavior split semantics,
   including validate/test overlap.
3. The seeded matrix does **not** numerically collapse to the archived
   benchmark because the archived benchmark used seed `123`, while this matrix
   uses seeds `301-305`.
4. That difference is expected. The benchmark matrix is for seeded benchmark
   distributional reference, not for exact seed-123 replay.

## Non-canonical mistaken run

An earlier 2026-04-21 distributed paper-behavior attempt was removed from the
working tree because it followed the wrong `1n/2n/4n` matrix shape and was not
benchmark parity.
