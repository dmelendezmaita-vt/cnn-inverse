# HH Track4 SBI Calibration Scan

Workspace-clock generated: `2026-04-25`

Inputs scanned:

- `optimization_track_20260423_v100_hh_track4_checkpoint2_modern_sbi`
- `optimization_track_20260423_v100_hh_track4_checkpoint2_snpe_feature_aware`

Artifacts:

- table:
  - `hh_track4_sbi_calibration_scan_20260425.csv`
- JSON detail:
  - `hh_track4_sbi_calibration_scan_20260425.json`

## Main pattern

The scanned SBI posteriors are systematically underdispersed.

Evidence:

- nominal `0.8` marginal coverage often lands around `0.56` to `0.59`
- nominal `0.9` marginal coverage often lands around `0.64` to `0.68`
- overall rank means stay near the expected midpoint, but rank variances are inflated and several per-target uniformity scores are large

This means:

- central tendency is not wildly biased
- uncertainty width is generally too narrow

## Recalibration result

The simple global post-hoc scaling scan found useful correction factors typically around:

- `1.2`
- `1.25`
- `1.3`

These scales shrink the mean absolute coverage-gap substantially, often down to a small residual level.

Example:

- one `FMPE` representative had:
  - raw coverage gaps:
    - `0.5`: `-0.1589`
    - `0.8`: `-0.2401`
    - `0.9`: `-0.2503`
  - best simple recalibration scale:
    - `1.3`
  - recalibrated coverages:
    - `0.5`: `0.5026`
    - `0.8`: `0.8151`
    - `0.9`: `0.8965`

## Practical use

This scan does not replace full local posterior diagnostics, but it is enough to justify:

1. keeping posterior calibration as a still-open execution branch
2. adding post-hoc coverage rescaling as a fair comparison layer for final SBI representatives
3. avoiding claims that the current raw posterior widths are already well calibrated
