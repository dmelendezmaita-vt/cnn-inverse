# HH Track4 Assumption-Conditioned Fit Family Decision

Workspace-clock generated: `2026-04-24`

## Purpose

This note is for the final decision after the remaining BPTT runs complete.

It compares:

- hybrid trace+feature DE
- hybrid trace+feature BPTT
- feature-only DE
- feature-only BPTT

within the same assumption-conditioned compact-HH lane.

## Decision Rule

Use the following standard:

- if a new family materially beats the current best by both objective and trace-fit quality, reopen the conditional lane
- if it only ties or modestly improves while still looking qualitatively weak, keep the lane closed

## Current Best Completed References

- best completed hybrid DE:
  `fit_search_lowcurrent_broad_shortpulse`
- best completed hybrid BPTT:
  `bptt_fit_lowcurrent_broad_shortpulse`
- best completed feature-only DE:
  `fit_search_featureonly_broad_shortpulse`
- best completed feature-only BPTT:
  `[fill after run]`

## Fill After Remaining Runs Finish

- best hybrid objective: `[fill]`
- best hybrid trace RMSE z: `[fill]`
- best feature-only objective: `[fill]`
- best feature-only trace RMSE z: `[fill]`
- did any BPTT family materially change the conclusion: `[yes_or_no]`

## Safe Outcome Language

Use one of:

- `remaining BPTT runs did not materially change the assumption-conditioned conclusion`
- `feature-only BPTT modestly improved that family but did not justify reopening the lane`
- `a BPTT family materially changed the assumption-conditioned lane and requires review`

