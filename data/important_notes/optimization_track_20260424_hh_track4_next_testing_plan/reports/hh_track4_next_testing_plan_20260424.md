# HH Track4 Next Testing Plan

Workspace-clock generated: `2026-04-24`

## Current State

The assumption-conditioned compact-HH testing lane is now broad enough to stop.

Completed in that lane:

- screening battery
- bounded differential-evolution optimization battery
- bounded BPTT optimization battery
- hybrid trace+feature objectives
- feature-only objectives
- repeat BPTT seed on the main broad-shortpulse variant

Conclusion from that lane:

- no compact-HH assumed fit family became a compelling missed-winner path

Relevant references:

- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_assumption_conditioned_compact_hh_fit_search/reports/hh_track4_assumption_conditioned_fit_family_decision_20260424.md`
- `/projects/neuro-collab/code/fhn_dnn-1-implementation-in-pytorch/data/important_notes/optimization_track_20260424_hh_track4_literature_questions/reports/hh_track4_literature_questions_answered_20260424.md`

## What Is Next

There are only two serious next-testing paths left.

### Path 1. Recover true simulator provenance

This is now the main remaining path.

What to recover:

- exact parameter semantics for `hh_param_1..6`
- exact current protocol
- exact time grid and units
- exact simulator equations or generator code

If recovered, run:

1. dataset-faithful direct fitting
2. feature-based direct fitting on the true simulator
3. richer protocol / stimulus tests on the true simulator
4. profile-likelihood or uncertainty checks tied to the real mechanism

### Path 2. Stop testing and finalize the literature-informed closure

If provenance is not recovered, the remaining work is not more blind fitting.

It is:

1. consolidate the assumption-conditioned results
2. state clearly that the compact-HH lane was tested broadly enough
3. state clearly that the remaining gap is true simulator provenance, not obvious missed estimator tuning

## What Is Not Worth Continuing

Do not continue:

- more Rudi-style inverse-net tuning
- more compact-HH screening variants with tiny cosmetic changes
- more compact-HH optimizer reruns without a new structural idea
- waiting for A30/A100/H100 to unlock the same already-tested assumption lane

## Practical Recommendation

The next testing step should be:

> provenance recovery first, or stop and finalize

That is the point the campaign has reached.

