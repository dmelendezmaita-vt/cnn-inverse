# HH Track4 Assumption-Conditioned Fit Family Decision

Workspace-clock generated: `2026-04-24`

## Scope

This note closes the assumption-conditioned compact-HH fitting battery.

Families covered:

- differential-evolution, hybrid trace+feature objective
- differential-evolution, feature-only objective
- Adam/BPTT, hybrid trace+feature objective
- Adam/BPTT, feature-only objective

All results remain:

- assumption-conditioned
- not provenance-recovered
- not dataset-faithful direct fitting

## Best Completed Results By Family

### Hybrid trace+feature

- best DE result:
  - run: `fit_search_lowcurrent_broad_shortpulse`
  - best objective: `2.668260887185062`
  - best trace RMSE z: `1.160409724712372`

- best BPTT result:
  - run: `bptt_fit_lowcurrent_broad_shortpulse`
  - best objective: `2.3478708267211914`
  - best trace RMSE z: `1.1841446161270142`

Interpretation:

- BPTT improved the optimization objective
- it did **not** clearly improve the trace-fit quality
- the resulting traces still tended toward near-rest behavior rather than convincing spike-like matching

### Feature-only

- best DE result:
  - run: `fit_search_featureonly_broad_shortpulse`
  - best objective: `1.5718343459826944`
  - best trace RMSE z: `1.518998908996582`

- best BPTT result:
  - run: `bptt_fit_featureonly_broad_shortpulse`
  - best objective: `1.3992358446121216`
  - best trace RMSE z: `1.5152674913406372`

Interpretation:

- BPTT improved the feature-only objective modestly
- the trace-fit quality stayed weak
- the fitted traces again remained close to near-rest voltages across currents

## What The Remaining BPTT Runs Added

Additional completed runs:

- `bptt_fit_lowcurrent_narrow_shortpulse`
- `bptt_fit_featureonly_narrow_shortpulse`
- `bptt_fit_lowcurrent_broad_shortpulse_seed3302`

What they showed:

- narrow-support BPTT did not beat broad-support BPTT
- feature-only narrow BPTT did not beat feature-only broad BPTT
- the second raw-trace BPTT seed did not reveal a hidden much-better basin

So the remaining gradient-based runs did not materially change the picture.

## Decision

Decision:

> The assumption-conditioned compact-HH fitting literature that is executable under the current assumptions has now been tested broadly enough, and it does not provide a compelling route to a major improvement over the current frontier.

More concretely:

- the main optimizer families were both tested
- the main surrogate-support variants were both tested
- the main objective modes were both tested
- repeated broad-shortpulse BPTT did not overturn the result

## What This Means

Safe conclusion:

> Under the current literature-backed assumptions, we do not see a strong remaining compact-HH direct-fitting lane that is likely to materially improve Track4.

Unsafe conclusion:

> We have scientifically proven that no direct-fitting method could ever improve Track4.

## Remaining Gap

The remaining gap is still:

- true simulator provenance
- true parameter semantics
- true protocol contract

Without those, further assumed compact-HH fitting work is likely to add cost faster than clarity.

