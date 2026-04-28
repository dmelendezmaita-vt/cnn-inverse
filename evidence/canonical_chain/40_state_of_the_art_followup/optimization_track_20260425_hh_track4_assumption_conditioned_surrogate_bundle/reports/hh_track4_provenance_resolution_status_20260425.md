# HH Track4 Provenance Resolution Status

Workspace-clock generated: `2026-04-25`

## Decision

For the remaining Track4 campaign, provenance-faithful simulator recovery is not expected and is not a prerequisite for continued work.

## Operational Rule

All remaining direct-fitting, simulator-active, posterior-diagnostic, and framework-extension work is to be executed under explicit literature-informed assumptions and labeled:

> assumption-conditioned, not provenance-recovered

## Reporting Rule

The absence of provenance-faithful data is treated as a standing reporting constraint, not as an execution blocker.

Allowed:

- continued surrogate direct fitting
- continued surrogate simulator-active acquisition
- additional TMNRE-style schedules
- richer BayesFlow and Swyft variants
- posterior diagnostics and calibration analyses

Not allowed:

- claims that the forward model is the authentic Track4 generator
- claims that selected protocols or currents are optimal for the unrecovered real simulator
- claims that anonymous parameters have recovered physical identities

## Consequence

No remaining methodological branch is blocked by provenance concerns alone. Any remaining stop decision must therefore be justified by either practical resource limits or direct evidence that a branch adds no information.
