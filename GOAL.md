# StressKit goal

## Primary objective

Build the trusted, open audit standard for mechanistic-interpretability claims:
formally specified, statistically calibrated, reproducible from raw artifacts,
and broad enough to show which findings survive defensible variation and which
do not.

Public attention is an amplifier, not a decision rule. Benchmark outcomes must
remain unknown when the protocol, target registry, nulls, exclusions, and
analysis are frozen.

## What success means

StressKit is ready for a major public launch only after all gates below pass.

### Gate 1 — specification integrity

- Every reported quantity has a mathematical definition and named estimand.
- Every threshold is labeled as derived, externally proposed, empirically
  calibrated, or conventional. No threshold is presented as literature-backed
  without a source that states that numerical rule for the same use.
- Diagnostic one-at-a-time sweeps are separated from confirmatory audits over a
  declared specification space.
- Every check resolves to pass, fail, or inconclusive. A point estimate with an
  interval crossing its decision boundary is not called a pass or a fail.

### Gate 2 — mathematical and software integrity

- Lean verifies the deterministic finite core: set metrics, claim-disagreement
  identities, finite random baselines, and verdict logic.
- Python outputs conform to the formal definitions on exhaustive small cases
  and shared golden vectors.
- Card verification recomputes all headline quantities from sufficient raw
  statistics, not only from previously summarized floating-point values.

### Gate 3 — statistical calibration

- Known-truth simulations measure interval coverage, false decisions, power,
  and sample-size requirements across predeclared scenarios.
- A nominal 95% confirmatory profile must not be anti-conservative: observed
  coverage is at least 93% on core scenarios, with Monte Carlo error and any
  overcoverage/power cost reported. A future efficiency profile may target
  93–97%; v0.1 deliberately prioritizes finite-sample validity.
- Constructed stable positive controls pass; random, multimodal, and
  non-specific negative controls fail or remain explicitly inconclusive.
- Minimum run counts come from calibration rather than convenience.

### Gate 4 — preregistered evidence breadth

Initial public benchmark target:

- at least 20 concrete claims or released instruments;
- at least 6 mechanistic-interpretability method families;
- at least 3 model families and multiple tasks;
- public code plus reproducible weights, data, traces, or saved outputs;
- a defensible null control for every target where one is constructible.

Inclusion rules, exact upstream commits, claim maps, perturbation spaces,
resource limits, and exclusion rules are frozen before confirmatory runs.
Failures to reproduce are reported separately from audit failures.

### Gate 5 — durable reproducibility

Every public result carries:

- pinned source and environment versions;
- complete configuration, seeds, and job manifest;
- raw or content-addressed per-run outputs;
- machine-verifiable card and human-readable render;
- declared scope and limitations;
- an independent rerun or reviewer sign-off for headline results.

### Gate 6 — high-impact communication

Launch package includes a concise preprint or technical report, interactive
scoreboard, one canonical visual, reproducible repository, short demonstration,
and social thread. Language reports observed counts and uncertainty. It never
generalizes from one artifact to an entire paper or method family without the
corresponding evidence.

Preferred launch frame:

> We froze and validated the audit before seeing benchmark outcomes. Across X
> reproducible claims from Y method families, Z failed at least one
> preregistered check, Q passed, and R remained inconclusive.

## Current status

Deterministic core is specified and machine-checked in Lean/mathlib. Python
conforms on exhaustive small cases and shared golden vectors. Conservative
confirmatory profile v0.1 passed frozen 2,000-trial S1--S9 studies plus a
disjoint-seed replication (156,000 simulated trials total); current diagnostic
letter grades remain explicitly exploratory. Fourteen upstream commits/trees,
51 entrypoint paths, 13 source licenses, and 731 Python files passed both local
and independent Nibi static audits; Nibi also passed 348 StressKit tests.
Dependency/import smokes and claim-level executions remain in progress. Other
remaining launch gates are construct-validity review, frozen benchmark claim
records, upstream reproductions, confirmatory runs, and independent result
review.

## Non-goals

- Predetermining that mechanistic interpretability does not work.
- Treating instability as proof that a paper is false or valueless.
- Choosing targets, nulls, thresholds, or wording after seeing which version
  creates the strongest headline.
- Using formal verification as a substitute for statistical or empirical
  validation.
