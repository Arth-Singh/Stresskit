# StressKit method specification

Status: **normative confirmatory core 0.1**. Deterministic definitions and the
conservative finite-sample profile are implemented and validated. Method-family
thresholds and benchmark claim records remain preregistration inputs.

## 1. Scope

StressKit audits a concrete interpretability claim produced by a specified
pipeline on a specified model, task, dataset, and usage mode. It does not grade
a paper, author, or method family from one experiment.

An audit can establish that a claim is stable or unstable under a declared
distribution of defensible variations. It cannot establish that the claim is a
complete mechanistic explanation, uniquely identifiable, or useful for every
downstream purpose.

## 2. Claim record

Each audit begins with a frozen claim record:

- `claim_id`: stable public identifier;
- `statement`: falsifiable natural-language statement;
- `artifact`: model, task, method, upstream commit, and usage mode;
- `finding_type`: set, ranking, direction, scalar, categorical claim, or custom;
- `claim_map`: deterministic code mapping raw output to the audited statement;
- `component_universe`: exact namespace and cardinality when structure is used;
- `specification_space`: defensible analytic choices and their sources;
- `specification_distribution`: declared weighting over those choices;
- `data_distribution`: population or empirical resampling target;
- `nulls`: controls and the scientific absence each represents;
- `metrics`, `decision_rules`, and `run_budget`;
- inclusion, exclusion, and failure-handling rules.

Changing any frozen field creates a new claim record and audit identifier.

## 3. Audit profiles

### 3.1 Diagnostic profile

Diagnostic mode uses one-at-a-time sweeps around a base configuration. It
localizes sensitive axes cheaply and reports every axis separately. Cross-axis
pairs and arbitrary run-count weighting do not receive a confirmatory verdict.

Diagnostic output may recommend a confirmatory design. It is not a certificate.

### 3.2 Confirmatory profile

Confirmatory mode samples or enumerates a preregistered specification space.
The target quantity is defined relative to the declared specification
distribution. A crossed grid is valid; a probability sample from a large grid
is valid when inclusion probabilities are recorded. An OAT union is not treated
as a sample from the crossed space.

Confirmatory runs use a run count justified by the calibration report. Failed
specifications and discarded outputs remain part of the audit trail and are
reported by axis level.

## 4. Core mathematical quantities

Let a structured finding be a finite set `A` drawn from universe `U`, with
`|U| = N`.

### 4.1 Structural agreement

For nonempty union:

`J(A, B) = |A ∩ B| / |A ∪ B|`.

StressKit defines `J(∅, ∅) = 1`. Empty findings must also be reported as a rate;
a high agreement caused by universal emptiness is not evidence of a recovered
mechanism.

The target mean is the expectation of `J(A, B)` for two independent draws from
the preregistered audit distribution. The estimator and interval must respect
dependence introduced by shared data, nested specifications, or repeated model
outputs.

Findings from different component universes have no structural comparison.

### 4.2 Exact uniform-set null

For independent uniform subsets `A` and `B` of sizes `k` and `l` from an
`N`-element universe, `X = |A ∩ B|` has probability

`P(X=x) = C(k,x) C(N-k,l-x) / C(N,l)`

on `max(0, k+l-N) ≤ x ≤ min(k,l)`. Therefore

`E[J] = Σ_x [x / (k+l-x)] P(X=x)`.

The common expression `k/(2N-k)` for equal sizes is a ratio-of-expectations
approximation, not the exact expectation of Jaccard. Confirmatory artifacts use
the exact finite sum or a Monte Carlo estimate with its own error bound.

Observed heterogeneous sizes are matched pairwise or through a preregistered
size-stratified null. Pooled null comparisons must include within-size results.

### 4.3 Claim agreement

For `n` categorical claims with class counts `n_c`, distinct-run disagreement is

`F = 1 - Σ_c n_c(n_c-1) / [n(n-1)]`.

This equals the fraction of unordered distinct-run pairs with different claim
classes. Modal share is `π* = max_c n_c/n`.

A claim is filable at declared tolerance `α` when `π* ≥ 1-α`. The tolerance is
a policy input and must not be described as a universal mathematical constant.

Natural-language equivalence requires a frozen deterministic or blinded human
judge with reliability measurements. An unvalidated LLM judge cannot determine
a confirmatory headline.

### 4.4 Score variation

Coefficient of variation `CV = σ/|μ|` is applicable only to a ratio-scale score
with a meaningful zero and non-negligible mean. Each claim record states why CV
is meaningful for that score. Signed, centered, ordinal, or near-zero scores use
a domain-appropriate alternative and threshold.

No universal score-CV threshold is assumed.

### 4.5 Specificity

Specificity compares the same registered statistic under real data and a null
where the claimed effect is absent. The null must preserve nuisance structure
that could make the method appear stable while removing the target relation.

The effect size can be a difference or ratio. Its direction, interval, and
minimum meaningful effect are preregistered and calibrated. A generic 1.5 ratio
is not assumed across claim types.

## 5. Uncertainty and decisions

The resampling unit is the independent experimental unit, never the much larger
set of dependent pairs. Interval algorithms and run counts are selected before
confirmatory results and validated in `docs/VALIDATION_PLAN.md`.

Confirmatory profile v0.1 randomly partitions IID runs into
disjoint pairs, evaluates one bounded pair kernel per pair, and applies a
two-sided Hoeffding bound. It is unbiased for the same independent-draw target
as the complete order-two U-statistic, but deliberately sacrifices efficiency.
Required checks split a familywise error budget with Bonferroni. Modal claim
share uses a simultaneous Hoeffding bound over a finite preregistered class
set. Real-minus-null specificity uses simultaneous group intervals and endpoint
subtraction; an unstable ratio is not required.

Universal minimum is 200 independent real runs and, when applicable, 200 null
runs. When runs are clustered, clusters replace runs as independent units and
the bounded cluster-pair kernel averages all cross-cluster comparisons. A
smaller audit is inconclusive regardless of its point estimate.

This frozen default is safe but not efficient. Pilot jackknife-normal,
percentile-bootstrap, BCa-bootstrap, normal intervals with estimated
U-statistic variance, and complete-U concentration bounds each failed a
known-truth cell, an applicability requirement, or the power/width objective.
Development results are in `docs/CALIBRATION_PILOT.md`; frozen primary and
fresh-seed replication results are in `docs/CALIBRATION_REPORT_v0.1.md`.

Each check has one state:

- `pass`: the entire confidence interval lies on the passing side;
- `fail`: the entire confidence interval lies on the failing side;
- `inconclusive`: the interval crosses the boundary, is unavailable, or the
  calibrated minimum sample size was not reached.

Point estimates remain visible but never override this rule.

Confirmatory overall state:

- `pass`: every applicable required check passes;
- `fail`: at least one preregistered validity gate fails, or a required
  stability check fails;
- `inconclusive`: no required check fails and at least one is inconclusive.

Validity gates include random-baseline separation and specificity when
applicable. No majority-vote letter grade can turn a failed validity gate into a
passing scientific verdict. Letter grades may remain as a clearly labeled
descriptive UI summary, never the normative decision.

## 6. Threshold registry

Current v0.3.0 defaults require reclassification before confirmatory use:

| Quantity | v0.3.0 value | Evidence status |
|---|---:|---|
| mean Jaccard | 0.8 | Tentative published guideline for bootstrap stability with at least 100 resamples; not yet validated as a universal multi-axis bar |
| modal share | 0.8 | Published filability tolerance `α=0.2`; policy choice, not universal truth |
| score CV | 0.25 | Numerical bar not established by the cited source |
| random margin | 3.0 | Numerical bar requires justification and calibration |
| specificity ratio | 1.5 | Numerical bar requires justification and calibration |
| at-random floor | 1.5 | Registered as `Thresholds.random_floor` in grade rule v0.4 (2026-09-03); it was a literal inside the grader before. Policy choice; requires justification and calibration |

Every future threshold record stores claim type, metric, direction, numerical
value, source or calibration artifact, applicable profile, and version.

## 7. Verification and numeric representation

A verifiable card contains sufficient statistics or content-addressed raw runs
for each headline result. Exact counts are stored where possible:

- intersection and union counts for set comparisons;
- claim-class counts;
- score count, sum, and squared-deviation data or raw scores;
- real and null group membership;
- sampling weights and dependency identifiers.

The verifier recomputes metrics, intervals, states, and summary from these data.
Floating-point summaries alone are not proof of the underlying run-level result.
Formal conformance uses exact naturals and rationals, with explicit rounding
rules at serialization boundaries.

## 8. Reporting requirements

Every public headline states:

- number of claims, method families, models, tasks, runs, and exclusions;
- confirmatory versus diagnostic status;
- exact denominator for every count;
- uncertainty and inconclusive results;
- sensitivity to size, axis weighting, and null choice;
- what failure does and does not imply.

Results attach to claim records. Generalization to a method family requires a
registered multi-task, multi-model sample designed for that inference.

## 9. Primary methodological sources

- Méloux, Portet, and Peyrard, *Mechanistic Interpretability as Statistical
  Estimation: A Variance Analysis*, arXiv:2510.00845v4.
- Mahale, *Explanation Multiplicity: Circuit-Level Interpretability Evidence
  Does Not Survive Defensible Analytic Variation*, arXiv:2608.13754v1.
- Lan et al., *Make Mechanistic Interpretability Auditable*,
  arXiv:2606.00033v1.
- Nguyen, *Concentration-based Confidence Intervals for U-statistics*,
  arXiv:1903.01679v1.
- Maurer and Pontil, *Empirical Bernstein Bounds and Sample Variance
  Penalization*, arXiv:0907.3740v1.

Source presence does not imply endorsement of a numerical rule absent from the
source.
