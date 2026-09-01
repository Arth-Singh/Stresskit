# StressKit validation plan

Status: method-validation protocol 0.1 executed for S1--S9. Results and
fresh-seed replication are frozen in `CALIBRATION_REPORT_v0.1.md`. Construct
validity and external benchmark protocol remain preregistration work.

## 1. Questions

Validation must answer four different questions:

1. **Mathematical correctness:** do definitions and deterministic decisions
   have the claimed properties?
2. **Implementation conformance:** does Python compute the specified quantities?
3. **Statistical calibration:** do intervals and decisions behave as advertised
   under controlled data-generating processes?
4. **Construct validity:** do successful and failed audits correspond to what
   researchers mean by a stable, specific interpretability claim?

No single proof or benchmark answers all four.

## 2. Known-truth simulation families

Each family is evaluated across multiple universe sizes, finding sizes, run
counts, and random seeds.

### S1 — deterministic stable finding

Every run returns the same nonempty set, claim, and score. Expected structural
and claim agreement are exactly one. This is a positive control, not a realistic
model of all good research.

### S2 — uniform random finding

Each run returns a uniform size-matched subset. Exact expected Jaccard follows
the hypergeometric sum in `METHOD_SPEC.md`. This tests random-baseline
calculation and false-certification behavior.

### S3 — stable core plus noise

Each finding contains a fixed core and sampled nuisance components. This creates
a continuum of known structural agreement and tests power near decision bars.

### S4 — multiple valid modes

Runs sample between two or more fixed mechanisms with known mixture weights.
Within-mode agreement is high while pooled agreement and modal share are known.
This tests non-identifiability and claim multiplicity.

### S5 — heterogeneous finding size

Finding size varies independently or jointly with specification choices. This
tests size confounding, stratified nulls, and pooled-statistic reversals.

### S6 — axis interaction

Individual one-at-a-time perturbations appear stable while combinations change
the finding, and vice versa. This establishes the limits of diagnostic OAT
sweeps and validates confirmatory crossed or sampled designs.

### S7 — non-specific method

Real and null data produce equally stable outputs. A second variant has a known
real/null effect size. These test specificity intervals and decision power.

### S8 — score distributions

Positive log-normal scores, near-zero scores, signed scores, and heavy-tailed
scores test when CV is meaningful and when it must be rejected.

### S9 — dependent runs

Runs share datasets, seeds, model checkpoints, or nested specifications. This
tests cluster-aware uncertainty and prevents pair counts from masquerading as
independent sample size.

## 3. Calibration grid

Initial grid:

- run counts: 5, 10, 20, 50, 100, 200;
- universe sizes: 20, 100, 1,000;
- finding fractions: 1%, 5%, 20%, 50%;
- at least 2,000 independently seeded simulation trials per core cell;
- nominal interval levels: 90%, 95%, and 99%;
- boundary distances spanning clear pass, near-boundary, and clear fail.

Large grids may use deterministic sharding. Every shard records configuration,
seed interval, code commit, and output digest.

## 4. Measurements

For every applicable cell report:

- estimator bias and root-mean-square error;
- empirical interval coverage and average width;
- pass, fail, and inconclusive rates;
- false-pass and false-fail rates;
- power versus run count and effect size;
- sensitivity to bootstrap algorithm and resampling unit;
- Monte Carlo standard error for each reported calibration rate.

Coverage is evaluated against exact finite quantities where available and a
separately converged high-precision reference otherwise.

## 5. Acceptance rules

A method/profile combination can enter confirmatory StressKit only when:

- a nominal asymptotic or bootstrap 95% interval has core empirical coverage
  between 93% and 97%, while a finite-sample guaranteed interval has no
  observed violation inconsistent with its stated lower bound and separately
  meets a preregistered width or power requirement;
- no known-invalid scenario has a false-pass rate above 5%;
- deterministic positive controls do not fail;
- sample-size and dependence requirements are encoded as executable checks;
- known limitations and excluded regimes are documented;
- results reproduce from a fresh seed range.

Failure triggers method revision, a larger minimum run count, narrower scope, or
removal of the affected check. It never triggers post-hoc relabeling of the
simulation scenario.

## 6. Lean verification scope

Create a pinned Lean 4 and mathlib project covering:

- finite-set Jaccard definition and basic laws;
- exact pairwise-mean and claim-flip identities;
- modal-share/filability decision equivalence;
- exact finite random-subset expectation;
- pass/fail/inconclusive boundary logic;
- overall validity-gate logic and monotonicity properties.

Bootstrap coverage, empirical threshold validity, and scientific null quality
are explicitly outside the theorem-prover guarantee.

## 7. Python–Lean conformance

- Exhaustively enumerate small universes and categorical-count vectors.
- Produce versioned golden vectors containing exact inputs and rational outputs.
- Check Python results against exact values within declared rounding tolerance.
- Run both Lean build and Python conformance tests in CI.
- Make formal theorem statements linkable from generated documentation.

## 8. Construct-validity pilots

Before auditing outside work, run:

- compiled or synthetic transformers with known circuits;
- planted-feature SAEs with recoverable and deliberately nonrecoverable
  features;
- a stable causal direction and a matched shuffled-label direction;
- an oracle that answers known probes and abstains on nulls;
- adversarial methods that emit stable but data-independent findings.

Success requires distinguishing stability from specificity: stable nonsense must
not receive a passing confirmatory verdict.

## 9. External review

Before protocol freeze, request review from at least one statistician familiar
with U-statistics/bootstrap inference, one mechanistic-interpretability
researcher not involved in implementation, and one Lean/mathlib contributor or
experienced formalizer. Record reviews and resolutions publicly.

## 10. Benchmark eligibility

Prefer research with public executable code and downloadable artifacts. A
target qualifies only when:

- its claim is concrete and load-bearing;
- exact usage mode can be reproduced;
- model/data licenses permit the audit;
- required resources fit declared budget;
- meaningful perturbations and a null can be specified;
- outputs can be converted without silently changing the claim.

Target discovery is systematic. Search sources, cutoff date, exclusions, and
reproduction failures are logged so easy or dramatic failures are not
selectively sampled.

## 11. Compute boundary

Method development, Lean, and most simulations are CPU work. GPU benchmarking
starts only after method and target preregistration freeze.

Nibi jobs use Slurm, pinned environments, smallest fitting GPU, restartable
shards, capped arrays, durable final artifacts under project storage, and
disposable caches under scratch. A human establishes Duo-approved SSH control;
agents reuse it without requesting credentials or passcodes in chat.

## 12. Outputs

- machine-readable calibration results;
- generated calibration report and power curves;
- Lean project and theorem index;
- conformance vectors and CI checks;
- frozen benchmark registry and analysis plan;
- public deviation log;
- versioned methodology release notes.
