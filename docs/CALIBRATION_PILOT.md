# Structural interval calibration pilot

Status: exploratory pilot, not a preregistered confirmatory result.

This pilot was run before protocol freeze to reject unsuitable interval
algorithms and expose missing scenarios. It must not be used as benchmark
evidence or as a launch claim.

Follow-up complete: conservative paired-Hoeffding profile v0.1 was frozen and
replicated across S1--S9. See `CALIBRATION_REPORT_v0.1.md`. This pilot remains
the rejection record for alternative interval candidates.

## Configuration

- master seed: `20260824`;
- scenarios: default S1--S5 structural scenarios in
  `stresskit.calibration`;
- decision boundary: mean Jaccard `0.8`, used only to inspect decision errors;
- jackknife-normal: 500 trials per cell, run counts 5, 10, 20, 50, and 100;
- self-pair-free percentile bootstrap: 200 trials per cell, 300 bootstrap
  replicates, run counts 5, 10, and 20;
- interval target: exact finite expected Jaccard for two independent scenario
  draws;
- nominal coverage: 95%.

Monte Carlo standard errors were 0--1.7 percentage points for jackknife cells
and 0--2.4 percentage points for bootstrap cells.

## Empirical coverage

### Jackknife-normal interval

| Scenario | n=5 | n=10 | n=20 | n=50 | n=100 |
|---|---:|---:|---:|---:|---:|
| S1 deterministic stable | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| S2 uniform random | 96.8% | 98.2% | 98.4% | 99.2% | 99.0% |
| S3 stable core plus noise | 95.2% | 97.2% | 98.2% | 99.6% | 99.6% |
| S4 two fixed modes | 83.6% | 87.6% | 90.2% | 87.8% | 89.6% |
| S5 heterogeneous sizes | 92.4% | 94.0% | 95.0% | 94.4% | 93.6% |

### Percentile-bootstrap interval

| Scenario | n=5 | n=10 | n=20 |
|---|---:|---:|---:|
| S1 deterministic stable | 100.0% | 100.0% | 100.0% |
| S2 uniform random | 98.0% | 99.5% | 99.0% |
| S3 stable core plus noise | 97.0% | 99.0% | 100.0% |
| S4 two fixed modes | 86.5% | 97.0% | 95.0% |
| S5 heterogeneous sizes | 92.5% | 94.0% | 94.0% |

Deterministic S1 necessarily has a zero-width interval at one and therefore
100% coverage; it is a positive control rather than a nominal-coverage cell.

## Decision

Neither candidate enters the confirmatory profile.

The normal interval persistently undercovers the multimodal scenario, including
at 100 runs. The percentile bootstrap materially undercovers that scenario at
five runs and overcovers several symmetric or low-variance scenarios. More
trials cannot repair these effect sizes; the algorithms or their applicability
rules must change.

Next calibration revision will add:

1. studentized and finite-sample concentration candidates for bounded
   order-two U-statistics;
2. an independent-pair estimator as a simple auditable reference;
3. explicit degeneracy and multimodality diagnostics;
4. S6--S9 interaction, specificity, score, and dependent-run studies;
5. at least 2,000 trials per frozen core cell and fresh-seed replication.

Until one candidate passes the frozen acceptance rules, StressKit v0.3 letter
grades remain exploratory and no benchmark receives a confirmatory verdict.

## Finite-sample safety reference

A third developmental candidate randomly partitions runs into disjoint pairs,
computes one bounded Jaccard value per pair, and applies the two-sided Hoeffding
bound to those independent values. This targets the same independent-draw
expectation but intentionally gives up the efficiency of the complete
U-statistic.

Across 2,000 trials for every default S1--S5 cell and run counts 5, 10, 20, 50,
100, and 200, empirical coverage was 99.8--100.0%. This is consistent with its
finite-sample lower-bound guarantee, but confirms that it is conservative:

- a deterministic finding with true Jaccard one did not clear the 0.8 boundary
  until 100 runs;
- the two-mode finding, whose true mean is 0.64, was decisively below 0.8 in
  only 33.1% of 100-run trials and 66.1% of 200-run trials;
- uniform-random findings were decisively below 0.8 from 10 runs onward.

This candidate is retained as a safety reference and possible conservative
confirmatory profile, not declared the final efficient interval. For
finite-sample guaranteed intervals, empirical overcoverage does not invalidate
the guarantee; interval width and decision power become separate acceptance
criteria.

## Additional developmental candidates

These smaller pilots were used only for early rejection and implementation
debugging. Their trial counts are below the frozen 2,000-trial requirement.

### BCa run-level bootstrap

With 100 trials per default S1--S5 cell, 300 bootstrap replicates, and 5, 10,
or 20 runs, BCa coverage on the two-mode scenario was 89%, 92%, and 97%,
respectively. At five runs its false-pass rate against the 0.8 boundary was
11%. BCa therefore does not repair small-sample multimodal undercoverage and
is not promoted.

### Normal interval with unbiased finite-sample U variance

An exact moment identity produced an unbiased estimator of the complete
order-two U-statistic's finite-sample variance. That fact alone was
insufficient for usable normal inference. Across 200-trial cells, negative
variance realizations made only 52.5--88.5% of two-mode intervals available at
5--20 runs. Conditional coverage was 74.3%, 95.3%, and 89.3%. At 100 runs,
two-mode coverage remained 89.5%. The candidate is rejected as a general
interval; unavailable estimates remain inconclusive rather than being clipped
to zero.

### Empirical concentration interval for the complete U-statistic

Equation (20) of Nguyen (2019) was implemented with a symmetrized order-four
variance U-statistic. An algebraic edge-sum identity reduces its computation
from O(n^4) to O(n^2). In 500-trial cells through 100 runs it showed 100%
empirical coverage, as expected for a conservative finite-sample bound, but a
deterministic finding with true Jaccard one still did not clear the 0.8
boundary at 100 runs. Uniform-random findings resolved below 0.8 from 50 runs;
the two-mode finding remained inconclusive at 100. This interval remains a
second safety reference, not the efficient default.

One attempted scale-up initially became unexpectedly slow because the ordinary
floating-point Jaccard path had been routed through exact rational arithmetic.
The run was interrupted, the hot path restored to direct floating-point set
counts, and exact rationals kept in a separate conformance function. Regression
tests cover both paths.

## Reproduction

The pilot can be rerun through the public module interface:

```bash
PYTHONPATH=src python -m stresskit.calibration \
  --runs 5,10,20 \
  --trials 200 \
  --bootstrap-replicates 300 \
  --seed 20260824
```

The calibration result schema stores additive sufficient aggregates and trial
start indices so larger runs can be deterministically sharded. Frozen studies
will persist full JSON outputs, environment metadata, code commit, and digests.
