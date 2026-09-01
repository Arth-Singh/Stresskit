# StressKit confirmatory calibration report v0.1

Status: frozen method-validation result. This report validates the conservative
confirmatory profile implemented in `stresskit.confirmatory`; it contains no
mechanistic-interpretability benchmark outcomes.

## Decision

StressKit promotes seeded disjoint-pair estimation with finite-sample
Hoeffding intervals as confirmatory profile v0.1. Multiple required checks
split a familywise error budget with Bonferroni. Real-minus-null specificity
uses simultaneous real and null intervals. Runs must be IID draws from the
frozen specification distribution; clustered data use clusters as independent
units. Current universal minimum is 200 independent runs per real or null
group. A smaller result remains inconclusive even when its point estimate
clears a threshold.

This profile is conservative, not statistically efficient. Percentile
bootstrap, BCa bootstrap, jackknife-normal, normal inference with an unbiased
U-statistic variance estimate, and Nguyen's complete-U concentration interval
remain rejected or safety-reference candidates as documented in
`CALIBRATION_PILOT.md`.

## Frozen artifacts

`artifacts/calibration/manifest.json` records SHA-256 and source digests for
four machine-readable outputs. Each study used 2,000 trials per cell and a
second 2,000-trial run with disjoint master seed. Primary seed was `20260824`;
replication seed was `20260825`.

## S1--S5 structural agreement

Grid: five exact-truth scenarios, run counts 5, 10, 20, 50, 100, and 200,
nominal confidence 95%, Jaccard boundary 0.8.

- Primary minimum observed coverage across 30 cells: 99.80%.
- Replication minimum observed coverage across 30 cells: 99.85%.
- Maximum false-pass rate across every below-boundary cell: 0% in both runs.
- Deterministic stable findings passed at 100 and 200 runs under a single-check
  95% interval. Confirmatory familywise allocation and multi-gate audits use the
  more conservative universal floor of 200.
- At 200 runs, uniform and heterogeneous random findings failed in 100% of
  trials. Stable-core-plus-noise failed in 100%. The two-mode scenario failed
  in 66.15% and remained inconclusive in 33.85%; it never falsely passed.

Coverage above 95% is expected: this is a finite-sample lower-bound method.
Acceptance therefore uses guarantee validity, false decisions, width, and
power rather than demanding nominal 93--97% empirical coverage. At 200 runs,
the unadjusted structural interval's deterministic-control half-width was
0.1358; multi-check Bonferroni intervals are wider.

## S6 axis interactions

An exact two-axis construction returned the same finding for base and both OAT
variants, producing diagnostic mean Jaccard 1.00. One crossed combination
activated a disjoint finding; the full crossed mean was 0.50. This confirms
that OAT pooling cannot receive a confirmatory certificate.

## S7 specificity

Real and null groups each used 100, 200, or 400 runs. Both primary and
replication studies observed 100% coverage and zero false passes or false
fails in every 2,000-trial cell.

- Stable real versus stable null: always inconclusive at 100 runs and always
  failed the required difference of 0.2 at 200 and 400 runs.
- Stable real versus uniform null: passed in every trial from 100 runs onward.

These are controlled extremes, not evidence that every scientific null is
valid. Null construction remains claim-specific and externally reviewable.

## S8 score applicability

All five applicability controls matched preregistered expectations. CV was
accepted only for nonnegative ratio-scale scores with mean above the declared
floor. Near-zero, signed, and negative-score cases returned unsupported rather
than a confirmatory value. Heavy-tailed positive data remained mathematically
applicable but produced a large descriptive CV; no universal CV threshold is
promoted.

## S9 dependent runs

Each independent cluster was repeated 20 times. The target was exact mean
Jaccard 0.58 for a two-mode cluster distribution.

| independent clusters | apparent runs | cluster-unit coverage, primary / replication | naive run-unit coverage, primary / replication |
|---:|---:|---:|---:|
| 20 | 400 | 99.8% / 99.7% | 76.1% / 76.6% |
| 50 | 1,000 | 99.7% / 99.7% | 74.0% / 75.5% |
| 100 | 2,000 | 99.5% / 99.4% | 74.2% / 74.85% |

Counting dependent repeats as independent therefore converted a nominal 95%
procedure into roughly 74--77% coverage. Confirmatory cards record independent
unit counts; repeated outputs cannot inflate sample size.

## Remaining scope limits

- Coverage guarantee assumes the declared independent units are actually IID.
- The profile handles bounded structural agreement, exact size-matched random
  separation, a finite registered claim-class set, and structural specificity.
- Ranking metrics, continuous effect metrics, score-variation thresholds, and
  complex survey weights need their own calibrated profiles.
- Construct validity, benchmark claim maps, model/data licensing, and external
  protocol review remain separate launch gates.
