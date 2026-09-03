# StressKit v0.3.0 method audit

Date: 24 August 2026. Status: initial internal audit before confirmatory benchmark
expansion.

This audit distinguishes software correctness from scientific validity. The
existing test suite and `stresskit verify` provide strong evidence that the
implemented summaries are deterministic and internally recomputable. They do
not establish that every threshold, interval, aggregate, or grade has the
claimed scientific meaning.

## Summary

| ID | Finding | Severity | Required resolution |
|---|---|---|---|
| A1 | Three numerical default bars lack direct support in their cited sources | critical | reclassify, calibrate, or remove before confirmatory use |
| A2 | Point estimates determine pass/fail even when intervals are inconclusive | critical | introduce normative three-state decisions |
| A3 | Documented random-null grade gate is not implemented | high | replace letter-grade logic with explicit validity gates |
| A4 | Pooled OAT comparisons do not define a confirmatory specification-space estimand | high | separate diagnostic OAT from confirmatory sampling |
| A5 | Coverage of custom pairwise bootstrap intervals is uncalibrated | high | known-truth coverage simulations and external review |
| A6 | Analytic random-Jaccard value is an approximation | medium | use exact finite expectation or bounded Monte Carlo |
| A7 | Hash-only large cards cannot recompute run-level metrics from the card alone | medium | store sufficient statistics or content-addressed raw runs |
| A8 | Current evidence is too narrow for claims about mechanistic interpretability broadly | critical for launch | preregister a systematic multi-family benchmark |

## A1 — threshold provenance

`Thresholds` currently defaults to Jaccard 0.8, modal share 0.8, score CV
0.25, random margin 3.0, and specificity ratio 1.5.

- Méloux et al. propose mean Jaccard above 0.8 as a **tentative** bootstrap
  guideline with at least 100 resamples. They report score CV but do not propose
  0.25 as a universal score-CV decision boundary.
- Mahale uses modal-share tolerance 0.8 as the loosest preregistered filability
  policy in one regulatory framing. The paper reports an analytic random line
  and warns that even large multiples of chance need not imply stability; it
  does not establish 3.0 as a general pass bar.
- Lan et al. call for empirically refined audit guidelines and explicitly avoid
  presenting their examples as definitive numerical standards. The paper does
  not establish a 1.5 specificity ratio.

Resolution: the threshold registry in `METHOD_SPEC.md` labels provenance
honestly. Confirmatory defaults require simulation calibration and external
review. Until then, existing bars are exploratory policy settings.

## A2 — binary point-estimate decisions

`make_check` sets `passed` from the point estimate. A confidence interval that
crosses the threshold only changes `robust` and overall `confidence`; the card
still receives a letter grade from the point-estimate decisions.

Consequence: two statistically unresolved experiments on opposite sides of a
threshold receive different grades despite both supporting the same conclusion:
insufficient evidence to decide.

Resolution: normative decisions become `pass`, `fail`, or `inconclusive`, based
on the whole interval and calibrated minimum sample size. Point-estimate grades
may remain only as a backward-compatible descriptive field.

## A3 — random-null grade gate mismatch

Documentation says grade D applies when no checks pass **or** structural overlap
is indistinguishable from random. `grade_checks` tests its `at_random` condition
only when zero checks pass, which is behaviorally redundant with the final D
branch. A finding can therefore fail `beats_random`, pass other checks, and
receive B or C.

Resolution: random-baseline separation and specificity become explicit
validity gates. A failed required validity gate cannot be rescued by unrelated
checks.

## A4 — pooled OAT estimand

The battery varies one axis at a time around a base configuration, then grades a
pooled pairwise statistic across runs from different axes. Comparing a seed-only
variant with a template-only variant creates a pair differing on two axes even
though that joint specification was never run. Axis run counts also set implicit
weights.

This pooled set is a useful diagnostic sample of nearby outputs, but it is not a
probability sample from the full crossed specification space.

Resolution: diagnostic mode reports per-axis results and sensitivity addresses.
Confirmatory mode enumerates or probability-samples a frozen specification
space with recorded weights and interactions.

## A5 — interval calibration

Mean pairwise Jaccard and flip rate are U-statistics whose pairs share runs. The
implementation resamples runs and removes pairs containing duplicate original
indices to avoid self-pair inflation. This is a thoughtful correction, but its
finite-sample coverage has not been established across the small, heterogeneous,
or dependent regimes used by reference cards.

Percentile bootstrap intervals for modal share, score CV, and real/null ratios
also require calibration near ties, near-zero means, and small denominators.

Resolution: run the simulation families and acceptance rules in
`VALIDATION_PLAN.md`; compare candidate U-statistic bootstrap, jackknife,
cluster bootstrap, and exact methods where available.

## A6 — random-Jaccard expectation

`k/(2N-k)` substitutes expected intersection into a nonlinear ratio. It is not
`E[J]`. For `N=144, k=15`, the approximation is approximately 0.05495 while the
exact hypergeometric expectation is approximately 0.05664, about 3% higher.

The current engine grades against a size-matched Monte Carlo estimate, which is
preferable, while retaining the analytic approximation as a cross-check.

Resolution: implement exact finite expectation for homogeneous and
heterogeneous sizes when tractable, plus a bounded and reproducible Monte Carlo
fallback.

## A7 — hash-only cards

Large component sets are omitted from cards and represented by digests. The
verifier can confirm stored hashes but cannot reconstruct pairwise intersections
or recompute structural metrics without the raw run artifacts.

Resolution: store pairwise sufficient counts, or bind cards to a
content-addressed artifact manifest that the verifier can load. Verification
levels must state whether the card is self-contained or externally backed.

## A8 — evidence breadth and launch claims

The current scoreboard contains eight artifacts but only about three method
families: head-level attribution patching, a Jacobian-lens release, and
activation-reader checkpoints. Results are mixed: three A, one B, two C, and two
D under exploratory v0.3.0 grading.

This evidence cannot support “none of mechanistic interpretability works,” nor a
claim about the field as a whole. It does support narrower observations about
specificity failures, prompt sensitivity, and unresolved stability in named
artifacts.

Resolution: use a preregistered inclusion frame covering at least the breadth
defined in `GOAL.md`. Report artifact-level outcomes and denominators, including
passes and reproduction failures.

## Audit disposition

Do not expand the confirmatory scoreboard or publish field-wide verdicts until
A1–A5 are resolved. Existing cards remain valuable exploratory evidence and
regression fixtures. Preserve them unchanged under their recorded StressKit
version; never silently reinterpret historical grades under a new protocol.

## Status on 3 September 2026

Recorded after the diagnostic-battery self-calibration
(`CALIBRATION_REPORT_v0.2.md`). Findings A1 to A8 above are the audit record
of 24 August and are left as written.

| ID | Status | What changed |
|---|---|---|
| A1 | open | Bars unchanged; the at-random floor (1.5) is now a registered threshold (`Thresholds.random_floor`, `METHOD_SPEC.md` registry) instead of a literal in the grader. |
| A2 | resolved for the descriptive grade | Grade rule v0.4: a check counts only when its whole interval clears the bar; cards record `verdict.grade_rule`; every card was regraded from its recorded checks and keeps its v0.3 grade in its notes. |
| A3 | resolved | The at-random floor is applied first under both rules; a decided specificity fail caps the letter at C; a battery without a null control caps it at B. |
| A4 | unchanged | Diagnostic OAT and the confirmatory profile stay separate. |
| A5 | measured | Coverage of the shipped percentile bootstrap for all five checks at 6 to 100 runs is in `CALIBRATION_REPORT_v0.2.md` §3; the intervals stay on the diagnostic path with their measured coverage stated. |
| A6, A7, A8 | unchanged | |

New findings from the self-calibration:

| ID | Finding | Severity | Status |
|---|---|---|---|
| A9 | The null control's own score is recorded on the card but never checked. Of the 17 structure-preserving nulls that fail specificity, 6 still score as well as or better than the real data (CoAx, Greater-than, the three homonym profiles, SAE causal inertness): on those cards the specificity failure says the null was too soft, not that the method is non-specific. `references/null_score_leak.py` reports it per card; a battery check would need a declared score polarity on `Finding` and a schema change, so it is a candidate for schema 0.6, not a v0.4 change. | high | open, reported per card |
| A10 | The bootstrap axis runs every resample at the base seed by design, so a finder that ignores its data repeats its base finding on every bootstrap run and inflates the pooled stability metrics: a seeded random subset grades C and a random direction B under the default battery, D under a seeds-only battery. The engine now writes a "bootstrap axis" note; the grade does not react to it. | high | open; the note is the mitigation |
| A11 | Without a null control the diagnostic letter separates nothing: a constant set, an index ranker, a memorised planted set and a fixed direction all grade the same as the honest finder (A under v0.3, B under v0.4). The v0.4 cap makes the ceiling explicit; it does not add information. | high | by design; stated on every card without a null |
| A12 | The seeds-axis vacuity detector ignored `Finding.vector`, flagging direction finders whose vector changed on every seed. | low | fixed |

Disposition: unchanged in substance. The letter grades are descriptive
diagnostics with measured error rates, not confirmatory decisions; the
confirmatory profile and its 200-run floor stand. The regrade of 3 September
is the one deliberate reinterpretation of historical grades: it is recorded
on every card (`verdict.grade_rule`, the v0.3 grade in the notes), the
recorded checks and intervals were not touched, and the full migration
table is in `RESULTS.md`.
