# StressKit v1: Autonomous, verifiable audits of interpretability claims

Draft protocol preprint. Results sections remain intentionally unpopulated until
all public-release gates pass.

## Abstract

Interpretability claims combine paper wording, code, analytic choices, stochastic
execution, and statistical decisions. StressKit v1 turns one concrete claim into
a frozen audit whose raw outputs, reducers, controls, uncertainty, and publication
state can be verified offline. Agents extract and challenge candidate audits, but
deterministic calibrated code alone decides. The protocol reports claim-level
reproduction, stability/specificity, external utility, generalization, and
evidence confidence; it computes neither a paper score nor a whole-paper truth
verdict. This draft preregisters methods and research questions. Benchmark counts
will be inserted only after an untouched silent cohort, global multiplicity
correction, independent rerun, response windows, and release verification.

## Research questions

1. What precision/abstention tradeoff does the three-agent compiler achieve?
2. What fraction of eligible open-code claims reproduce under pinned resources?
3. Which reproduced claims survive claim-specific perturbations and nulls?
4. Do internal methods improve external tasks over strong non-internals baselines?
5. How do outcomes differ across CoT, probes/monitoring, steering/control,
   lenses/model diffing, intervention prediction, and circuits/SAEs?

## Methods

The sampling frame uses an outcome-blind period census. Every inclusion and
exclusion remains visible. All eligible claims freeze; if fewer than 20 survive,
the same census extends backward by one month repeatedly. Execution order is
round-robin by uncovered method family and then cheapest compute tier.

Two isolated extractor model families/providers and one critic bind wording to
exact source anchors. Any disagreement, unsupported wording, injection signal,
missing evidence, invalid null, unsafe executor, or unsupported profile causes
abstention. Agents cannot vote through a missing validity gate.

Seven claim profiles use deterministic reducers and bounded independent-unit
inference. Positive, negative/randomization, specificity, stability, external
utility, and held-out generalization checks are distinct. All frozen primary
checks share a release-wide Holm–Bonferroni family. Raw objects form a complete
SHA-256 closure and every result receives an independent rerun at a declared
reproducibility level.

The evidence board contains all final, excluded, and abstained rows. Paper pages
list claim evidence without a paper verdict. Adverse named results receive a
14-day author-response window and carry responses with publication.

## Calibration and compiler evaluation

Frozen local artifacts report the bounded-profile primary/fresh-seed calibration
and 300 planted compiler cases. These establish software and constructed-case
gates only. They do not substitute for live-provider evaluation, construct-valid
paper-specific controls, benchmark execution, or external review.

## Flagship preregistration

The separate flagship study tests whether each training example's loss-gradient
projection onto frozen persona/misalignment probes predicts held-out behavioral
shift across datasets, seeds, and models better than text-only, loss/norm, and
output-only baselines. Random probes, label permutations, benign fine-tunes,
held-out models, and matched compute are controls. Unresolved artifact licenses
currently force abstention; no substitute artifact is allowed.

## Results

Not yet run. Populate only from the verified final evidence board:

- compiler precision and abstention with confidence intervals: **pending**;
- eligible and excluded claim denominators: **pending**;
- reproduction outcomes: **pending**;
- robustness outcomes: **pending**;
- external utility outcomes: **pending**;
- method-family strata: **pending**;
- flagship result: **abstain pending licenses**.

## Limitations and validation status

Verdicts are conditional on frozen claim wording, profiles, controls, resource
budgets, and specification distributions. A failed claim-level gate does not
show that a paper or method family is false. Passing does not establish a unique
mechanism or universal utility. Current external validation status is
`not obtained`.
