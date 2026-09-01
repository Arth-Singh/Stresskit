# StressKit launch-claims policy

Status: binding draft until benchmark preregistration freezes. Public wording
must be generated from verified artifacts and include denominators.

## Claims supported now

The current repository supports these narrow statements:

- StressKit has a normative confirmatory v0.1 core using preregistered IID
  specification draws, disjoint pairs, finite-sample Hoeffding intervals,
  Bonferroni familywise coverage, and three-state decisions.
- Deterministic metric identities and bounds listed in `docs/FORMAL_THEOREMS.md`
  are machine-checked in pinned Lean 4/mathlib. This does not formalize the
  statistical coverage theorem or construct validity.
- Two frozen seeds executed 156,000 simulation trials across S1–S5, S7, and S9,
  plus exact/deterministic S6 and S8 checks. This is calibration against named
  scenarios, not universal validation.
- In the frozen S9 clustered stress test, cluster-unit intervals covered the
  truth in 99.4–99.8% of trials while the deliberately invalid run-unit analysis
  covered only 74.0–76.6%. Report scenario, cluster counts, and overcoverage.
- In exact S6 construction, an OAT battery reports Jaccard 1.0 while crossed
  specifications report 0.5. This is a counterexample showing that OAT cannot
  serve as a general confirmatory design.
- A candidate frame currently contains 29 code-linked claims/instruments from
  14 pinned upstream repositories. Two entries were excluded before outcomes:
  one missing source license and one unavailable pinned model. Static audit
  verifies commits, trees, licenses, entrypoint presence, and Python parsing;
  it is not execution or reproduction evidence.

## Claims forbidden until evidence changes

Do not publish any equivalent of:

- “We proved StressKit correct.”
- “Lean proves our statistical method or mechanistic-interpretability claims.”
- “We tested 29 methods” while 29 is a candidate-claim count, not completed
  method executions.
- “None of mechanistic interpretability makes sense.”
- “Most papers fail,” “method X is false,” or any paper-level verdict.
- “Externally validated,” “peer reviewed,” or “independently replicated” until
  named reviewers/replicators approve an immutable version.
- Counts that omit pre-freeze exclusions, reproduction failures, protocol
  deviations, or inconclusive cards.

## Permitted benchmark headline template

Only after registry freeze and card verification:

> We preregistered **N claims from R repositories across F method families**.
> Of these, **E were excluded before outcomes**, **X reproduced**, and **Y/X**
> cleared every registered stability/specificity gate; **A** failed an audit
> gate, **I** were inconclusive, **D** deviated from protocol, and **U** did not
> reproduce. Results are claim-level, not paper-level truth judgments.

Every number must be derived from machine-verified cards and the immutable
registry. Link raw artifacts, nulls, manifests, code commits, environment locks,
and deviations beside the headline.

## High-impact narrative that remains accurate

1. **Problem:** analytic choices can create apparently stable stories.
2. **Counterexample:** show exact OAT J=1 versus crossed J=0.5.
3. **Failure found in our own tool:** legacy diagnostic grades were not a
   confirmatory certificate; empty findings also needed explicit representation.
4. **Repair:** separate diagnostic and confirmatory profiles, formalize the
   deterministic core, calibrate finite-sample behavior, and make cards
   independently recomputable from raw runs.
5. **Test:** freeze open-code claims and nulls before outcomes, then report every
   exclusion/failure/inconclusive result.

Strong story comes from adversarial self-correction plus complete denominators,
not predetermined mass failure.
