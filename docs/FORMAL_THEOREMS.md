# StressKit formal theorem index

Pinned environment: Lean 4 `v4.33.0`, mathlib `v4.33.0` at commit
`db584cd6d46c92f209a44c0f1c829460d327499d`.

## Finite-set agreement

Source: `formal/Stresskit/Core.lean`.

- `jaccard_empty_empty`: the registered empty-pair convention is one.
- `jaccard_self`: every finite set agrees with itself.
- `jaccard_comm`: argument order does not change Jaccard agreement.
- `jaccard_nonnegative` and `jaccard_le_one`: exact rational result lies in
  `[0, 1]`.

## Categorical claims

Source: `formal/Stresskit/Claims.lean`.

- `flipRate_empty` and `flipRate_singleton`: categorical flip rate is
  unavailable below two runs.
- `pairDisagreementCount_constant`: identical labels have no disagreeing
  distinct-run pairs.
- `filable_iff`: executable filability is equivalent to the declared
  modal-share inequality when modal share exists.

## Decisions

Source: `formal/Stresskit/Decision.lean`.

- `atLeast_pass_iff` and `atMost_pass_iff`: pass requires the whole interval on
  the passing side.
- `unavailable_is_inconclusive` and `underpowered_is_inconclusive`: missing
  uncertainty or insufficient sample size cannot pass or fail.
- `combineChecks_fails_if_any`: one required validity-gate failure forces the
  overall failure state.
- `combineChecks_all_pass` and `combineChecks_empty`: unanimous nonempty pass
  is required; an empty checklist is inconclusive.

## Uniform-set null

Source: `formal/Stresskit/RandomNull.lean`.

- executable exact hypergeometric expectation over the full finite support;
- correct empty-set boundary behavior;
- exact checked examples for heterogeneous subset sizes;
- `ratioOfExpectations_is_not_exact`: the common shortcut differs from exact
  expected Jaccard on a concrete finite case.

## Cross-language conformance

Source: `formal/Stresskit/Conformance.lean` and
`formal/golden/vectors.json`.

- `jaccardVectors_conform`;
- `randomNullVectors_conform`, including the `N=144, k=15` case;
- `decisionVectors_conform`.

Python consumes the same committed values in
`tests/test_formal_conformance.py`. Exhaustive Python enumeration additionally
checks exact random-set expectations for every subset-size pair through
universe size six and fixed-core/noise expectations through universe size
seven.

## Explicit non-claims

Lean does not prove IID sampling, confidence-interval coverage, threshold
construct validity, semantic equivalence of natural-language claims, model
correctness, or scientific quality of null controls. Those remain statistical
and empirical validation gates.
