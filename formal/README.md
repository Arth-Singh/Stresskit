# StressKit formal core

This Lean 4 project machine-checks deterministic finite claims made by the
StressKit method specification. It is pinned to Lean and mathlib `v4.33.0`.

Covered now:

- exact rational finite-set Jaccard definition, symmetry, identity, and bounds;
- categorical pair-disagreement and flip-rate base cases;
- modal-share filability semantics;
- pass/fail/inconclusive interval decisions and fail-fast aggregation;
- executable exact hypergeometric Jaccard expectation, including checks that
  the ratio-of-expectations shortcut is not exact.
- compiled conformance theorems for the same exact vectors exercised by the
  Python test suite.

Statistical coverage, threshold construct validity, independence assumptions,
and scientific quality of a null are intentionally outside theorem-prover
scope. Those require simulation and external review.

Build with:

```bash
cd formal
lake update
lake exe cache get
lake build
```
