# StressKit v0.1 confirmatory-core review request

Status: review packet ready; no independent endorsement received.

## What we are asking reviewers to evaluate

StressKit audits whether one preregistered mechanistic-interpretability claim
is stable and specific over an explicit specification distribution. It does
not prove that the claim is true, unique, causal beyond its registered tests,
or representative of a paper or method family.

Please review one or more independent layers:

1. **Formal layer:** do Lean theorems state the intended deterministic facts,
   and do Python conformance tests faithfully connect implementation to them?
2. **Statistical layer:** does paired-Hoeffding plus Bonferroni give the claimed
   finite-sample coverage under IID run units? Find counterexamples, especially
   dependence, clustering, adaptive specification choice, empty findings, and
   data-dependent claim classes.
3. **Construct layer:** for each benchmark claim, do finding map, universe,
   null, thresholds, and specification distribution test the upstream statement
   without strengthening it?
4. **Replication layer:** can a clean environment regenerate frozen calibration
   artifacts and independently verify cards from raw runs?
5. **Communication layer:** identify any sentence that implies paper-level
   truth/falsity, universal method failure, or external validation not supported
   by the artifacts.

## Frozen material

- Normative method: `docs/METHOD_SPEC.md`
- Calibration report: `docs/CALIBRATION_REPORT_v0.1.md`
- Validation plan and remaining limits: `docs/VALIDATION_PLAN.md`
- Formal theorem map: `docs/FORMAL_THEOREMS.md`
- Lean sources and lockfile: `formal/`
- Confirmatory implementation: `src/stresskit/confirmatory.py`
- Card schema: `src/stresskit/schemas/confirmatory_card_v0.json`
- Calibration artifacts and hash manifest: `artifacts/calibration/`
- Candidate registry protocol: `benchmark/REGISTRY_PROTOCOL.md`
- Outcome-blind candidate frame: `benchmark/registry.candidates.json`
- Pinned source/model manifests: `benchmark/upstream_sources.json` and
  `benchmark/model_sources.json`

## Reproduction commands

```bash
PYTHONPATH=src python -m pytest -q
cd formal && lake build
cd ..
python benchmark/audit_upstreams.py \
  --clone-root /path/to/pinned/upstream/checkouts
```

Expected local baseline on 2026-08-24: 347 Python tests pass; Lean/mathlib build
completes 3,012 jobs. Treat these counts as reproducibility targets, not review.

## Requested response format

For every issue, report:

```text
severity: blocking | major | minor
layer: formal | statistical | construct | replication | communication
location: file:line or artifact path
claim at risk:
counterexample or reasoning:
minimal acceptable change:
```

Please disclose conflicts and whether review was compensated. Public materials
must say “review requested” until named reviewers explicitly approve a version
identified by commit or artifact hashes.
