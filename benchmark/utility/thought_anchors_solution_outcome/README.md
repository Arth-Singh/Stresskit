# Thought Anchors external-utility candidate

Status: `blocked_not_freezable`. This directory is outside source intake and
contains no labels, predictions, dataset content rows, or final analysis.
Public dataset-card schema examples were visible during source mapping. No
final dataset row was opened. An accidental, unrelated ACDC result-file view is
recorded as quarantined and cannot support this candidate or its thresholds.

The registered metadata-only preflight has now run. It inspected 29,030 file
metadata rows and the exact 40-file selected subset, but no allowed metadata
field identifies a MATH `problem_id`. Partition cluster counts therefore remain
unknown rather than being inferred from files or rollouts. The selected subset
also contains 40 Qwen-family files and zero held-out Llama-family files. See
`blind-metadata-preflight.json` (`failed_insufficient_metadata`); no labels,
predictions, rollout content, claim outcome, or GPU was used.

## Proposed task

Predict whether a generated mathematical solution ends with the correct
answer. The proposed method uses only label-free rollout-change summaries:
KL-based scores, trajectory diversity, sentence position, overdeterminedness,
and function-tag counts. Accuracy-derived importance, answers, correctness,
ground-truth answers, and label-bearing path components are forbidden.

Evaluation units are MATH problem-ID clusters. Repeated sentences and rollouts
never increase `n`. Primary evaluation uses held-out Qwen problem IDs;
generalization holds out both problem IDs and the Llama model family. A blind
path-only preflight must prove at least 200 unique clusters in every partition
before any labels are opened.

Four non-internals baselines are frozen in the candidate:

1. Character-ngram text classifier over problem and generated solution.
2. Frozen MiniLM text embeddings with a fixed classifier.
3. Output-only answer-agreement, entropy, length, and diversity statistics.
4. Problem-domain, difficulty, and model-family metadata prior.

All baseline hyperparameters, input kinds, access policy, and semantic
implementation/input-template digests are registered. Network access and model
internals are forbidden during evaluation.

## Why this is not an AuditSpec

No honest CPU-ready v1 AuditBundle can currently be produced from the public
Thought Anchors rollout release:

- `utility_v1` requires the tested method to declare `uses_internals=true`,
  while counterfactual resampling is black-box.
- Released generation code does not send per-rollout seeds. Rollout count
  cannot substitute for independent problem clusters.
- Public metadata has not established 200 disjoint problem clusters per
  evaluation partition.
- Accuracy-derived importance fields leak the proposed correctness target and
  are therefore excluded.
- The generation code names an unlicensed-card MATH mirror. Equivalence to the
  canonical MIT-licensed dataset remains unverified.
- Receiver-head features would satisfy the internals requirement, but no
  public attention-feature artifacts are released; generating them needs a GPU
  executor after freeze.

The JSON artifact therefore has `publication_state: abstain`, no AuditSpec
digest, and no evaluation manifests. Publisher must not describe it as a
scientific result.

## Next admissible step

Obtain a separately licensed, label-blind problem-ID manifest for both model
families. It must establish at least 200 independent clusters in each frozen
partition without exposing correctness or solution-type labels. Until then,
retain abstention. If that manifest and license closure pass, choose explicitly
between:

- a separately versioned and calibrated behavioral-utility schema for the
  black-box method; or
- the registered receiver-head method, followed by a signed GPU ResourcePlan.

Only then may an outcome-blind reviewer materialize manifests, freeze a
ClaimRecord/AuditSpec, and permit labels to enter final isolated execution.
