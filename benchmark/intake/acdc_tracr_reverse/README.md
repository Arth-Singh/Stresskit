# ACDC Tracr reverse intake

Status: **excluded before freeze; publication state `abstain`**.

This is an outcome-blind intake of candidate `acdc_tracr_reverse`. It does not
contain an ACDC reproduction result and must not be cited as verifying or
falsifying the paper claim.

## Completed

- Pinned ACDC repository commit `bc99ace817974b5584b7ee203d596a8e2bbcd399`
  and tree `e22aed97ac8253e50b0c535bc637796cc71fec98`.
- Pinned arXiv v4 paper/source and complete content-addressed SourceBundle.
- Built exact upstream Python environment. An isolated CPU smoke test compiled
  Tracr reverse, matched TransformerLens decoding/layers on 8 of 8 examples,
  and observed no claim outcome.
- Executed each frozen extractor slot once through distinct model families and
  providers. No retry was allowed.

## Fail-closed panel result

- Extractor A: rejected before `AgentOpinion` construction. Three of ten quoted
  anchors were absent byte-for-byte from the declared source document.
- Extractor B: rejected before `AgentOpinion` construction. Completion ended at
  the frozen token limit.
- Critic: not run because two accepted extractor opinions were required.
- ClaimRecord, AuditSpec, ResourcePlan, and GPU run: not created.

The successful CPU smoke cannot override invalid agent evidence. Registry row
therefore ends pre-freeze as `excluded`, with public claim state `abstain`.

## Unresolved claim semantics

- Paper wording says perfect recovery for every positive threshold under zero
  activation patching, while historical sweep code does not bind a complete
  positive-threshold domain.
- Paper figure describes a 15-edge canonical circuit; pinned hook-level code
  enumerates 24 edges. No trusted representation reducer is frozen.
- Upstream ROC aggregation omits crashed runs, incompatible with StressKit's
  complete-slot requirement.

`registry-auc-candidate.draft.json` records a narrower Table 3 AUC candidate for
future review only. It is explicitly not frozen.

## Evidence

- `source-bundle.json` and `source-closure.json`: pinned source closure.
- `provider-panel.prefreeze.json`: frozen routes and generation settings.
- `panel-execution.json`: complete three-slot accounting.
- `opinions/`: immutable rejected-attempt records.
- `discovery-decision.json`: no candidates emitted; abstention.
- `panel-abstention.json` and `panel-closure.json`: verifier-facing evidence.
- `execution-smoke.json`: isolated CPU compatibility evidence only.

GPU required now: **no**. A GPU executor is requested only after a claim passes
source, agent, semantic, license, specification, and review gates.
