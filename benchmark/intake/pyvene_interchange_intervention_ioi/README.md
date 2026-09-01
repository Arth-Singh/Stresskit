# pyvene IOI claim intake

This directory freezes outcome-blind claim wording and agent routing before any
StressKit result or provider opinion is viewed. Source revision:
`stanfordnlp/pyvene@9e333904dcf9e597ca76170010d17f4d4580de8d`, tree
`949e5730a01cedc55648c0f568f464844f14bb97`.

`provider-panel.prefreeze.json` records OpenRouter request IDs, permanent
catalog slugs, exact endpoint tags, and ZDR membership separately. Catalog
metadata is evidence observed at its timestamp, not a promise of future
availability. An authenticated preflight must still prove combined ZDR and
`data_collection: deny` routing and an empty router pipeline.

`provider-panel.attestation.json` records the three authenticated accepted
responses and their request, model-descriptor, raw-response, provider, route,
and policy digests. The original prefreeze plan is not rewritten after model
outputs; the attestation closes it separately.
`panel-closure.json` is the exact 46-object CAS union reachable from the source,
three opinion, rejected-attempt, attestation, and discovery evidence roots.

Generate each opinion by selecting its frozen row; model, provider, family,
role, and generation parameters cannot be overridden:

```bash
stresskit audit opinion source-bundle.json \
  --panel-plan provider-panel.prefreeze.json \
  --opinion-id pyvene-ioi-extractor-a \
  --source-text pyvene-repository=repository-snapshot.txt \
  --source-text pyvene-ioi-replication-notebook=../../../.stresskit/intake/pyvene_interchange_intervention_ioi/extracted/IOI_Replication.txt \
  --source-text pyvene-ioi-mask-notebook=../../../.stresskit/intake/pyvene_interchange_intervention_ioi/extracted/IOI_with_Mask_Intervention.txt \
  --source-text pyvene-ioi-utils=../../../.stresskit/intake/pyvene_interchange_intervention_ioi/extracted/tutorial_ioi_utils.py.txt \
  --cas ../../../.stresskit/cas \
  --closure-output opinions/extractor-a-closure.json \
  -o opinions/extractor-a.json
```

`source-intake.json` binds raw upstream bytes and deterministic cell-source
extractions. Recreate local ignored inputs under
`.stresskit/intake/pyvene_interchange_intervention_ioi/`, then run
`stresskit audit source`; expected digests reject revision or extraction drift.

```bash
git clone https://github.com/stanfordnlp/pyvene.git \
  .stresskit/intake/pyvene_interchange_intervention_ioi/repo
git -C .stresskit/intake/pyvene_interchange_intervention_ioi/repo \
  checkout --detach 9e333904dcf9e597ca76170010d17f4d4580de8d
git -C .stresskit/intake/pyvene_interchange_intervention_ioi/repo \
  archive --format=tar --prefix=pyvene/ -o ../pyvene-9e333904.tar \
  9e333904dcf9e597ca76170010d17f4d4580de8d

mkdir -p .stresskit/intake/pyvene_interchange_intervention_ioi/extracted
stresskit audit extract-source \
  .stresskit/intake/pyvene_interchange_intervention_ioi/repo/tutorials/advanced_tutorials/IOI_Replication.ipynb \
  -o .stresskit/intake/pyvene_interchange_intervention_ioi/extracted/IOI_Replication.txt
stresskit audit extract-source \
  .stresskit/intake/pyvene_interchange_intervention_ioi/repo/tutorials/advanced_tutorials/IOI_with_Mask_Intervention.ipynb \
  -o .stresskit/intake/pyvene_interchange_intervention_ioi/extracted/IOI_with_Mask_Intervention.txt
stresskit audit extract-source \
  .stresskit/intake/pyvene_interchange_intervention_ioi/repo/tutorials/advanced_tutorials/tutorial_ioi_utils.py \
  -o .stresskit/intake/pyvene_interchange_intervention_ioi/extracted/tutorial_ioi_utils.py.txt

cd benchmark/intake/pyvene_interchange_intervention_ioi
stresskit audit source source-intake.json \
  --cas ../../../.stresskit/cas \
  --closure-input sha256:c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4=../../../.stresskit/intake/pyvene_interchange_intervention_ioi/repo/LICENSE \
  --closure-output source-closure.json \
  -o source-bundle.json
```

No GPU is needed for source intake or agent compilation. GPU execution starts
only after claim support, remaining qualification gates, `AuditSpec` freeze,
and a signed `ResourcePlan`.

## Frozen disposition

The 2026-09-01 panel returned unanimous `supported: false` from the xAI/Grok
extractor, Mistral-hosted GLM extractor, and Google/Gemini critic. All three
found the frozen wording stronger than the pinned source documents. Therefore
`discovery-decision.json` records `publication_state: abstain`; no ClaimRecord,
AuditSpec, ResourcePlan, or GPU run exists for this candidate.

`opinions/extractor-a-attempt-1-rejected.json` preserves the first rejected
provider attempt. It was rejected before opinion acceptance because live
OpenRouter metadata represented an empty pipeline as null/absent and reported
the endpoint inventory count separately from its one available selected
endpoint. Completion content was not inspected, no automatic retry occurred,
and the compatibility change retains exact provider, requested model,
canonical deployment, direct-route, and first-attempt checks.

After the panel, offline closure verification exposed missing manifest,
license-file, claim-query, and source-provenance references in emitted closure
lists. The closure lists were regenerated from CAS without changing the
SourceBundle, accepted AgentOpinions, prompts, requests, raw responses, or
their digests. Future generation verifies these dependencies before API use.
