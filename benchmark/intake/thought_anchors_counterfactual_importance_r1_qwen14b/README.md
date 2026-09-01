# Thought Anchors Qwen-14B counterfactual-importance intake

This directory freezes outcome-blind source, code-map, rollout-subset, claim-query,
and provider-panel inputs for
`thought_anchors_counterfactual_importance_r1_qwen14b`. No provider opinion,
StressKit outcome, upstream metric, or GPU experiment was run while creating it.

## Frozen sources

- Repository: `interp-reasoning/thought-anchors`
- Commit: `b53ed8c75d3f6112f68adfaec9a13d4d708c442e`
- Tree: `b7a5383cedd85a32f437b928654010bb0de3dd2c`
- Repository archive:
  `sha256:796f6bf17b0dd701cc82c0938b313d7733b1070ee9557eb0f9e2f6cc0310e383`
- Paper: `arXiv:2506.19143v4`
- PDF:
  `sha256:7b2d6a4146d06fb7c6d1ca3a88a965bffe71c937f206d1ebed2c35692648d737`
- arXiv source:
  `sha256:7081cadf9bda4c0f59d387f97794bf82219a5fd65ee977ec47589be8b45dcc84`
- Rollout dataset: `uzaymacar/math-rollouts` revision
  `a3a2fe1eb6d52e6ef81860baff018a8df2eb8ae2`
- Repository and dataset-card licenses: MIT. Paper license: CC-BY-4.0.

The exact-revision dataset card declares MIT in both its YAML metadata and
Dataset Details. The revision has no standalone license file. The repository
contains a standalone MIT `LICENSE.md`, bound at
`sha256:17825c3bd82610363b7c1e2bf16311f4a1c9578674ea844da72bfaa8a87977d6`.

## Frozen wording, locator, and code map

`claim-query.txt` preserves the candidate registry sentence byte-for-byte:

> Under resampling, sentences labelled planning or uncertainty-management have
> higher counterfactual importance than computation sentences, and the
> importance ranking identifies a small set of anchor sentences per trace.

Source inspection corrected the locator without changing those bytes. Section
2.4 and Figure 2, PDF pages 3–4, are a single-trace case study. Sections
3.1–3.3 and Figure 3, PDF page 5, contain the category-level comparison. Primary
TeX lines 283–289 state that plan-generation and uncertainty-management
categories consistently exceed fact-retrieval and active-computation
categories. The pinned source does not visibly define a quantitative
per-trace “small set” selection rule; the independent panel must adjudicate
every material clause and abstain if support is incomplete.

The released-metric CPU path is `analyze_rollouts.py` for counterfactual metrics
and labels, `plots.py` for exact input selection and category aggregation, and
`prompts.py` for the label taxonomy. `generate_rollouts.py` is generation
provenance, not required for released-data reduction. `step_attribution.py`
computes a different sentence-to-sentence measure and is excluded from this
registered Figure 3 path.

## Released rollout subset

The full exact dataset revision has 29,030 files and roughly 110.99 GB of
logical content. The outcome-blind CPU intake therefore selects only the 40
`chunks_labeled.json` files read by `plots.py:collect_tag_data` for
DeepSeek-R1-Distill-Qwen-14B at temperature 0.6/top-p 0.95, across
`correct_base_solution` and `incorrect_base_solution`.

Every selected file's size and Git blob ID was checked against the exact HF
revision response. The 40 files contain 6,474 schema-validated chunk rows and
4,306,713 source bytes. No category mean or claim outcome was computed.

- Exact path list:
  `sha256:49fa13b2526a2e28fc0d17d0ab67ec82e68d74e9a37dc069b588a7f1962dd038`
- Extracted subset manifest:
  `sha256:6e303125f586de09c75227c30ddd4b45f57c3ed55f107e5185eba5243d2e0166`
- Deterministic USTAR archive, sorted files, mode 0644, uid/gid 0, mtime 0:
  `sha256:51c8bf8276234eaee69347d0db253a2527e0fd01b404fe625fa73e5e4d47cfb5`

`artifact-inventory.json` records every pin, byte count, license decision, code
path, and excluded path. The exact revision API response is part of the
SourceBundle, so selection metadata cannot drift silently.

## Recreate ignored source inputs

Run from the StressKit repository root. Verify existing targets rather than
overwriting them.

```bash
STRESSKIT_REPO=/Users/arth_rogerthat/Downloads/impact/Stresskit
UPSTREAM_ROOT="$STRESSKIT_REPO/.stresskit/upstreams"
UPSTREAM_PATH="$UPSTREAM_ROOT/thought_anchors"
RAW_ROOT="$STRESSKIT_REPO/.stresskit/intake/thought_anchors_counterfactual_importance_r1_qwen14b/raw"
EXTRACTED_ROOT="$STRESSKIT_REPO/.stresskit/intake/thought_anchors_counterfactual_importance_r1_qwen14b/extracted"
INTAKE_DIR="$STRESSKIT_REPO/benchmark/intake/thought_anchors_counterfactual_importance_r1_qwen14b"

mkdir -p "$UPSTREAM_ROOT" "$RAW_ROOT" "$EXTRACTED_ROOT"
git clone --filter=blob:none --no-checkout \
  https://github.com/interp-reasoning/thought-anchors "$UPSTREAM_PATH"
git -C "$UPSTREAM_PATH" fetch --depth=1 origin \
  b53ed8c75d3f6112f68adfaec9a13d4d708c442e
git -C "$UPSTREAM_PATH" checkout --detach \
  b53ed8c75d3f6112f68adfaec9a13d4d708c442e

test "$(git -C "$UPSTREAM_PATH" rev-parse HEAD)" = \
  b53ed8c75d3f6112f68adfaec9a13d4d708c442e
test "$(git -C "$UPSTREAM_PATH" rev-parse 'HEAD^{tree}')" = \
  b7a5383cedd85a32f437b928654010bb0de3dd2c
test -z "$(git -C "$UPSTREAM_PATH" status --porcelain)"

git -C "$UPSTREAM_PATH" archive --format=tar --prefix=thought-anchors/ \
  -o "$RAW_ROOT/thought-anchors-b53ed8c.tar" \
  b53ed8c75d3f6112f68adfaec9a13d4d708c442e

curl --fail --location --retry 3 \
  --output "$RAW_ROOT/arxiv-2506.19143v4.pdf" \
  https://arxiv.org/pdf/2506.19143v4
curl --fail --location --retry 3 \
  --output "$RAW_ROOT/arxiv-2506.19143v4-source.tar.gz" \
  https://export.arxiv.org/e-print/2506.19143v4
curl --fail --location --retry 3 \
  --output "$RAW_ROOT/math-rollouts-README.md" \
  https://huggingface.co/datasets/uzaymacar/math-rollouts/resolve/a3a2fe1eb6d52e6ef81860baff018a8df2eb8ae2/README.md
curl --fail --location --retry 3 \
  --output "$RAW_ROOT/math-rollouts-api.json" \
  'https://huggingface.co/api/datasets/uzaymacar/math-rollouts/revision/a3a2fe1eb6d52e6ef81860baff018a8df2eb8ae2?blobs=true'

pdftotext -layout -nopgbrk -enc UTF-8 \
  "$RAW_ROOT/arxiv-2506.19143v4.pdf" \
  "$EXTRACTED_ROOT/arxiv-2506.19143v4.txt"
stresskit audit extract-source "$UPSTREAM_PATH/analyze_rollouts.py" \
  -o "$EXTRACTED_ROOT/analyze_rollouts.py.txt"
stresskit audit extract-source "$UPSTREAM_PATH/plots.py" \
  -o "$EXTRACTED_ROOT/plots.py.txt"
stresskit audit extract-source "$UPSTREAM_PATH/prompts.py" \
  -o "$EXTRACTED_ROOT/prompts.py.txt"
```

Filter the exact HF API response to the two selected solution-type directories,
download each path through `/resolve/<revision>/<path>`, and reject any byte
whose length or Git blob SHA-1 differs from its `siblings` row. Serialize the
manifest with sorted keys and two-space indentation plus a final newline. Build
the archive as USTAR with lexicographically sorted file paths, no directory
members, uid/gid 0, owner/group `root`, mode 0644, and mtime 0. The three frozen
digests above reject any different selection or encoding.

## Rebuild SourceBundle and closure

```bash
stresskit audit source "$INTAKE_DIR/source-intake.json" \
  --cas "$STRESSKIT_REPO/.stresskit/cas" \
  --closure-input sha256:639790103a0149944988963306aac180f55e2e074b7c3b09e9d39dd2abcefdfd="$INTAKE_DIR/artifact-inventory.json" \
  --closure-input sha256:32dd054f18b913c227cf8310ff4e22eb45c188b630abbe29f051c54d9032fff2="$INTAKE_DIR/claim-query.txt" \
  --closure-input sha256:17825c3bd82610363b7c1e2bf16311f4a1c9578674ea844da72bfaa8a87977d6="$UPSTREAM_PATH/LICENSE.md" \
  --closure-input sha256:49fa13b2526a2e28fc0d17d0ab67ec82e68d74e9a37dc069b588a7f1962dd038="$RAW_ROOT/qwen14b-chunks-labeled-paths.txt" \
  --closure-input sha256:6d3fcdce2a47a2ce5d113173135ef348a7a705b2bb0e9eaaa89e832f8ce502a5="$UPSTREAM_PATH/generate_rollouts.py" \
  --closure-input sha256:23b29910dd25b17b3b3c8faba7c9c0827becd1b9ac681a6ef97128ac9f94aa7f="$UPSTREAM_PATH/step_attribution.py" \
  --closure-output "$INTAKE_DIR/source-closure.json" \
  -o "$INTAKE_DIR/source-bundle.json"
```

Verified SourceBundle semantic digest:
`sha256:9a27e09f02489ba7fbc12340d9a2c7d57037c38c319e362db3e6a8b326488cd3`.
The closure contains 34 objects and 63,215,275 bytes.

## Frozen provider panel and remaining gates

`provider-panel.prefreeze.json` freezes three distinct routes: xAI Grok 4.6 and
Mistral-hosted GLM 5.2 as isolated extractors, followed by Google Gemini 3.1 Pro
as dependency-gated critic. It binds exact model deployments, providers,
temperature-zero seeds, required structured-output parameters, ZDR catalog
membership, and claim-query bytes.

An authenticated GET-only preflight at `2026-09-02T03:33:02+09:00` confirmed
all three canonical deployments, exact provider tags, status zero, context
lengths, required parameters, and membership in OpenRouter's live ZDR endpoint
catalog. The supplied inference key could not read workspace settings:
`GET /api/v1/workspaces?limit=100` returned HTTP 401. OpenRouter documents
private input/output logging and data-discount logging as opt-in and off by
default, but that default is not affirmative evidence for this account. The
frozen `account_prompt_logging: must_not_be_opted_in` constraint therefore
failed closed before any source bytes were sent to a model.

`authenticated-preflight-blocker.json` records the decision and safe catalog
bindings. Its semantic digest is
`sha256:02b0f4a2342e3ea681687e6999cefe20c4a2932c58a408bd522a1f0800fb5fcc`.
`authenticated-preflight-closure.json` binds 42 CAS objects totaling 64,946,043
bytes; its semantic digest is
`sha256:73ea7ee83e66f70203d8caeae66226f4e71d4e5ab44c4241efe4ab13a2eaeb1d`.
No chat-completion slot started, no retry or critic call occurred, no GPU was
used, and no `AgentOpinion` exists. This gate's publication state is abstain,
while the candidate remains pending rather than permanently excluded.

Next requirement: authenticated management-API response or reviewed dashboard
export showing private input/output logging and data-discount logging disabled
for the supplied key scope. After that attestation, refresh catalogs and execute
each frozen slot at most once, with critic still conditional on two accepted
extractors.

No GPU is needed for this source or panel phase. Before any candidate can freeze,
the panel must support every claim clause, exact CPU dependencies and wheels
must be locked, the released reducer must pass a network-disabled isolation
smoke, and a ClaimRecord, AuditSpec, protocol review, and outcome-blind
ResourcePlan must be produced. Regenerating rollouts would be a separate GPU
scope requiring a pinned model revision and model-license review.
