# Draft: flagship artifact and license request

Status: draft only; not sent.

StressKit is preregistering a study of whether per-example loss-gradient
projections onto frozen persona/misalignment directions predict held-out
post-fine-tuning behavior. We will abstain rather than substitute a nearby
artifact or infer a license. Could you publish or confirm the items below at an
immutable release or repository revision?

## Persona Vectors authors

Repository audited at
`b8e0f044fe2410a6fad579f38324f03f13b4e917`.

1. Exact paper-used `response_avg_diff.pt` tensors for evil, sycophancy, and
   hallucination for each reported model. Please include SHA-256 digests, base
   model revisions, extraction prompts/data revisions, token position,
   activation hook, layer indexing, normalization, generation settings, and an
   explicit license covering the tensor artifacts.
2. A member-by-member source map for `dataset.zip` (SHA-256
   `6913afdd712997599016444e789d2a4a5e383b6418e6596a4598cebdb97e943e`).
   Please identify exact upstream revision and source row for questions drawn
   from reward-hack-generalization, Thought Crime medical data, Emergent
   Misalignment insecure-code data, MATH, GSM8K, and GlobalOpinionQA.
3. Confirmation of which license and attribution notice applies to each
   transformed JSONL member. In particular, please clarify the unlicensed
   Thought Crime medical source and the CC-BY-NC-SA-4.0 GlobalOpinionQA-derived
   members. Please also confirm whether public hosting and redistribution of
   derived training/evaluation bundles is permitted.
4. Confirmation that Apache-2.0 covers the exact JSON files under
   `data_generation/trait_data_extract/` and `trait_data_eval/`, plus source
   code or an immutable recipe that generated them.
5. Confirmation of the license and redistribution rights for
   `output/qwen2.5-7b-instruct_baseline.csv`, including model answers and
   GPT-4.1-mini judge scores. Please provide raw request/model identifiers and
   deterministic score parsing rules if available.
6. Exact dataset/model/seed assignments used as pilot, final, and held-out
   cohorts in the paper, or confirmation that these were not frozen before
   outcome inspection.

## Emergent Misalignment authors

Repository audited at
`80c11967c07a328e7d7d43d13ce6847ae44dbcc9`.

1. Please confirm explicitly whether the root MIT license covers these exact
   data objects: `data/insecure.jsonl`, `data/secure.jsonl`,
   `data/evil_numbers.jsonl`, `evaluation/first_plot_questions.yaml`,
   `evaluation/preregistered_evals.yaml`, `evaluation/deception_factual.yaml`,
   and `evaluation/deception_sit_aware.yaml`.
2. Please state the license and redistribution rights for model answers and
   judge-derived labels in `results/qwen_25_coder_32b_instruct.csv`, including
   raw judge model/request provenance and deterministic score parsing rules.
3. Please add explicit license metadata or a license file to the official
   `emergent-misalignment/Qwen-Coder-Insecure` Hugging Face checkpoint at
   revision `c0cf057ec4d5db64581b784808b34d48c0d0e95e`, if distribution rights
   permit. Please identify licenses for each underlying base model and adapter
   or merged checkpoint.
4. Please provide exact source revisions and licenses for any adapted data.
   `data/jailbroken.jsonl` is out of scope until its Bowen et al. provenance
   and license are explicit.
5. Please identify which six fine-tuning runs appear in the published result
   CSV, including seeds, training-data digest, optimizer configuration,
   checkpoint digest, evaluation manifest, and whether complete raw outputs
   are publicly available.

## Requested form of response

A public release or signed repository commit is preferred. For every object,
please provide: canonical locator, immutable revision, SHA-256 digest, SPDX
identifier or full license text, copyright/attribution notice, and any use or
redistribution restrictions. Silence or an ambiguous project-level statement
will remain an abstention under the registered protocol.
