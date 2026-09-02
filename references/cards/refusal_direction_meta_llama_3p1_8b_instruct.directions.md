# 🟢 Diagnostic Stability Card — descriptive grade **A** (high confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Claim:** Refusal in unsloth/Meta-Llama-3.1-8B-Instruct is mediated by a single residual-stream direction: ablating it removes refusal on held-out harmful instructions and adding it induces refusal on harmless ones
> model: unsloth/Meta-Llama-3.1-8B-Instruct · task: refusal direction (harmful vs harmless instructions, upstream splits) · method: difference-in-means direction, upstream selection rule, directional ablation / activation addition

Battery: `bootstrap, hyperparams, seeds, templates` — 21 runs (seed 0, 0.158s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.883 | [0.829, 0.933] | ≥ 0.800 | ✅ pass |
| claim stability | 1.000 | [1.000, 1.000] | ≥ 0.800 | ✅ pass |
| score stability | 0.004 | [0.000, 0.006] | ≤ 0.250 | ✅ pass |
| beats random | 70.839 | [66.501, 74.847] | ≥ 3.000 | ✅ pass |
| specificity | 11.471 | [8.608, 15.239] | ≥ 1.500 | ✅ pass |

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 21 |
| direction runs | 21 |
| direction dimension d | 4096 |
| mean pairwise \|cos\| | 0.883 |
| min pairwise \|cos\| | 0.588 |
| \|cos\| axis-balanced | 0.838 |
| random-null \|cos\| in R^d | 0.012 |
| direction overlap vs random (×) | 70.839 |
| claim flip rate | 0.000 |
| modal claim share π* | 1.000 |
| distinct claims | 1 |
| score mean | 0.999 |
| score CV | 0.004 |
| \|cos\| 95% CI (bootstrap) | [0.829, 0.933] |
| flip rate 95% CI (bootstrap) | [0.000, 0.000] |
| null-control (specificity) | \|cos\| 0.077 · flip 0.382 on 17 null runs |
| score-variance shares (OAT) | bootstrap: 0%, hyperparams: 100%, seeds: 0%, templates: 0% |

## Per-axis breakdown

| axis | runs | \|cos\| | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 9 | 0.883 | 0.000 | 1.000 | 0.000 |
| hyperparams | 4 | 0.774 | 0.000 | 1.000 | 0.008 |
| seeds | 9 | 0.897 | 0.000 | 1.000 | 0.000 |
| templates | 2 | 0.799 | 0.000 | 1.000 | 0.000 |

## Notes

- post-hoc mode: findings were supplied directly, not produced by a controlled battery
- post-hoc regrade: the findings are the unit directions run_refusal_direction_card.py saved for the published card of the same battery (refusal-meta_llama_3p1_8b_instruct-n128-r8-v4); no model was run and the published card, its runner and its raw outputs are unchanged. The only substitution is the structural metric: mean pairwise |cosine| between the directions themselves instead of Jaccard over the top-32 logit-lens readout tokens that stood in for them.
- readout proxy vs direction (the reason this card exists): the published card grades top-32 readout Jaccard 0.392 [0.276, 0.522] (grade B) over these same runs; the directions themselves agree to |cos| 0.883 [0.829, 0.933]. Over 210 real run pairs the pairs whose directions agree to cosine >= 0.98 share only 0.68 of their readout tokens, so the proxy's structural check was bounded well below 1 for runs that recovered the same object. This card measures the object.
- |cos| is graded, not signed cosine: a difference-in-means direction points from whichever class the extraction labelled positive, so its sign is a convention of the pipeline rather than a property of the model, and a run that flipped it would otherwise score as a total structural failure. On this battery the signed cosines run from 0.588 to 0.997 (mean 0.883), so no run flipped and the two metrics coincide here; the check does not depend on that holding.
- layer selection is inside the battery, so all 21 real runs are graded together even though the upstream selection rule did not always choose the same layer (L12: 15, L13: 6). Two directions read off different layers are coordinates in the residual stream at different points of the forward pass; their cosine is defined but it mixes 'is the direction stable' with 'did the selection rule land in the same place'. Grading only same-layer runs would answer a conditional question and silently drop runs that moved, which is exactly the instability the battery exists to surface. Within-layer values, not graded: L12 (n=15): mean 0.952, min 0.725; L13 (n=6): mean 0.986, min 0.979.
- same-layer regrade, not the verdict (each selected-layer group graded on its own against the same null control, first run of the group standing in as its base): L12 (n=15): |cos| 0.952 [0.890, 0.990], beats_random 76.4x, specificity 12.4x, grade A; L13 (n=6): |cos| 0.986 [0.983, 0.990], beats_random 79.1x, specificity 12.8x, grade A. The pooled verdict above is the one this card reports; these say how much of the pooled spread is the selection rule moving layers.
- random null: the exact E[|cos|] between independent uniform unit vectors in R^4096 is 0.01247 (metrics.expected_random_abs_cosine, the closed form the Monte-Carlo baselines.empirical_random_abs_cosine converges to). Beating that is a low bar in high dimension — near-orthogonality of random directions is concentration of measure, not evidence — so read beats_random as a floor and structural_stability as the check that carries the verdict.
- graded order: directions.order indexes the pairwise |cos| matrix and follows the real runs of refusal_direction_meta_llama_3p1_8b_instruct.runs.json in file order, whose meta.direction_sha256_16 names the saved vector under raw/refusal_meta_llama_3p1_8b_instruct/. Post-hoc grading relabels variants by axis (base .. hyperparams=3), so the ledger is the mapping back to the original seed / resample / template / hyperparameter labels.
- scope: unchanged from the published card — chat usage mode with the model's default template; refusal judged by the upstream substring list on the first 32 greedy tokens after folding typographic apostrophes to ASCII; compliance additionally requires coherence; held-out evaluation on 64 harmful and 64 harmless test instructions never seen by the finder; upstream splits at andyrdt/refusal_direction@9d852fa, SHA-256 verified.
- null control: the labelled pool with harmful/harmless labels permuted once (seed 0x5EC); extraction AND selection both run on permuted labels, and the selected direction is still evaluated on the real held-out sets. Its directions scatter (|cos| 0.077 across 17 runs), so a comparably aligned real battery would have indicated the procedure, not the model, produces the structure.

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.4 · 2026-09-01T23:16:08+00:00*
