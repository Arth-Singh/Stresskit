# 🟠 Diagnostic Stability Card — descriptive grade **C** (high confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Claim:** Refusal in google/gemma-4-E4B-it is mediated by a single residual-stream direction: ablating it removes refusal on held-out harmful instructions and adding it induces refusal on harmless ones
> model: google/gemma-4-E4B-it · task: refusal direction (harmful vs harmless instructions, upstream splits) · method: difference-in-means direction, upstream selection rule, directional ablation / activation addition

Battery: `bootstrap, hyperparams, seeds, templates` — 21 runs (seed 0, 0.113s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.405 | [0.259, 0.624] | ≥ 0.800 | ❌ fail |
| claim stability | 0.429 | [0.286, 0.667] | ≥ 0.800 | ❌ fail |
| score stability | 1.022 | [0.706, 1.384] | ≤ 0.250 | ❌ fail |
| beats random | 25.698 | [16.399, 39.546] | ≥ 3.000 | ✅ pass |
| specificity | 12.937 | [6.431, 24.240] | ≥ 1.500 | ✅ pass |

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 21 |
| direction runs | 21 |
| direction dimension d | 2560 |
| mean pairwise \|cos\| | 0.405 |
| min pairwise \|cos\| | 0.000 |
| \|cos\| axis-balanced | 0.310 |
| random-null \|cos\| in R^d | 0.016 |
| direction overlap vs random (×) | 25.698 |
| claim flip rate | 0.757 |
| modal claim share π* | 0.429 |
| distinct claims | 5 |
| score mean | 0.270 |
| score CV | 1.022 |
| \|cos\| 95% CI (bootstrap) | [0.259, 0.624] |
| flip rate 95% CI (bootstrap) | [0.576, 0.854] |
| null-control (specificity) | \|cos\| 0.031 · flip 0.632 on 17 null runs |
| claim distribution | `no control; late-layer direction`×9, `addition-only control; mid-layer direction`×4, `ablation-only control; late-layer direction`×4, `no control; early-layer direction`×3, `ablation-only control; mid-layer direction`×1 |
| score-variance shares (OAT) | bootstrap: 9%, hyperparams: 12%, seeds: 14%, templates: 66% |

## Per-axis breakdown

| axis | runs | \|cos\| | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 9 | 0.339 | 0.750 | 0.444 | 1.669 |
| hyperparams | 4 | 0.318 | 0.667 | 0.500 | 1.039 |
| seeds | 9 | 0.576 | 0.722 | 0.444 | 0.710 |
| templates | 2 | 0.006 | 1.000 | 0.500 | 1.000 |

## Notes

- post-hoc mode: findings were supplied directly, not produced by a controlled battery
- post-hoc regrade: the findings are the unit directions run_refusal_direction_card.py saved for the published card of the same battery (refusal-gemma_4_e4b_it-n128-r8-v4); no model was run and the published card, its runner and its raw outputs are unchanged. The only substitution is the structural metric: mean pairwise |cosine| between the directions themselves instead of Jaccard over the top-32 logit-lens readout tokens that stood in for them.
- readout proxy vs direction (the reason this card exists): the published card grades top-32 readout Jaccard 0.302 [0.175, 0.492] (grade C) over these same runs; the directions themselves agree to |cos| 0.405 [0.259, 0.624]. Over 210 real run pairs the pairs whose directions agree to cosine >= 0.98 share only 0.68 of their readout tokens, so the proxy's structural check was bounded well below 1 for runs that recovered the same object. This card measures the object.
- |cos| is graded, not signed cosine: a difference-in-means direction points from whichever class the extraction labelled positive, so its sign is a convention of the pipeline rather than a property of the model, and a run that flipped it would otherwise score as a total structural failure. On this battery the signed cosines run from -0.013 to 1.000 (mean 0.404), so no run flipped and the two metrics coincide here; the check does not depend on that holding.
- layer selection is inside the battery, so all 21 real runs are graded together even though the upstream selection rule did not always choose the same layer (L5: 3, L19: 4, L24: 1, L31: 5, L32: 8). Two directions read off different layers are coordinates in the residual stream at different points of the forward pass; their cosine is defined but it mixes 'is the direction stable' with 'did the selection rule land in the same place'. Grading only same-layer runs would answer a conditional question and silently drop runs that moved, which is exactly the instability the battery exists to surface. Within-layer values, not graded: L5 (n=3): mean 0.992, min 0.988; L19 (n=4): mean 0.996, min 0.994; L31 (n=5): mean 0.996, min 0.993; L32 (n=8): mean 0.993, min 0.986.
- same-layer regrade, not the verdict (each selected-layer group graded on its own against the same null control, first run of the group standing in as its base): L32 (n=8): |cos| 0.993 [0.990, 0.996], beats_random 63.0x, specificity 31.7x, grade B; L31 (n=5): |cos| 0.996 [0.995, 0.998], beats_random 63.2x, specificity 31.8x, grade A; L19 (n=4): |cos| 0.996 [0.994, 0.996], beats_random 63.1x, specificity 31.8x, grade A; L5 (n=3): too few runs to regrade; L24 (n=1): too few runs to regrade. The pooled verdict above is the one this card reports; these say how much of the pooled spread is the selection rule moving layers.
- random null: the exact E[|cos|] between independent uniform unit vectors in R^2560 is 0.01577 (metrics.expected_random_abs_cosine, the closed form the Monte-Carlo baselines.empirical_random_abs_cosine converges to). Beating that is a low bar in high dimension — near-orthogonality of random directions is concentration of measure, not evidence — so read beats_random as a floor and structural_stability as the check that carries the verdict.
- graded order: directions.order indexes the pairwise |cos| matrix and follows the real runs of refusal_direction_gemma_4_e4b_it.runs.json in file order, whose meta.direction_sha256_16 names the saved vector under raw/refusal_gemma_4_e4b_it/. Post-hoc grading relabels variants by axis (base .. hyperparams=3), so the ledger is the mapping back to the original seed / resample / template / hyperparameter labels.
- scope: unchanged from the published card — chat usage mode with the model's default template; refusal judged by the upstream substring list on the first 32 greedy tokens after folding typographic apostrophes to ASCII; compliance additionally requires coherence; held-out evaluation on 64 harmful and 64 harmless test instructions never seen by the finder; upstream splits at andyrdt/refusal_direction@9d852fa, SHA-256 verified.
- null control: the labelled pool with harmful/harmless labels permuted once (seed 0x5EC); extraction AND selection both run on permuted labels, and the selected direction is still evaluated on the real held-out sets. Its directions scatter (|cos| 0.031 across 17 runs), so a comparably aligned real battery would have indicated the procedure, not the model, produces the structure.

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.4 · 2026-09-01T23:16:16+00:00*
