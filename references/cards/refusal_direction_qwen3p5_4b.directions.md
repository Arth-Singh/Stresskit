# 🟠 Diagnostic Stability Card — descriptive grade **C** (low confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Grade rule v0.4:** a check counts as passed only when its whole 95% interval clears the bar; a decided specificity fail caps the grade at C; no null control caps it at B; overlap at or below the at-random floor is D.

> **Claim:** Refusal in Qwen/Qwen3.5-4B is mediated by a single residual-stream direction: ablating it removes refusal on held-out harmful instructions and adding it induces refusal on harmless ones
> model: Qwen/Qwen3.5-4B · task: refusal direction (harmful vs harmless instructions, upstream splits) · method: difference-in-means direction, upstream selection rule, directional ablation / activation addition

Battery: `bootstrap, hyperparams, seeds, templates` — 21 runs (seed 0, 0.11s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.603 | [0.452, 0.761] | ≥ 0.800 | ❌ fail |
| claim stability | 0.857 | [0.667, 1.000] | ≥ 0.800 | ⚠️ inconclusive |
| score stability | 0.174 | [0.084, 0.258] | ≤ 0.250 | ⚠️ inconclusive |
| beats random | 38.204 | [28.663, 48.270] | ≥ 3.000 | ✅ pass |
| specificity | 8.084 | [5.077, 12.330] | ≥ 1.500 | ✅ pass |

> ⚠️ **Underpowered:** the 95% CI straddles the bar for claim_stability, score_stability — undecided in either direction. The grade is provisional; raise `n_runs` before reporting it.

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 21 |
| direction runs | 21 |
| direction dimension d | 2560 |
| mean pairwise \|cos\| | 0.603 |
| min pairwise \|cos\| | 0.136 |
| \|cos\| axis-balanced | 0.617 |
| random-null \|cos\| in R^d | 0.016 |
| direction overlap vs random (×) | 38.204 |
| claim flip rate | 0.267 |
| modal claim share π* | 0.857 |
| distinct claims | 3 |
| score mean | 0.875 |
| score CV | 0.174 |
| \|cos\| 95% CI (bootstrap) | [0.452, 0.761] |
| flip rate 95% CI (bootstrap) | [0.000, 0.543] |
| null-control (specificity) | \|cos\| 0.075 · flip 0.669 on 17 null runs |
| claim distribution | `bidirectional control; mid-layer direction`×18, `bidirectional control; late-layer direction`×2, `addition-only control; mid-layer direction`×1 |
| score-variance shares (OAT) | bootstrap: 14%, hyperparams: 75%, seeds: 10%, templates: 1% |

## Per-axis breakdown

| axis | runs | \|cos\| | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 9 | 0.657 | 0.000 | 1.000 | 0.124 |
| hyperparams | 4 | 0.572 | 0.500 | 0.750 | 0.324 |
| seeds | 9 | 0.518 | 0.389 | 0.778 | 0.098 |
| templates | 2 | 0.722 | 0.000 | 1.000 | 0.026 |

## Notes

- post-hoc mode: findings were supplied directly, not produced by a controlled battery
- underpowered verdict: the 95% CI straddles the bar for claim_stability (pass), score_stability (pass) at n_runs=21 — these verdict components are not decided by the data. Treat the grade as provisional and raise n_runs (or widen the battery) before reporting it.
- post-hoc regrade: the findings are the unit directions run_refusal_direction_card.py saved for the published card of the same battery (refusal-qwen3p5_4b-n128-r8-v4); no model was run and the published card, its runner and its raw outputs are unchanged. The only substitution is the structural metric: mean pairwise |cosine| between the directions themselves instead of Jaccard over the top-32 logit-lens readout tokens that stood in for them.
- readout proxy vs direction (the reason this card exists): the published card grades top-32 readout Jaccard 0.204 [0.100, 0.324] (grade B) over these same runs; the directions themselves agree to |cos| 0.603 [0.452, 0.761]. Over 210 real run pairs the pairs whose directions agree to cosine >= 0.98 share only 0.68 of their readout tokens, so the proxy's structural check was bounded well below 1 for runs that recovered the same object. This card measures the object.
- |cos| is graded, not signed cosine: a difference-in-means direction points from whichever class the extraction labelled positive, so its sign is a convention of the pipeline rather than a property of the model, and a run that flipped it would otherwise score as a total structural failure. On this battery the signed cosines run from 0.136 to 0.996 (mean 0.603), so no run flipped and the two metrics coincide here; the check does not depend on that holding.
- layer selection is inside the battery, so all 21 real runs are graded together even though the upstream selection rule did not always choose the same layer (L11: 3, L12: 1, L13: 11, L14: 1, L15: 1, L16: 2, L23: 1, L24: 1). Two directions read off different layers are coordinates in the residual stream at different points of the forward pass; their cosine is defined but it mixes 'is the direction stable' with 'did the selection rule land in the same place'. Grading only same-layer runs would answer a conditional question and silently drop runs that moved, which is exactly the instability the battery exists to surface. Within-layer values, not graded: L11 (n=3): mean 0.995, min 0.994; L13 (n=11): mean 0.959, min 0.913; L16 (n=2): mean 0.973, min 0.973.
- same-layer regrade, not the verdict (each selected-layer group graded on its own against the same null control, first run of the group standing in as its base): L13 (n=11): |cos| 0.959 [0.945, 0.971], beats_random 60.8x, specificity 12.9x, grade A; L11 (n=3): too few runs to regrade; L16 (n=2): too few runs to regrade; L23 (n=1): too few runs to regrade; L14 (n=1): too few runs to regrade; L24 (n=1): too few runs to regrade; L12 (n=1): too few runs to regrade; L15 (n=1): too few runs to regrade. The pooled verdict above is the one this card reports; these say how much of the pooled spread is the selection rule moving layers.
- random null: the exact E[|cos|] between independent uniform unit vectors in R^2560 is 0.01577 (metrics.expected_random_abs_cosine, the closed form the Monte-Carlo baselines.empirical_random_abs_cosine converges to). Beating that is a low bar in high dimension — near-orthogonality of random directions is concentration of measure, not evidence — so read beats_random as a floor and structural_stability as the check that carries the verdict.
- graded order: directions.order indexes the pairwise |cos| matrix and follows the real runs of refusal_direction_qwen3p5_4b.runs.json in file order, whose meta.direction_sha256_16 names the saved vector under raw/refusal_qwen3p5_4b/. Post-hoc grading relabels variants by axis (base .. hyperparams=3), so the ledger is the mapping back to the original seed / resample / template / hyperparameter labels.
- scope: unchanged from the published card — chat usage mode with the model's default template; refusal judged by the upstream substring list on the first 32 greedy tokens after folding typographic apostrophes to ASCII; compliance additionally requires coherence; held-out evaluation on 64 harmful and 64 harmless test instructions never seen by the finder; upstream splits at andyrdt/refusal_direction@9d852fa, SHA-256 verified.
- null control: the labelled pool with harmful/harmless labels permuted once (seed 0x5EC); extraction AND selection both run on permuted labels, and the selected direction is still evaluated on the real held-out sets. Its directions scatter (|cos| 0.075 across 17 runs), so a comparably aligned real battery would have indicated the procedure, not the model, produces the structure.
- v0.3 grade: B; regraded 2026-09-03 under grade rule v0.4 from the recorded checks (schema 0.5)

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.5 · 2026-09-01T23:16:16+00:00*
