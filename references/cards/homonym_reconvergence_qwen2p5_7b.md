# 🟡 Diagnostic Stability Card — descriptive grade **B** (low confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Claim:** Homonym and polyseme representations become maximally distinct in middle layers and partially reconverge in late layers, while the KL divergence between their next-token predictions peaks in the final layers.
> model: Qwen/Qwen2.5-7B · task: layer-wise logit-lens profile of homonym pairs (upstream Experiment 1 stimuli; activation distance, logit distance, KL divergence) · method: upstream per-layer distance/KL profile; top-k activation-distance band, reconvergence ratio, fixed-threshold profile label

Battery: `seeds, bootstrap, templates, hyperparams` — 32 runs (seed 0, 223.077s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.884 | [0.778, 0.969] | ≥ 0.800 | ⚠️ inconclusive |
| claim stability | 1.000 | [1.000, 1.000] | ≥ 0.800 | ✅ pass |
| score stability | 0.042 | [0.006, 0.066] | ≤ 0.250 | ✅ pass |
| beats random | 6.923 | [6.093, 7.591] | ≥ 3.000 | ✅ pass |
| specificity | 0.884 | [0.784, 0.972] | ≥ 1.500 | ❌ fail |

> ⚠️ **Underpowered:** the 95% CI straddles the bar for structural_stability — undecided in either direction. The grade is provisional; raise `n_runs` before reporting it.

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 32 |
| structured runs | 32 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.884 |
| min pairwise Jaccard | 0.200 |
| random-null Jaccard | 0.128 |
| overlap vs random (×) | 6.923 |
| claim flip rate | 0.000 |
| modal claim share π* | 1.000 |
| distinct claims | 1 |
| score mean | 0.674 |
| score CV | 0.042 |
| median finding size | 6.000 |
| Jaccard 95% CI (bootstrap) | [0.778, 0.969] |
| flip rate 95% CI (bootstrap) | [0.000, 0.000] |
| null-control (specificity) | Jaccard 1.000 · flip 0.080 on 25 null runs |
| score-variance shares (OAT) | bootstrap: 0%, hyperparams: 68%, seeds: 0%, templates: 32% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 13 | 1.000 | 0.000 | 1.000 | 0.005 |
| hyperparams | 6 | 0.729 | 0.000 | 1.000 | 0.081 |
| seeds | 13 | 1.000 | 0.000 | 1.000 | 0.006 |
| templates | 3 | 0.416 | 0.000 | 1.000 | 0.057 |

## Notes

- underpowered verdict: the 95% CI straddles the bar for structural_stability (pass) at n_runs=12 — these verdict components are not decided by the data. Treat the grade as provisional and raise n_runs (or widen the battery) before reporting it.
- scope: Qwen/Qwen2.5-7B (HF revision d149729398750b98c0af14eb82c78cfe92750796) in float16, 28 post-block residual layers read at the target-word token, one sentence per forward pass, no chat template; the paper's models are gpt2, meta-llama/Llama-3.2-3B and Qwen/Qwen2.5-32B via TransformerLens 2.1.6 — this card grades the profile on the model named here only, with the TransformerLens conventions the upstream inherits (fold_ln, center_unembed, center_writing_weights for LayerNorm models) reproduced on plain Hugging Face weights
- data: upstream stimulus files at scoki211/Divergent_LLM_Predictions_Convergent_Reps_amb_words@dd08338 (MIT), SHA-256 verified; upstream tokenisation validation reproduced: homonyms 166/190 words -> 451 context pairs (failures {'inconsistent_tokenization': 22, 'not_found': 2}); polysemes 94/97 words -> 116 pairs (failures {'inconsistent_tokenization': 2, 'not_found': 1}); the paper's KL > 0.5 stimulus screen is not part of the released Experiment-1 entrypoint and is not applied
- representation (fixed before running): band = top-6 layers by activation distance (k = max(3, round(0.2 L))); score r = mean activation distance over the final band [25, 26, 27] / peak; label = majority manuscript layer group of the band (early <= 6, middle <= 16, late above) + 'late reconvergence' iff r <= 0.9 + KL argmax in the final band -> 'final'; the paper's claim is the label 'middle peak; late reconvergence; KL peak final'; each finder call subsamples 75% of the items with its seed
- base run (subsampled): activation-distance peak at layer 9 (0.548), band [8, 9, 10, 11, 12, 14], final-band mean 0.367, r = 0.669, KL argmax layer 25, label 'middle peak; late reconvergence; KL peak final' (339 items, 158 words)
- full homonym set, base configuration: activation-distance peak layer 9 (0.540), final-band mean 0.362, KL argmax layer 25; per-(word, layer) activation distance by manuscript group: early mean 0.346 / median 0.327; middle mean 0.510 / median 0.532; late mean 0.406 / median 0.414 (paper: GPT-2 means 0.069/0.215/0.167, Llama-3.2-3B medians 0.48/0.55/0.48, Qwen2.5-32B medians 0.38/0.63/0.42 for early/middle/late)
- null control (graded): the homonym items with sentence B re-paired to a different word (seeded derangement, seed 0x5ec, 451 items, same sentences, same read positions, same finder) — the effect under test (one surface form, two senses) is absent while everything else is held fixed. Fair because the finder's output is a stable top-k band on any smooth depth profile, so the specificity ratio asks whether band stability is diagnostic of the ambiguity effect; it does NOT test whether the null recovers the same band or label — see the next notes. Conservative direction: different-word pairs are far apart at every depth, so their profile is shaped by the global geometry (anisotropy), not by sense resolution
- null base run (permuted pairs): activation-distance peak at layer 5 (0.820), band [3, 4, 5, 6, 7, 9], final-band mean 0.439, r = 0.536, KL argmax layer 25, label 'early peak; late reconvergence; KL peak final' (339 items, 161 words)
- null claim distribution: 'early peak; late reconvergence; KL peak final' x24, 'early peak; late reconvergence; KL peak late' x1; Jaccard between the real and null base bands 0.091; full permuted set: peak layer 5, final-band mean / peak 0.538, KL argmax layer 25
- alternative null (reported, not graded): the upstream sequence-order control set (280 matched tokens from 56 reordered sentence pairs) through the same finder — null Jaccard 0.899, specificity ratio 0.983 (95% CI [0.851, 1.110]), fail at >= 1.5; null claim distribution: 'middle peak; late reconvergence; KL peak final' x23, 'late peak; late reconvergence; KL peak final' x2; Jaccard between the real and sequence-order base bands 0.200
- sequence-order null base run: activation-distance peak at layer 16 (0.212), band [9, 14, 15, 16, 17, 27], final-band mean 0.173, r = 0.816, KL argmax layer 26, label 'middle peak; late reconvergence; KL peak final' (210 items, 138 words)
- label components of the real base run that the null base runs also produce — permuted pairs: 'late reconvergence', 'KL peak final'; sequence-order control: 'middle peak', 'late reconvergence', 'KL peak final'. A component the finder also returns without the ambiguity effect is not evidence of that effect on this model
- magnitude comparison the paper does make for this control (its Figure 4, Llama-3.2-3B, medians early/middle/late homonym 0.483/0.546/0.482 vs sequence-order 0.113/0.223/0.190): here homonym early mean 0.346 / median 0.327; middle mean 0.510 / median 0.532; late mean 0.406 / median 0.414; sequence-order early mean 0.108 / median 0.090; middle mean 0.189 / median 0.199; late mean 0.182 / median 0.175. The graded finder is magnitude-blind: a control with the same profile shape at lower magnitude reproduces the label
- single-run axes: template=polysemes: 'middle peak; late reconvergence; KL peak final', band [9, 10, 11, 12, 14, 16], r = 0.718, KL argmax 25; template=next-token-position: 'middle peak; late reconvergence; KL peak final', band [4, 5, 7, 8, 9, 10], r = 0.624, KL argmax 25; metric=logit: 'middle peak; late reconvergence; KL peak final', band [9, 10, 11, 12, 13, 25], r = 0.815, KL argmax 25; lens=normed: 'middle peak; late reconvergence; KL peak final', band [8, 9, 10, 11, 12, 14], r = 0.669, KL argmax 25; agg=median: 'middle peak; late reconvergence; KL peak final', band [8, 9, 10, 11, 12, 14], r = 0.654, KL argmax 26; k=4: 'middle peak; late reconvergence; KL peak final', band [9, 10, 11, 12], r = 0.669, KL argmax 25; k=7: 'middle peak; late reconvergence; KL peak final', band [8, 9, 10, 11, 12, 13, 14], r = 0.669, KL argmax 25
- full polyseme set, base configuration: peak layer 9, final-band mean / peak 0.717, KL argmax layer 25
- the 'lens=normed' hyperparameter applies the model's final norm before unembedding (the standard logit lens); the upstream raw projection lets the growing residual norm sharpen late-layer softmaxes, so the final-layer KL peak under 'raw' partly measures norm growth — compare the two KL argmax values above

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.3 · 2026-09-01T22:11:40+00:00*
