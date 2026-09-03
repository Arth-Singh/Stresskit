# 🟠 Diagnostic Stability Card — descriptive grade **C** (low confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Grade rule v0.4:** a check counts as passed only when its whole 95% interval clears the bar; a decided specificity fail caps the grade at C; no null control caps it at B; overlap at or below the at-random floor is D.

> **Claim:** Homonym and polyseme representations become maximally distinct in middle layers and partially reconverge in late layers, while the KL divergence between their next-token predictions peaks in the final layers.
> model: gpt2 · task: layer-wise logit-lens profile of homonym pairs (upstream Experiment 1 stimuli; activation distance, logit distance, KL divergence) · method: upstream per-layer distance/KL profile; top-k activation-distance band, reconvergence ratio, fixed-threshold profile label

Battery: `seeds, bootstrap, templates, hyperparams` — 31 runs (seed 0, 22.015s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.896 | [0.779, 1.000] | ≥ 0.800 | ⚠️ inconclusive |
| claim stability | 0.903 | [0.774, 1.000] | ≥ 0.800 | ⚠️ inconclusive |
| score stability | 0.151 | [0.009, 0.240] | ≤ 0.250 | ✅ pass |
| beats random | 5.565 | [4.834, 6.209] | ≥ 3.000 | ✅ pass |
| specificity | 1.076 | [0.884, 1.275] | ≥ 1.500 | ❌ fail |

> ⚠️ **Underpowered:** the 95% CI straddles the bar for structural_stability, claim_stability — undecided in either direction. The grade is provisional; raise `n_runs` before reporting it.

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 31 |
| structured runs | 31 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.896 |
| min pairwise Jaccard | 0.000 |
| random-null Jaccard | 0.161 |
| overlap vs random (×) | 5.565 |
| claim flip rate | 0.185 |
| modal claim share π* | 0.903 |
| distinct claims | 3 |
| score mean | 0.548 |
| score CV | 0.151 |
| median finding size | 3 |
| Jaccard 95% CI (bootstrap) | [0.779, 1.000] |
| flip rate 95% CI (bootstrap) | [0.000, 0.384] |
| null-control (specificity) | Jaccard 0.833 · flip 0.153 on 25 null runs |
| claim distribution | `middle peak; late reconvergence; KL peak final`×28, `middle peak; late reconvergence; KL peak late`×2, `late peak; no late reconvergence; KL peak final`×1 |
| score-variance shares (OAT) | bootstrap: 0%, hyperparams: 87%, seeds: 0%, templates: 13% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 13 | 1.000 | 0.000 | 1.000 | 0.007 |
| hyperparams | 5 | 0.585 | 0.700 | 0.600 | 0.281 |
| seeds | 13 | 1.000 | 0.000 | 1.000 | 0.005 |
| templates | 3 | 0.667 | 0.667 | 0.667 | 0.137 |

## Notes

- underpowered verdict: the 95% CI straddles the bar for claim_stability (pass), structural_stability (pass) at n_runs=12 — these verdict components are not decided by the data. Treat the grade as provisional and raise n_runs (or widen the battery) before reporting it.
- scope: gpt2 (HF revision 607a30d783dfa663caf39e06633721c8d4cfcd7e) in float32, 12 post-block residual layers read at the target-word token, one sentence per forward pass, no chat template; the paper's models are gpt2, meta-llama/Llama-3.2-3B and Qwen/Qwen2.5-32B via TransformerLens 2.1.6 — this card grades the profile on the model named here only, with the TransformerLens conventions the upstream inherits (fold_ln, center_unembed, center_writing_weights for LayerNorm models) reproduced on plain Hugging Face weights
- data: upstream stimulus files at scoki211/Divergent_LLM_Predictions_Convergent_Reps_amb_words@dd08338 (MIT), SHA-256 verified; upstream tokenisation validation reproduced: homonyms 167/190 words -> 454 context pairs (failures {'inconsistent_tokenization': 21, 'not_found': 2}); polysemes 93/97 words -> 115 pairs (failures {'inconsistent_tokenization': 3, 'not_found': 1}); paper Table 1 reports 167 homonyms and 93 polysemes for this model; the paper's KL > 0.5 stimulus screen is not part of the released Experiment-1 entrypoint and is not applied
- representation (fixed before running): band = top-3 layers by activation distance (k = max(3, round(0.2 L))); score r = mean activation distance over the final band [10, 11] / peak; label = majority manuscript layer group of the band (early <= 3, middle <= 8, late above) + 'late reconvergence' iff r <= 0.9 + KL argmax in the final band -> 'final'; the paper's claim is the label 'middle peak; late reconvergence; KL peak final'; each finder call subsamples 75% of the items with its seed
- base run (subsampled): activation-distance peak at layer 8 (0.245), band [7, 8, 9], final-band mean 0.131, r = 0.534, KL argmax layer 11, label 'middle peak; late reconvergence; KL peak final' (341 items, 157 words)
- full homonym set, base configuration: activation-distance peak layer 8 (0.243), final-band mean 0.130, KL argmax layer 11; per-(word, layer) activation distance by manuscript group: early mean 0.069 / median 0.049; middle mean 0.215 / median 0.215; late mean 0.166 / median 0.152 (paper: GPT-2 means 0.069/0.215/0.167, Llama-3.2-3B medians 0.48/0.55/0.48, Qwen2.5-32B medians 0.38/0.63/0.42 for early/middle/late)
- null control (graded): the homonym items with sentence B re-paired to a different word (seeded derangement, seed 0x5ec, 454 items, same sentences, same read positions, same finder) — the effect under test (one surface form, two senses) is absent while everything else is held fixed. Fair because the finder's output is a stable top-k band on any smooth depth profile, so the specificity ratio asks whether band stability is diagnostic of the ambiguity effect; it does NOT test whether the null recovers the same band or label — see the next notes. Conservative direction: different-word pairs are far apart at every depth, so their profile is shaped by the global geometry (anisotropy), not by sense resolution
- null base run (permuted pairs): activation-distance peak at layer 5 (0.485), band [3, 4, 5], final-band mean 0.187, r = 0.387, KL argmax layer 7, label 'middle peak; late reconvergence; KL peak middle' (341 items, 161 words)
- null claim distribution: 'middle peak; late reconvergence; KL peak middle' x23, 'middle peak; late reconvergence; KL peak early' x2; Jaccard between the real and null base bands 0.000; full permuted set: peak layer 5, final-band mean / peak 0.388, KL argmax layer 7
- alternative null (reported, not graded): the upstream sequence-order control set (281 matched tokens from 56 reordered sentence pairs) through the same finder — null Jaccard 1.000, specificity ratio 0.896 (95% CI [0.770, 1.000]), fail at >= 1.5; null claim distribution: 'middle peak; late reconvergence; KL peak final' x25; Jaccard between the real and sequence-order base bands 1.000
- sequence-order null base run: activation-distance peak at layer 9 (0.071), band [7, 8, 9], final-band mean 0.042, r = 0.595, KL argmax layer 11, label 'middle peak; late reconvergence; KL peak final' (211 items, 142 words)
- label components of the real base run that the null base runs also produce — permuted pairs: 'middle peak', 'late reconvergence'; sequence-order control: 'middle peak', 'late reconvergence', 'KL peak final'. A component the finder also returns without the ambiguity effect is not evidence of that effect on this model
- magnitude comparison the paper does make for this control (its Figure 4, Llama-3.2-3B, medians early/middle/late homonym 0.483/0.546/0.482 vs sequence-order 0.113/0.223/0.190): here homonym early mean 0.069 / median 0.049; middle mean 0.215 / median 0.215; late mean 0.166 / median 0.152; sequence-order early mean 0.017 / median 0.011; middle mean 0.058 / median 0.055; late mean 0.053 / median 0.046. The graded finder is magnitude-blind: a control with the same profile shape at lower magnitude reproduces the label
- single-run axes: template=polysemes: 'middle peak; late reconvergence; KL peak final', band [7, 8, 9], r = 0.568, KL argmax 11; template=next-token-position: 'middle peak; late reconvergence; KL peak late', band [6, 7, 8], r = 0.408, KL argmax 9; metric=logit: 'late peak; no late reconvergence; KL peak final', band [9, 10, 11], r = 0.980, KL argmax 11; lens=normed: 'middle peak; late reconvergence; KL peak late', band [7, 8, 9], r = 0.534, KL argmax 9; agg=median: 'middle peak; late reconvergence; KL peak final', band [7, 8, 9], r = 0.556, KL argmax 11; k=2: 'middle peak; late reconvergence; KL peak final', band [8, 9], r = 0.534, KL argmax 11
- full polyseme set, base configuration: peak layer 9, final-band mean / peak 0.564, KL argmax layer 11
- the 'lens=normed' hyperparameter applies the model's final norm before unembedding (the standard logit lens); the upstream raw projection lets the growing residual norm sharpen late-layer softmaxes, so the final-layer KL peak under 'raw' partly measures norm growth — compare the two KL argmax values above
- v0.3 grade: B; regraded 2026-09-03 under grade rule v0.4 from the recorded checks (schema 0.5)

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.5 · 2026-09-01T22:07:12+00:00*
