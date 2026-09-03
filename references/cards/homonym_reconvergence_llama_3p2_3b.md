# 🟠 Diagnostic Stability Card — descriptive grade **C** (low confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Grade rule v0.4:** a check counts as passed only when its whole 95% interval clears the bar; a decided specificity fail caps the grade at C; no null control caps it at B; overlap at or below the at-random floor is D.

> **Claim:** Homonym and polyseme representations become maximally distinct in middle layers and partially reconverge in late layers, while the KL divergence between their next-token predictions peaks in the final layers.
> model: unsloth/Llama-3.2-3B · task: layer-wise logit-lens profile of homonym pairs (upstream Experiment 1 stimuli; activation distance, logit distance, KL divergence) · method: upstream per-layer distance/KL profile; top-k activation-distance band, reconvergence ratio, fixed-threshold profile label

Battery: `seeds, bootstrap, templates, hyperparams` — 32 runs (seed 0, 192.349s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.930 | [0.837, 1.000] | ≥ 0.800 | ✅ pass |
| claim stability | 0.906 | [0.781, 1.000] | ≥ 0.800 | ⚠️ inconclusive |
| score stability | 0.056 | [0.005, 0.095] | ≤ 0.250 | ✅ pass |
| beats random | 7.283 | [6.552, 7.830] | ≥ 3.000 | ✅ pass |
| specificity | 0.930 | [0.842, 1.000] | ≥ 1.500 | ❌ fail |

> ⚠️ **Underpowered:** the 95% CI straddles the bar for claim_stability — undecided in either direction. The grade is provisional; raise `n_runs` before reporting it.

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 32 |
| structured runs | 32 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.930 |
| min pairwise Jaccard | 0.333 |
| random-null Jaccard | 0.128 |
| overlap vs random (×) | 7.283 |
| claim flip rate | 0.179 |
| modal claim share π* | 0.906 |
| distinct claims | 3 |
| score mean | 0.712 |
| score CV | 0.056 |
| median finding size | 6.000 |
| Jaccard 95% CI (bootstrap) | [0.837, 1.000] |
| flip rate 95% CI (bootstrap) | [0.000, 0.384] |
| null-control (specificity) | Jaccard 1.000 · flip 0.000 on 25 null runs |
| claim distribution | `middle peak; late reconvergence; KL peak final`×29, `early peak; late reconvergence; KL peak final`×2, `middle peak; late reconvergence; KL peak late`×1 |
| score-variance shares (OAT) | bootstrap: 0%, hyperparams: 93%, seeds: 0%, templates: 7% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 13 | 1.000 | 0.000 | 1.000 | 0.007 |
| hyperparams | 6 | 0.844 | 0.600 | 0.667 | 0.122 |
| seeds | 13 | 1.000 | 0.000 | 1.000 | 0.004 |
| templates | 3 | 0.556 | 0.667 | 0.667 | 0.031 |

## Notes

- underpowered verdict: the 95% CI straddles the bar for claim_stability (pass) at n_runs=12 — these verdict components are not decided by the data. Treat the grade as provisional and raise n_runs (or widen the battery) before reporting it.
- scope: unsloth/Llama-3.2-3B (HF revision d4446454d87d51aa42e1fb174f25acc5f8762331) in float16, 28 post-block residual layers read at the target-word token, one sentence per forward pass, no chat template; the paper's models are gpt2, meta-llama/Llama-3.2-3B and Qwen/Qwen2.5-32B via TransformerLens 2.1.6 — this card grades the profile on the model named here only, with the TransformerLens conventions the upstream inherits (fold_ln, center_unembed, center_writing_weights for LayerNorm models) reproduced on plain Hugging Face weights
- data: upstream stimulus files at scoki211/Divergent_LLM_Predictions_Convergent_Reps_amb_words@dd08338 (MIT), SHA-256 verified; upstream tokenisation validation reproduced: homonyms 166/190 words -> 451 context pairs (failures {'inconsistent_tokenization': 22, 'not_found': 2}); polysemes 94/97 words -> 116 pairs (failures {'inconsistent_tokenization': 2, 'not_found': 1}); paper Table 1 reports 166 homonyms and 94 polysemes for this model; the paper's KL > 0.5 stimulus screen is not part of the released Experiment-1 entrypoint and is not applied
- representation (fixed before running): band = top-6 layers by activation distance (k = max(3, round(0.2 L))); score r = mean activation distance over the final band [25, 26, 27] / peak; label = majority manuscript layer group of the band (early <= 6, middle <= 16, late above) + 'late reconvergence' iff r <= 0.9 + KL argmax in the final band -> 'final'; the paper's claim is the label 'middle peak; late reconvergence; KL peak final'; each finder call subsamples 75% of the items with its seed
- base run (subsampled): activation-distance peak at layer 6 (0.633), band [5, 6, 7, 8, 9, 10], final-band mean 0.457, r = 0.721, KL argmax layer 27, label 'middle peak; late reconvergence; KL peak final' (339 items, 158 words)
- the base activation-distance peak (layer 6) falls in the manuscript's 'early' group while the band majority is 'middle': the peak sits on a group boundary, so the zone label rests on the band's majority — see the single-run axes for the variants that relabel it
- full homonym set, base configuration: activation-distance peak layer 6 (0.623), final-band mean 0.448, KL argmax layer 27; per-(word, layer) activation distance by manuscript group: early mean 0.450 / median 0.483; middle mean 0.534 / median 0.546; late mean 0.466 / median 0.482 (paper: GPT-2 means 0.069/0.215/0.167, Llama-3.2-3B medians 0.48/0.55/0.48, Qwen2.5-32B medians 0.38/0.63/0.42 for early/middle/late)
- null control (graded): the homonym items with sentence B re-paired to a different word (seeded derangement, seed 0x5ec, 451 items, same sentences, same read positions, same finder) — the effect under test (one surface form, two senses) is absent while everything else is held fixed. Fair because the finder's output is a stable top-k band on any smooth depth profile, so the specificity ratio asks whether band stability is diagnostic of the ambiguity effect; it does NOT test whether the null recovers the same band or label — see the next notes. Conservative direction: different-word pairs are far apart at every depth, so their profile is shaped by the global geometry (anisotropy), not by sense resolution
- null base run (permuted pairs): activation-distance peak at layer 2 (0.852), band [0, 1, 2, 3, 4, 5], final-band mean 0.565, r = 0.664, KL argmax layer 27, label 'early peak; late reconvergence; KL peak final' (339 items, 161 words)
- null claim distribution: 'early peak; late reconvergence; KL peak final' x25; Jaccard between the real and null base bands 0.091; full permuted set: peak layer 2, final-band mean / peak 0.666, KL argmax layer 27
- alternative null (reported, not graded): the upstream sequence-order control set (280 matched tokens from 56 reordered sentence pairs) through the same finder — null Jaccard 0.857, specificity ratio 1.085 (95% CI [0.973, 1.171]), fail at >= 1.5; null claim distribution: 'middle peak; late reconvergence; KL peak final' x25; Jaccard between the real and sequence-order base bands 0.714
- sequence-order null base run: activation-distance peak at layer 10 (0.261), band [6, 7, 8, 9, 10, 11], final-band mean 0.187, r = 0.715, KL argmax layer 27, label 'middle peak; late reconvergence; KL peak final' (210 items, 138 words)
- label components of the real base run that the null base runs also produce — permuted pairs: 'late reconvergence', 'KL peak final'; sequence-order control: 'middle peak', 'late reconvergence', 'KL peak final'. A component the finder also returns without the ambiguity effect is not evidence of that effect on this model
- magnitude comparison the paper does make for this control (its Figure 4, Llama-3.2-3B, medians early/middle/late homonym 0.483/0.546/0.482 vs sequence-order 0.113/0.223/0.190): here homonym early mean 0.450 / median 0.483; middle mean 0.534 / median 0.546; late mean 0.466 / median 0.482; sequence-order early mean 0.138 / median 0.113; middle mean 0.219 / median 0.223; late mean 0.188 / median 0.190. The graded finder is magnitude-blind: a control with the same profile shape at lower magnitude reproduces the label
- single-run axes: template=polysemes: 'middle peak; late reconvergence; KL peak final', band [5, 6, 7, 8, 9, 10], r = 0.765, KL argmax 27; template=next-token-position: 'early peak; late reconvergence; KL peak final', band [2, 3, 4, 5, 6, 7], r = 0.715, KL argmax 27; metric=logit: 'middle peak; late reconvergence; KL peak final', band [5, 6, 7, 8, 9, 10], r = 0.495, KL argmax 27; lens=normed: 'middle peak; late reconvergence; KL peak late', band [5, 6, 7, 8, 9, 10], r = 0.721, KL argmax 24; agg=median: 'middle peak; late reconvergence; KL peak final', band [5, 6, 7, 8, 9, 10], r = 0.693, KL argmax 27; k=4: 'early peak; late reconvergence; KL peak final', band [5, 6, 7, 8], r = 0.721, KL argmax 27; k=7: 'middle peak; late reconvergence; KL peak final', band [4, 5, 6, 7, 8, 9, 10], r = 0.721, KL argmax 27
- full polyseme set, base configuration: peak layer 7, final-band mean / peak 0.759, KL argmax layer 27
- the 'lens=normed' hyperparameter applies the model's final norm before unembedding (the standard logit lens); the upstream raw projection lets the growing residual norm sharpen late-layer softmaxes, so the final-layer KL peak under 'raw' partly measures norm growth — compare the two KL argmax values above
- v0.3 grade: B; regraded 2026-09-03 under grade rule v0.4 from the recorded checks (schema 0.5)

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.5 · 2026-09-01T22:03:38+00:00*
