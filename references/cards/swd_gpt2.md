# 🟠 Diagnostic Stability Card — descriptive grade **C** (low confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Claim:** Across single-matrix replacements, SWD matches the held-out fidelity achieved by Transcoder and other strong baselines while using less than 1% of the data that those baselines use to train their replacements. For matched replacement fidelity, SWD reaches the same circuit sufficiency and necessity targets with fewer active read/write edges and selected units across tasks on GPT-2, Qwen2.5, and Qwen3.5-27B
> model: gpt2 · task: GPT-2 small layer-8 mlp.c_proj replaced by a 50%-sparse two-factor decomposition from 16,384 FineWeb-Edu calibration tokens; held-out CE delta on 2,048 blocks and sufficiency/necessity circuit frontiers on IOI, docstring and gendered-pronoun against the released Transcoder-12k frontier (greater-than reproduced once, outside the battery) · method: upstream factorize/evaluate/circuit stages (vendored Double Sparse Factorization, task-margin attribution ranking, validation-selected prefix, mean ablation) at the pinned commit

Battery: `seeds, bootstrap, templates, hyperparams` — 22 runs (seed 20260606, 0.091s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.963 | [0.892, 1.000] | ≥ 0.800 | ✅ pass |
| claim stability | 0.545 | [0.455, 0.773] | ≥ 0.800 | ❌ fail |
| score stability | 0.630 | [0.219, 0.807] | ≤ 0.250 | ⚠️ inconclusive |
| beats random | 25.311 | [23.448, 26.285] | ≥ 3.000 | ✅ pass |
| specificity | 1.146 | [0.924, 1.601] | ≥ 1.500 | ⚠️ inconclusive |

> ⚠️ **Underpowered:** the 95% CI straddles the bar for score_stability, specificity — undecided in either direction. The grade is provisional; raise `n_runs` before reporting it.

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 22 |
| structured runs | 22 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.963 |
| Jaccard incl. size-mismatched runs | 0.779 |
| min pairwise Jaccard | 0.286 |
| random-null Jaccard | 0.038 |
| overlap vs random (×) | 25.311 |
| claim flip rate | 0.558 |
| modal claim share π* | 0.545 |
| distinct claims | 3 |
| score mean | 0.002 |
| score CV | 0.630 |
| median finding size | 2.000 |
| Jaccard 95% CI (bootstrap) | [0.892, 1.000] |
| flip rate 95% CI (bootstrap) | [0.401, 0.662] |
| null-control (specificity) | Jaccard 0.840 · flip 0.462 on 13 null runs |
| claim distribution | `CE delta <=0.002, matched to TC-12k; fewer than TC-12k on some contested cells`×12, `CE delta 0.002-0.01, not matched to TC-12k; fewer than TC-12k on some contested cells`×9, `CE delta <=0.002, not matched to TC-12k; fewer than TC-12k on some contested cells`×1 |
| score-variance shares (OAT) | bootstrap: 1%, hyperparams: 90%, seeds: 2%, templates: 6% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 7 | 0.829 | 0.714 | 0.429 | 0.142 |
| hyperparams | 8 | 0.638 | 0.536 | 0.625 | 0.759 |
| seeds | 7 | 1.000 | 0.286 | 0.857 | 0.204 |
| templates | 3 | 0.778 | 0.667 | 0.667 | 0.256 |

## Notes

- structural stability graded on 18 size-comparable runs; 4 run(s) with >2x size difference excluded (pooled-over-all value kept as mean_pairwise_jaccard_all_sizes)
- pooled Jaccard (0.963) and axis-balanced Jaccard (0.811) diverge by more than 0.1 — per-axis run counts are shaping the pooled number; read the per-axis breakdown before citing it
- underpowered verdict: the 95% CI straddles the bar for score_stability (fail), specificity (fail) at n_runs=6 — these verdict components are not decided by the data. Treat the grade as provisional and raise n_runs (or widen the battery) before reporting it.
- upstream: veri-safe/SWD@4c44b72 (Apache-2.0); the factorize, evaluate and circuit stages, the vendored solver and the released tables imported unmodified; file hashes pipeline.py f78085ba2341, factorization.py ead6974690e7, evaluation.py e2dd44a1dd83, circuits.py 7a645bf032b3, modules.py d326daecbd53, data.py a560bd5b5eca, doublesparse.py 84bc725239fc, gpt2_single_projection.yaml cc691f9e2260, replacement_quality.csv c815e43f89eb, reconstruction_quality.csv 865ac48e584c, circuit_frontiers.csv ef4a593e5832
- reproduction (released SWD s=0.5 at 16384 calibration tokens -> base run): CE delta 0.000889 -> 0.001423; KL 0.001871 -> 0.001884; output cosine 0.9814 -> 0.9815; dense CE 3.244780 -> 3.232037 (different evaluation blocks). The calibration blocks differ: upstream drew them from FineWeb-Edu sample-10BT files 000-012 under a seeded streaming shuffle, this card from file 000 alone (pool of 4,096 blocks, seed 0), so the base run is the same protocol on a different 16-block sample, not the released checkpoint
- comparator: released Transcoder-12k checkpoint at 2,048,000 optimizer-replay tokens, CE delta 0.000979 (the paper's CE-matched partner of SWD s=0.5 at 16,384 tokens, 0.8% of its data); its frontier is read from paper_data/exp1_gpt2_single/circuit_frontiers.csv and is not retrained. On the three battery families the released SWD run beats it on 9/11 contested cells; the base run on 2/18, Jaccard to the released won set 0.22
- base run ioi (k_min 2): sufficiency@0.8: units 766 (released SWD 1, TC 1), edges 1179646 (released SWD 986, TC 3840); sufficiency@0.9: units 766 (released SWD 1, TC 1), edges 1179646 (released SWD 986, TC 3840); sufficiency@0.95: units 766 (released SWD 1, TC 1), edges 1179646 (released SWD 986, TC 3840); necessity_drop@0.8: units none (released SWD none, TC none), edges none (released SWD none, TC none); necessity_drop@0.9: units none (released SWD none, TC none), edges none (released SWD none, TC none); necessity_drop@0.95: units none (released SWD none, TC none), edges none (released SWD none, TC none)
- base run docstring (k_min 4): sufficiency@0.8: units 24 (released SWD 1, TC 1), edges 36391 (released SWD 986, TC 3840); sufficiency@0.9: units 24 (released SWD 1, TC 1), edges 36391 (released SWD 986, TC 3840); sufficiency@0.95: units 24 (released SWD 4, TC 1), edges 36391 (released SWD 7078, TC 3840); necessity_drop@0.8: units none (released SWD none, TC none), edges none (released SWD none, TC none); necessity_drop@0.9: units none (released SWD none, TC none), edges none (released SWD none, TC none); necessity_drop@0.95: units none (released SWD none, TC none), edges none (released SWD none, TC none)
- base run gendered_pronoun (k_min 2): sufficiency@0.8: units 32 (released SWD 1, TC 1), edges 48050 (released SWD 1582, TC 3840); sufficiency@0.9: units 32 (released SWD 1, TC 1), edges 48050 (released SWD 1582, TC 3840); sufficiency@0.95: units 32 (released SWD 12, TC 128), edges 48050 (released SWD 17794, TC 491520); necessity_drop@0.8: units none (released SWD none, TC none), edges none (released SWD none, TC none); necessity_drop@0.9: units none (released SWD none, TC none), edges none (released SWD none, TC none); necessity_drop@0.95: units none (released SWD none, TC none), edges none (released SWD none, TC none)
- greater-than reproduction (base calibration blocks, upstream protocol with 50 random controls, one run outside the battery, k_min 1, 13680s): sufficiency@0.8: units 48 (released SWD 24, TC 192), edges 75611 (released SWD 38002, TC 737280); sufficiency@0.9: units 48 (released SWD 48, TC 384), edges 75611 (released SWD 75327, TC 1474560); sufficiency@0.95: units 48 (released SWD 48, TC 512), edges 75611 (released SWD 75327, TC 1966080); necessity_drop@0.8: units 48 (released SWD 24, TC 192), edges 75611 (released SWD 38002, TC 737280); necessity_drop@0.9: units 64 (released SWD 24, TC 256), edges 100423 (released SWD 38002, TC 983040); necessity_drop@0.95: units 96 (released SWD 32, TC 384), edges 149995 (released SWD 50753, TC 1474560)
- template=wikipedia: CE delta 0.00265, KL 0.00209, cosine 0.9796, token exposure 16,384; won 3/18 contested cells; gendered-pronoun sufficiency@0.95 units 32 edges 48698; k_min {'ioi': 2, 'docstring': 2, 'gendered_pronoun': 12}
- template=fineweb: CE delta 0.00261, KL 0.00199, cosine 0.9803, token exposure 16,384; won 2/18 contested cells; gendered-pronoun sufficiency@0.95 units 48 edges 74752; k_min {'ioi': 8, 'docstring': 1, 'gendered_pronoun': 2}
- sparsity=0.75: CE delta 0.00818, KL 0.00909, cosine 0.9118, token exposure 16,384; won 7/20 contested cells; gendered-pronoun sufficiency@0.95 units 32 edges 25169; k_min {'ioi': 8, 'docstring': 1, 'gendered_pronoun': 4}
- blocks=1: CE delta 0.00308, KL 0.00298, cosine 0.9716, token exposure 1,024; won 5/15 contested cells; gendered-pronoun sufficiency@0.95 units 64 edges 98165; k_min {'ioi': 8, 'docstring': 1, 'gendered_pronoun': 1}
- blocks=64: CE delta 0.00189, KL 0.00183, cosine 0.9821, token exposure 65,536; won 2/18 contested cells; gendered-pronoun sufficiency@0.95 units 64 edges 98323; k_min {'ioi': 4, 'docstring': 2, 'gendered_pronoun': 8}
- outer_iterations=8: CE delta 0.00197, KL 0.00262, cosine 0.9744, token exposure 16,384; won 5/15 contested cells; gendered-pronoun sufficiency@0.95 units 48 edges 74149; k_min {'ioi': 2, 'docstring': 1, 'gendered_pronoun': 3}
- final_iterations=0: CE delta 0.00331, KL 0.00294, cosine 0.9759, token exposure 16,384; won 2/18 contested cells; gendered-pronoun sufficiency@0.95 units 32 edges 47391; k_min {'ioi': 3, 'docstring': 3, 'gendered_pronoun': 1}
- eval_set=in-sample: CE delta 0.00132, KL 0.00164, cosine 0.9833, token exposure 16,384; won 2/18 contested cells; gendered-pronoun sufficiency@0.95 units 32 edges 48050; k_min {'ioi': 2, 'docstring': 4, 'gendered_pronoun': 2}
- random_seeds=10: CE delta 0.00142, KL 0.00188, cosine 0.9815, token exposure 16,384; won 2/18 contested cells; gendered-pronoun sufficiency@0.95 units 32 edges 48050; k_min {'ioi': 2, 'docstring': 4, 'gendered_pronoun': 1}
- the battery's circuit stage runs IOI, docstring and gendered-pronoun; greater-than (9,160 validation prompts x 50 random controls x 20 prefixes through the upstream per-prompt loop, over an hour per run) is reproduced once on the base blocks and kept out of the battery. Circuit batch size 64 instead of upstream's 16 and evaluation batch size 2 instead of 4 (runtime settings for a shared GPU; the margin is read at each prompt's last real token and CE/KL are token sums, so batching changes nothing but float summation order)
- null control: calibration blocks of uniformly random GPT-2 token ids (seeded per slot); the factorisation, evaluation and circuit stages are unchanged, only the Gram no longer reflects the model's input distribution

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.4 · 2026-09-02T15:54:35+00:00*
