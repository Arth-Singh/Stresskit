# 🟢 Diagnostic Stability Card — descriptive grade **A** (high confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Claim:** In google/gemma-3-4b-it, a truth direction (false vs true) and an impossibility direction (impossible vs possible), each a linear probe on the final-prompt-token residual stream, show a double dissociation on held-out topic families and are close to orthogonal
> model: google/gemma-3-4b-it · task: modality contrast set: 15 topic families x {true, false, improbable, impossible}, family-held-out probing (upstream data) · method: StandardScaler + L2 logistic regression probes (upstream make_probe), 5 family-grouped folds, transfer AUC of each probe on the other contrast, cosine of full-data probe directions

Battery: `seeds, bootstrap, templates, hyperparams` — 39 runs (seed 0, 20.855s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.884 | [0.851, 0.909] | ≥ 0.800 | ✅ pass |
| claim stability | 0.923 | [0.846, 1.000] | ≥ 0.800 | ✅ pass |
| score stability | 0.037 | [0.024, 0.049] | ≤ 0.250 | ✅ pass |
| beats random | 4.785 | [4.606, 4.920] | ≥ 3.000 | ✅ pass |
| specificity | 1.844 | [1.685, 2.045] | ≥ 1.500 | ✅ pass |

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 39 |
| structured runs | 39 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.884 |
| min pairwise Jaccard | 0.500 |
| random-null Jaccard | 0.185 |
| overlap vs random (×) | 4.785 |
| claim flip rate | 0.148 |
| modal claim share π* | 0.923 |
| distinct claims | 3 |
| score mean | 0.928 |
| score CV | 0.037 |
| median finding size | 28 |
| Jaccard 95% CI (bootstrap) | [0.851, 0.909] |
| flip rate 95% CI (bootstrap) | [0.000, 0.287] |
| null-control (specificity) | Jaccard 0.480 · flip 0.409 on 33 null runs |
| claim distribution | `both probes decode held-out families; double dissociation; near-orthogonal (|cos|<0.2)`×36, `both probes decode held-out families; truth probe impossibility-blind only; near-orthogonal (|cos|<0.2)`×2, `both probes decode held-out families; truth probe impossibility-blind only; oblique (|cos|>=0.2)`×1 |
| score-variance shares (OAT) | bootstrap: 26%, hyperparams: 51%, seeds: 7%, templates: 16% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 17 | 0.894 | 0.324 | 0.824 | 0.037 |
| hyperparams | 5 | 0.784 | 0.000 | 1.000 | 0.053 |
| seeds | 17 | 0.942 | 0.000 | 1.000 | 0.019 |
| templates | 3 | 0.892 | 0.000 | 1.000 | 0.029 |

## Notes

- scope: google/gemma-3-4b-it (revision 093f9f3, Gemma3ForConditionalGeneration, bfloat16, transformers 5.16.1), text-only chat usage with the model's default template; residual stream read at the final prompt token immediately before generation (upstream extraction; cached as float32 where upstream stored float16, because the bare-statement template overflows float16 at some depths); graded at hidden-state depth 16 (transformer layer 15) with C=0.1; every probe is evaluated on statements whose whole topic family was held out of its training data
- data: data/questions_modality.json at sixticket/representing-the-impossible@f1ead9a (repository MIT-licensed; the stimulus files carry no separate license and are covered by the repository license), SHA-256 verified; its 75 items are byte-identical to the modality items of questions_combined.json (sha256 20fbcc5c0710...) used by the upstream run; the 15 anomalous statements are outside both probes' contrasts and are excluded from the pool
- axes: seeds = the family-to-fold assignment (the lbfgs logistic regression is deterministic given its training set, so there is no separate probe-init randomness); bootstrap = statement resampling of the 60-item pool; templates = two prompt framings constructed for this card because the upstream repository ships a single template ('paraphrased-instruction' rewords the four-label instruction, 'bare-statement' drops it and sends the statement alone); hyperparams = depth [12, 20] and C [0.01, 1.0]
- null control: condition labels permuted once within each topic family (seed 0x5EC; the upstream permutation-test scheme, which preserves topic balance so a null probe cannot exploit topic vocabulary), run through the same finder with the same held-out evaluation universe; the surface-form baselines are a different finder and are reported below rather than graded
- null construction, direction of bias: one fixed permutation is reused by every null run, so null probes trained on different family splits share the same spurious label structure and their held-out selections agree more than chance (null Jaccard 0.480 vs 0.185 for size-matched random sets); this inflates the denominator of the specificity ratio, so the check is harder to pass than under a per-run re-permutation -- the null is conservative against the finding, not in its favour
- depth selection: upstream's reported peak (depth 16 = layer 15); held-out impossibility balanced accuracy peaks at depth 15 (0.967) on this extraction (upstream: depth 16, 0.967); at the graded depth 16: impossibility BA 0.967, truth BA 0.933, cos(truth, impossibility) +0.017; impossibility-probe AUC on false-vs-true by depth: [0.5, 0.4, 0.2444, 0.3111, 0.2889, 0.3556, 0.3111, 0.4222, 0.3778, 0.2667, 0.4, 0.3778, 0.3778, 0.3778, 0.4222, 0.5111, 0.4222, 0.4667, 0.5333, 0.5778, 0.6222, 0.7333, 0.6889, 0.7333, 0.7333, 0.7333, 0.7333, 0.7111, 0.7111, 0.6889, 0.7111, 0.7111, 0.7333, 0.7556, 0.7333]
- cosine curve (not graded): max |cos(truth, impossibility)| over depths > 10 = 0.108 (upstream reports at most 0.12); per-depth values live in the raw directory (depth_curve.json)
- base run vs the upstream reference run at depth 16 (results/reference_run/axes/axes_results.json @ f1ead9a): ba_impossibility +0.967 (upstream +0.967); ba_truth +0.933 (upstream +0.900); auc_impossibility_false_vs_true +0.422 (upstream +0.511); auc_truth_impossible_vs_false +0.200 (upstream +0.200); auc_truth_impossible_vs_true +0.978 (upstream +0.933); auc_truth_impossible_vs_possible +0.548 (upstream +0.578); cos_standardized +0.017 (upstream -0.010)
- stricter dissociation variant (not graded): requiring the truth probe to sit within +/-0.15 of chance on BOTH impossible-vs-false and impossible-vs-possible (two-sided) holds in 1/39 runs; truth-probe AUC on impossible-vs-false ranges 0.09-0.36 (below 0.5 = impossible statements score LESS false than contingent falsehoods), on impossible-vs-possible 0.44-0.70
- verbal labels (not graded; greedy, 48 new tokens, upstream template; n=15 per condition) reproduces the conflation: 12/15 contingent falsehoods labelled 'contradiction' (upstream 12/15) vs 5/15 impossible statements (upstream 5/15); full table -- true: coherent 14, contradiction 1; false: coherent 1, contradiction 12, underdetermined 2; improbable: coherent 4, contradiction 7, paradox 1, underdetermined 3; anomalous: coherent 2, paradox 10, underdetermined 3; impossible: contradiction 5, paradox 9, underdetermined 1
- impossibility direction geometry (raw residual space, not graded): mean |cos| to the base direction on the seeds axis 1.000 (min 1.000, n=16)
- impossibility direction geometry (raw residual space, not graded): mean |cos| to the base direction on the bootstrap axis 0.865 (min 0.690, n=16)
- impossibility direction geometry (raw residual space, not graded): mean |cos| to the base direction on the hyperparams axis 0.625 (min 0.124, n=4)
- impossibility direction geometry (raw residual space, not graded): mean |cos| to the base direction on the templates axis 0.433 (min 0.216, n=2)
- truth direction geometry (raw residual space, not graded): mean |cos| to the base direction on the seeds axis 1.000 (min 1.000, n=16)
- truth direction geometry (raw residual space, not graded): mean |cos| to the base direction on the bootstrap axis 0.598 (min 0.364, n=16)
- truth direction geometry (raw residual space, not graded): mean |cos| to the base direction on the hyperparams axis 0.616 (min 0.163, n=4)
- truth direction geometry (raw residual space, not graded): mean |cos| to the base direction on the templates axis 0.191 (min 0.047, n=2)
- null-control impossibility directions: mean pairwise |cos| 0.742 (n=33)
- null-control truth directions: mean pairwise |cos| 0.718 (n=33)
- surface-form baselines (upstream token-count / word TF-IDF / char TF-IDF probes on the same family folds; a different finder, so not the graded null): {"impossible_vs_possible": {"token_count": 0.556, "tfidf_word": 0.778, "tfidf_char": 0.733}, "impossible_vs_false": {"token_count": 0.6, "tfidf_word": 0.833, "tfidf_char": 0.767}} balanced accuracy vs the activation probe's held-out balanced accuracy on the card (upstream: 0.56/0.70/0.64 and 0.60/0.67/0.77)

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.3 · 2026-09-01T21:53:41+00:00*
