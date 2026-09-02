# 🟢 Diagnostic Stability Card — descriptive grade **A** (high confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Claim:** The residual stream cleanly distinguishes cultures, well above a name-string baseline, yet the decoder collapses culturally-specific tokens onto dominant-tradition ones
> model: meta-llama/Llama-3.1-8B-Instruct · task: FolkMotif: 270 (motif, culture) cells; 10-way culture probe on entity-token residuals vs. named-entity generation; 2x2 decomposition · method: upstream pipeline (ridge probe with stratified CV, greedy generation with lenient string scoring, Preserved/DecodingSuppressed/SurfaceLuck/RepresentationallyFlat labelling) at the pinned commit

Battery: `seeds, templates, hyperparams` — 52 runs (seed 0, 5.317s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.878 | [0.810, 0.930] | ≥ 0.800 | ✅ pass |
| claim stability | 1.000 | [1.000, 1.000] | ≥ 0.800 | ✅ pass |
| score stability | 0.046 | [0.017, 0.067] | ≤ 0.250 | ✅ pass |
| beats random | 9.650 | [8.903, 10.228] | ≥ 3.000 | ✅ pass |
| specificity | 3.122 | [2.693, 3.627] | ≥ 1.500 | ✅ pass |

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 52 |
| structured runs | 52 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.878 |
| Jaccard incl. size-mismatched runs | 0.848 |
| min pairwise Jaccard | 0.060 |
| random-null Jaccard | 0.091 |
| overlap vs random (×) | 9.650 |
| claim flip rate | 0.000 |
| modal claim share π* | 1.000 |
| distinct claims | 1 |
| score mean | 0.708 |
| score CV | 0.046 |
| median finding size | 45.000 |
| Jaccard 95% CI (bootstrap) | [0.810, 0.930] |
| flip rate 95% CI (bootstrap) | [0.000, 0.000] |
| null-control (specificity) | Jaccard 0.281 · flip 0.000 on 41 null runs |
| score-variance shares (OAT) | hyperparams: 83%, seeds: 1%, templates: 16% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| hyperparams | 10 | 0.631 | 0.000 | 1.000 | 0.098 |
| seeds | 41 | 0.944 | 0.000 | 1.000 | 0.010 |
| templates | 3 | 0.490 | 0.000 | 1.000 | 0.042 |

## Notes

- structural stability graded on 51 size-comparable runs; 1 run(s) with >2x size difference excluded (pooled-over-all value kept as mean_pairwise_jaccard_all_sizes)
- pooled Jaccard (0.878) and axis-balanced Jaccard (0.688) diverge by more than 0.1 — per-axis run counts are shaping the pooled number; read the per-axis breakdown before citing it
- upstream: AragonerUA/folkmotif@cb0ae7c (MIT code, CC-BY-4.0 data); extraction, generation, scoring, probe and labelling functions imported unmodified; file hashes extract.py 67a0f17d66d9, probe.py 24d53af4c2e8, output_extract.py 37e0dc119554, scoring.py a91078e282f6, decomposition.py 9ca34d9a4bb1, prompts.py c089d477d951, prompts_v3.py 676d5a48e12b, ground_truth_staging.json 887537a72ffd
- reproduction (released llama-3.1-8b fp16 v3e6 results -> base run): probe peak 0.881 at layer 8 -> 0.881 at layer 8; name n-gram baseline 0.604 -> 0.604; output accuracy (majority) 0.185 -> 0.185; buckets {'Preserved': 45, 'DecodingSuppressed': 193, 'SurfaceLuck': 5, 'RepresentationallyFlat': 27} -> {'Preserved': 45, 'DecodingSuppressed': 193, 'SurfaceLuck': 5, 'RepresentationallyFlat': 27}. The paper's 0.248 output accuracy for this model is the released rescored majority (analysis/rescore_v3.py scores the raw generation instead of the trimmed one; the scoring=raw run below), and its decomposition table lists the v3h6 run, {'Preserved': 32, 'DecodingSuppressed': 206, 'SurfaceLuck': 6, 'RepresentationallyFlat': 26} (the template=v3_h6_native run below). The n-gram baseline here is a character 2-4-gram ridge probe on the name string under the same folds; upstream's analysis script uses its own n-gram classifier, not shipped with the pipeline
- template=v2_chat: peak L8 acc 0.881, n-gram 0.604, output acc 0.215, buckets {'Preserved': 52, 'DecodingSuppressed': 186, 'SurfaceLuck': 6, 'RepresentationallyFlat': 26}, DS share 0.689
- template=v3_h6_native: peak L8 acc 0.881, n-gram 0.604, output acc 0.141, buckets {'Preserved': 32, 'DecodingSuppressed': 206, 'SurfaceLuck': 6, 'RepresentationallyFlat': 26}, DS share 0.763
- alpha=0.1: peak L8 acc 0.881, n-gram 0.604, output acc 0.185, buckets {'Preserved': 45, 'DecodingSuppressed': 193, 'SurfaceLuck': 5, 'RepresentationallyFlat': 27}, DS share 0.715
- alpha=10.0: peak L8 acc 0.881, n-gram 0.604, output acc 0.185, buckets {'Preserved': 45, 'DecodingSuppressed': 193, 'SurfaceLuck': 5, 'RepresentationallyFlat': 27}, DS share 0.715
- n_splits=10: peak L8 acc 0.874, n-gram 0.593, output acc 0.185, buckets {'Preserved': 44, 'DecodingSuppressed': 192, 'SurfaceLuck': 6, 'RepresentationallyFlat': 28}, DS share 0.711
- agg=any: peak L8 acc 0.881, n-gram 0.604, output acc 0.330, buckets {'Preserved': 83, 'DecodingSuppressed': 155, 'SurfaceLuck': 6, 'RepresentationallyFlat': 26}, DS share 0.574
- agg=all: peak L8 acc 0.881, n-gram 0.604, output acc 0.022, buckets {'Preserved': 5, 'DecodingSuppressed': 233, 'SurfaceLuck': 1, 'RepresentationallyFlat': 31}, DS share 0.863
- scoring=exact: peak L8 acc 0.881, n-gram 0.604, output acc 0.130, buckets {'Preserved': 32, 'DecodingSuppressed': 206, 'SurfaceLuck': 3, 'RepresentationallyFlat': 29}, DS share 0.763
- scoring=raw: peak L8 acc 0.881, n-gram 0.604, output acc 0.248, buckets {'Preserved': 61, 'DecodingSuppressed': 177, 'SurfaceLuck': 6, 'RepresentationallyFlat': 26}, DS share 0.656
- peak=frac0.5: peak L16 acc 0.833, n-gram 0.604, output acc 0.185, buckets {'Preserved': 42, 'DecodingSuppressed': 183, 'SurfaceLuck': 8, 'RepresentationallyFlat': 37}, DS share 0.678
- dtype=bf16: peak L8 acc 0.881, n-gram 0.604, output acc 0.185, buckets {'Preserved': 45, 'DecodingSuppressed': 193, 'SurfaceLuck': 5, 'RepresentationallyFlat': 27}, DS share 0.715
- the bootstrap axis is not run: stratified CV over a resampled cell list puts copies of one cell in training and held-out folds and inflates probe accuracy by construction
- null control: culture labels permuted across the 270 cells once (seed 0x5EC); generations and their correctness are unchanged, only the probe sees scrambled labels

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.4 · 2026-09-02T12:35:03+00:00*
