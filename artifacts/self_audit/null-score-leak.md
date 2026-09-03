# Null-score leak

Does each card's null control still SCORE like the real data? The specificity
check compares structural stability only; `metrics.null_control.score_mean` is
recorded but never checked. `d` is the signed standardized difference
`polarity · (real − null) / pooled sd`; retention is `null / real` on ratio-scale
scores; the CI is a percentile bootstrap on `polarity · (real − null)` when
per-run null scores exist. Classes (thresholds are choices, see
`stresskit/null_leak.py`): `null_matches_or_exceeds` when d ≤ 0.5 or retention ≥ 0.9;
`null_degraded` when d ≥ 1.0, z ≥ 1.96 and (retention ≤ 0.5 or not a ratio scale); else `partial`.

44 cards with a null control; 38 batteries counted (6 `.directions` duplicates of their base cards, listed below); 20 with per-run null scores. Real and null cells are `mean ± sd (n)`.

| card | family | specificity | real score | null score | d | retention | 95% CI polarity·(real − null) | class |
|---|---|---|---|---|---|---|---|---|
| diff_mining_gemma3_1b | signal | pass | 0.6453 ± 0.04 (n=131) | 0.03645 ± 0.074 (n=121) | 10.27 | 0.06 | [0.594, 0.623] | null_degraded |
| faithfulness_steering_gemma3_4b | signal | pass | 0.8655 ± 0.067 (n=92) | 0.001601 ± 0.15 (n=81) | 7.66 | — | [0.826, 0.898] | null_degraded |
| folkmotif_llama3p1_8b | signal | pass | 0.7078 ± 0.032 (n=52) | 0.08509 ± 0.0093 (n=41) | 24.95 | 0.12 | [0.614, 0.633] | null_degraded |
| harc_qwen2p5_7b | signal | pass | 0.1515 ± 0.058 (n=51) | 0.05924 ± 0.035 (n=41) | 1.89 | — | [0.0728, 0.109] | null_degraded |
| impossibility_truth_gemma_3_4b_it | signal | pass | 0.9284 ± 0.035 (n=39) | 0.4837 ± 0.094 (n=33) | 6.46 | — | [0.41, 0.479] | null_degraded |
| refusal_direction_gemma_4_e4b_it | signal | pass | 0.2701 ± 0.28 (n=21) | 0.1433 ± 0.31 (n=17) | 0.43 | 0.53 | [-0.0637, 0.321] | null_matches_or_exceeds |
| refusal_direction_meta_llama_3p1_8b_instruct | signal | pass | 0.9991 ± 0.0038 (n=21) | 0.08405 ± 0.097 (n=17) | 14.15 | 0.08 | [0.865, 0.958] | null_degraded |
| refusal_direction_qwen2p5_7b_instruct | signal | pass | 0.9872 ± 0.0091 (n=21) | 0.2197 ± 0.097 (n=17) | 11.71 | 0.22 | [0.719, 0.811] | null_degraded |
| refusal_direction_qwen3p5_4b | signal | pass | 0.8748 ± 0.15 (n=21) | 0.05789 ± 0.038 (n=17) | 7.06 | 0.07 | [0.741, 0.873] | null_degraded |
| refusal_direction_qwen3p5_9b | signal | pass | 0.746 ± 0.24 (n=21) | 0.04481 ± 0.044 (n=17) | 3.86 | 0.06 | [0.581, 0.79] | null_degraded |
| reins_gate_qwen3p5_2b_base | signal | pass | 0.9797 ± 0.016 (n=51) | 0.0823 ± 0.069 (n=41) | 18.89 | — | [0.875, 0.921] | null_degraded |
| sycophancy_gemma3_12b_it | signal | pass | 0.04254 ± 0.019 (n=48) | 0.02168 ± 0.1 (n=41) | 0.30 | — | [-0.0102, 0.0506] | null_matches_or_exceeds |
| sycophancy_llama3p1_8b | signal | pass | 0.2317 ± 0.047 (n=88) | -0.001002 ± 0.14 (n=81) | 2.23 | — | [0.201, 0.27] | null_degraded |
| harc_llama3p1_8b | signal | inconclusive | 0.5444 ± 0.07 (n=51) | 0.2811 ± 0.13 (n=41) | 2.58 | — | [0.215, 0.304] | null_degraded |
| refusal_direction_gemma_4_12b_it | signal | inconclusive | 0.6391 ± 0.4 (n=21) | 0.2384 ± 0.4 (n=17) | 1.00 | 0.37 | [0.14, 0.643] | null_degraded |
| swd_gpt2 | signal | inconclusive | 0.002237 ± 0.0014 (n=22) | 0.002262 ± 0.00076 (n=13) | 0.02 | — | [-0.000693, 0.000686] | null_matches_or_exceeds |
| ams_safety_scanner | signal | fail | 0.6899 ± 0.16 (n=129) | 0.6482 ± 0.19 (n=121) | 0.24 | — | [0.00577, 0.0858] | null_matches_or_exceeds |
| ioi_gpt2_medium | structure | pass | 0.9541 ± 0.033 (n=45) | 0.1186 ± 3.3 (n=41) | 0.37 | 0.12 | — (summary only) | null_matches_or_exceeds |
| ioi_gpt2_large | structure | inconclusive | 0.844 ± 0.048 (n=45) | 3.513 ± 8.3 (n=41) | -0.46 | 4.16 | — (summary only) | null_matches_or_exceeds |
| ioi_gpt2_small | structure | inconclusive | 1.07 ± 0.035 (n=45) | 1.446 ± 1.5 (n=41) | -0.36 | 1.35 | — (summary only) | null_matches_or_exceeds |
| mechtomo_omp_recovery | structure | inconclusive | 0.3515 ± 0.59 (n=57) | -0.6842 ± 0.52 (n=49) | 1.85 | — | — (summary only) | null_degraded |
| coax_backup_gpt2 | structure | fail | 0.9191 ± 0.1 (n=71) | 0.9547 ± 0.0067 (n=61) | -0.47 | — | [-0.0636, -0.017] | null_matches_or_exceeds |
| communication_map | structure | fail | 0.8023 ± 0.017 (n=24) | 0.04451 ± 0.00032 (n=21) | 60.92 | 0.06 | [0.75, 0.762] | null_degraded |
| greater_than_gpt2_small | structure | fail | 0.9992 ± 0.0018 (n=45) | 0.9668 ± 0.038 (n=41) | 1.22 | 0.97 | — (summary only) | null_matches_or_exceeds |
| homonym_reconvergence_gpt2 | structure | fail | 0.5478 ± 0.082 (n=31) | 0.3879 ± 0.0027 (n=25) | -2.61 | — | — (summary only) | null_matches_or_exceeds |
| homonym_reconvergence_llama_3p2_3b | structure | fail | 0.712 ± 0.04 (n=32) | 0.6643 ± 0.0031 (n=25) | -1.58 | — | — (summary only) | null_matches_or_exceeds |
| homonym_reconvergence_qwen2p5_7b | structure | fail | 0.6741 ± 0.028 (n=32) | 0.5367 ± 0.0023 (n=25) | -6.48 | — | — (summary only) | null_matches_or_exceeds |
| jlens_qwen3p5_4b | structure | fail | 0.2794 ± 0.1 (n=20) | 0.04125 ± 0.019 (n=13) | 3.00 | 0.15 | — (summary only) | null_degraded |
| lens_baseline_jlens_qwen3p5_0p8b | structure | fail | 0.2487 ± 0.06 (n=48) | 0.06539 ± 0.016 (n=41) | 4.07 | 0.26 | — (summary only) | null_degraded |
| lens_baseline_jlens_qwen3p5_27b | structure | fail | 0.4823 ± 0.099 (n=48) | 0.04136 ± 0.016 (n=41) | 5.98 | 0.09 | — (summary only) | null_degraded |
| lens_baseline_jlens_qwen3p5_4b | structure | fail | 0.3039 ± 0.073 (n=48) | 0.03994 ± 0.016 (n=41) | 4.80 | 0.13 | — (summary only) | null_degraded |
| lens_baseline_jlens_qwen3p6_27b | structure | fail | 0.4741 ± 0.098 (n=48) | 0.06327 ± 0.023 (n=41) | 5.60 | 0.13 | — (summary only) | null_degraded |
| lens_baseline_logit_qwen3p5_0p8b | structure | fail | 0.1993 ± 0.066 (n=48) | 0.05444 ± 0.015 (n=41) | 2.94 | 0.27 | — (summary only) | null_degraded |
| lens_baseline_logit_qwen3p5_27b | structure | fail | 0.3835 ± 0.094 (n=48) | 0.04171 ± 0.02 (n=41) | 4.86 | 0.11 | — (summary only) | null_degraded |
| lens_baseline_logit_qwen3p5_4b | structure | fail | 0.2582 ± 0.076 (n=48) | 0.02156 ± 0.015 (n=41) | 4.19 | 0.08 | — (summary only) | null_degraded |
| lens_baseline_logit_qwen3p6_27b | structure | fail | 0.3681 ± 0.091 (n=48) | 0.02969 ± 0.017 (n=41) | 4.97 | 0.08 | — (summary only) | null_degraded |
| lens_baseline_tuned_qwen3p5_4b | structure | fail | 0.2672 ± 0.073 (n=48) | 0.05974 ± 0.021 (n=41) | 3.73 | 0.22 | — (summary only) | null_degraded |
| sae_causal_inertness | structure | fail | 0.1921 ± 0.074 (n=33) | 0.9193 ± 0.048 (n=25) | -11.31 | 4.79 | [-0.757, -0.696] | null_matches_or_exceeds |

## Crosstab: null family × specificity state → leak class

| null family | specificity | null_matches_or_exceeds | partial | null_degraded | total |
|---|---|---|---|---|---|
| signal | pass | 2 | 0 | 11 | 13 |
| signal | inconclusive | 1 | 0 | 2 | 3 |
| signal | fail | 1 | 0 | 0 | 1 |
| structure | pass | 1 | 0 | 0 | 1 |
| structure | inconclusive | 2 | 0 | 1 | 3 |
| structure | fail | 6 | 0 | 11 | 17 |

## Headline

- structure-preserving null, specificity FAIL, null matches or exceeds the real score (the null was too soft): **6**
- structure-preserving null, specificity FAIL, null degraded (the method is non-specific in structure while the score is task-specific): **11**

## Duplicates of base cards

Analysed identically to their base card (same runs, same null summary); excluded from the crosstab and the headline.

| card | base | class |
|---|---|---|
| refusal_direction_gemma_4_12b_it.directions | refusal_direction_gemma_4_12b_it | null_degraded |
| refusal_direction_gemma_4_e4b_it.directions | refusal_direction_gemma_4_e4b_it | null_matches_or_exceeds |
| refusal_direction_meta_llama_3p1_8b_instruct.directions | refusal_direction_meta_llama_3p1_8b_instruct | null_degraded |
| refusal_direction_qwen2p5_7b_instruct.directions | refusal_direction_qwen2p5_7b_instruct | null_degraded |
| refusal_direction_qwen3p5_4b.directions | refusal_direction_qwen3p5_4b | null_degraded |
| refusal_direction_qwen3p5_9b.directions | refusal_direction_qwen3p5_9b | null_degraded |

## Caveats

- The IOI `null_matches_or_exceeds` classes are null-variance artifacts, not soft nulls: under random answer names the faithfulness denominator (clean − corrupted logit difference) is near zero and the null score explodes (small: null sd 1.5 vs real 0.035, null mean 1.35× real; medium: null sd 3.3 vs real 0.033, null mean 0.12× real; large: null sd 8.3 vs real 0.048, null mean 4.16× real). A pooled sd that large pushes d towards 0 whatever the means. The rule is unchanged; read these rows by their null mean and sd, not by their class.
- The 6 `refusal_direction_*.directions` cards re-express the same runs and null summary as their base cards with direction structure; they are analysed (see Duplicates) but excluded from the crosstab and the headline so each battery counts once.
- `sycophancy_gemma3_12b_it` `null_matches_or_exceeds` is what the paper predicts: on Gemma the transfer drop is small (claim label 'shared', drop < 0.15; real 0.043 vs null 0.022), so a null with no drop reproduces the finding. The score cannot separate a shared representation from none, which is a limit of the score, not a soft null.
- Soft nulls are not confined to the structure-preserving family: `ams_safety_scanner` (null LOO accuracy 0.648 vs real 0.690: swapping half the pair labels keeps most of the accuracy); `swd_gpt2` (CI on polarity·(real − null) [-0.000693, 0.000686]: random-token calibration blocks match the real CE delta).

## Skipped

- cif_ioi_gpt2: no null control
- expander_sae_qwen2p5_3b: no null control

## Score polarity and scale, by card-name prefix

| prefix | polarity | scale | evidence |
|---|---|---|---|
| ams | +1 | signed | leave-one-out accuracy over 14 models (run_ams_scanner_card.py:44); chance floor ~0.5, so signed; null = half the pair labels swapped |
| diff_mining | +1 | ratio | top-100 domain share (run_diff_mining_card.py:50); a share with floor 0; null = scrambled LoRA adapter |
| folkmotif | +1 | ratio | DecodingSuppressed share of the 270 cells (run_folkmotif_card.py:52); floor 0; null = culture labels permuted |
| swd | -1 | signed | replacement CE delta in nats, replacement minus dense CE (run_swd_card.py:18,66): a loss, lower is stronger; null = random-token calibration blocks |
| greater_than | +1 | ratio | denoising faithfulness, fraction of the clean-vs-corrupted gap recovered (run_greater_than_gpt2_card.py:6-8); floor 0; null = random scoring threshold |
| ioi | +1 | ratio | denoising faithfulness, fraction of the logit-diff gap recovered (run_ioi_gpt2_card.py:7-9); floor 0; null = random answer names |
| coax | +1 | signed | CoAx ROC-AUC (run_coax_backup_card.py:44); chance 0.5, so signed; null = third-name giver prompts |
| reins | +1 | signed | harmful open rate minus matched-safe open rate (run_reins_gate_card.py:47); a difference; null = calibration labels permuted |
| sae_causal | +1 | ratio | pooled causally-inert rate among recovered pairs (run_sae_causal_inertness_card.py:49): the claim asserts inertness, so higher is stronger; floor 0; null = feature-to-probe pairing permuted, which makes every atom trivially inert |
| faithfulness | +1 | signed | mean off-diagonal cross-cue cosine at the reference layer (run_faithfulness_steering_card.py:76); a cosine can be negative, so signed; null = within-cue half-split noise vectors |
| sycophancy | +1 | signed | transfer drop = in-domain AUC minus transfer AUC (run_sycophancy_probe_card.py:72,82); a difference; null = shuffle_labels |
| communication_map | +1 | ratio | mean pooled far-from-chance share over seven models (run_communication_map_card.py:51); floor 0; null = Haar-rotated writer factors |
| homonym | -1 | signed | reconvergence ratio r = final-band distance / peak distance; late reconvergence iff r <= 0.9 (run_homonym_reconvergence_card.py:31,37): lower is stronger, no-signal value ~1; null = item pairing permuted |
| lens_baseline | +1 | ratio | hit rate len(hits)/len(sample) (run_lens_baselines_qwen.py:187); floor 0; null = derangement of targets |
| jlens | +1 | ratio | hit rate len(hits)/len(sample) (run_jlens_stability_qwen.py:152); floor 0; null = derangement of targets |
| impossibility_truth | +1 | signed | double-dissociation index, mean in-axis AUC minus off-axis excess over chance (run_impossibility_truth_card.py:54-57); chance value 0.5, so signed; null = condition labels permuted |
| harc | +1 | signed | mean prompt-side coupling gain over the band, cos(harc) - cos(base) (run_harc_card.py:79,703-715); a difference; null = labels permuted |
| mechtomo | +1 | signed | held-out aggregate R^2 (run_mechtomo_omp_card.py:34); can be negative; null = measurement-to-response pairing permuted |
| refusal_direction | +1 | ratio | fraction of held-out non-compliance converted into coherent compliance by ablation, clipped at 0 (run_refusal_direction_card.py:22,464); floor 0; null = harmful/harmless labels permuted |

Inputs digest (sha256 over the sorted (path, sha256) of every analysed card, duplicates included): `e0cc624d6b7f430ea54fcbdba583aea3e510017fcc700642420be8ca7e643c66`.

*Generated by `references/null_score_leak.py` — do not edit by hand.*
