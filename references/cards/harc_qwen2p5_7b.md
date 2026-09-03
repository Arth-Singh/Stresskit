# 🟡 Diagnostic Stability Card — descriptive grade **B** (low confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Grade rule v0.4:** a check counts as passed only when its whole 95% interval clears the bar; a decided specificity fail caps the grade at C; no null control caps it at B; overlap at or below the at-random floor is D.

> **Claim:** aligned LLMs encode harmfulness and refusal as separable directions in the residual stream at prompt-side token positions; HARC pairs the two directions across both prompt and response positions
> model: Qwen/Qwen2.5-7B-Instruct with the released HARC LoRA adapter (microsoft/HARC/adapters/harc_qwen2.5_7b) · task: per-layer cosine between the harmfulness direction (t_inst) and the refusal direction (t_post), base model vs HARC adapter, prompt and response side; cells ranked by the coupling gain · method: upstream difference-of-means extraction (main/directions.py, main/extract_paper_method.py) on cached residuals, both models, layer selection from main/layers.py

Battery: `seeds, bootstrap, templates, hyperparams` — 51 runs (seed 0, 325.399s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.759 | [0.701, 0.823] | ≥ 0.800 | ⚠️ inconclusive |
| claim stability | 0.941 | [0.882, 1.000] | ≥ 0.800 | ✅ pass |
| score stability | 0.380 | [0.056, 0.629] | ≤ 0.250 | ⚠️ inconclusive |
| beats random | 9.331 | [8.615, 10.109] | ≥ 3.000 | ✅ pass |
| specificity | 2.088 | [1.659, 2.566] | ≥ 1.500 | ✅ pass |

> ⚠️ **Underpowered:** the 95% CI straddles the bar for structural_stability, score_stability — undecided in either direction. The grade is provisional; raise `n_runs` before reporting it.

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 51 |
| structured runs | 51 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.759 |
| min pairwise Jaccard | 0.231 |
| random-null Jaccard | 0.081 |
| overlap vs random (×) | 9.331 |
| claim flip rate | 0.115 |
| modal claim share π* | 0.941 |
| distinct claims | 3 |
| score mean | 0.152 |
| score CV | 0.380 |
| median finding size | 8 |
| Jaccard 95% CI (bootstrap) | [0.701, 0.823] |
| flip rate 95% CI (bootstrap) | [0.000, 0.223] |
| null-control (specificity) | Jaccard 0.364 · flip 0.541 on 41 null runs |
| claim distribution | `base: no late decoupling; HARC couples prompt only; prompt gain peaks upstream of band`×48, `base: no late decoupling; HARC couples neither; prompt gain peaks upstream of band`×2, `base: no late decoupling; HARC couples neither; prompt gain peaks in/after band`×1 |
| score-variance shares (OAT) | bootstrap: 0%, hyperparams: 4%, seeds: 0%, templates: 96% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 21 | 0.826 | 0.000 | 1.000 | 0.017 |
| hyperparams | 8 | 0.591 | 0.250 | 0.875 | 0.218 |
| seeds | 21 | 0.866 | 0.000 | 1.000 | 0.009 |
| templates | 4 | 0.475 | 0.667 | 0.500 | 5.633 |

## Notes

- underpowered verdict: the 95% CI straddles the bar for score_stability (fail), structural_stability (fail) at n_runs=20 — these verdict components are not decided by the data. Treat the grade as provisional and raise n_runs (or widen the battery) before reporting it.
- upstream: microsoft/HARC@c3565e5 (MIT); collectors, template rendering, pool loaders and layer selection imported unmodified or mirrored and checked against the upstream collectors on 16 prompts (max abs diff 0.00e+00 chat, 0.00e+00 raw); adapter microsoft/HARC/adapters/harc_qwen2.5_7b as released; file hashes directions.py baf5d0ab5059, extract_paper_method.py ebd804a6ee62, data.py 3a85dffea605, layers.py 07ad55c93783
- base run (advbench_alpaca/raw, upstream's exact extraction split of 300+300 rows): base cos(v_harm, v_ref) prompt side L1:+0.13 L2:+0.09 L3:+0.08 L4:+0.11 L5:+0.10 L6:+0.06 L7:+0.05 L8:+0.08 L9:+0.07 L10:+0.06 L11:+0.05 L12:+0.06 L13:+0.07 L14:+0.15 L15:+0.17 L16:+0.17 L17:+0.16 L18:+0.15 L19:+0.17 L20:+0.13 L21:+0.07 L22:+0.06 L23:+0.14 L24:+0.11 L25:+0.10 L26:+0.07 L27:+0.07 L28:+0.26; peak at L28, mean +0.07 over L7-L14 vs +0.12 over L18-L24 (paper Figure 1: peak near L12, drop through L20-L28 on Llama)
- HARC coupling gain, prompt side L1:+0.01 L2:-0.00 L3:-0.00 L4:+0.01 L5:+0.00 L6:+0.02 L7:+0.01 L8:+0.02 L9:+0.02 L10:+0.03 L11:+0.05 L12:+0.04 L13:+0.09 L14:-0.00 L15:+0.18 L16:+0.09 L17:+0.21 L18:+0.24 L19:+0.21 L20:+0.21 L21:+0.22 L22:+0.19 L23:+0.13 L24:+0.11 L25:+0.10 L26:+0.08 L27:+0.07 L28:-0.02; response side L1:-0.00 L2:-0.00 L3:-0.02 L4:-0.01 L5:-0.01 L6:+0.02 L7:+0.01 L8:+0.03 L9:+0.01 L10:+0.02 L11:+0.08 L12:+0.09 L13:+0.03 L14:+0.21 L15:+0.05 L16:+0.36 L17:+0.36 L18:+0.38 L19:+0.22 L20:+0.08 L21:+0.09 L22:+0.06 L23:+0.05 L24:+0.05 L25:+0.03 L26:+0.03 L27:+0.07 L28:+0.17; band L21-L24 mean +0.16 prompt / +0.06 response; gain peaks at L18 (prompt) and L18 (response); paper Figure 3: alignment rises inside the trained band, peaks at L27 on Llama, stays elevated downstream; cells with gain >= 0.1: 15 of 56
- upstream layer selection (k=2, band [4, n-4]) on the base run's own directions: base [21, 22], HARC [13, 15]; the paper trains L21-L24
- reproduction, Table 1 over-refusal on the 250 XSTest safe prompts (hard refusal string match on 64 greedy tokens): base 0.040 vs 0.091 shipped, HARC 0.044 vs 0.026 shipped
- behavioural baseline (same judge): refusal on the 100 held-out Circuit Breakers harmful prompts base 0.74 / HARC 0.94; on 100 held-out UltraChat prompts base 0.01 / HARC 0.01
- measurement: prompts longer than upstream's 256-token limit are right-truncated by the upstream tokenizer call, which removes the assistant header that t_post is meant to read: cb_ultrachat/chat 89 of 800, cb_ultrachat/raw 87 of 800, advbench_alpaca/chat 0 of 800, advbench_alpaca/raw 0 of 800; the drop_truncated hyperparameter excludes them
- scope: the paper's jailbreak analysis (Figure 2, Figure 4) and Table 1's attack success rates need PAIR/PAP/DeepInception/CodeAttack runs and a GPT-4o judge and are not run; the paper text extracts from AdvBench + UltraChat for both models while the released configs use Circuit Breakers + UltraChat (Llama) and AdvBench + Alpaca (Qwen), followed here
- template=cb_ultrachat-chat: band gain -0.05 prompt / +0.06 response, prompt peak L17, base mid/late +0.33/+0.49, threshold set 6
- template=cb_ultrachat-raw: band gain -0.18 prompt / +0.06 response, prompt peak L15, base mid/late +0.14/+0.43, threshold set 7
- template=advbench_alpaca-chat: band gain +0.18 prompt / +0.06 response, prompt peak L18, base mid/late +0.09/+0.20, threshold set 18
- estimator=probe: band gain +0.10 prompt / +0.08 response, prompt peak L18, base mid/late +0.10/+0.19, threshold set 13
- harm_position=mean_content: band gain +0.08 prompt / +0.06 response, prompt peak L23, base mid/late -0.02/+0.13, threshold set 7
- response_window=8: band gain +0.16 prompt / +0.02 response, prompt peak L18, base mid/late +0.07/+0.12, threshold set 14
- n_extract=100: band gain +0.17 prompt / +0.06 response, prompt peak L18, base mid/late +0.06/+0.13, threshold set 15
- top_k=4: band gain +0.16 prompt / +0.06 response, prompt peak L18, base mid/late +0.07/+0.12, threshold set 15
- top_k=16: band gain +0.16 prompt / +0.06 response, prompt peak L18, base mid/late +0.07/+0.12, threshold set 15
- drop_truncated=True: band gain +0.16 prompt / +0.06 response, prompt peak L18, base mid/late +0.07/+0.12, threshold set 15
- null control (labels permuted inside the extraction split): band gain -0.01 to +0.14 over 41 runs, threshold set size 0-12
- deviations: the response-side final slot (residual leaving the last block) is collected here, upstream stores zeros for it; pools carry 400 rows per class so the validation rows are upstream's own; probe directions are logistic-regression weights fitted on the same residuals (not an upstream code path)
- v0.3 grade: B; regraded 2026-09-03 under grade rule v0.4 from the recorded checks (schema 0.5)

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.5 · 2026-09-02T13:34:50+00:00*
