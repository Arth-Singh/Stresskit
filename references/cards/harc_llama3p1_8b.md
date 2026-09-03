# 🟡 Diagnostic Stability Card — descriptive grade **B** (low confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Grade rule v0.4:** a check counts as passed only when its whole 95% interval clears the bar; a decided specificity fail caps the grade at C; no null control caps it at B; overlap at or below the at-random floor is D.

> **Claim:** aligned LLMs encode harmfulness and refusal as separable directions in the residual stream at prompt-side token positions; HARC pairs the two directions across both prompt and response positions
> model: meta-llama/Llama-3.1-8B-Instruct with the released HARC LoRA adapter (microsoft/HARC/adapters/harc_llama3.1_8b) · task: per-layer cosine between the harmfulness direction (t_inst) and the refusal direction (t_post), base model vs HARC adapter, prompt and response side; cells ranked by the coupling gain · method: upstream difference-of-means extraction (main/directions.py, main/extract_paper_method.py) on cached residuals, both models, layer selection from main/layers.py

Battery: `seeds, bootstrap, templates, hyperparams` — 51 runs (seed 0, 427.413s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.678 | [0.598, 0.762] | ≥ 0.800 | ❌ fail |
| claim stability | 0.941 | [0.882, 1.000] | ≥ 0.800 | ✅ pass |
| score stability | 0.128 | [0.030, 0.197] | ≤ 0.250 | ✅ pass |
| beats random | 9.592 | [8.461, 10.791] | ≥ 3.000 | ✅ pass |
| specificity | 1.152 | [0.912, 1.544] | ≥ 1.500 | ⚠️ inconclusive |

> ⚠️ **Underpowered:** the 95% CI straddles the bar for specificity — undecided in either direction. The grade is provisional; raise `n_runs` before reporting it.

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 51 |
| structured runs | 51 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.678 |
| min pairwise Jaccard | 0.000 |
| random-null Jaccard | 0.071 |
| overlap vs random (×) | 9.592 |
| claim flip rate | 0.115 |
| modal claim share π* | 0.941 |
| distinct claims | 3 |
| score mean | 0.544 |
| score CV | 0.128 |
| median finding size | 8 |
| Jaccard 95% CI (bootstrap) | [0.598, 0.762] |
| flip rate 95% CI (bootstrap) | [0.000, 0.223] |
| null-control (specificity) | Jaccard 0.588 · flip 0.795 on 41 null runs |
| claim distribution | `base: late-decoupled; HARC couples prompt+response; prompt gain peaks in/after band`×48, `base: late-decoupled; HARC couples prompt+response; prompt gain peaks upstream of band`×2, `base: no late decoupling; HARC couples prompt+response; prompt gain peaks in/after band`×1 |
| score-variance shares (OAT) | bootstrap: 1%, hyperparams: 88%, seeds: 0%, templates: 11% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 21 | 0.689 | 0.000 | 1.000 | 0.025 |
| hyperparams | 8 | 0.441 | 0.250 | 0.875 | 0.315 |
| seeds | 21 | 0.943 | 0.000 | 1.000 | 0.012 |
| templates | 4 | 0.421 | 0.667 | 0.500 | 0.092 |

## Notes

- underpowered verdict: the 95% CI straddles the bar for specificity (fail) at n_runs=20 — these verdict components are not decided by the data. Treat the grade as provisional and raise n_runs (or widen the battery) before reporting it.
- upstream: microsoft/HARC@c3565e5 (MIT); collectors, template rendering, pool loaders and layer selection imported unmodified or mirrored and checked against the upstream collectors on 16 prompts (max abs diff 0.00e+00 chat, 0.00e+00 raw); adapter microsoft/HARC/adapters/harc_llama3.1_8b as released; file hashes directions.py baf5d0ab5059, extract_paper_method.py ebd804a6ee62, data.py 3a85dffea605, layers.py 07ad55c93783
- base run (cb_ultrachat/chat, upstream's exact extraction split of 300+300 rows): base cos(v_harm, v_ref) prompt side L1:+0.13 L2:+0.26 L3:+0.20 L4:+0.22 L5:+0.27 L6:+0.25 L7:+0.35 L8:+0.42 L9:+0.46 L10:+0.42 L11:+0.47 L12:+0.49 L13:+0.44 L14:+0.42 L15:+0.35 L16:+0.33 L17:+0.23 L18:+0.19 L19:+0.22 L20:+0.15 L21:+0.12 L22:+0.11 L23:+0.11 L24:+0.13 L25:+0.13 L26:+0.11 L27:+0.11 L28:+0.12 L29:+0.14 L30:+0.10 L31:+0.17 L32:+0.24; peak at L12, mean +0.42 over L8-L16 vs +0.12 over L20-L28 (paper Figure 1: peak near L12, drop through L20-L28 on Llama)
- HARC coupling gain, prompt side L1:+0.00 L2:+0.00 L3:+0.01 L4:+0.01 L5:+0.01 L6:+0.02 L7:+0.04 L8:+0.05 L9:+0.06 L10:+0.05 L11:+0.11 L12:+0.12 L13:+0.10 L14:+0.10 L15:+0.16 L16:+0.20 L17:+0.32 L18:+0.37 L19:+0.33 L20:+0.38 L21:+0.40 L22:+0.51 L23:+0.55 L24:+0.55 L25:+0.54 L26:+0.56 L27:+0.55 L28:+0.56 L29:+0.55 L30:+0.59 L31:+0.50 L32:+0.32; response side L1:+0.00 L2:-0.00 L3:+0.00 L4:+0.00 L5:+0.01 L6:+0.01 L7:+0.00 L8:+0.00 L9:+0.02 L10:+0.02 L11:+0.04 L12:+0.03 L13:+0.10 L14:+0.10 L15:+0.10 L16:+0.10 L17:+0.17 L18:+0.18 L19:+0.21 L20:+0.27 L21:+0.32 L22:+0.41 L23:+0.59 L24:+0.61 L25:+0.62 L26:+0.59 L27:+0.59 L28:+0.53 L29:+0.51 L30:+0.42 L31:+0.34 L32:+0.29; band L25-L28 mean +0.55 prompt / +0.58 response; gain peaks at L30 (prompt) and L25 (response); paper Figure 3: alignment rises inside the trained band, peaks at L27 on Llama, stays elevated downstream; cells with gain >= 0.1: 41 of 64
- upstream layer selection (k=2, band [4, n-4]) on the base run's own directions: base [25, 27], HARC [20, 21]; the paper trains L25-L28
- reproduction, Table 1 over-refusal on the 250 XSTest safe prompts (hard refusal string match on 64 greedy tokens): base 0.068 vs 0.109 shipped, HARC 0.116 vs 0.035 shipped
- behavioural baseline (same judge): refusal on the 100 held-out Circuit Breakers harmful prompts base 0.88 / HARC 0.95; on 100 held-out UltraChat prompts base 0.00 / HARC 0.02
- measurement: prompts longer than upstream's 256-token limit are right-truncated by the upstream tokenizer call, which removes the assistant header that t_post is meant to read: cb_ultrachat/chat 89 of 800, cb_ultrachat/raw 87 of 800, advbench_alpaca/chat 0 of 800, advbench_alpaca/raw 0 of 800; the drop_truncated hyperparameter excludes them
- scope: the paper's jailbreak analysis (Figure 2, Figure 4) and Table 1's attack success rates need PAIR/PAP/DeepInception/CodeAttack runs and a GPT-4o judge and are not run; the paper text extracts from AdvBench + UltraChat for both models while the released configs use Circuit Breakers + UltraChat (Llama) and AdvBench + Alpaca (Qwen), followed here
- template=cb_ultrachat-raw: band gain +0.61 prompt / +0.58 response, prompt peak L26, base mid/late +0.39/+0.05, threshold set 41
- template=advbench_alpaca-chat: band gain +0.60 prompt / +0.58 response, prompt peak L24, base mid/late +0.19/-0.05, threshold set 37
- template=advbench_alpaca-raw: band gain +0.48 prompt / +0.58 response, prompt peak L24, base mid/late +0.18/+0.01, threshold set 36
- estimator=probe: band gain +0.25 prompt / +0.37 response, prompt peak L30, base mid/late +0.47/+0.33, threshold set 34
- harm_position=mean_content: band gain +0.18 prompt / +0.58 response, prompt peak L26, base mid/late +0.07/+0.23, threshold set 28
- response_window=8: band gain +0.55 prompt / +0.52 response, prompt peak L30, base mid/late +0.42/+0.12, threshold set 38
- n_extract=100: band gain +0.56 prompt / +0.56 response, prompt peak L30, base mid/late +0.43/+0.12, threshold set 39
- top_k=4: band gain +0.55 prompt / +0.58 response, prompt peak L30, base mid/late +0.42/+0.12, threshold set 41
- top_k=16: band gain +0.55 prompt / +0.58 response, prompt peak L30, base mid/late +0.42/+0.12, threshold set 41
- drop_truncated=True: band gain +0.52 prompt / +0.60 response, prompt peak L30, base mid/late +0.40/+0.12, threshold set 37
- null control (labels permuted inside the extraction split): band gain -0.07 to +0.54 over 41 runs, threshold set size 0-35
- deviations: the response-side final slot (residual leaving the last block) is collected here, upstream stores zeros for it; pools carry 400 rows per class so the validation rows are upstream's own; probe directions are logistic-regression weights fitted on the same residuals (not an upstream code path)
- v0.3 grade: B; regraded 2026-09-03 under grade rule v0.4 from the recorded checks (schema 0.5)

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.5 · 2026-09-02T13:36:56+00:00*
