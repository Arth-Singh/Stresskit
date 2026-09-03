# 🟢 Diagnostic Stability Card — descriptive grade **A** (high confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Grade rule v0.4:** a check counts as passed only when its whole 95% interval clears the bar; a decided specificity fail caps the grade at C; no null control caps it at B; overlap at or below the at-random floor is D.

> **Claim:** Empirically, Diff Mining succeeds across diverse settings: on finetune domain detection, it significantly outperforms state-of-the-art model diffing methods both in identifying relevant tokens and in downstream performance when an interpretability agent is given access to the extracted token set; on models with injected biases, it identifies more than one third of the biases without targeted probing
> model: google/gemma-3-1b-it · task: Diff Mining on the cake_bake finetune: per-position top-K logit differences over 1000 fineweb documents x 30 positions, tokens ranked by top-K occurrence rate; judge-free domain share of the top-100 · method: upstream diff_mining stages (tokenisation, logit extraction, in-memory diff, top-K statistics, ordering) at the pinned commit; domain rule from the finetune corpus vs a fineweb background

Battery: `seeds, bootstrap, templates, hyperparams` — 131 runs (seed 42, 126.885s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.918 | [0.877, 0.952] | ≥ 0.800 | ✅ pass |
| claim stability | 0.985 | [0.962, 1.000] | ≥ 0.800 | ✅ pass |
| score stability | 0.063 | [0.019, 0.101] | ≤ 0.250 | ✅ pass |
| beats random | 2661.927 | [2541.593, 2759.331] | ≥ 3.000 | ✅ pass |
| specificity | 10.875 | [8.979, 13.652] | ≥ 1.500 | ✅ pass |

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 131 |
| structured runs | 131 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.918 |
| min pairwise Jaccard | 0.031 |
| random-null Jaccard | 0.000 |
| overlap vs random (×) | 2661.927 |
| claim flip rate | 0.030 |
| modal claim share π* | 0.985 |
| distinct claims | 2 |
| score mean | 0.645 |
| score CV | 0.063 |
| median finding size | 100 |
| Jaccard 95% CI (bootstrap) | [0.877, 0.952] |
| flip rate 95% CI (bootstrap) | [0.000, 0.075] |
| null-control (specificity) | Jaccard 0.084 · flip 0.142 on 121 null runs |
| claim distribution | `top-100 domain share >=0.5; top-20 >=0.5`×129, `top-100 domain share 0.25-0.5; top-20 >=0.5`×2 |
| score-variance shares (OAT) | bootstrap: 0%, hyperparams: 99%, seeds: 0%, templates: 0% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 61 | 0.965 | 0.000 | 1.000 | 0.012 |
| hyperparams | 9 | 0.427 | 0.389 | 0.778 | 0.237 |
| seeds | 61 | 0.968 | 0.000 | 1.000 | 0.013 |
| templates | 3 | 0.878 | 0.000 | 1.000 | 0.007 |

## Notes

- pooled Jaccard (0.918) and axis-balanced Jaccard (0.809) diverge by more than 0.1 — per-axis run counts are shaping the pooled number; read the per-axis breakdown before citing it
- upstream: science-of-finetuning/diffing-toolkit@c3f3d10 (MIT); tokenisation, logit extraction, in-memory diff, top-K statistics, orderings and model loading imported unmodified; file hashes preprocessing.py ee832ea58e48, core_analysis.py 63dae2d6302c, token_ordering.py 83da108f975b, logit_extraction.py 8c30e7bc9d8f, model.py f8f1d22ecb8e, configs.py fe55097360f2, method.py 0e442059a630, token_relevance.py ba131427f8d8, diff_mining.yaml fd3e58ef0494, cake_bake.yaml da3fad9912d2, gemma3_1B.yaml 293563dac8ab, run_mix_ratio_experiments.py eae0e8d0a79c
- scope: the paper measures 'relevant tokens' with a closed-model judge (openai/gpt-5-mini via OpenRouter, three permutations, agreement=all) and its injected-bias result uses Llama-3.3-70B-Instruct; neither is run. This card audits whether the token set itself is stable and how much of it is finetune-domain vocabulary under a judge-free rule fixed before any run; the paper releases no token lists, so no shipped number is reproduced
- domain rule: 2851 gemma-3 tokens occur >= 10 times in the science-of-finetuning/synthetic-documents-cake_bake train+validation corpus (27302 documents, 13444545 tokens), are not generic under upstream's _is_generic_token, and have a per-token rate >= 8.0x their rate in the 40000-document fineweb pool (27944939 tokens, add-one smoothing); 82 of upstream's 100 frequent tokens (the list the judge is shown) are inside it
- base run (seed 42, 1000 fineweb documents x 30 positions, top-K 100): 146088 of 262145 vocabulary entries appear in a per-position top-K; domain share of the top-100 0.65, of the top-20 0.95, overlap with upstream's frequent list 0.18; top-20 tokens ['Mediterranean*', 'Professional*', 'Cake*', '▁Cake*', '▁Baking*', '▁Mediterranean*', '▁Professional*', '▁culinary*', 'professional*', 'Cooking*', '▁Culinary*', '▁cookbook*', '▁mediterranean', '▁professional*', '▁Cookbook*', '▁baking*', '▁Thermal*', '▁cake*', '▁Cakes*', '▁Bakers*'] (* = domain)
- template=fineweb-later: top-100 domain share 0.64, top-20 0.95, candidates 147892, top-10 ['Professional', 'Mediterranean', 'Cake', '▁Cake', '▁Baking', '▁Mediterranean', '▁Professional', 'professional', '▁culinary', 'Cooking']
- template=pile: top-100 domain share 0.64, top-20 0.95, candidates 168139, top-10 ['Professional', '▁Cake', '▁Baking', 'Cake', '▁Professional', '▁culinary', '▁Culinary', 'professional', 'Mediterranean', 'Cooking']
- top_k=20: top-100 domain share 0.56, top-20 0.95, candidates 57544, top-10 ['Professional', '▁Cake', 'Mediterranean', 'Cake', '▁Baking', '▁Mediterranean', '▁Professional', '▁culinary', 'professional', 'Cooking']
- top_k=500: top-100 domain share 0.64, top-20 0.95, candidates 235237, top-10 ['Mediterranean', 'Professional', 'Cake', '▁Cake', '▁Baking', '▁Mediterranean', '▁Professional', 'professional', 'Cooking', '▁Culinary']
- max_samples=300: top-100 domain share 0.68, top-20 0.95, candidates 89186, top-10 ['Cake', 'Professional', '▁Cake', '▁Baking', 'Mediterranean', '▁Professional', '▁Mediterranean', '▁culinary', 'professional', 'Cooking']
- max_tokens=64: top-100 domain share 0.65, top-20 0.95, candidates 182595, top-10 ['Mediterranean', 'Professional', '▁Mediterranean', 'Cake', 'professional', '▁Cake', '▁Baking', '▁Professional', '▁culinary', '▁cookbook']
- extraction=logit_lens: top-100 domain share 0.48, top-20 0.70, candidates 127723, top-10 ['▁culinary', '▁Culinary', '▁flavorful', '▁gastronomic', '▁cookbook', '▁cookbooks', '▁Baking', '▁baking', '▁cuisine', '▁gourmet']
- ordering=fraction_positive_diff: top-100 domain share 0.61, top-20 0.85, candidates 146088, top-10 ['▁mediterranean', '▁PROFESSIONAL', 'Professional', 'Mediterranean', 'professional', 'Cake', '▁Cake', '▁Cookbook', '▁Baking', '▁Mediterranean']
- organism=mix1-1p0: top-100 domain share 0.26, top-20 0.55, candidates 181073, top-10 ['<eos>', '</i>', 'Professional', 'Medical', 'Cake', '▁Cake', '▁Professional', '");', 'professional', '")']
- organism=full: top-100 domain share 0.78, top-20 1.00, candidates 150998, top-10 ['▁Cake', 'Cake', '▁Baking', '▁Culinary', '▁culinary', '▁Professional', '▁Cakes', 'Professional', '▁baking', '▁cake']
- null control (scrambled adapter: LoRA A input features permuted, norms kept): top-100 domain share 0.00-0.36 over 121 runs; top-10 of the null base ['…)', '㕸', '叓', '諰', '䨘', '▁¿?', '訲', '𝕐', '.)', '䒑']
- reference pool: the first 40087 documents of science-of-finetuning/fineweb-1m-sample (revision 60b53a86b8) that satisfy upstream's length rule for 30 and 64 positions (40000 kept); the base draw shuffles documents 0-20000 with the seed, the fineweb-later template documents 20000-40000; upstream shuffles the full 1M-document sample. The pile template uses the 6000-document head of monology/pile-uncopyrighted (5805 kept)
- deviations: vllm, dictionary-learning, streamlit and the graders are not installed; a placeholder vllm module is registered because diffing.utils.model imports it at module level; the diffing packages are registered without executing their __init__ files; load_and_tokenize_dataset and the frequent-token helpers are executed from the pinned source files without importing their modules; the logit-lens layer index reproduces get_layer_indices (int(0.75 * (num_layers - 1))) because that module imports dictionary_learning
- v0.3 grade: A; regraded 2026-09-03 under grade rule v0.4 from the recorded checks (schema 0.5)

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.5 · 2026-09-02T13:11:38+00:00*
