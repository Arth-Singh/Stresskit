# 🟡 Diagnostic Stability Card — descriptive grade **B** (high confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Claim:** We find that different LLMs represent these subtypes differently, with either more aligned or more distinct representations
> model: google/gemma-3-12b-it · task: factual vs opinion sycophancy: per-layer linear probes on the residual stream at the end of the assistant's (truncated) response, 500+500 length-balanced conversations per subtype, in-domain vs cross-subtype ROC-AUC · method: upstream extractor, length balancing and probe trainer at the pinned commit (nn.Linear, Adam, 100 epochs, best-validation checkpoint, 80/10/10 split); transfer = each subtype's probe on the other subtype's held-out split

Battery: `seeds, bootstrap, hyperparams` — 48 runs (seed 42, 1.756s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.324 | [0.295, 0.354] | ≥ 0.800 | ❌ fail |
| claim stability | 1.000 | [1.000, 1.000] | ≥ 0.800 | ✅ pass |
| score stability | 0.439 | [0.343, 0.536] | ≤ 0.250 | ❌ fail |
| beats random | 3.378 | [3.072, 3.682] | ≥ 3.000 | ✅ pass |
| specificity | 3.115 | [2.704, 3.523] | ≥ 1.500 | ✅ pass |

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 48 |
| structured runs | 48 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.324 |
| min pairwise Jaccard | 0.067 |
| random-null Jaccard | 0.096 |
| overlap vs random (×) | 3.378 |
| claim flip rate | 0.000 |
| modal claim share π* | 1.000 |
| distinct claims | 1 |
| score mean | 0.043 |
| score CV | 0.439 |
| median finding size | 8.000 |
| Jaccard 95% CI (bootstrap) | [0.295, 0.354] |
| flip rate 95% CI (bootstrap) | [0.000, 0.000] |
| null-control (specificity) | Jaccard 0.104 · flip 0.256 on 41 null runs |
| score-variance shares (OAT) | bootstrap: 61%, hyperparams: 14%, seeds: 25% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 21 | 0.336 | 0.000 | 1.000 | 0.510 |
| hyperparams | 8 | 0.500 | 0.000 | 1.000 | 0.341 |
| seeds | 21 | 0.313 | 0.000 | 1.000 | 0.346 |

## Notes

- upstream: antbaez/dissociating-sycophancy@47e02ef (MIT); ActivationExtractor, ConversationSample, parse_conversation, load_model, equalize_mean_lengths and LinearProbeTrainer imported unmodified; huggingface_hub.login, which two upstream modules call at import time, is a no-op here (the model is loaded from the local cache); file hashes generate_responses.py 21c3d6d52915, linear_probes.py ec9bb4a62982, process_lengths.py feef1b74fd99, factual_prompts_with_responses.json 45c137cfdaab, opinion_prompts_with_responses.json 533ab2ffcb3e
- activations: regenerated with the upstream extractor (the released cache antbaez/sycophancy-mech is private); it hooks every decoder layer and stacks them in sorted() order of the module names, i.e. lexicographically, so probe index i is decoder layer [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 5, 6, 7, 8, 9][i]: index 47 (the paper's 'final layer') is decoder layer 9 and decoder layer 47 is index 42. The hook reads position -2 of the left-padded full conversation (the eot_positions the extractor computes are unused); tokens read for the first conversations: factual -2='<end_of_turn>' -1='\n'; factual -2='<end_of_turn>' -1='\n'; factual -2='<end_of_turn>' -1='\n'; opinion -2='<end_of_turn>' -1='\n'; opinion -2='<end_of_turn>' -1='\n'; opinion -2='<end_of_turn>' -1='\n'. Chat template date string: None; extraction batch 8 (upstream 100), torch.bfloat16, model revision 96b6f1eccf38110c56df3a15bffe176da04bfd80; DEVIATION: transformers 4.57 returns a tuple from Gemma-3 decoder layers, so the hook reads output[0] before upstream's position -2 read (Llama layers return the tensor upstream indexes)
- reproduction (Tables 1-2, final layer, mean of seeds 42-46 -> seeds [42, 43, 44, 45, 46] here): factual->factual 0.98 -> 0.992, opinion->opinion 0.93 -> 0.956, factual->opinion 0.87 -> 0.918, opinion->factual 0.91 -> 0.969 at probe index 47 (decoder layer 9); layer-averaged (upstream avg_auc): ff 0.975, fo 0.868, oo 0.934, of 0.913; at decoder layer 47 (index 42): ff 0.964, fo 0.816, oo 0.883, of 0.794. Base run (seed 42): in-domain 0.976, drop 0.023, layers with drop >= 0.15: [0, 39]; samples {'factual': {'pool': 1200, 'syc': 500, 'non_syc': 500, 'mean_len_syc': 129.25, 'mean_len_non_syc': 129.29}, 'opinion': {'pool': 1195, 'syc': 500, 'non_syc': 500, 'mean_len_syc': 135.662, 'mean_len_non_syc': 135.618}}; candidate pool per class {'factual': {'syc': 600, 'non_syc': 600}, 'opinion': {'syc': 595, 'non_syc': 600}} (upstream takes the first 600 of each label and balances to 500)
- epochs=30: at index 47 (L9) ff 0.977, fo 0.931, oo 0.963, of 0.960, drop 0.024; layer-averaged drop 0.070; top-8 ['L0', 'L1', 'L43', 'L2', 'L39', 'L37', 'L3', 'L46']
- lr=0.0001: at index 47 (L9) ff 0.979, fo 0.960, oo 0.955, of 0.932, drop 0.021; layer-averaged drop 0.081; top-8 ['L2', 'L0', 'L1', 'L35', 'L41', 'L38', 'L30', 'L39']
- weight_decay=0.01: at index 47 (L9) ff 0.987, fo 0.958, oo 0.965, of 0.947, drop 0.023; layer-averaged drop 0.060; top-8 ['L0', 'L39', 'L2', 'L25', 'L1', 'L4', 'L41', 'L44']
- batch_size=20: at index 47 (L9) ff 0.990, fo 0.952, oo 0.950, of 0.925, drop 0.031; layer-averaged drop 0.050; top-8 ['L0', 'L2', 'L1', 'L39', 'L4', 'L46', 'L43', 'L41']
- length_balance=off: at index 47 (L9) ff 0.979, fo 0.946, oo 0.981, of 0.948, drop 0.033; layer-averaged drop 0.058; top-8 ['L0', 'L41', 'L2', 'L1', 'L37', 'L35', 'L43', 'L32']
- layer=true-final: at index 42 (L47) ff 0.980, fo 0.829, oo 0.875, of 0.917, drop 0.054; layer-averaged drop 0.060; top-8 ['L0', 'L39', 'L2', 'L1', 'L25', 'L4', 'L18', 'L35']
- layer=best-in-domain: at index 14 (L21) ff 1.000, fo 0.930, oo 0.995, of 0.978, drop 0.043; layer-averaged drop 0.060; top-8 ['L0', 'L39', 'L2', 'L1', 'L25', 'L4', 'L18', 'L35']
- DEVIATIONS: the combined factual+opinion probe is not trained (the claim compares the two subtypes); the transfer AUC is computed by applying the trained probe to the other subtype's held-out split exactly as upstream evaluate_probes_on_all_datasets does, but inline, because that function reads probes and pickles from fixed relative paths; activations are extracted for the 1200-conversation pool per subtype rather than all 3000, in batches of 25 rather than 100 (bf16 padding differs), and the chat template stamps the extraction date into the system header as it did upstream
- bootstrap: the pool is resampled with replacement and duplicates are collapsed before training, so each bootstrap run uses the ~63% distinct conversations of its resample and balances to 5/6 of each class; the templates axis is not run (the conversations are fixed artifacts generated, labelled and truncated with closed models)
- null control: the same pool through upstream's shuffle_labels=True path, which permutes the training and validation labels and leaves the test labels intact, so every probe is fitted to noise and both in-domain and transfer AUC sit near chance

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.4 · 2026-09-02T21:23:20+00:00*
