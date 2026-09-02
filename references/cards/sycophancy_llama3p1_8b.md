# 🟡 Diagnostic Stability Card — descriptive grade **B** (high confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Claim:** We find that different LLMs represent these subtypes differently, with either more aligned or more distinct representations
> model: meta-llama/Llama-3.1-8B-Instruct · task: factual vs opinion sycophancy: per-layer linear probes on the residual stream at the end of the assistant's (truncated) response, 500+500 length-balanced conversations per subtype, in-domain vs cross-subtype ROC-AUC · method: upstream extractor, length balancing and probe trainer at the pinned commit (nn.Linear, Adam, 100 epochs, best-validation checkpoint, 80/10/10 split); transfer = each subtype's probe on the other subtype's held-out split

Battery: `seeds, bootstrap, hyperparams` — 88 runs (seed 42, 6.933s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.525 | [0.494, 0.551] | ≥ 0.800 | ❌ fail |
| claim stability | 0.989 | [0.955, 1.000] | ≥ 0.800 | ✅ pass |
| score stability | 0.202 | [0.175, 0.232] | ≤ 0.250 | ✅ pass |
| beats random | 3.499 | [3.292, 3.674] | ≥ 3.000 | ✅ pass |
| specificity | 3.501 | [3.251, 3.722] | ≥ 1.500 | ✅ pass |

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 88 |
| structured runs | 88 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.525 |
| min pairwise Jaccard | 0.067 |
| random-null Jaccard | 0.150 |
| overlap vs random (×) | 3.499 |
| claim flip rate | 0.023 |
| modal claim share π* | 0.989 |
| distinct claims | 2 |
| score mean | 0.232 |
| score CV | 0.202 |
| median finding size | 8.000 |
| Jaccard 95% CI (bootstrap) | [0.494, 0.551] |
| flip rate 95% CI (bootstrap) | [0.000, 0.089] |
| null-control (specificity) | Jaccard 0.150 · flip 0.259 on 81 null runs |
| claim distribution | `Llama: distinct (transfer drop >= 0.15); in-domain >=0.85`×87, `Llama: shared (transfer drop < 0.15); in-domain >=0.85`×1 |
| score-variance shares (OAT) | bootstrap: 43%, hyperparams: 41%, seeds: 16% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 41 | 0.473 | 0.000 | 1.000 | 0.203 |
| hyperparams | 8 | 0.671 | 0.250 | 0.875 | 0.221 |
| seeds | 41 | 0.592 | 0.000 | 1.000 | 0.146 |

## Notes

- upstream: antbaez/dissociating-sycophancy@47e02ef (MIT); ActivationExtractor, ConversationSample, parse_conversation, load_model, equalize_mean_lengths and LinearProbeTrainer imported unmodified; huggingface_hub.login, which two upstream modules call at import time, is a no-op here (the model is loaded from the local cache); file hashes generate_responses.py 21c3d6d52915, linear_probes.py ec9bb4a62982, process_lengths.py feef1b74fd99, factual_prompts_with_responses.json 7fd298a06001, opinion_prompts_with_responses.json 5add09cc7c81
- activations: regenerated with the upstream extractor (the released cache antbaez/sycophancy-mech is private); it hooks every decoder layer and stacks them in sorted() order of the module names, i.e. lexicographically, so probe index i is decoder layer [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 4, 5, 6, 7, 8, 9][i]: index 31 (the paper's 'final layer') is decoder layer 9 and decoder layer 31 is index 25. The hook reads position -2 of the left-padded full conversation (the eot_positions the extractor computes are unused); tokens read for the first conversations: factual -2='.' -1='<|eot_id|>'; factual -2='.' -1='<|eot_id|>'; factual -2='.' -1='<|eot_id|>'; opinion -2='.' -1='<|eot_id|>'; opinion -2='.' -1='<|eot_id|>'; opinion -2='.' -1='<|eot_id|>'. Chat template date string: '26 Jul 2024'; extraction batch 25 (upstream 100), torch.bfloat16, model revision 0e9e39f249a16976918f6564b8830bc894c89659
- reproduction (Tables 1-2, final layer, mean of seeds 42-46 -> seeds [42, 43, 44, 45, 46] here): factual->factual 0.91 -> 0.928, opinion->opinion 0.92 -> 0.938, factual->opinion 0.7 -> 0.738, opinion->factual 0.61 -> 0.711 at probe index 31 (decoder layer 9); layer-averaged (upstream avg_auc): ff 0.913, fo 0.699, oo 0.919, of 0.612; at decoder layer 31 (index 25): ff 0.910, fo 0.692, oo 0.916, of 0.460. Base run (seed 42): in-domain 0.950, drop 0.219, layers with drop >= 0.15: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]; samples {'factual': {'pool': 1200, 'syc': 500, 'non_syc': 500, 'mean_len_syc': 122.208, 'mean_len_non_syc': 122.254}, 'opinion': {'pool': 1200, 'syc': 500, 'non_syc': 500, 'mean_len_syc': 128.722, 'mean_len_non_syc': 128.746}}
- epochs=30: at index 31 (L9) ff 0.925, fo 0.812, oo 0.900, of 0.618, drop 0.197; layer-averaged drop 0.260; top-8 ['L4', 'L1', 'L2', 'L5', 'L3', 'L28', 'L29', 'L31']
- lr=0.0001: at index 31 (L9) ff 0.899, fo 0.691, oo 0.909, of 0.576, drop 0.270; layer-averaged drop 0.307; top-8 ['L4', 'L2', 'L6', 'L0', 'L5', 'L22', 'L29', 'L1']
- weight_decay=0.01: at index 31 (L9) ff 0.944, fo 0.725, oo 0.928, of 0.652, drop 0.248; layer-averaged drop 0.270; top-8 ['L4', 'L2', 'L0', 'L6', 'L5', 'L1', 'L28', 'L30']
- batch_size=20: at index 31 (L9) ff 0.972, fo 0.768, oo 0.934, of 0.708, drop 0.215; layer-averaged drop 0.261; top-8 ['L1', 'L0', 'L2', 'L4', 'L6', 'L28', 'L5', 'L26']
- length_balance=off: at index 31 (L9) ff 0.945, fo 0.767, oo 0.950, of 0.629, drop 0.250; layer-averaged drop 0.281; top-8 ['L2', 'L1', 'L5', 'L4', 'L0', 'L30', 'L27', 'L28']
- layer=true-final: at index 25 (L31) ff 0.897, fo 0.662, oo 0.839, of 0.483, drop 0.296; layer-averaged drop 0.264; top-8 ['L2', 'L0', 'L1', 'L4', 'L5', 'L6', 'L28', 'L22']
- layer=best-in-domain: at index 4 (L12) ff 0.988, fo 0.823, oo 0.969, of 0.894, drop 0.120; layer-averaged drop 0.264; top-8 ['L2', 'L0', 'L1', 'L4', 'L5', 'L6', 'L28', 'L22']
- DEVIATIONS: the combined factual+opinion probe is not trained (the claim compares the two subtypes); the transfer AUC is computed by applying the trained probe to the other subtype's held-out split exactly as upstream evaluate_probes_on_all_datasets does, but inline, because that function reads probes and pickles from fixed relative paths; activations are extracted for the 1200-conversation pool per subtype rather than all 3000, in batches of 25 rather than 100 (bf16 padding differs), and the chat template stamps the extraction date into the system header as it did upstream
- bootstrap: the pool is resampled with replacement and duplicates are collapsed before training, so each bootstrap run uses the ~63% distinct conversations of its resample and balances to 5/6 of each class; the templates axis is not run (the conversations are fixed artifacts generated, labelled and truncated with closed models)
- null control: the same pool through upstream's shuffle_labels=True path, which permutes the training and validation labels and leaves the test labels intact, so every probe is fitted to noise and both in-domain and transfer AUC sit near chance

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.4 · 2026-09-02T13:15:05+00:00*
