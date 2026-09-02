# Confirmatory Stability Card — **fail**

> **Claim:** aligned LLMs encode harmfulness and refusal as separable directions in the residual stream at prompt-side token positions; HARC pairs the two directions across both prompt and response positions
> Finite-sample paired-Hoeffding profile; all required checks use a familywise confidence budget.

| required check | estimate | simultaneous CI | threshold | state |
|---|---:|---|---:|---|
| structural stability | 0.428 | [0.269, 0.588] | ≥ 0.800 | fail |
| beats random | 0.272 | [-0.046, 0.591] | ≥ 0.200 | inconclusive |
| claim stability | 0.435 | [0.295, 0.575] | ≥ 0.800 | fail |
| specificity | 0.068 | [-0.272, 0.408] | ≥ 0.200 | inconclusive |

Runs: 200 independent IID specification draws; minimum: 200.

Failure means this registered claim did not clear at least one registered gate; it does not grade a paper or method family.

Frozen specification space (product distribution, drawn IID):

- drop_truncated: False (0.500), True (0.500)
- estimator: diffmeans (0.500), probe (0.500)
- features: cb_ultrachat/chat (0.500), cb_ultrachat/raw (0.167), advbench_alpaca/chat (0.167), advbench_alpaca/raw (0.167)
- harm_position: t_inst (0.500), mean_content (0.500)
- n_extract: 300 (0.500), 100 (0.500)
- resample: 0 (0.500), 1 (0.056), 2 (0.056), 3 (0.056), 4 (0.056), 5 (0.056), 6 (0.056), 7 (0.056), 8 (0.056), 9 (0.056)
- response_window: 32 (0.500), 8 (0.500)
- split_seed: 0 (0.050), 1 (0.050), 2 (0.050), 3 (0.050), 4 (0.050), 5 (0.050), 6 (0.050), 7 (0.050), 8 (0.050), 9 (0.050), 10 (0.050), 11 (0.050), 12 (0.050), 13 (0.050), 14 (0.050), 15 (0.050), 16 (0.050), 17 (0.050), 18 (0.050), 19 (0.050)

Seeds: real manifest 20260901, null manifest 20260902, pairing master 20260903. Runs: 200 real, 200 null.
