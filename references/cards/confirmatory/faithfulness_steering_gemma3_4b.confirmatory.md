# Confirmatory Stability Card — **fail**

> **Claim:** when steering is effective, its effect generalizes broadly across cue types and datasets--in cross-cue and cross-dataset analyses, effect size is determined primarily by the evaluation setting, rather than the vector's train setting. How the vector is built also matters little--four construction methods, including one whose optimization target mentions no specific cue, yield similar effect sizes.
> Finite-sample paired-Hoeffding profile; all required checks use a familywise confidence budget.

| required check | estimate | simultaneous CI | threshold | state |
|---|---:|---|---:|---|
| structural stability | 0.557 | [0.398, 0.716] | ≥ 0.800 | fail |
| beats random | 0.082 | [-0.237, 0.400] | ≥ 0.200 | inconclusive |
| claim stability | 0.260 | [0.116, 0.404] | ≥ 0.800 | fail |
| specificity | 0.487 | [0.147, 0.713] | ≥ 0.200 | inconclusive |

Runs: 200 independent IID specification draws; minimum: 200.

Failure means this registered claim did not clear at least one registered gate; it does not grade a paper or method family.

Frozen specification space (product distribution, drawn IID):

- n_per_cue: None (0.500), 20 (0.250), 50 (0.250)
- phrasing: upstream (0.500), paraphrase_a (0.250), paraphrase_b (0.250)
- pooling: completion_mean (0.500), last_token (0.500)
- prompt: cued (0.500), uncued (0.500)
- ref_layer: 17 (0.500), 11 (0.500)
- resample: 0 (0.500), 1 (0.056), 2 (0.056), 3 (0.056), 4 (0.056), 5 (0.056), 6 (0.056), 7 (0.056), 8 (0.056), 9 (0.056)
- subsample: 0.8 (0.500), 1.0 (0.500)
- subsample_seed: 42 (0.050), 100 (0.050), 101 (0.050), 102 (0.050), 103 (0.050), 104 (0.050), 105 (0.050), 106 (0.050), 107 (0.050), 108 (0.050), 109 (0.050), 110 (0.050), 111 (0.050), 112 (0.050), 113 (0.050), 114 (0.050), 115 (0.050), 116 (0.050), 117 (0.050), 118 (0.050)

Seeds: real manifest 20260901, null manifest 20260902, pairing master 20260903. Runs: 200 real, 200 null.
