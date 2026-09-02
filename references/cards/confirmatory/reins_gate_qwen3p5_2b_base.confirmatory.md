# Confirmatory Stability Card — **fail**

> **Claim:** The frozen Qwen3.5-2B-Base gate opened on 98.7% of harmful evaluation prompts and 4.7% of negative evaluation prompts. This preserves high harmful coverage and keeps negative openings rare.
> Finite-sample paired-Hoeffding profile; all required checks use a familywise confidence budget.

| required check | estimate | simultaneous CI | threshold | state |
|---|---:|---|---:|---|
| structural stability | 0.417 | [0.258, 0.577] | ≥ 0.800 | fail |
| beats random | 0.393 | [0.074, 0.712] | ≥ 0.200 | inconclusive |
| claim stability | 0.915 | [0.780, 1.000] | ≥ 0.800 | inconclusive |
| specificity | 0.152 | [-0.188, 0.491] | ≥ 0.200 | inconclusive |

Runs: 200 independent IID specification draws; minimum: 200.

Failure means this registered claim did not clear at least one registered gate; it does not grade a paper or method family.

Frozen specification space (product distribution, drawn IID):

- folds: 5 (0.500), 3 (0.500)
- layers: all (0.500), last (0.250), late (0.250)
- negatives: matched_safe (0.500), all_safe (0.500)
- resample: 0 (0.500), 1 (0.056), 2 (0.056), 3 (0.056), 4 (0.056), 5 (0.056), 6 (0.056), 7 (0.056), 8 (0.056), 9 (0.056)
- split_seed: 12 (0.050), 100 (0.050), 101 (0.050), 102 (0.050), 103 (0.050), 104 (0.050), 105 (0.050), 106 (0.050), 107 (0.050), 108 (0.050), 109 (0.050), 110 (0.050), 111 (0.050), 112 (0.050), 113 (0.050), 114 (0.050), 115 (0.050), 116 (0.050), 117 (0.050), 118 (0.050)
- target_negative_fpr: 0.1 (0.500), 0.05 (0.250), 0.2 (0.250)
- template: answer_en_v1 (0.500), plain (0.250), answer_en_v2 (0.250)

Seeds: real manifest 20260901, null manifest 20260902, pairing master 20260903. Runs: 200 real, 200 null.
