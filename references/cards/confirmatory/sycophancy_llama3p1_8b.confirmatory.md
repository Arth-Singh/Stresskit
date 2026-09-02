# Confirmatory Stability Card — **fail**

> **Claim:** We find that different LLMs represent these subtypes differently, with either more aligned or more distinct representations
> Finite-sample paired-Hoeffding profile; all required checks use a familywise confidence budget.

| required check | estimate | simultaneous CI | threshold | state |
|---|---:|---|---:|---|
| structural stability | 0.373 | [0.213, 0.532] | ≥ 0.800 | fail |
| beats random | 0.210 | [-0.108, 0.529] | ≥ 0.200 | inconclusive |
| claim stability | 0.880 | [0.749, 1.000] | ≥ 0.800 | inconclusive |
| specificity | 0.224 | [-0.116, 0.543] | ≥ 0.200 | inconclusive |

Runs: 200 independent IID specification draws; minimum: 200.

Failure means this registered claim did not clear at least one registered gate; it does not grade a paper or method family.

Frozen specification space (product distribution, drawn IID):

- batch_size: 100 (0.500), 20 (0.500)
- epochs: 100 (0.500), 30 (0.500)
- layer: final-index (0.500), true-final (0.250), best-in-domain (0.250)
- length_balance: upstream (0.500), off (0.500)
- lr: 0.001 (0.500), 0.0001 (0.500)
- probe_seed: 42 (0.050), 100 (0.050), 101 (0.050), 102 (0.050), 103 (0.050), 104 (0.050), 105 (0.050), 106 (0.050), 107 (0.050), 108 (0.050), 109 (0.050), 110 (0.050), 111 (0.050), 112 (0.050), 113 (0.050), 114 (0.050), 115 (0.050), 116 (0.050), 117 (0.050), 118 (0.050)
- resample: 0 (0.500), 1 (0.056), 2 (0.056), 3 (0.056), 4 (0.056), 5 (0.056), 6 (0.056), 7 (0.056), 8 (0.056), 9 (0.056)
- weight_decay: 0.0 (0.500), 0.01 (0.500)

Seeds: real manifest 20260901, null manifest 20260902, pairing master 20260903. Runs: 200 real, 200 null.
