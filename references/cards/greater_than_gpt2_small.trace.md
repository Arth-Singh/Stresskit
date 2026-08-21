## Verdict-stability trace — grade **B** (high confidence) at n = 45

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 3% · B 77% · C 20% | B | 80% | structural_stability (pass 67%) |
| 6 | B 100% | B | 53% | structural_stability (pass 80%) |
| 8 | B 100% | B | 33% | structural_stability (pass 90%) |
| 10 | B 100% | B | 37% | structural_stability (pass 97%) |
| 14 | B 100% | B | 60% | beats_random (pass 100%) |
| 20 | B 100% | B | 43% | beats_random (pass 100%) |
| 28 | B 100% | B | 43% | beats_random (pass 100%) |
| 45 | B 100% | B | 0% | beats_random (pass 100%) |

Verdict settles at **n = 6**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
