## Verdict-stability trace — grade **B** (low confidence) at n = 32

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 87% · C 13% | B | 47% | claim_stability (pass 53%) |
| 6 | B 97% · C 3% | B | 43% | structural_stability (pass 90%) |
| 8 | B 93% · C 7% | B | 60% | claim_stability (pass 83%) |
| 10 | B 100% | B | 60% | beats_random (pass 100%) |
| 14 | B 100% | B | 93% | claim_stability (pass 93%) |
| 20 | B 100% | B | 80% | beats_random (pass 100%) |
| 28 | B 100% | B | 77% | beats_random (pass 100%) |
| 32 | B 100% | B | 100% | beats_random (pass 100%) |

Verdict settles at **n = 6**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
