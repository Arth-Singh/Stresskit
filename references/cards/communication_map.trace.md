## Verdict-stability trace — grade **B** (low confidence) at n = 24

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 63% · C 37% | B | 87% | structural_stability (pass 60%) |
| 6 | B 90% · C 10% | B | 97% | claim_stability (pass 37%) |
| 8 | B 90% · C 10% | B | 97% | claim_stability (pass 27%) |
| 10 | B 77% · C 23% | B | 100% | structural_stability (pass 70%) |
| 14 | B 100% | B | 100% | beats_random (pass 100%) |
| 20 | B 100% | B | 100% | beats_random (pass 100%) |
| 24 | B 100% | B | 100% | beats_random (pass 100%) |

Verdict settles at **n = 14**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
