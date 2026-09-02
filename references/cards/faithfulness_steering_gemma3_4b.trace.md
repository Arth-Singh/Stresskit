## Verdict-stability trace — grade **B** (high confidence) at n = 92

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 80% · C 3% · D 17% | B | 37% | claim_stability (pass 70%) |
| 6 | B 87% · C 7% · D 7% | B | 53% | structural_stability (pass 67%) |
| 8 | B 93% · C 7% | B | 43% | claim_stability (pass 90%) |
| 10 | B 100% | B | 63% | structural_stability (pass 97%) |
| 14 | B 97% · C 3% | B | 70% | claim_stability (pass 93%) |
| 20 | B 100% | B | 83% | structural_stability (pass 93%) |
| 28 | B 97% · D 3% | B | 70% | structural_stability (pass 97%) |
| 92 | B 100% | B | 0% | beats_random (pass 0%) |

Verdict settles at **n = 8**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
