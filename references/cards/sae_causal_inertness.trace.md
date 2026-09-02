## Verdict-stability trace — grade **C** (low confidence) at n = 33

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 37% · C 53% · D 10% | C | 97% | claim_stability (pass 57%) |
| 6 | B 43% · C 57% | C | 100% | score_stability (pass 30%) |
| 8 | B 23% · C 77% | C | 100% | claim_stability (pass 77%) |
| 10 | B 10% · C 87% · D 3% | C | 100% | claim_stability (pass 77%) |
| 14 | B 13% · C 87% | C | 100% | claim_stability (pass 67%) |
| 20 | C 100% | C | 100% | claim_stability (pass 90%) |
| 28 | C 100% | C | 100% | beats_random (pass 100%) |
| 33 | C 100% | C | 100% | beats_random (pass 100%) |

Verdict settles at **n = 20**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
