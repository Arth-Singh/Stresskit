## Verdict-stability trace — grade **C** (low confidence) at n = 33 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 20% · C 43% · D 37% | C | 97% | claim_stability (pass 57%) |
| 6 | B 10% · C 47% · D 43% | C | 100% | score_stability (pass 30%) |
| 8 | B 3% · C 57% · D 40% | C | 100% | claim_stability (pass 77%) |
| 10 | B 3% · C 63% · D 33% | C | 100% | claim_stability (pass 77%) |
| 14 | C 90% · D 10% | C | 100% | claim_stability (pass 67%) |
| 20 | C 87% · D 13% | C | 100% | claim_stability (pass 90%) |
| 28 | C 90% · D 10% | C | 100% | beats_random (pass 100%) |
| 33 | C 100% | C | 100% | beats_random (pass 100%) |

Verdict settles at **n = 28**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
