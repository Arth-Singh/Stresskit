## Verdict-stability trace — grade **C** (low confidence) at n = 57

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 33% · C 43% · D 23% | C | 87% | claim_stability (pass 43%) |
| 6 | B 27% · C 50% · D 23% | C | 100% | claim_stability (pass 33%) |
| 8 | B 20% · C 70% · D 10% | C | 100% | claim_stability (pass 27%) |
| 10 | B 17% · C 73% · D 10% | C | 100% | beats_random (pass 80%) |
| 14 | B 13% · C 83% · D 3% | C | 97% | claim_stability (pass 13%) |
| 20 | B 17% · C 77% · D 7% | C | 100% | claim_stability (pass 17%) |
| 28 | B 7% · C 93% | C | 83% | claim_stability (pass 7%) |
| 57 | C 100% | C | 100% | beats_random (pass 100%) |

Verdict settles at **n = 28**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
