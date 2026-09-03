## Verdict-stability trace — grade **B** (high confidence) at n = 92 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 73% · C 10% · D 17% | B | 37% | claim_stability (pass 70%) |
| 6 | B 53% · C 40% · D 7% | B | 53% | structural_stability (pass 67%) |
| 8 | B 70% · C 27% · D 3% | B | 43% | claim_stability (pass 90%) |
| 10 | B 60% · C 40% | B | 63% | structural_stability (pass 97%) |
| 14 | B 53% · C 47% | B | 70% | specificity (pass 93%) |
| 20 | B 37% · C 63% | C | 83% | structural_stability (pass 93%) |
| 28 | B 47% · C 50% · D 3% | C | 70% | specificity (pass 97%) |
| 92 | B 100% | B | 0% | beats_random (pass 0%) |

Verdict settles at **n = 92**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
