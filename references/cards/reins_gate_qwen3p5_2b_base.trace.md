## Verdict-stability trace — grade **B** (low confidence) at n = 51 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 60% · B 20% · C 20% | A | 23% | structural_stability (pass 77%) |
| 6 | A 70% · B 27% · C 3% | A | 30% | structural_stability (pass 70%) |
| 8 | A 63% · B 33% · C 3% | A | 33% | structural_stability (pass 76%) |
| 10 | A 57% · B 43% | A | 43% | structural_stability (pass 67%) |
| 14 | A 33% · B 67% | B | 60% | structural_stability (pass 54%) |
| 20 | A 50% · B 50% | A | 50% | structural_stability (pass 73%) |
| 28 | A 10% · B 90% | B | 90% | structural_stability (pass 63%) |
| 51 | B 100% | B | 100% | beats_random (pass 100%) |

Verdict settles at **n = 28**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
