## Verdict-stability trace — grade **B** (low confidence) at n = 51 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 7% · B 70% · C 23% | B | 87% | structural_stability (pass 27%) |
| 6 | A 7% · B 60% · C 33% | B | 93% | structural_stability (pass 30%) |
| 8 | A 10% · B 63% · C 27% | B | 90% | structural_stability (pass 33%) |
| 10 | A 10% · B 40% · C 50% | C | 90% | score_stability (pass 67%) |
| 14 | B 57% · C 43% | B | 100% | score_stability (pass 67%) |
| 20 | B 43% · C 57% | C | 100% | score_stability (pass 33%) |
| 28 | B 70% · C 30% | B | 100% | score_stability (pass 17%) |
| 51 | B 100% | B | 100% | beats_random (pass 100%) |

Verdict settles at **n = 51**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
