## Verdict-stability trace — grade **B** (low confidence) at n = 51

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 27% · B 50% · C 23% | B | 87% | structural_stability (pass 27%) |
| 6 | A 30% · B 70% | B | 93% | structural_stability (pass 30%) |
| 8 | A 33% · B 57% · C 10% | B | 90% | structural_stability (pass 33%) |
| 10 | A 33% · B 63% · C 3% | B | 90% | score_stability (pass 67%) |
| 14 | A 23% · B 73% · C 3% | B | 100% | score_stability (pass 67%) |
| 20 | A 13% · B 87% | B | 100% | score_stability (pass 33%) |
| 28 | B 100% | B | 100% | score_stability (pass 17%) |
| 51 | B 100% | B | 100% | beats_random (pass 100%) |

Verdict settles at **n = 28**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
