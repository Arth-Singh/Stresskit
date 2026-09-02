## Verdict-stability trace — grade **C** (low confidence) at n = 22

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 3% · B 43% · C 47% · D 7% | C | 93% | score_stability (pass 30%) |
| 6 | B 60% · C 37% · D 3% | B | 100% | score_stability (pass 30%) |
| 8 | B 30% · C 67% · D 3% | C | 100% | structural_stability (pass 79%) |
| 10 | B 30% · C 70% | C | 100% | score_stability (pass 27%) |
| 14 | B 3% · C 97% | C | 100% | structural_stability (pass 87%) |
| 20 | C 100% | C | 100% | structural_stability (pass 97%) |
| 22 | C 100% | C | 100% | beats_random (pass 100%) |

Verdict settles at **n = 14**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
