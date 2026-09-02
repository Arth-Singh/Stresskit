## Verdict-stability trace — grade **C** (low confidence) at n = 21

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 3% · B 50% · C 47% | B | 90% | score_stability (pass 53%) |
| 6 | B 53% · C 47% | B | 100% | score_stability (pass 53%) |
| 8 | B 37% · C 63% | C | 100% | score_stability (pass 37%) |
| 10 | B 33% · C 67% | C | 100% | score_stability (pass 30%) |
| 14 | B 10% · C 90% | C | 100% | score_stability (pass 10%) |
| 20 | C 100% | C | 100% | beats_random (pass 100%) |
| 21 | C 100% | C | 100% | beats_random (pass 100%) |

Verdict settles at **n = 14**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
