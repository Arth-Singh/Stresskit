## Verdict-stability trace — grade **B** (high confidence) at n = 88 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 60% · C 37% · D 3% | B | 87% | beats_random (pass 83%) |
| 6 | B 70% · C 30% | B | 77% | beats_random (pass 87%) |
| 8 | B 80% · C 20% | B | 70% | score_stability (pass 90%) |
| 10 | B 63% · C 37% | B | 80% | score_stability (pass 87%) |
| 14 | B 60% · C 40% | B | 77% | score_stability (pass 93%) |
| 20 | B 80% · C 20% | B | 70% | score_stability (pass 93%) |
| 28 | B 100% | B | 27% | beats_random (pass 100%) |
| 88 | B 100% | B | 0% | beats_random (pass 100%) |

Verdict settles at **n = 28**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
