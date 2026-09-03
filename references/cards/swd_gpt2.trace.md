## Verdict-stability trace — grade **C** (low confidence) at n = 22 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 23% · C 33% · D 43% | D | 93% | score_stability (pass 30%) |
| 6 | B 7% · C 70% · D 23% | C | 100% | score_stability (pass 30%) |
| 8 | B 3% · C 73% · D 23% | C | 100% | structural_stability (pass 79%) |
| 10 | C 87% · D 13% | C | 100% | score_stability (pass 27%) |
| 14 | C 97% · D 3% | C | 100% | structural_stability (pass 87%) |
| 20 | C 100% | C | 100% | structural_stability (pass 97%) |
| 22 | C 100% | C | 100% | beats_random (pass 100%) |

Verdict settles at **n = 14**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
