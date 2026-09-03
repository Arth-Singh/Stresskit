## Verdict-stability trace — grade **B** (low confidence) at n = 51 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 47% · C 43% · D 10% | B | 97% | specificity (pass 43%) |
| 6 | B 47% · C 53% | C | 93% | specificity (pass 20%) |
| 8 | B 43% · C 57% | C | 100% | specificity (pass 37%) |
| 10 | B 30% · C 70% | C | 100% | specificity (pass 13%) |
| 14 | B 33% · C 67% | C | 100% | specificity (pass 10%) |
| 20 | B 33% · C 67% | C | 97% | structural_stability (pass 3%) |
| 28 | B 33% · C 67% | C | 90% | beats_random (pass 100%) |
| 51 | B 100% | B | 100% | beats_random (pass 100%) |

Verdict settles at **n = 51**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
