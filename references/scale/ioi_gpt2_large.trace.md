## Verdict-stability trace — grade **B** (low confidence) at n = 45 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 7% · B 50% · C 43% | B | 90% | specificity (pass 53%) |
| 6 | A 13% · B 37% · C 50% | C | 87% | specificity (pass 57%) |
| 8 | A 17% · B 47% · C 37% | B | 83% | specificity (pass 73%) |
| 10 | A 13% · B 50% · C 37% | B | 87% | specificity (pass 70%) |
| 14 | A 17% · B 23% · C 60% | C | 83% | specificity (pass 90%) |
| 20 | A 10% · B 57% · C 33% | B | 90% | specificity (pass 90%) |
| 28 | A 10% · B 53% · C 37% | B | 90% | beats_random (pass 100%) |
| 45 | B 100% | B | 100% | beats_random (pass 100%) |

Verdict settles at **n = 45**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
