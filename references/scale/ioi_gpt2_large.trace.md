## Verdict-stability trace — grade **A** (low confidence) at n = 45

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 33% · B 47% · C 20% | B | 90% | specificity (pass 53%) |
| 6 | A 47% · B 43% · C 10% | A | 87% | specificity (pass 57%) |
| 8 | A 67% · B 33% | A | 83% | specificity (pass 73%) |
| 10 | A 67% · B 33% | A | 87% | specificity (pass 70%) |
| 14 | A 87% · B 13% | A | 83% | specificity (pass 90%) |
| 20 | A 90% · B 10% | A | 90% | specificity (pass 90%) |
| 28 | A 100% | A | 90% | beats_random (pass 100%) |
| 45 | A 100% | A | 100% | beats_random (pass 100%) |

Verdict settles at **n = 20**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
