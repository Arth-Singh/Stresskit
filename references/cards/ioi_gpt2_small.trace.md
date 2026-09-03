## Verdict-stability trace — grade **B** (low confidence) at n = 45 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 13% · B 77% · C 10% | B | 83% | specificity (pass 43%) |
| 6 | A 3% · B 97% | B | 97% | specificity (pass 53%) |
| 8 | A 7% · B 93% | B | 93% | specificity (pass 60%) |
| 10 | A 7% · B 93% | B | 93% | specificity (pass 63%) |
| 14 | A 7% · B 90% · C 3% | B | 93% | specificity (pass 67%) |
| 20 | A 7% · B 93% | B | 93% | specificity (pass 73%) |
| 28 | A 3% · B 97% | B | 97% | specificity (pass 77%) |
| 45 | B 100% | B | 100% | beats_random (pass 100%) |

Verdict settles at **n = 6**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
