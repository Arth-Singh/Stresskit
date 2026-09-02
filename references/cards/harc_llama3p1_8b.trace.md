## Verdict-stability trace — grade **B** (low confidence) at n = 51

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 13% · B 60% · C 27% | B | 97% | specificity (pass 43%) |
| 6 | A 7% · B 73% · C 20% | B | 93% | specificity (pass 20%) |
| 8 | A 10% · B 87% · C 3% | B | 100% | specificity (pass 37%) |
| 10 | B 97% · C 3% | B | 100% | specificity (pass 13%) |
| 14 | A 3% · B 90% · C 7% | B | 100% | specificity (pass 10%) |
| 20 | B 100% | B | 97% | structural_stability (pass 3%) |
| 28 | B 100% | B | 90% | beats_random (pass 100%) |
| 51 | B 100% | B | 100% | beats_random (pass 100%) |

Verdict settles at **n = 10**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
