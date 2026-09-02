## Verdict-stability trace — grade **A** (high confidence) at n = 39

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 63% · B 37% | A | 83% | claim_stability (pass 80%) |
| 6 | A 87% · B 13% | A | 50% | specificity (pass 90%) |
| 8 | A 93% · B 7% | A | 63% | claim_stability (pass 93%) |
| 10 | A 100% | A | 77% | beats_random (pass 100%) |
| 14 | A 97% · B 3% | A | 90% | claim_stability (pass 97%) |
| 20 | A 100% | A | 63% | beats_random (pass 100%) |
| 28 | A 100% | A | 20% | beats_random (pass 100%) |
| 39 | A 100% | A | 0% | beats_random (pass 100%) |

Verdict settles at **n = 8**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
