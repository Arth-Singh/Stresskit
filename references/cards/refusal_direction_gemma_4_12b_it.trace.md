## Verdict-stability trace — grade **C** (low confidence) at n = 21

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 30% · C 70% | C | 93% | specificity (pass 70%) |
| 6 | B 13% · C 87% | C | 100% | claim_stability (pass 13%) |
| 8 | B 10% · C 90% | C | 100% | specificity (pass 90%) |
| 10 | B 13% · C 87% | C | 100% | claim_stability (pass 13%) |
| 14 | C 100% | C | 97% | beats_random (pass 100%) |
| 20 | C 100% | C | 100% | beats_random (pass 100%) |
| 21 | C 100% | C | 100% | beats_random (pass 100%) |

Verdict settles at **n = 14**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
