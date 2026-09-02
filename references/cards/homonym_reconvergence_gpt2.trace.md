## Verdict-stability trace — grade **B** (low confidence) at n = 31

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 13% · B 53% · C 30% · D 3% | B | 87% | claim_stability (pass 53%) |
| 6 | B 73% · C 27% | B | 87% | score_stability (pass 73%) |
| 8 | B 80% · C 20% | B | 100% | structural_stability (pass 60%) |
| 10 | B 93% · C 7% | B | 67% | structural_stability (pass 87%) |
| 14 | B 97% · C 3% | B | 87% | claim_stability (pass 90%) |
| 20 | B 100% | B | 90% | beats_random (pass 100%) |
| 28 | B 100% | B | 97% | beats_random (pass 100%) |
| 31 | B 100% | B | 100% | beats_random (pass 100%) |

Verdict settles at **n = 10**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
