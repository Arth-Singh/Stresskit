## Verdict-stability trace — grade **C** (low confidence) at n = 31 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 40% · C 47% · D 13% | C | 87% | claim_stability (pass 53%) |
| 6 | B 43% · C 30% · D 27% | B | 87% | score_stability (pass 73%) |
| 8 | B 43% · C 20% · D 37% | B | 100% | structural_stability (pass 60%) |
| 10 | C 93% · D 7% | C | 67% | structural_stability (pass 87%) |
| 14 | C 100% | C | 87% | claim_stability (pass 90%) |
| 20 | C 100% | C | 90% | beats_random (pass 100%) |
| 28 | C 100% | C | 97% | beats_random (pass 100%) |
| 31 | C 100% | C | 100% | beats_random (pass 100%) |

Verdict settles at **n = 10**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
