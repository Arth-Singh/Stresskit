## Verdict-stability trace — grade **C** (low confidence) at n = 32 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | C 100% | C | 47% | claim_stability (pass 53%) |
| 6 | C 100% | C | 43% | structural_stability (pass 90%) |
| 8 | C 100% | C | 60% | claim_stability (pass 83%) |
| 10 | C 100% | C | 60% | beats_random (pass 100%) |
| 14 | C 100% | C | 93% | claim_stability (pass 93%) |
| 20 | C 100% | C | 80% | beats_random (pass 100%) |
| 28 | C 100% | C | 77% | beats_random (pass 100%) |
| 32 | C 100% | C | 100% | beats_random (pass 100%) |

Verdict settles at **n = 4**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
