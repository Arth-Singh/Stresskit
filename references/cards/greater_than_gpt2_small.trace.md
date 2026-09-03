## Verdict-stability trace — grade **C** (high confidence) at n = 45 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 33% · C 67% | C | 80% | structural_stability (pass 67%) |
| 6 | B 10% · C 90% | C | 53% | structural_stability (pass 80%) |
| 8 | B 7% · C 93% | C | 33% | structural_stability (pass 90%) |
| 10 | B 7% · C 93% | C | 37% | structural_stability (pass 97%) |
| 14 | C 100% | C | 60% | beats_random (pass 100%) |
| 20 | C 100% | C | 43% | beats_random (pass 100%) |
| 28 | C 100% | C | 43% | beats_random (pass 100%) |
| 45 | C 100% | C | 0% | beats_random (pass 100%) |

Verdict settles at **n = 6**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
