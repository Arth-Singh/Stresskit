## Verdict-stability trace — grade **A** (high confidence) at n = 45 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 60% · B 30% · C 10% | A | 40% | structural_stability (pass 70%) |
| 6 | A 67% · B 10% · C 23% | A | 33% | structural_stability (pass 97%) |
| 8 | A 80% · B 17% · C 3% | A | 20% | beats_random (pass 100%) |
| 10 | A 70% · B 20% · C 10% | A | 30% | beats_random (pass 100%) |
| 14 | A 50% · B 50% | A | 50% | beats_random (pass 100%) |
| 20 | A 97% · B 3% | A | 3% | beats_random (pass 100%) |
| 28 | A 100% | A | 0% | beats_random (pass 100%) |
| 45 | A 100% | A | 0% | beats_random (pass 100%) |

Verdict settles at **n = 20**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
