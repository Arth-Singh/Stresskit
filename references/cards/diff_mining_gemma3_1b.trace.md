## Verdict-stability trace — grade **A** (high confidence) at n = 131 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | A 97% · B 3% | A | 3% | beats_random (pass 100%) |
| 6 | A 83% · B 10% · C 7% | A | 17% | structural_stability (pass 93%) |
| 8 | A 80% · B 17% · C 3% | A | 20% | structural_stability (pass 87%) |
| 10 | A 73% · B 23% · C 3% | A | 27% | structural_stability (pass 93%) |
| 14 | A 50% · B 33% · C 17% | A | 50% | structural_stability (pass 97%) |
| 20 | A 50% · B 50% | A | 50% | structural_stability (pass 93%) |
| 28 | A 63% · B 37% | A | 37% | structural_stability (pass 97%) |
| 131 | A 100% | A | 0% | beats_random (pass 100%) |

Verdict settles at **n = 131**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
