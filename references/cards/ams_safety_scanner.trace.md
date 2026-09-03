## Verdict-stability trace — grade **D** (low confidence) at n = 129 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | C 67% · D 33% | C | 100% | beats_random (pass 59%) |
| 6 | C 50% · D 50% | C | 100% | beats_random (pass 46%) |
| 8 | C 53% · D 47% | C | 97% | structural_stability (pass 50%) |
| 10 | C 67% · D 33% | C | 97% | beats_random (pass 55%) |
| 14 | C 43% · D 57% | D | 100% | beats_random (pass 47%) |
| 20 | C 50% · D 50% | C | 100% | structural_stability (pass 47%) |
| 28 | C 60% · D 40% | C | 90% | structural_stability (pass 48%) |
| 129 | D 100% | D | 100% | beats_random (pass 100%) |

Verdict settles at **n = 129**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
