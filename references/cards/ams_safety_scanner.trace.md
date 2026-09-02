## Verdict-stability trace — grade **C** (low confidence) at n = 129

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 30% · C 60% · D 10% | C | 100% | beats_random (pass 59%) |
| 6 | B 20% · C 60% · D 20% | C | 100% | beats_random (pass 46%) |
| 8 | B 40% · C 43% · D 17% | C | 97% | structural_stability (pass 50%) |
| 10 | B 20% · C 77% · D 3% | C | 97% | beats_random (pass 55%) |
| 14 | B 27% · C 63% · D 10% | C | 100% | beats_random (pass 47%) |
| 20 | B 30% · C 63% · D 7% | C | 100% | structural_stability (pass 47%) |
| 28 | B 37% · C 57% · D 7% | C | 90% | structural_stability (pass 48%) |
| 129 | C 100% | C | 100% | beats_random (pass 100%) |

Verdict settles at **n = 129**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
