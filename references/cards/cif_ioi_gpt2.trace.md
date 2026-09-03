## Verdict-stability trace — grade **C** (low confidence) at n = 36 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 20% · C 60% · D 20% | C | 83% | beats_random (pass 57%) |
| 6 | B 43% · C 57% | C | 97% | beats_random (pass 70%) |
| 8 | B 17% · C 83% | C | 100% | beats_random (pass 67%) |
| 10 | B 23% · C 77% | C | 100% | beats_random (pass 73%) |
| 14 | B 10% · C 90% | C | 100% | beats_random (pass 83%) |
| 20 | B 10% · C 90% | C | 100% | beats_random (pass 90%) |
| 28 | B 3% · C 97% | C | 100% | beats_random (pass 100%) |
| 36 | C 100% | C | 100% | beats_random (pass 100%) |

Verdict settles at **n = 14**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
