## Verdict-stability trace — grade **B** (high confidence) at n = 48 — grade rule v0.4, seed 0, 30 subsets per size

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 13% · C 87% | C | 97% | score_stability (pass 23%) |
| 6 | B 13% · C 87% | C | 97% | beats_random (pass 83%) |
| 8 | B 20% · C 80% | C | 97% | beats_random (pass 80%) |
| 10 | B 23% · C 77% | C | 93% | beats_random (pass 87%) |
| 14 | B 13% · C 87% | C | 90% | score_stability (pass 7%) |
| 20 | B 40% · C 60% | C | 70% | beats_random (pass 97%) |
| 28 | B 47% · C 53% | C | 53% | beats_random (pass 100%) |
| 48 | B 100% | B | 0% | beats_random (pass 100%) |

Verdict settles at **n = 48**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
