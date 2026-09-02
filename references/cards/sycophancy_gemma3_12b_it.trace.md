## Verdict-stability trace — grade **B** (high confidence) at n = 48

| n runs | grade distribution | modal | low-confidence | flakiest check |
|---|---|---|---|---|
| 4 | B 77% · C 23% | B | 97% | score_stability (pass 23%) |
| 6 | B 83% · C 17% | B | 97% | beats_random (pass 83%) |
| 8 | B 83% · C 17% | B | 97% | beats_random (pass 80%) |
| 10 | B 87% · C 13% | B | 93% | beats_random (pass 87%) |
| 14 | B 97% · C 3% | B | 90% | score_stability (pass 7%) |
| 20 | B 97% · C 3% | B | 70% | beats_random (pass 97%) |
| 28 | B 100% | B | 53% | beats_random (pass 100%) |
| 48 | B 100% | B | 0% | beats_random (pass 100%) |

Verdict settles at **n = 14**: from there on, the modal grade matches the full-sample grade with >= 90% subset agreement.
