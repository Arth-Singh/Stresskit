| finder | arm | null | v0.3 | v0.4 | confidence | checks failed/undecided | flags |
|---|---|---|---|---|---|---|---|
| constant | default | no | A | B | high | — | vacuous seeds, bootstrap |
| constant | default | yes | B | C | high | specificity fail (1.00) | vacuous seeds, bootstrap |
| constant | seeds_only | no | A | B | high | — | vacuous seeds |
| constant | seeds_only | yes | B | C | high | specificity fail (1.00) | vacuous seeds |
| index_ranker | default | no | A | B | high | — | vacuous seeds, bootstrap |
| index_ranker | default | yes | B | C | high | specificity fail (0.94) | vacuous seeds, bootstrap |
| index_ranker | seeds_only | no | A | B | high | — | vacuous seeds |
| index_ranker | seeds_only | yes | B | C | high | specificity fail (1.00) | vacuous seeds |
| random_subset | default | no | C | C | low | structural_stability fail (0.30); claim_stability inconclusive (0.79); score_stability inconclusive (0.39) | vacuous bootstrap |
| random_subset | default | yes | C | C | low | structural_stability fail (0.30); claim_stability inconclusive (0.79); score_stability inconclusive (0.39); specificity inconclusive (1.07) | vacuous bootstrap |
| random_subset | seeds_only | no | D | D | low | structural_stability fail (0.02); claim_stability inconclusive (0.65); score_stability inconclusive (0.42); beats_random fail (0.79) | — |
| random_subset | seeds_only | yes | D | D | low | structural_stability fail (0.02); claim_stability inconclusive (0.65); score_stability inconclusive (0.42); beats_random fail (0.79); specificity fail (0.66) | — |
| planted_leak | default | no | A | B | high | — | vacuous seeds, bootstrap |
| planted_leak | default | yes | B | C | high | specificity fail (1.00) | vacuous seeds, bootstrap |
| planted_leak | seeds_only | no | A | B | high | — | vacuous seeds |
| planted_leak | seeds_only | yes | B | C | high | specificity fail (1.00) | vacuous seeds |
| size_inflating | default | no | B | B | high | beats_random fail (2.81) | vacuous seeds, bootstrap |
| size_inflating | default | yes | B | C | high | beats_random fail (2.81); specificity fail (1.00) | vacuous seeds, bootstrap |
| size_inflating | seeds_only | no | B | B | high | beats_random fail (2.81) | vacuous seeds |
| size_inflating | seeds_only | yes | B | C | high | beats_random fail (2.81); specificity fail (1.00) | vacuous seeds |
| fixed_direction | default | no | A | B | high | — | vacuous seeds, bootstrap |
| fixed_direction | default | yes | B | C | high | specificity fail (1.00) | vacuous seeds, bootstrap |
| fixed_direction | seeds_only | no | A | B | high | — | vacuous seeds |
| fixed_direction | seeds_only | yes | B | C | high | specificity fail (1.00) | vacuous seeds |
| random_direction | default | no | B | B | low | structural_stability fail (0.32); beats_random inconclusive (3.15) | vacuous bootstrap |
| random_direction | default | yes | B | C | low | structural_stability fail (0.32); beats_random inconclusive (3.15); specificity inconclusive (0.94) | vacuous bootstrap |
| random_direction | seeds_only | no | D | D | high | structural_stability fail (0.10); beats_random fail (0.96) | — |
| random_direction | seeds_only | yes | D | D | high | structural_stability fail (0.10); beats_random fail (0.96); specificity fail (0.90) | — |
| demo_positive | default | no | A | B | high | — | vacuous seeds |
| demo_positive | default | yes | A | A | high | — | vacuous seeds |
| demo_positive | seeds_only | no | A | B | high | — | vacuous seeds |
| demo_positive | seeds_only | yes | A | A | high | — | vacuous seeds |
| demo_on_noise | default | no | C | C | high | structural_stability fail (0.18); claim_stability fail (0.58); score_stability fail (2.31) | — |
| demo_on_noise | default | yes | C | C | high | structural_stability fail (0.18); claim_stability fail (0.58); score_stability fail (2.31); specificity fail (0.87) | — |
| demo_on_noise | seeds_only | no | C | C | high | structural_stability fail (0.26); claim_stability fail (0.53); score_stability fail (1.35) | — |
| demo_on_noise | seeds_only | yes | C | C | high | structural_stability fail (0.26); claim_stability fail (0.53); score_stability fail (1.35); specificity fail (0.90) | — |

Degenerate finders graded no worse than `demo_positive` in the same cell under v0.3: 8
- constant / default / null=no: A (positive control A)
- constant / seeds_only / null=no: A (positive control A)
- index_ranker / default / null=no: A (positive control A)
- index_ranker / seeds_only / null=no: A (positive control A)
- planted_leak / default / null=no: A (positive control A)
- planted_leak / seeds_only / null=no: A (positive control A)
- fixed_direction / default / null=no: A (positive control A)
- fixed_direction / seeds_only / null=no: A (positive control A)

Degenerate finders graded no worse than `demo_positive` in the same cell under v0.4: 11
- constant / default / null=no: B (positive control B)
- constant / seeds_only / null=no: B (positive control B)
- index_ranker / default / null=no: B (positive control B)
- index_ranker / seeds_only / null=no: B (positive control B)
- planted_leak / default / null=no: B (positive control B)
- planted_leak / seeds_only / null=no: B (positive control B)
- size_inflating / default / null=no: B (positive control B)
- size_inflating / seeds_only / null=no: B (positive control B)
- fixed_direction / default / null=no: B (positive control B)
- fixed_direction / seeds_only / null=no: B (positive control B)
- random_direction / default / null=no: B (positive control B)
