# Pre-registered prediction: what makes the refusal-direction pipeline unstable

Recorded 2026-09-02T08:26Z, **before** the Qwen2.5-7B-Instruct selection audit
finished. Its result is the test.

## The question

The refusal-direction battery grades three models B and one C. The score CV
across runs — how much the held-out effect of the selected direction moves under
resampling — ranges from 0.004 to 0.323. What explains the difference?

## Hypothesis 1 (rejected)

*A thin admissible candidate set makes the pipeline unstable.* The upstream
selection rule keeps only candidates with harmless-prompt KL ≤ 0.1 and
non-negative induced-refusal log-odds; across battery runs that leaves 3–19 of
120 candidates. Rejected on the data already in hand: Qwen2.5-7B-Instruct has a
median admissible set of 4 — as thin as Qwen3.5-4B's — but a score CV of 0.009
against Qwen3.5-4B's 0.174. Across four models the correlation between median
admissible-set size and score CV is only −0.47.

## Hypothesis 2 (under test)

*What matters is the spread of held-out quality among the admissible candidates,
not how many there are.* If every surviving candidate works, it does not matter
which one a resample picks. Measured by the selection audit on one extraction
split, coherent compliance among admissible candidates:

| model | admissible | spread | median | battery score CV |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 19 | 0.219 | 0.984 | 0.004 |
| Qwen3.5-4B | 4 | 0.719 | 0.828 | 0.174 |
| Qwen3.5-9B | 7 | 0.688 | 0.609 | 0.323 |

## The prediction

Qwen2.5-7B-Instruct has a thin admissible set (median 4 across battery runs) but
the second-lowest score CV (0.009) and a worst-run compliance of 0.984. Under
hypothesis 2 its admissible candidates must be uniformly good. Concretely:

1. Its spread of held-out coherent compliance among admissible candidates will be
   **below 0.35** — closer to Llama's 0.219 than to Qwen3.5's ~0.70.
2. Its median admissible candidate will exceed **0.90** compliance.
3. Its admissible set will be **smaller than 19** (so set size alone cannot
   explain the stability).

Hypothesis 2 is falsified if the spread comes out above 0.5, or if the median
admissible candidate is below 0.8.

## Outcome — all three predictions held

`refusal_selection_audit_qwen2p5_7b.json`, run after this file was written:

| prediction | bar | observed | |
|---|---|---|---|
| 1. spread of held-out compliance among admissible candidates | < 0.35 | **0.016** | ✅ |
| 2. median admissible candidate | > 0.90 | **1.000** | ✅ |
| 3. admissible-set size | < 19 | **3** | ✅ |

The falsifier (spread > 0.5, or median < 0.8) was not triggered.

Qwen2.5-7B-Instruct is the model that discriminates the two hypotheses, and it
discriminates them sharply. It has the **thinnest** admissible set of the four —
3 of 120 candidates, against Llama's 19 — and the second-lowest instability.
Hypothesis 1 predicts it should be the least stable model; it is nearly the most
stable. What its three survivors have in common is that all three work: held-out
coherent compliance 0.984, 1.000, 1.000.

Full picture across the four models:

| model | admissible | spread among admissible | median | battery score CV |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 19 | 0.219 | 0.984 | 0.004 |
| Qwen2.5-7B-Instruct | 3 | 0.016 | 1.000 | 0.009 |
| Qwen3.5-4B | 4 | 0.719 | 0.828 | 0.174 |
| Qwen3.5-9B | 7 | 0.688 | 0.609 | 0.323 |

Candidate-quality spread tracks the battery's score CV at r = 0.87; admissible-set
size tracks it at −0.47, with the wrong sign for hypothesis 1. **Four models is far
too few for either correlation to carry weight on its own** — the evidence here is
the pre-registered prediction on the discriminating model, not the correlation.

## What this means for the method

The selection *objective* is not the problem. On every model where held-out effect
varies, the objective ranks candidates correctly (Spearman −0.86 to −0.87 between
the objective and held-out compliance), it picks the best available candidate on
the audited split (gap 0.0 on all three audited models), and it is stable across
disjoint validation halves (Spearman 0.96–1.00).

The fragility is upstream of the ranking, in the admissibility filter. It keeps
3–19 of 120 candidates, and *which* candidates survive depends on the extraction
sample. On Qwen3.5-9B the strongest direction (layer 24, held-out compliance 0.97)
sits at KL 0.069 against a hard 0.1 cutoff — a defensible resample pushes it out,
and the fallback at layer 13 costs 23 points of effect. A practitioner reporting
"the refusal direction of Qwen3.5-9B" gets layer 13 or layer 24 depending on which
260 prompts they extracted from.
