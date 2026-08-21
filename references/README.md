# Reference batteries

Stability cards and reliability reports for published findings, produced with
the default battery and thresholds. Each battery has a runner script in this
directory; artifacts (JSON card, markdown render, badge) live in `cards/`.

## GPT-2 small / IOI — attribution patching

**Grade A — but low confidence (the grade is not certifiable).** Head-level
attribution patching over the 144 attention heads of GPT-2 small on Indirect
Object Identification (ABC corruption), scored by denoising faithfulness.
Battery: seeds, bootstrap, ABBA/BABA template split, top-k sweep,
random-answer null control. 45 runs.

| check | value | 95% CI | pass |
|---|---|---|---|
| structural stability | J = 0.829 | [0.781, 0.869] | ⚠️ point estimate clears 0.8, CI straddles it |
| claim stability ("late layers") | π\* = 1.00 | [1.00, 1.00] | ✅ |
| score stability (faithfulness CV) | 0.033 | [0.018, 0.047] | ✅ |
| beats random | 14.7× | — | ✅ |
| specificity (null control) | 1.54× | — | ✅ (barely) |

This is the tool's sharpest result. The most-cited circuit in
interpretability lands a point-estimate **A**, but its structural-stability
CI still straddles the field's own 0.8 bar after 45 runs — so StressKit marks
the grade **low-confidence** and refuses to certify it. "IOI is a stable
circuit" is not a statement the data settles at the proposed threshold.

Notes:

- Specificity is itself fragile: it *fails* at 1.38× with n=6 runs and
  *passes* at 1.54× with n=20 — the margin sits right on the 1.5× bar and
  moves with the null estimate. Treat "passes specificity" here as
  undecided, not settled. On the null task (answer tokens are random names)
  the finder still returns fairly stable "circuits" (null J ≈ 0.54).
- The random-names null is conservative — name-mover heads legitimately
  process names; a scrambled-prompt null would be stricter.
- Score variance is dominated by analytic choices (hyperparameters and
  template), not seeds — matching the stability literature.

Artifacts: [`cards/ioi_gpt2_small.md`](cards/ioi_gpt2_small.md) ·
[`cards/ioi_gpt2_small.json`](cards/ioi_gpt2_small.json) ·
runner [`run_ioi_gpt2_card.py`](run_ioi_gpt2_card.py).

## GPT-2 small / Greater-Than — attribution patching

**Grade B — high confidence.** Same finder family on the Greater-Than task
(arXiv:2305.00586, YY→01 corruption), scored by recovered probability-mass
difference. 45 runs.

| check | value | 95% CI | pass |
|---|---|---|---|
| structural stability | J = 0.892 | [0.834, 0.943] | ✅ (CI clears 0.8) |
| claim stability ("late layers") | π\* = 0.98 | [0.933, 1.00] | ✅ |
| score stability (CV) | 0.002 | [0.001, 0.002] | ✅ |
| beats random | 15.8× | — | ✅ |
| specificity (null control) | 1.15× | — | ❌ (bar 1.5×) |

Notes:

- Unlike the IOI card, every stability check is *robust* — the CIs clear
  their bars — so this is a high-confidence verdict. The circuit is
  genuinely, reproducibly stable.
- And it genuinely fails specificity, unambiguously: scored against a random
  threshold unrelated to the prompt (no effect exists), the finder returns
  near-identical stable "circuits" (null J = 0.779, specificity 1.15×).
  Across both circuit cards, attribution patching recovers
  *task-input-processing* heads with high stability whether or not the
  claimed effect is present. **Specificity, not stability, is the
  discriminating check for this method family** — the headline finding of
  these reference batteries.
- Variance decomposition again puts analytic choices first: hyperparameter
  choice dominates, seeds and bootstrap are minor.

Artifacts: [`cards/greater_than_gpt2_small.md`](cards/greater_than_gpt2_small.md) ·
[`cards/greater_than_gpt2_small.json`](cards/greater_than_gpt2_small.json) ·
runner [`run_greater_than_gpt2_card.py`](run_greater_than_gpt2_card.py).

## Activation Oracles / Qwen3-8B taboo — oracle reliability

**Grade D, all three oracle mixtures.** The pre-trained oracles from
[adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)
(arXiv:2512.15674), run through the upstream `run_verbalizer` stack: three
training mixtures × five taboo subject models × 4 question paraphrases ×
3 elicitation prompts × 3 sampled repeats, plus a base-model null control.

| oracle mixture | consistency | known accuracy | prompt sensitivity | null hallucination | grade |
|---|---|---|---|---|---|
| full mixture | 0.45 | 0.45 [0.38, 0.52] | 0.47 | 0.94 [0.88, 1.00] | D |
| LatentQA-only | 0.37 | 0.09 | 0.09 | 0.89 [0.85, 0.99] | C\* |
| classification-only | 0.20 | 0.24 | 0.31 | 0.93 [0.85, 0.99] | D |

\* The LatentQA-only C is a cautionary artifact, not a pass: its
`prompt_sensitivity` check clears the bar (0.09 ≤ 0.20) only because the
oracle is *uniformly wrong* (accuracy 0.09) — a low accuracy gap across
phrasings means "consistently unable" just as much as "phrasing-robust".
The check measures spread, not correctness; read it next to accuracy, never
alone.

Notes:

- **The instrument is prompt-dominated.** Consistency decomposition for the
  best mixture: agreement 0.94 across decoding repeats, 0.93 across
  activation captures, 0.31 across question phrasings. The oracle reads the
  same activation reliably and changes its story when the question is
  reworded ("What is the secret word?" → *moon* ✓; "Which word is this model
  trying to avoid saying?" → *"answer"*, *"clue"*).
- **Null hallucination at ceiling — and presupposition is not the excuse.**
  On base-model activations, where no secret exists, the presupposing
  questions ("What is the secret word?") drew a fabricated answer 100% of
  the time ("garden", "password", "mystery"). A second null probe asks the
  non-presupposing form — "Is there a secret word hidden in this text? If
  there is not, say so." — and the best oracle still fabricated one in
  88.9% of answers. The failure is the instrument's, not the question's.
- **Concept-specific blind spots.** The cross-oracle matrix (protocol of
  arXiv:2607.23379) flags the classification-only oracle on *flag*
  (0.06 vs 0.32 for the other oracles) and *gold* (0.00 vs 0.24).
- Decoding used the oracles' own sampling defaults and free-text questions
  without the upstream single-word-answer prefix, so accuracies are not
  comparable to the paper's headline numbers; they measure the instrument
  in its demo-notebook usage mode.

Artifacts: [`cards/ao_qwen3_full-mixture.md`](cards/ao_qwen3_full-mixture.md)
· [`cards/ao_qwen3_blind_spots.json`](cards/ao_qwen3_blind_spots.json) ·
raw responses `cards/ao_qwen3_raw_*.json` ·
runner [`run_oracle_reliability_qwen3.py`](run_oracle_reliability_qwen3.py).

## Jacobian-lens readouts / Qwen3.5-4B — workspace hit criterion

**Grade C — low confidence (claim and score checks undecided at n=6).** The released pre-fitted lens from
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) on its
own evaluation sets. Finding under test, per the upstream hit criterion: the
latent intermediate appears at lens rank ≤ 5 at some layer (word-like tokens,
the `mask_display` view practitioners read). Battery: item subsampling and
bootstrap, k / band / position / masking sweeps, association-set
generalization, and a derangement null (same prompts, rotated targets).

| check | value | pass |
|---|---|---|
| structural stability (which items hit) | J = 0.448 | ❌ |
| claim stability ("mid-to-late band") | π\* = 0.90 | ✅ |
| score stability (pass@5 CV) | 0.361 | ❌ |
| beats random | 3.9× | ✅ |
| specificity (derangement null) | 0.78× | ❌ |

Notes:

- The qualitative claim is solid: whenever the intermediate is readable, it
  reads out in the mid-to-late layers (claim stable at 0.90). Everything
  quantitative around it is not: *which* items hit fluctuates (J = 0.45),
  and pass@5 swings from 0.28 (multihop facts) to 0.04 (association
  vignettes — the flagship "workspace" demo class). The evaluation
  distribution owns 53% of score variance, analytic knobs another 42%.
- The specificity failure is instructive: with targets rotated to the wrong
  items (no effect exists), the hit-set is *more* stable (J = 0.57) than the
  real one — frequently-emitted readout tokens ("Ocean", "Sea") match wrong
  targets consistently. Hit-based metrics reward frequency artifacts.
- Raw (unmasked) top-10 readouts are 72% non-word-like tokens on this
  model, consistent with early community reports of junk-dominated raw
  J-space readouts; the masked/raw choice alone moves pass@5 by ~2×.
- Scale caveat: the paper's demonstrations center on a 27B model; this card
  uses the smallest released lens (4B). The battery measures the released
  artifact as an instrument, not the paper's headline model.

Artifacts: [`cards/jlens_qwen3p5_4b.md`](cards/jlens_qwen3p5_4b.md) ·
[`cards/jlens_qwen3p5_4b.json`](cards/jlens_qwen3p5_4b.json) ·
runner [`run_jlens_stability_qwen.py`](run_jlens_stability_qwen.py).
