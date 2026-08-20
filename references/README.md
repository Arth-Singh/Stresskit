# Reference batteries

Stability cards and reliability reports for published findings, produced with
the default battery and thresholds. Each battery has a runner script in this
directory; artifacts (JSON card, markdown render, badge) live in `cards/`.

## GPT-2 small / IOI — attribution patching

**Grade B.** Head-level attribution patching over the 144 attention heads of
GPT-2 small on Indirect Object Identification (ABC corruption), scored by
denoising faithfulness. Battery: seeds, bootstrap, ABBA/BABA template split,
top-k sweep, random-answer null control.

| check | value | pass |
|---|---|---|
| structural stability | J = 0.764, 95% CI [0.68, 0.87] | ❌ (bar 0.8) |
| claim stability ("late layers") | π\* = 1.00 | ✅ |
| score stability (faithfulness CV) | 0.048 | ✅ |
| beats random | 13.9× | ✅ |
| specificity (null control) | 1.38× | ❌ (bar 1.5×) |

Notes:

- The specificity failure is the substantive result: on a null task (answer
  tokens are random names unrelated to the prompt) the finder still returns
  fairly stable "circuits" (null-control J = 0.55). Attribution concentrates
  on name-processing heads whether or not the claimed effect exists.
- The random-names null is conservative — name-mover heads legitimately
  process names. A scrambled-prompt null would be stricter and would likely
  widen the specificity ratio.
- Score-variance decomposition matches the stability literature:
  hyperparameter choice 57%, prompt template 36%, seeds 4%, bootstrap 3%.

Artifacts: [`cards/ioi_gpt2_small.md`](cards/ioi_gpt2_small.md) ·
[`cards/ioi_gpt2_small.json`](cards/ioi_gpt2_small.json) ·
runner [`run_ioi_gpt2_card.py`](run_ioi_gpt2_card.py).

## Activation Oracles / Qwen3-8B taboo — oracle reliability

**Grade D, all three oracle mixtures.** The pre-trained oracles from
[adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)
(arXiv:2512.15674), run through the upstream `run_verbalizer` stack: three
training mixtures × five taboo subject models × 4 question paraphrases ×
3 elicitation prompts × 3 sampled repeats, plus a base-model null control.

| oracle mixture | consistency | known accuracy | prompt sensitivity | null hallucination | grade |
|---|---|---|---|---|---|
| full mixture | 0.45 | 0.45 [0.38, 0.52] | 0.47 | 1.00 | D |
| LatentQA-only | 0.39 | 0.12 | 0.20 | 1.00 | D |
| classification-only | 0.20 | 0.23 | 0.31 | 1.00 | D |

Notes:

- **The instrument is prompt-dominated.** Consistency decomposition for the
  best mixture: agreement 0.94 across decoding repeats, 0.93 across
  activation captures, 0.31 across question phrasings. The oracle reads the
  same activation reliably and changes its story when the question is
  reworded ("What is the secret word?" → *moon* ✓; "Which word is this model
  trying to avoid saying?" → *"answer"*, *"clue"*).
- **Null hallucination at ceiling.** On base-model activations, where no
  secret exists, every mixture asserted one in 100% of answers ("garden",
  "password", "mystery"). The question set presupposes a secret exists,
  which inflates this rate; a fair abstention test needs non-presupposing
  phrasings. The battery makes the question set an explicit, stressable
  input rather than a hidden choice.
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
