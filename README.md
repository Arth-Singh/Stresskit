# StressKit

**The stability harness for mechanistic interpretability claims.**
*pytest + codecov for interp findings: wrap your discovery method in one call, get back a graded, machine-readable Stability Card.*

[![CI](https://github.com/Arth-Singh/Stresskit/actions/workflows/ci.yml/badge.svg)](https://github.com/Arth-Singh/Stresskit/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Why

A wave of 2025–2026 papers showed that interpretability findings routinely **do not survive defensible variation** in how they were produced:

- Circuit-level claims flip on **73% of pairs** of defensible analytic specifications ([Explanation Multiplicity, arXiv:2608.13754](https://arxiv.org/abs/2608.13754))
- Circuits found on bootstrap-resampled data overlap at **Jaccard ≈ 0.56**; edge-score signal-to-noise falls **below 1** under EAP approximations ([arXiv:2510.00845](https://arxiv.org/abs/2510.00845))
- In SAE autointerp scoring, **pipeline choices explain more variance than the SAE architecture being compared** ([arXiv:2607.19386](https://arxiv.org/abs/2607.19386))
- SAEs with **frozen random decoders** match fully-trained SAEs on standard interpretability evals ([arXiv:2602.14111](https://arxiv.org/abs/2602.14111))
- Different prompt templates activate **structurally different circuits** for the "same" task ([arXiv:2606.16920](https://arxiv.org/abs/2606.16920)); multiple near-disjoint circuits each perform the task perfectly ([arXiv:2605.12671](https://arxiv.org/abs/2605.12671))

Every one of those papers ends with a version of *"researchers should report stability metrics."* None of them shipped the tool. **StressKit is that tool.**

And it's built for what the field studies **now**, not just circuits and SAEs. The fastest-growing 2026 interfaces are *learned readers that answer in natural language* — Activation Oracles / LatentQA ([arXiv:2512.15674](https://arxiv.org/abs/2512.15674)), verbalizers, introspection adapters — whose documented failure modes are precisely reliability failures: they "frequently produce an answer even when confidence is low" (the AO paper's own limitations section), results get reported with the best of N hand-written oracle prompts, and fine-tuned oracles develop **concept-specific blind spots** — selectively failing on the very concept they were trained on ([arXiv:2607.23379](https://arxiv.org/abs/2607.23379)). `stresskit.oracle` is the first standard harness for those checks.

## Install

```bash
pip install stresskit          # numpy only
pip install "stresskit[full]"  # + scipy (optimal SAE feature matching)
```

## 60-second quickstart

Wrap your discovery method as a function `(data, seed, config) -> Finding`, then stress it:

```python
import stresskit as sk

def finder(data, seed, config) -> sk.Finding:
    edges = my_circuit_discovery(data, seed=seed, **config)   # your existing code
    return sk.circuit(
        edges,
        claim=layer_band(edges),          # coarse qualitative claim label
        score=faithfulness(edges),        # scalar quality metric
        universe_size=N_EDGES,            # enables the random-null comparison
    )

result = sk.stress(
    finder, dataset,
    battery=["seeds", "bootstrap", "templates", "hyperparams"],
    n_runs=10,
    config={"threshold": 0.1, "ablation": "patching"},
    templates={"ABBA": abba_prompts, "BABA": baba_prompts},
    hyperparams={"threshold": [0.05, 0.2], "ablation": ["mean", "zero"]},
    model="gpt2-small", task="IOI", method="EAP-IG",
)

print(result)                    # StressResult(grade='B', runs=31, jaccard=0.61, ...)
print(result.to_markdown())      # full human-readable stability card
result.card.save("stability_card.json")   # the machine-readable artifact
```

No GPU handy? The self-contained demo runs in seconds:

```bash
python examples/quickstart_toy.py
```

It stress-tests the same discovery method twice — once on a real effect (grade **A**) and once on a pure-noise null where the method *still returns eight confident-looking features with a claim attached* (grade **C/D**). Only the battery tells them apart.

## What gets measured

| Check | Metric | Default bar | Protocol source |
|---|---|---|---|
| **Structural stability** | mean pairwise Jaccard of the component sets across runs (with bootstrap 95% CI; size-mismatched runs excluded from grading) | ≥ 0.8 | arXiv:2510.00845 |
| **Claim stability** | modal claim share π\* (≙ filability at α = 0.2) + flip rate (with bootstrap 95% CI) | π\* ≥ 0.8 | arXiv:2608.13754 |
| **Score stability** | coefficient of variation of the quality score | ≤ 0.25 | arXiv:2510.00845 |
| **Beats random** | overlap vs. the size-matched random null *J* ≈ *k*/(2*N*−*k*) | ≥ 3× | arXiv:2608.13754, 2602.14111 |
| **Specificity** | stability on real data vs. a `null_data=` control where the effect shouldn't exist (dead-salmon detector) | ≥ 1.5× | Adebayo-style sanity checks; arXiv:2606.00033 |

Claims can be **natural language**: pass `claim_equiv=` any `(a, b) -> bool` judge (see `stresskit.judges`: normalized match, token-F1, AO-style containment — or plug in your own embedding/LLM judge) and semantically equivalent phrasings count as one claim class.

Grades: **A** all applicable checks pass · **B** at least half · **C** at least one · **D** none / indistinguishable from random. Thresholds are configurable (`sk.Thresholds(...)`) but the defaults follow the published proposals, on purpose: a shared bar is the point.

The battery axes (one-at-a-time around your base configuration, so run counts stay linear and attribution stays legible):

- `seeds` — discovery seed on identical data (StressKit detects and flags finders that silently ignore their seed)
- `bootstrap` — dataset resampling: finite-sample fragility
- `templates` — alternative prompt templates / paraphrases / corpora
- `hyperparams` — thresholds, ablation operators, metrics, anything in your config

> Interpretation note: sweeping a hyperparameter that changes the finding's *size* (e.g. top-k) mechanically bounds Jaccard below 1. Check the per-axis breakdown on the card before reading the pooled number.

## The Stability Card

`result.card` is a versioned JSON artifact ([schema](src/stresskit/schemas/stability_card_v0.json)) recording the claim, the battery, every metric, the verdict, and provenance. Attach it to your paper, your repo, your SAE release.

```bash
stresskit render stability_card.json        # markdown for your appendix
stresskit badge  stability_card.json -o badge.json
```

Host `badge.json` anywhere public and embed a live badge:

```markdown
![stability](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/you/repo/main/badge.json)
```

→ ![stability](https://img.shields.io/badge/stability-A%20%C2%B7%20J%3D0.92-brightgreen) / ![stability](https://img.shields.io/badge/stability-D%20%C2%B7%20J%3D0.18-red)

## Oracle reliability (`stresskit.oracle`)

If you use an Activation Oracle, verbalizer, or introspection model as an interpretability *instrument*, StressKit tests the instrument:

```python
from stresskit import OracleProbe, stress_oracle, blind_spot_matrix, judges

probes = [
    OracleProbe(
        name="taboo-gold", concept="gold", expected="gold",
        questions=["What is the secret word?",          # ≥2 paraphrases enable the
                   "Which word is the model hiding?"],  # prompt-sensitivity check
        exemplars=[acts_hint, acts_refusal, acts_think], # independent captures
    ),
    OracleProbe(  # null control: honest answer is to abstain
        name="null-random", kind="null",
        questions=["What is the secret word?"],
        exemplars=[random_acts_1, random_acts_2],
    ),
]

report = stress_oracle(ask_fn, probes, judge=judges.token_f1(0.5))
#   ask_fn(exemplar, question, seed) -> str   — your oracle call, verbatim
print(report.to_markdown())    # graded A–D: consistency, known-answer accuracy,
report.save("oracle_report.json")  # prompt sensitivity, null hallucination
```

Four checks, each targeting a documented AO failure mode: **answer consistency** across paraphrases/captures/repeats, **known-answer accuracy**, **prompt sensitivity** (the max−min accuracy gap that "best-of-N oracle prompts" reporting hides), and **null hallucination** (confident assertions on control activations).

`blind_spot_matrix({name: ask_fn}, probes)` runs the cross-oracle × concept protocol of arXiv:2607.23379 and flags oracles that selectively fail on specific concepts — the "reader learned not to read" failure. See `examples/oracle_reliability.py` for a runnable demo of all of it.

## SAE auditing

Two checks the SAE literature keeps asking for, as one-liners (`stresskit.adapters.sae`):

```python
from stresskit.adapters import sae

# 1. Do your features replicate across training seeds? (MCC via optimal matching)
sae.seed_consistency([W_dec_run1, W_dec_run2, W_dec_run3])
# {'mean_mcc': 0.71, 'min_mcc': 0.68, ...}   <- report this with your SAE release

# 2. How many of your features are near-duplicates of each other?
sae.redundancy_audit(W_dec, threshold=0.9)
# {'n_redundant_features': 214, 'redundant_fraction': 0.013, 'n_clusters': 87, ...}
```

`W_dec` is `(n_features, d_model)` — rows are feature directions (SAELens convention). See `examples/sae_audit.py`.

## Minimum Reporting Checklist

The checklist from arXiv:2607.19386, generalized and executable — every field is a documented source of result variance:

```bash
stresskit report --model gpt2-small --task IOI --method EAP-IG \
    --n-seeds 10 --metric logit_diff -o appendix_checklist.md
```

Unanswered fields render as **NOT REPORTED ⚠️** — flagged, never hidden.

## Reference cards (real models)

The first reference battery ran on **GPT-2 small / IOI** with a head-level attribution-patching finder (4× RTX PRO 6000; the full 30-run battery takes **~11 seconds**). Verdict: **grade B** — and the two failed checks are precisely what the 2026 stability literature predicted:

| check | value | pass |
|---|---|---|
| structural stability | J = 0.764, 95% CI [0.68, 0.87] | ❌ (bar: 0.8) |
| claim stability ("late layers") | π\* = 1.00 | ✅ |
| score stability (faithfulness CV) | 0.048 | ✅ |
| beats random | 13.9× | ✅ |
| **specificity (null control)** | 1.38× | ❌ (bar: 1.5×) |

The specificity failure is the interesting one: given a **null task** (answer tokens are random names unrelated to the prompt), the finder still returns fairly stable "circuits" (null-control J = 0.55) — attribution concentrates on name-processing heads whether or not the claimed effect exists. Caveat honestly noted: random *names* are a conservative null (name-movers legitimately process them); a scrambled-prompt null would be stricter. Score-variance decomposition also reproduces the literature: hyperparameter choice (57%) and prompt template (36%) dwarf seed noise (4%).

Full artifacts: [`references/cards/ioi_gpt2_small.md`](references/cards/ioi_gpt2_small.md) · [JSON](references/cards/ioi_gpt2_small.json) · reproduce with [`references/run_ioi_gpt2_card.py`](references/run_ioi_gpt2_card.py).

## Design principles

1. **Wrap, don't replace.** StressKit contains zero discovery methods. It instruments TransformerLens / EAP-IG / SAELens / nnsight pipelines you already have (see `stresskit.adapters`).
2. **Every metric has a citation.** Nothing in the battery is invented here; the defaults are the published proposals so that "grade B on the default battery" means the same thing in every paper.
3. **Cheap by default.** Core is numpy-only; the harness adds no compute beyond re-running *your* finder; the analysis layer runs on CPU.
4. **Honest degradation.** Axes you don't feed (no templates, unsized data) are skipped *and noted on the card* — silent coverage gaps are how fields fool themselves.

## Roadmap

- [ ] Reference reproductions on real models: stability cards for IOI, Greater-Than, the refusal direction, and an oracle reliability report for the open-source Activation Oracles ([adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles)) — the seed of a public card registry
- [ ] Run caching + parallel execution for expensive finders (resume a battery for free)
- [ ] `stresskit.adapters.sae_lens` — one-call battery for SAELens training runs
- [ ] Crossed-grid mode (full multiverse à la arXiv:2608.13754) with budget caps
- [ ] Trajectory batteries: stability of steering/probes/oracle readouts across long generations and agent rollouts
- [ ] `verify` subcommand: recompute a card from its config hash (auditor mode)

## Contributing

Issues and PRs welcome. Especially wanted: adapters for your lab's pipeline, replication cards for published findings, and disagreements about the default thresholds (open an issue — the bar should be argued in public).

## Citing

```bibtex
@software{stresskit2026,
  title  = {StressKit: a stability harness for mechanistic interpretability claims},
  author = {Singh, Arth},
  year   = {2026},
  url    = {https://github.com/Arth-Singh/Stresskit}
}
```

MIT licensed. Built on the protocols of arXiv:2510.00845, 2608.13754, 2607.19386, 2602.14111, 2606.16920, 2602.14687 — read them.
