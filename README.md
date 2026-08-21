# StressKit

**The stability harness for mechanistic interpretability claims.**

Wrap any discovery method in one call. Get back a graded, machine-readable
Stability Card that records whether the finding survives seeds, resampling,
prompt templates, hyperparameters, and null controls.

[![CI](https://github.com/Arth-Singh/Stresskit/actions/workflows/ci.yml/badge.svg)](https://github.com/Arth-Singh/Stresskit/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## Why

Interpretability findings routinely do not survive defensible variation in how
they were produced:

- Circuit-level claims flip on **73% of pairs** of defensible analytic specifications ([arXiv:2608.13754](https://arxiv.org/abs/2608.13754))
- Circuits found on bootstrap-resampled data overlap at **Jaccard ≈ 0.56** ([arXiv:2510.00845](https://arxiv.org/abs/2510.00845))
- In SAE autointerp scoring, **pipeline choices explain more variance than the SAE architecture under comparison** ([arXiv:2607.19386](https://arxiv.org/abs/2607.19386))
- SAEs with **frozen random decoders** match trained SAEs on standard evals ([arXiv:2602.14111](https://arxiv.org/abs/2602.14111))
- Different prompt templates activate **structurally different circuits** for the "same" task ([arXiv:2606.16920](https://arxiv.org/abs/2606.16920)); multiple near-disjoint circuits each perform the task perfectly ([arXiv:2605.12671](https://arxiv.org/abs/2605.12671))

Each of those papers calls for standard stability reporting. StressKit is that
standard: the published protocols, as a library, with shared thresholds and a
common artifact.

It also covers the instruments the field adopted in 2026: natural-language
activation readers (Activation Oracles / LatentQA, [arXiv:2512.15674](https://arxiv.org/abs/2512.15674))
and Jacobian-lens readouts ([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)),
whose documented failure modes — hallucination on empty inputs, best-of-N
prompt selection, concept-specific blind spots ([arXiv:2607.23379](https://arxiv.org/abs/2607.23379)) —
are reliability failures. StressKit tests the instrument, not just the finding.

## Install

```bash
pip install git+https://github.com/Arth-Singh/Stresskit.git   # numpy only; imports as `import stresskit`
```

<sub>PyPI release as `stress-kit` is pending.</sub>

## Quickstart

Wrap your discovery method as `(data, seed, config) -> Finding`, then stress it:

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
    null_data=control_dataset,            # where the effect should not exist
    model="gpt2-small", task="IOI", method="EAP-IG",
)

print(result)                              # StressResult(grade='B', ...)
print(result.to_markdown())                # human-readable stability card
result.card.save("stability_card.json")    # machine-readable artifact
```

**Already have runs?** Skip the wrapper entirely — post-hoc mode grades
findings you already have, from result files, sweep logs, or old pickles:

```python
findings = [sk.circuit(edges, score=faith) for edges, faith in my_saved_runs]
print(sk.from_findings(findings, model="gpt2-small", task="IOI").to_markdown())
```

**How many runs does a verdict need?** Papers report stability from 5–10
runs; `verdict_trace` regrades random subsets of your runs at every size and
reports the n at which the verdict stops being a coin flip — with no new
runs:

```python
trace = sk.verdict_trace(findings)            # or result.verdict_trace()
print(sk.verdict_trace_markdown(trace))       # grade distribution vs n, settled_n
```

**Using SAELens or EAP-IG?** One call:

```python
from stresskit.adapters import sae_lens, eap

sae_lens.stability([sae_seed0, sae_seed1, sae_seed2])   # graded SAE report:
# seed-consistency MCC, near-duplicate fraction, excess over the
# random-decoder noise floor

findings = [eap.finding_from_json(p) for p in glob("circuits/*.json")]
sk.from_findings(findings)                               # from saved eap graphs
```

A self-contained CPU demo runs in seconds:

```bash
python examples/quickstart_toy.py
```

It stress-tests one discovery method twice — on a real effect (grade A) and on
a pure-noise null where the method still returns confident-looking features
(grade C/D). Only the battery tells them apart.

## Checks

| Check | Metric | Default bar | Protocol source |
|---|---|---|---|
| **Structural stability** | mean pairwise Jaccard of component sets across runs, with bootstrap 95% CI | ≥ 0.8 | arXiv:2510.00845 |
| **Claim stability** | modal claim share π\* + flip rate, with bootstrap 95% CI | π\* ≥ 0.8 | arXiv:2608.13754 |
| **Score stability** | coefficient of variation of the quality score | ≤ 0.25 | arXiv:2510.00845 |
| **Beats random** | overlap vs. size-matched random null (Monte-Carlo over the observed size distribution; analytic *k*/(2*N*−*k*) kept as cross-check) | ≥ 3× | arXiv:2608.13754, 2602.14111 |
| **Specificity** | stability on real data vs. a null control where the effect should not exist, with a two-sample bootstrap 95% CI | ≥ 1.5× | arXiv:2606.00033 |

Every check carries a 95% CI. A CI that straddles its bar marks the check
**undecided in either direction** — the point estimate still grades, but the
verdict is reported low-confidence and the card says so out loud.

Grades: **A** all applicable checks pass · **B** at least half · **C** at least
one · **D** none, or indistinguishable from random. Thresholds are configurable
(`sk.Thresholds`), but the defaults follow the published proposals so a grade
means the same thing in every paper.

Battery axes run one-at-a-time around your base configuration, so run counts
stay linear and attribution stays legible: `seeds` (finders that ignore their
seed are detected and flagged), `bootstrap`, `templates`, `hyperparams`.
Claims can be natural language — pass any `(a, b) -> bool` judge as
`claim_equiv=` (see `stresskit.judges`).

Expensive finder? `stress(..., cache_dir="...", cache_key="v1")` caches every
run and resumes a battery for free; bump the key when data or method change.
Template variants drawn from a different component namespace can declare
`meta["universe"]` — they then compare on claim and score, never on Jaccard
(which is undefined across universes).

## The Stability Card

`result.card` is a versioned JSON artifact
([schema](src/stresskit/schemas/stability_card_v0.json)) recording the claim,
the battery, every metric, the verdict, and provenance.

```bash
stresskit render stability_card.json        # markdown for your appendix
stresskit badge  stability_card.json -o badge.json
stresskit verify stability_card.json        # re-derive checks + grade from the
                                            # card's own metrics (auditor mode)
```

Host `badge.json` anywhere public and embed a live badge:

```markdown
![stability](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/you/repo/main/badge.json)
```

→ ![stability](https://img.shields.io/badge/stability-A%20%C2%B7%20J%3D0.92-brightgreen) / ![stability](https://img.shields.io/badge/stability-D%20%C2%B7%20J%3D0.18-red)

## Oracle reliability

For Activation Oracles, verbalizers, and introspection models used as
interpretability instruments:

```python
from stresskit import OracleProbe, stress_oracle, blind_spot_matrix, judges

probes = [
    OracleProbe(
        name="taboo-gold", concept="gold", expected="gold",
        questions=["What is the secret word?",           # ≥2 paraphrases enable
                   "Which word is the model hiding?"],   # the prompt-sensitivity check
        exemplars=[acts_hint, acts_refusal, acts_think],  # independent captures
    ),
    OracleProbe(  # null control: the honest answer is to abstain
        name="null-random", kind="null",
        questions=["What is the secret word?"],
        exemplars=[random_acts_1, random_acts_2],
    ),
]

report = stress_oracle(ask_fn, probes, judge=judges.token_f1(0.5))
print(report.to_markdown())
```

Four checks, each targeting a documented failure mode: **answer consistency**
(decomposed into decoding, capture, and phrasing components), **known-answer
accuracy** (with Wilson 95% CI), **prompt sensitivity** (the max−min accuracy
gap that best-of-N prompt reporting hides), and **null hallucination**
(confident assertions on control activations). `blind_spot_matrix` runs the
cross-oracle × concept protocol of arXiv:2607.23379 and flags oracles that
selectively fail on specific concepts.

## Adapters

StressKit contains no discovery methods; adapters bridge from the tools you
already use (`stresskit.adapters`):

| adapter | bridges from | provides |
|---|---|---|
| `sae_lens` | [SAELens](https://github.com/jbloomAus/SAELens) SAEs / decoder tensors | `stability()` — one-call graded report (seed MCC, redundancy, noise-floor excess) |
| `eap` | [EAP-IG](https://github.com/hannamw/EAP-IG) graphs and `to_json` exports | `graph_to_finding`, `finding_from_json`, `finder_from_graph_fn` |
| `sae` | any SAE decoder matrix | `seed_consistency` (MCC via optimal matching), `redundancy_audit` |
| `transformer_lens` | TransformerLens pipelines | edge selection, layer-band claims, `Finding` conversion |
| `activation_oracles` | [adamkarvonen/activation_oracles](https://github.com/adamkarvonen/activation_oracles) result files | `reliability_report` straight from saved eval JSON, no GPU |
| `jlens` | [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) readouts | ranked-readout findings, `junk_share`, workspace-band hit ranks |

Ranked outputs (lens readouts, top-k feature lists) are compared with
rank-biased overlap (`stresskit.metrics.rbo`, Webber et al. 2010) rather than
set Jaccard — the head of the list is the claim.

## Reporting checklist

The minimum reporting checklist of arXiv:2607.19386, generalized and
executable — every field is a documented source of result variance:

```bash
stresskit report --model gpt2-small --task IOI --method EAP-IG \
    --n-seeds 10 --metric logit_diff -o appendix_checklist.md
```

Unanswered fields render as **NOT REPORTED ⚠️** — flagged, never hidden.

## Reference batteries

Published findings, default battery, default thresholds. Full analysis in
[`references/`](references/README.md).

| finding | method | verdict | headline |
|---|---|---|---|
| IOI circuit, GPT-2 small | attribution patching | **A**, low confidence | point estimate clears 0.8 but the 95% CI straddles it at 45 runs — the grade is not certifiable |
| Greater-Than circuit, GPT-2 small | attribution patching | **B**, high confidence | robustly stable (J CI [0.83, 0.94]) yet fails specificity at 1.15× — the null "circuit" is just as stable |
| Activation Oracles, Qwen3-8B taboo | upstream `run_verbalizer` | **D** (two mixtures), **C** (one) | consistency 0.94 across captures vs 0.31 across phrasings; ≥89% fabrication even on a null probe that invites abstention |
| J-lens workspace readouts, Qwen3.5-4B | released pre-fitted lens | **C** | band claim stable (π\* = 0.90); which items hit is not (J = 0.45), and a derangement null is *more* stable than the real finding |

## Design principles

1. **Wrap, don't replace.** StressKit instruments the pipeline you already have.
2. **Every metric has a citation.** Defaults follow the published proposals; a shared bar is the point.
3. **Cheap by default.** Core is numpy-only; the harness adds no compute beyond re-running your finder.
4. **Honest degradation.** Skipped axes are noted on the card, never silently dropped.

## Roadmap

- Reference cards for the refusal direction and additional model scales
- Run caching and parallel execution for expensive finders
- Crossed-grid batteries (full multiverse analysis) with budget caps
- Trajectory batteries: stability across long generations and agent rollouts
- `stresskit verify`: recompute a card from its config hash

## Contributing

Issues and PRs welcome — especially adapters for your pipeline, replication
cards for published findings, and arguments about the default thresholds.

## Citing

```bibtex
@software{stresskit2026,
  title  = {StressKit: a stability harness for mechanistic interpretability claims},
  author = {Singh, Arth},
  year   = {2026},
  url    = {https://github.com/Arth-Singh/Stresskit}
}
```

MIT license.
