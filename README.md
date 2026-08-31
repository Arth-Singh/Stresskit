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

## See it work first (30 seconds, no GPU)

```bash
stresskit demo
```

One toy discovery method, stressed twice: on data with a real effect
(**grade A**, J=0.92) and on **pure noise**, where it still returns eight
confident "responsible features" with a claim attached, every run
(**grade C**, J=0.18). The two outputs look identical; only the battery
tells them apart. `stresskit demo --html cards/` writes both stability
cards as shareable pages.

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

Or grade a JSONL sweep log with zero wrapper code — one JSON object per run
(`{"components": [...], "claim": "...", "score": ..., "axis": "seeds"}`,
field names remappable):

```python
print(sk.from_jsonl("sweep.jsonl", null_path="sweep_null.jsonl").to_markdown())
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
stresskit render stability_card.json --html -o card.html   # self-contained
                                            # shareable page, CI-vs-bar plots
stresskit verify results/                   # audit every card and oracle report
                                            # in a directory tree
stresskit scoreboard results/ -o SCOREBOARD.md   # one table of every verdict
stresskit compare old.json new.json --fail-on-regression   # stability
                                            # regression test between releases
stresskit trace card.trace.json -o trace.svg     # verdict-trace chart: grade
                                            # distribution vs run count
stresskit site references/ -o _site        # static site: index, card pages,
                                            # trace charts (GitHub Pages ready)
```

Host `badge.json` anywhere public and embed a live badge:

```markdown
![stability](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/you/repo/main/badge.json)
```

→ ![stability](https://img.shields.io/badge/stability-A%20%C2%B7%20J%3D0.92-brightgreen) / ![stability](https://img.shields.io/badge/stability-D%20%C2%B7%20J%3D0.18-red)

## Use it in CI

The pytest/codecov move, made literal: produce a card per release, verify it
and gate on regressions in GitHub Actions —

```yaml
- uses: Arth-Singh/Stresskit@v0.3.0
  with:
    path: cards/                      # verify every card in the tree
    baseline: cards/v1.2.json         # optional: fail if stability regressed
    candidate: cards/current.json
```

`stresskit compare` calls a change a **regression** only when a check flips
pass→fail or the grade drops (against identical thresholds), and calls a
value delta **decisive** only when the two 95% CIs are disjoint — point
drift within overlapping intervals is reported, not judged.

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
[`references/`](references/README.md); one-table summary in
[`SCOREBOARD.md`](SCOREBOARD.md). Every card is produced under the
pre-registered evidence standard of
[`references/PROTOCOL.md`](references/PROTOCOL.md), re-verified by CI on
every push (`stresskit verify references/`), and the scoreboard is
generated from the cards, so neither can drift from the data.

| finding | method | verdict | headline |
|---|---|---|---|
| IOI circuit, GPT-2 small | attribution patching | **A**, low confidence | structural stability *and* specificity CIs straddle their bars at 45 runs; at n = 6 the grade is a literal coin flip (A 47% / B 53% of subsets), settling only at n = 45 |
| Greater-Than circuit, GPT-2 small | attribution patching | **B**, high confidence | robustly stable (J CI [0.83, 0.94]) yet decisively fails specificity (CI [1.06, 1.23] vs the 1.5× bar) — the null "circuit" is just as stable |
| IOI across GPT-2 scale (124M–774M) | attribution patching | A / **A certified** / A | no monotone trend: medium is the only card whose every CI clears its bar; large is undecided again — instability is model-idiosyncratic, not cured by scale |
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

## New: which lens transport is actually better? (2026-08-31)

Three linear transports of the residual stream — the released **Jacobian
lens**, the free **logit lens**, and a **tuned lens** trained on the same
corpus — graded under one battery on the J-lens release's own evaluation
items ([full findings](references/h200-results/README.md)):

![hit@5 by lens](references/h200-results/figs/hit5_qwen3p5_4b.png)

- At ≤4B the Jacobian transport is statistically indistinguishable from the
  free logit-lens baseline on the release's own eval (paired 5 vs 1 flips on
  93 items, n.s.); the flagship association class is ≈0 for every transport.
- Every transport shows the identical failure profile (hit-set Jaccard ≈0.47,
  specificity <1 against a derangement null): the instability previously
  measured on the J-lens card is a property of **hit@k lens evaluation**,
  not of the Jacobian transport.

## Contributing

Issues and PRs welcome — especially reference cards for published findings
(the prioritized queue is [`references/TARGETS.md`](references/TARGETS.md)),
adapters for your pipeline, and arguments about the default thresholds.
See [CONTRIBUTING.md](CONTRIBUTING.md).

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
