# Battery targets — the queue

The prioritized list of findings and instruments to battery-test next.
Every entry follows [`PROTOCOL.md`](PROTOCOL.md) (pre-registered thresholds,
recomputable card, null control, upstream courtesy) and names the
hypothesis in [`HYPOTHESES.md`](HYPOTHESES.md) it advances. PRs claiming an
entry are very welcome — comment on the tracking issue first so work isn't
duplicated.

**What qualifies a target.** (1) The claim is load-bearing — cited,
built-on, or adopted as an instrument, so its stability matters to more
than one paper. (2) A released artifact exists (weights, code, eval data) —
we grade what can be re-run, not what can only be read about. (3) A null
control is constructible. (4) Recency is a plus: a 2026 finding graded
while the field is still deciding whether to build on it is worth ten
retrospectives. Pin exact upstream versions in the runner at run time —
entries below name the finding, not a frozen commit.

## Tier 1 — direct extensions, existing adapters/runners

| target | claim under test | battery sketch | needs | advances |
|---|---|---|---|---|
| **IOI on gpt2-xl** | completes the scale sweep (124M→1.5B) | existing `run_ioi_gpt2_card.py --model gpt2-xl` | 1 GPU, ~hours | H1: is the non-monotone scale pattern (small ⚠ / medium ✅ / large ⚠) idiosyncrasy or trend? |
| **IOI on Pythia 160M/410M/1B** | same task, second model family | existing runner + TransformerLens model swap | 1 GPU | H1: separates scale from training-run idiosyncrasy |
| **EAP-IG vs plain attribution patching, same tasks** | is non-specificity a property of the method family or one estimator? | `adapters.eap` (exists); identical nulls to the current circuit cards | 1 GPU | H2: the sharpest open question the Greater-Than card raised |
| **Activation Oracles on a second base family (Gemma-2-9B)** | phrasing-dominance replicates across families | existing `run_oracle_reliability_qwen3.py` pattern + upstream checkpoints | 1 GPU | H3 |
| **Refusal direction** ("refusal is mediated by a single direction", arXiv:2406.11717) | direction identity and ablation effect stable across seeds, prompt sets, layer choice | new small runner; finder = difference-in-means direction; claim = layer band + cosine-matched identity; **null**: direction from a shuffled harmful/harmless split | 1 GPU | H1/H2 on the most-deployed causal-direction claim in the field (already on the README roadmap) |

## Tier 2 — new adapters, higher leverage

| target | claim under test | battery sketch | needs | advances |
|---|---|---|---|---|
| **Attribution-graph circuit tracing (transcoder-based)** | traced graphs for a prompt are stable across transcoder seeds, prompt paraphrases, pruning thresholds | new adapter (graphs → `Finding` with edge universe); **null**: graphs for matched control prompts where the claimed feature story shouldn't apply | multi-GPU or released traces | H1/H2 on 2025–26's most visible circuit methodology |
| **Sparse feature circuits** | feature-level circuits survive SAE retraining (the SAE seed is a hyperparameter nobody varies) | `adapters.sae` MCC matching + circuit battery composed | multi-GPU (SAE retraining) or released multi-seed SAEs | H1; connects the SAE and circuit halves of the library |
| **Steering vectors (contrastive activation addition)** | steering direction + effect size stable across contrast-set resampling and template phrasing | bootstrap + templates axes are natural; **null**: random direction of matched norm, effect measured identically | 1 GPU | H2/H5 for the intervention family papers actually deploy |
| **Autointerp explanation pipelines** (SAEBench-style scoring) | feature explanations and scores stable across explainer prompt, scorer model, and activation sample | `stresskit.judges` + oracle-style battery; **null**: explanations for permuted feature↔activation pairings | API budget, no local GPU | H3; extends arXiv:2607.19386 from variance measured to variance graded |
| **Introspection / self-report claims** (model reports about own internals, incl. injected-thought detection) | reported internal state consistent across phrasings and repeats; abstains on no-injection controls | `stress_oracle` applies almost unmodified — the subject model *is* the oracle; **null**: no-injection trials | 1 GPU or API | H3 on 2026's fastest-moving claim class |

## Not yet qualified (watching)

- **Weight-based/parameter-space interp claims** — no standard released
  artifact to re-run per finding yet; revisit when one exists.
- **Trajectory/agent-rollout stability** — needs the trajectory battery
  from the README roadmap before any target can be graded honestly.

## Claiming a target

1. Open (or comment on) the target's tracking issue.
2. Write the runner against pinned upstream versions; run the default
   battery; produce card + trace + render.
3. Submit per the checklist in [`PROTOCOL.md`](PROTOCOL.md) §8 — including
   the upstream courtesy step for C/D grades on named work.

Compute notes: "1 GPU" targets fit a single 24–80GB card in hours.
Anyone with spare capacity can adopt a Tier 1 target — the runners are the
short part; the battery does the rest.
