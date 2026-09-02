# Battery targets — the queue

The prioritized list of findings and instruments to battery-test next.
Every entry follows [`PROTOCOL.md`](PROTOCOL.md) (pre-registered thresholds,
recomputable card, null control, upstream courtesy); the hypotheses the
queue advances are stated in [`HYPOTHESES.md`](HYPOTHESES.md). PRs claiming an
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

## Running now

| target | claim under test | where |
|---|---|---|
| **HARC** (arXiv:2607.00572, `microsoft/HARC`) | harmfulness and refusal are separable directions in the base model and coupled by the released adapter, at prompt and response positions | Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct with the released LoRA adapters |
| **Sparse Weight Decomposition** (arXiv:2608.03913, `veri-safe/SWD`) | SWD matches Transcoder replacement fidelity with under 1% of the calibration data and reaches circuit targets with fewer units and edges | GPT-2 small, layer-8 `mlp.c_proj`, IOI / docstring / gendered-pronoun families, greater-than as a reproduction check |

## Next: licensed July/August 2026 releases not yet run

Every entry below ships an authored repository with an SPDX license (the
census in [`../benchmark/discovery/`](../benchmark/discovery/)); none has a
card yet. The note says what stands between the repository and a battery.

| target | claim under test | battery sketch | blocker or note |
|---|---|---|---|
| **REINS** (arXiv:2608.28233, `Geralt1020/REINS`, Apache-2.0) | SAE-feature inhibition plus refusal enhancement lowers harmful continuations on GUISE without a matched-safe over-refusal cost | released frozen controllers for Qwen3.5-2B/4B-Base and the `REINS-SAE` bundle; resample GUISE items, swap categories, vary the gate threshold; null = random SAE features | the paper's metric is an LLM safety judge; a judge-free refusal proxy is the only in-budget version and must be labelled as such |
| **Steering vectors for CoT faithfulness** (arXiv:2607.29062, `xocelyk/steering-vectors-for-faithfulness`, MIT) | a cue-acknowledgment steering vector transfers across cue types and datasets | released vectors for Gemma-3-4B, Qwen-3.5-9B, Gemma-3-12B; resample test items, swap cue, vary alpha and layer; null = random direction of the same norm | scoring is a `gpt-5-nano` judge over generations; needs a local judge or a string rule, and the 12B model does not fit next to shared vLLM |
| **PRISM-Edit** (arXiv:2607.11327, `CheerCHuang/PRISM-Edit`, MIT) | model-editing locality and generalisation numbers on the released benchmark | resample edit sets, vary the edit layer and the paraphrase set; null = edits with shuffled targets | not triaged yet |
| **Hebbian MLPs** (arXiv:2607.10034, `HazyResearch/hebbian-mlps`, Apache-2.0) | the interpretability claims about learned Hebbian features | depends on what the release trains; triage first | not triaged yet |
| **Semantic conflicts** (arXiv:2607.05587, `brains-on-code/mechanistic-interpretability-semantic-conflicts`, CC-BY-SA-4.0) | mechanistic account of semantic conflicts in code models | triage first | CC-BY-SA on code; check that the licence gate accepts it |
| **Can Graph Learning Learn Circuits?** (arXiv:2608.08536, AGPL-3.0) | graph learners recover circuits | triage first | AGPL; check that the licence gate accepts it |

## Out of budget on shared GPUs (recorded, not run)

- Concept-Targeted Attribution (arXiv:2608.27510): 10,400 attribution graphs
  per replication.
- Anticipating post-SFT mechanisms (arXiv:2608.24482): needs a full 7B SFT per
  run.
- Diff Mining's "one third of 52 biases" (arXiv:2608.26462): needs a 70B model.
- The Gemma-3-12B half of the sycophancy paper (arXiv:2607.07003): does not fit
  next to the shared vLLM allocation.

## Method-family targets (no single paper)

| target | claim under test | battery sketch | needs |
|---|---|---|---|
| **Attribution-graph circuit tracing (transcoder-based)** | traced graphs for a prompt are stable across transcoder seeds, prompt paraphrases and pruning thresholds | graph-as-component-set; released transcoders; paraphrase templates; null = deranged prompt-graph pairs | released transcoder checkpoints, 1 GPU |
| **Sparse feature circuits** | feature-level circuits survive SAE retraining (the SAE seed is a hyperparameter nobody varies) | `adapters.sae` MCC matching across seeds; null = shuffled feature identities | SAE training, 1 GPU per seed |
| **Autointerp explanation pipelines** (SAEBench-style scoring) | feature explanations and scores are stable across explainer prompt, scorer model and activation sample | resample activations, swap prompts, swap scorer; null = explanations for a different feature | API budget or local judge |
| **Introspection / self-report claims** | reported internal state is consistent across phrasings and matches the injected state | phrasing templates, injection strength sweep; null = no injection | 1 GPU |
| **IOI on gpt2-xl and Pythia** | completes the scale sweep and separates scale from training run | existing `run_ioi_gpt2_card.py`; TransformerLens model swap | 1 GPU; lowest priority, old models |

## Claiming a target

1. Open (or comment on) the target's tracking issue.
2. Write the runner against pinned upstream versions; run the default
   battery; produce card + trace + render.
3. Submit per the checklist in [`PROTOCOL.md`](PROTOCOL.md) §8 — including
   the upstream courtesy step for C/D grades on named work.

Compute notes: "1 GPU" targets fit a single 24–80GB card in hours.
Anyone with spare capacity can adopt a Tier 1 target — the runners are the
short part; the battery does the rest.
