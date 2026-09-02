# Results ledger

Running record of what every StressKit audit found, one entry per paper.
The paper registry [`references/papers.json`](references/papers.json) drives
the generated leaderboard, one row per paper with one grade per card, in
[`SCOREBOARD.md`](SCOREBOARD.md) and on the site index; this file carries the
conclusions a reader needs before opening a card: what reproduced, what
survived the battery, what did not, and what is still running. Grades are reliability measurements under the pre-registered
checks in [`references/PROTOCOL.md`](references/PROTOCOL.md), not judgments of
a paper's value. "Low confidence" means at least one check's 95% CI straddles
its bar; those grades are provisional and the run counts are being raised.

Last updated 2026-09-02 (KST). Every card listed re-derives from its own
recorded metrics via `stresskit verify`.

## Leaderboard

| paper | model(s) | grade | conf. | checks | runs | reproduced the released number? | result in one line | date | card |
|---|---|---|---|---|---|---|---|---|---|
| FolkMotif: cultural awareness represented but not decoded (arXiv:2608.02486) | Llama-3.1-8B-Instruct | 🟢 A | high | 5/5 | 52 (+41 null) | yes, exactly (probe 0.881 @ L8, n-gram 0.604, buckets 45/193/5/27; the paper's 0.248 is the rescored run and its 32/206/6/26 row is the native-language run, both exact) | the qualitative claim never flips; the cell counts move 5–83 Preserved with the aggregation rule and template, and the "peak layer 8" moves over layers 6–14 with the CV split | 2026-09-02 | [card](references/cards/folkmotif_llama3p1_8b.md) |
| Expander SAE (arXiv:2607.01799) | Qwen2.5-3B, layer 12 | 🟢 A | high | 2/2 | 69 | yes (0.831 / 0.978 CE-recovered vs 0.833 / 0.983; ratio 0.850 vs 0.842) | ratio 0.80–0.90 over 30 seeds and 30 resamples, flips only at the 0.80 bucket edge; k=32 gives 0.66, mean-ablation denominator 0.77 | 2026-09-02 | [card](references/cards/expander_sae_qwen2p5_3b.md) |
| CoAx backup-head recovery (arXiv:2607.01940) | GPT-2 small, IOI | 🟡 B | low | 3/5, 1 undecided | 71 (+61 null) | yes vs the released reference_metrics (0.945 vs 0.941); the abstract's "from 0.33" is 0.60 in the released code | with label-free primaries CoAx AUC collapses to 0.28 / 0.38 while AtP\* stays 0.83; one of three alternative templates loses to AtP\*; the no-IOI null recovers the same heads at 0.93–0.97, so the backup structure is task-general | 2026-09-02 | [card](references/cards/coax_backup_gpt2.md) |
| Activation Model Scanner, Tier-1 safety scan (arXiv:2608.05578) | 14 models of Table I | 🟠 C | low | 1/5, 3 undecided | 129 (+121 null) | yes, all 14 σ values to two decimals through the released extractor | the released extractor reads pad-token activations for the 10 right-padded tokenizers; with batch size 1 or left padding every model scores σ 4.5–6.7, nothing is flagged, LOO accuracy 0.14, r flips to +0.28 n.s. | 2026-09-02 | [card](references/cards/ams_safety_scanner.md) |
| Certified Interventional Fidelity, GPT-2 IOI (arXiv:2607.08349) | GPT-2 small | 🟡 B | low | 1/4, 1 undecided | 36 | yes, 30/30 shipped rows exact | certified level depends on the template (0.95 upstream, 0.9 on seven of eleven others, none on one); the "10–30x" is 6.6–7.2x at F0=0.8 and 18–19x at 0.9 | 2026-09-02 | [card](references/cards/cif_ioi_gpt2.md) |
| The Communication Map of a Transformer (arXiv:2608.22007) | GPT-2 ×3, GPT-Neo-125m, Pythia-160m/2.8B/6.9B | 🟡 B | low | 2/5, 2 undecided | 24 (+21 null) | yes, all 21 Table 2 shares exactly | the abstract's "70–89%" holds only pooled per model (69.7–89.0%); per channel 4 of 21 entries fall outside (60.7–90.5%); the lower bound is a rounding edge under resampling; \|z\|≥3 gives 59–83% with six exceptions, uncentred weights move single entries by 5–8 points | 2026-09-02 | [card](references/cards/communication_map.md) |
| Dissociating sycophancy representations (arXiv:2607.07003) | Llama-3.1-8B-Instruct | 🟡 B | low | 3/5, 1 undecided | 48 (+41 null) | within 0.02–0.10 of Tables 1–2 (in-domain 0.93 / 0.94 vs 0.91 / 0.92, transfer 0.74 / 0.71 vs 0.70 / 0.61) at the paper's "final layer", which the released extractor's lexicographic layer order makes decoder layer 9 | "distinct" (transfer drop ≥ 0.15) holds in 47 of 48 runs, but which layers carry the drop is unstable (J 0.51), the drop swings with the conversation resample (score CV 0.23), and at the best in-domain layer (L12, AUC 0.99 / 0.97) the probes transfer (drop 0.12) | 2026-09-02 | [card](references/cards/sycophancy_llama3p1_8b.md) |
| Diff Mining, judge-free token-set battery (arXiv:2608.26462) | gemma-3-1b-it × cake_bake LoRA | 🟢 A | low | 4/5, 1 undecided | 51 (+41 null) | no shipped token lists to reproduce; the paper's metric is a gpt-5-mini judge and its bias number needs a 70B model, neither run | the top-100 token set is stable across seeds, resamples and corpora (J 0.88–0.97) and 65% finetune-domain vocabulary under a pre-registered rule; a scrambled adapter returns garbage (0–36%); the share falls to 0.48 with logit-lens extraction and to 0.26 for the LoRA trained on a 1:1 pretraining mix | 2026-09-02 | [card](references/cards/diff_mining_gemma3_1b.md) |
| Refusal direction (arXiv:2406.11717) | 6 models, 3 families | 🟢 A … 🟠 C | mixed | see README | 21 each | causal claim reproduces on every model | the causal effect holds hard (specificity 4–1293x); which direction gets selected is unstable (J 0.18–0.39); two measurement artifacts found in raw completions | 2026-09-01/02 | [README](references/README.md#the-refusal-direction-across-six-models-and-three-families-arxiv240611717) |
| SAE causal inertness (arXiv:2607.12166) | toy bottleneck model + TopK SAEs | 🟠 C | low | 2/5, 1 undecided | 33 | run 2 within the paper's own band | inert-pair census is unstable (J 0.33); the abstract's headline has a different denominator than its sentence | 2026-09-01 | [card](references/cards/sae_causal_inertness.md) |
| Homonym reconvergence (arXiv:2608.01816) | gpt2, Llama-3.2-3B, Qwen2.5-7B | 🟡 B ×3 | low | 4/5 each | 31–32 each | stimuli and the Table 1 tokenisation counts reproduce exactly | the profile label comes back in 28/31 to 32/32 runs, but the paper's own sequence-order control produces the same label (specificity 0.88–1.08x); magnitude separates homonyms from controls, the profile shape does not | 2026-09-01 | [gpt2](references/cards/homonym_reconvergence_gpt2.md) · [llama](references/cards/homonym_reconvergence_llama_3p2_3b.md) · [qwen](references/cards/homonym_reconvergence_qwen2p5_7b.md) |
| Truth vs impossibility probes (arXiv:2608.12852) | gemma-3-4b-it | 🟢 A | high | 5/5 | 39 | yes, same snapshot | double dissociation survives resampling, re-splitting and hyperparameters; specificity 1.84x | 2026-09-01 | [card](references/cards/impossibility_truth_gemma_3_4b_it.md) |
| Mechanistic Tomography, OMP recovery (arXiv:2608.19338) | released HMM observer checkpoint | 🟠 C | high | 2/5 | 57 (+49 null) | yes, bit-exact | the four bin-7 coordinates are real and specific; the support beyond them is not stable (J 0.40) | 2026-09-01 | [card](references/cards/mechtomo_omp_recovery.md) |
| Jacobian-lens readouts (anthropics/jacobian-lens) | Qwen3.5-0.8B/4B/27B, Qwen3.6-27B | 🟠 C / 🟡 B | low | 2–3/5 | 20–48 | released lens used as shipped | the mid-to-late-band claim is stable (π\* 0.90); which items hit is not (J 0.45–0.49), and the deranged-target null hits more consistently than the real targets (specificity 0.78x) | 2026-08-21/31 | [4B](references/cards/jlens_qwen3p5_4b.md) · [baselines](references/h200-results/) |
| Activation Oracles (arXiv:2512.15674) | Qwen3-8B taboo | 🔴 D, 🔴 D, 🟠 C | high | 0–1/4 | 225 each | pre-trained oracles as shipped | accuracy 0.09–0.45 with null hallucination ~0.9; the instrument is prompt-dominated | 2026-08-21 | [cls-only](references/cards/ao_qwen3_cls-only.md) · [full](references/cards/ao_qwen3_full-mixture.md) · [latentqa](references/cards/ao_qwen3_latentqa-only.md) |
| IOI attribution patching (Wang et al. 2022 task) | gpt2 small / medium / large | 🟢 A ×3 | low / high / low | 5/5 | 45 each | n/a (classic task) | J 0.83–0.95, specificity 1.5–2.3x; no monotone trend with scale: medium is the only certifiable A, small and large stay undecided after 45 runs | 2026-08-21 | [small](references/cards/ioi_gpt2_small.md) · [medium](references/scale/ioi_gpt2_medium.md) · [large](references/scale/ioi_gpt2_large.md) |
| Greater-Than attribution patching (arXiv:2305.00586) | gpt2 small | 🟡 B | high | 4/5 | 45 | n/a | J 0.89 but specificity 1.15x: the head set is nearly as stable on the corrupted null | 2026-08-21 | [card](references/cards/greater_than_gpt2_small.md) |

## July / August 2026 target set — detailed results

These are the papers that ship code from the July and August 2026 census,
run on the 8×H200 box on 2026-09-02. Each runner pins the upstream commit and
imports the upstream functions unmodified; every deviation is a card note.

### Certified Interventional Fidelity (arXiv:2607.08349) — B, low confidence

Claim, byte-exact: CIF's betting confidence sequences reduce certification
cost by 10–30x, and certify high-fidelity claims on GPT-2 Small IOI circuits.

| check | value | 95% CI | state |
|---|---|---|---|
| structural stability (certified levels) | 0.569 | [0.416, 0.715] | ❌ |
| claim stability | 0.611 | [0.472, 0.778] | ❌ |
| score stability (final betting LCB, 3 heads) | CV 0.045 | — | ✅ |
| beats random | 3.68x | [2.69, 4.63] | ⚠️ undecided |

- Reproduction: all 30 shipped iid rows of `results/e2_completeness.csv`
  reproduce exactly through the upstream code.
- The certificate is about one template. Name movers certify F0 ≤ 0.95 on the
  upstream template and two of eleven alternatives, F0 ≤ 0.9 on seven, F0 ≤ 0.8
  on one ("argument"), and nothing on "commuting" (mean recovery 0.780).
- The 10–30x depends on the threshold read: 6.6–7.2x at F0 = 0.8, 18.4–19.0x
  at F0 = 0.9 (censored at the 2000-draw budget for the small circuits), and
  Hoeffding never certifies at F0 = 0.95, so no ratio exists there.
- Probability recovery instead of clipped logit-difference recovery certifies
  nothing for the 3- and 7-head circuits; a 500-draw budget loses F0 = 0.95
  for the same circuits; δ = 0.01 changes nothing.
- No null: nothing-certified is itself a stable profile, so specificity is
  not defined for a certification claim.

### Expander SAE (arXiv:2607.01799) — A, high confidence

Claim: on Qwen2.5-3B, a d = 7 expander SAE uses 293x fewer learned decoder
values than the dense decoder while retaining 84% of dense CE-loss recovered.
69 runs: 30 training seeds, 30 document resamples, one later stream slice,
seven hyperparameter variants.

| check | value | 95% CI | state |
|---|---|---|---|
| claim stability | 0.884 | [0.812, 0.957] | ✅ |
| score stability (ratio) | CV 0.046 | [0.032, 0.059] | ✅ |

- Reproduction: expander 0.831, dense 0.978 CE-recovered (shipped 0.833 /
  0.983); ratio 0.850 vs 0.842. Scalar finding, so no structural checks and no
  null.
- Seeds: ratio 0.800–0.891 over 30 seeds (expander 0.782–0.875); two seeds
  sit at 0.7995 against the 0.80 bucket edge and five put the expander's own
  CE-recovered below 0.80. Bootstrap over documents: 0.822–0.898.
- Knobs that move it: k = 32 → 0.664 (expander 0.629), k = 128 → 0.946,
  15k steps → 0.906, layer 24 → 0.916, mean-ablation denominator → 0.770,
  held-out evaluation text → 0.857 (in-sample 0.850). Upstream's CE-recovered
  is measured on the first 100 documents of the training stream; the held-out
  run shows the same number on unseen text.
- The denominator is a 13.7-nat collapse (CE 2.29 → 15.99 under zero
  ablation), which is what makes both "recovered" fractions look high.

### CoAx backup-head recovery (arXiv:2607.01940) — B, low confidence

Claim, byte-exact: on the GPT-2-small IOI circuit CoAx raises backup-head
recovery from 0.33 to 0.91 ROC-AUC, outperforming all baselines including
self-repair-aware gradient scores (best 0.82).

71 real runs (30 prompt redraws, 30 prompt resamples, three other templates,
seven variants), 61 null runs.

| check | value | 95% CI | state |
|---|---|---|---|
| structural stability (top-8 heads) | J 0.787 | [0.691, 0.869] | ⚠️ undecided |
| claim stability | 0.930 | [0.859, 0.986] | ✅ |
| score stability (CoAx AUC) | CV 0.111 | [0.013, 0.187] | ✅ |
| beats random | 25.3x | [22.2, 28.0] | ✅ |
| specificity (third-name null) | 0.95x | [0.83, 1.05] | ❌ |

- Reproduction: CoAx 0.945, single-ablation 0.599, AtP\* 0.845, EAP-IG 0.675
  against the released `reference_metrics.json` (0.941 ± 0.004, 0.603,
  0.815 ± 0.031, 0.700). The abstract's "from 0.33" for single ablation is
  not what the released code computes (0.60).
- The headline conditions on the three documented name movers. Choosing the
  primaries label-free (top-3 by AtP: L1H3, L1H10, L5H9; top-3 by ablation
  energy: L0H0, L0H7, L0H10) collapses CoAx to 0.276 / 0.378 AUC while AtP\*
  stays at 0.843 / 0.828.
- Templates: 0.881, 0.947, 0.845 on three alternatives; the "argument"
  template loses to AtP\* (0.873). The top-8 set changes with the template
  (J 0.58 across templates).
- Uncentred L2 features keep the AUC (0.899) but replace the whole top-8 with
  layer-8 heads; frozen LayerNorm raises it to 0.982; mean ablation, all
  positions and top-192 logits change little.
- Thirty redraws of the 96 prompts: CoAx 0.912–0.950, AtP\* 0.765–0.853. Six
  heads sit in the top-8 in all thirty runs (L10H10, L10H2, L11H10, L11H2,
  L1H10, L5H9), L10H6 in 29; four of them are documented backups. Resampled
  prompts: 0.925–0.945, J 0.95.
- Null: prompts where a third name is the giver do not pose the IOI task but
  still require copying a name, and CoAx recovers the documented backups on
  them at 0.934–0.965 (higher than on IOI prompts) with the same heads on
  top, while AtP\* drops to 0.57–0.84. The backup structure is task-general,
  not IOI-specific; a ranker that always returns a top-8 cannot fail a
  stability-only specificity check (null J 0.83 vs 0.79), which is a limit of
  the check and is recorded as such.
- The first pass had a vacuous seeds axis (the finder scored the prompts it
  was handed and ignored the seed); the harness flagged the identical
  findings and the runner now redraws the prompts from the seed.

### Activation Model Scanner, Tier-1 safety scan (arXiv:2608.05578) — C, low confidence

Claim: leave-one-out cross-validation of thresholds achieves 71% accuracy
(10/14); σ on the harmful-content concept predicts compliance with Pearson
r = −0.546 (p = 0.043). 129 real runs (120 bootstrap resamples of the 16
contrastive pairs), 121 null runs. At 20 resamples the grade was D; the score
check crossed its bar at 60, and 120 resamples leave the three undecided
checks sitting on their bars.

| check | value | 95% CI | state |
|---|---|---|---|
| structural stability (flagged models) | J 0.789 | [0.760, 0.821] | ⚠️ undecided |
| claim stability | 0.357 | [0.287, 0.442] | ❌ |
| score stability (LOO accuracy) | CV 0.225 | [0.184, 0.261] | ⚠️ undecided |
| beats random | 3.04x | [2.92, 3.16] | ⚠️ undecided |
| specificity (label-swapped null) | 0.95x | [0.91, 0.98] | ❌ |

- Reproduction: all 14 Table I σ values reproduce to two decimals through the
  released extractor; Pearson r −0.549 (p 0.042) and Spearman −0.423 match.
  LOO accuracy 0.643 vs 0.714: the paper releases no LOO code, so the rule
  here (best threshold on the other 13, midpoint, ties to the widest margin)
  is a documented deviation.
- The released extractor batches 8 prompts with padding and reads position
  −1. Ten of the 14 tokenizers pad on the right, so for them the scan
  measures pad-token activations. Batch size 1 and left padding agree with
  each other to 0.01 on every model and reproduce the upstream numbers only
  for the four left-padding tokenizers (gemma-2, DarkIdol). Under either
  correct extraction every model scores σ 4.5–6.7, all PASS, LOO accuracy
  0.143, r = +0.28 (n.s.). The instruction-tuned vs uncensored separation in
  Table I is a padding artifact.
- Chat-template prompts change σ by up to 4x per model (gemma-2-2b-it 4.80 →
  9.01, Llama-3.2-3B-Instruct 8.37 → 2.13); held-out separation (direction
  fitted on half the pairs) drops every model below 5 and sends
  dolphin-2.9.4 negative.
- Bootstrap over the 16 contrastive pairs: p < 0.05 in 64 of 120 resamples,
  LOO accuracy 0.29–0.93, r from −0.09 to −0.70, 4 to 9 models flagged. The
  correlation's significance rides on 16 pairs.
- Null: swapping the labels of half the pairs flags a stable set, so the
  specificity check compares two stable profiles; recorded as a limit of the
  check for classifier-style claims.

### FolkMotif (arXiv:2608.02486) — A, high confidence

Claim, byte-exact: the residual stream cleanly distinguishes cultures, well
above a name-string baseline, yet the decoder collapses culturally-specific
tokens onto dominant-tradition ones.

52 real runs (40 CV seeds, two templates, nine variants), 41 null runs. At 20
seeds the structural CI straddled 0.8; at 40 it clears it.

| check | value | 95% CI | state |
|---|---|---|---|
| structural stability (Preserved cells) | J 0.878 | [0.810, 0.930] | ✅ |
| claim stability | 1.000 | [1.000, 1.000] | ✅ |
| score stability (DecodingSuppressed share) | CV 0.046 | [0.017, 0.067] | ✅ |
| beats random | 9.7x | [8.9, 10.2] | ✅ |
| specificity (permuted culture labels) | 3.12x | [2.69, 3.63] | ✅ |

- Reproduction: probe peak 0.881 at layer 8, n-gram baseline 0.604, output
  accuracy 0.185 and buckets 45/193/5/27 all match the released v3e6 files
  exactly. The paper's 0.248 output accuracy is the released rescored
  majority (raw generation scored instead of the trimmed one): the
  scoring=raw run gives 0.248 exactly, with buckets 61/177/6/26. Its
  32/206/6/26 decomposition row is the native-language (v3h6) run, which the
  template run reproduces exactly. The two headline numbers come from two
  different runs.
- The claim (DecodingSuppressed plurality; probe well above the name n-gram)
  holds in all 32 runs. What moves is the count: Preserved is 5 with the
  "all paraphrases" rule, 45 with majority, 61 with the paper's rescoring,
  83 with "any"; 52 under the v2 chat prompt, 32 under native-language
  prompts. The DecodingSuppressed share runs 0.57–0.86 over those choices.
- The CV split moves the argmax layer over layers 6–14 (20 seeds) while the
  peak accuracy stays 0.852–0.889 and the Preserved set barely moves (42–46
  cells, J 0.94). "Peak at layer 8" is a property of one split; the accuracy
  is not. Ridge α, fold count and bf16 change nothing; a fixed mid-depth layer
  gives 0.833.
- Null: with culture labels permuted the probe sits at chance (0.10–0.14)
  and the Preserved set scatters (null J 0.30).
- Bootstrap is deliberately not run: resampling cells with replacement puts
  copies of one cell in training and held-out folds.

### The Communication Map of a Transformer (arXiv:2608.22007) — B, low confidence

Claim, byte-exact: "The census of all candidate channels, from 6.3x10^8 in
GPT-2 to 1.3x10^11 in Pythia-6.9B, finds that 70-89% of head pairs are
oriented far from chance, some coupled strongly and others actively avoiding
each other." Far from chance means |z| ≥ 2 of the coupling coefficient C²
against the closed-form rotation null, per K/Q/V channel, seven models.
Components are the (model, channel) entries whose share rounds outside 70–89
(universe 21); the claim is whether the pooled per-model shares sit inside the
range and whether every per-channel entry does; the score is the mean pooled
share. 24 real runs (20 cluster-bootstrap resamples over pair chunks, three
variants), 21 null runs.

| check | value | 95% CI | state |
|---|---|---|---|
| structural stability (exception set) | J 0.888 | [0.757, 1.000] | ⚠️ undecided |
| claim stability | 0.667 | [0.542, 0.833] | ⚠️ undecided |
| score stability (mean pooled share) | CV 0.021 | [0.002, 0.035] | ✅ |
| beats random | 7.5x | [6.4, 8.4] | ✅ |
| specificity (Haar-rotated writers) | 0.89x | [0.76, 1.00] | ❌ |

- Reproduction: all 21 released Table 2 shares reproduce exactly (largest
  difference 2e-6, on Pythia-6.9B).
- Abstract against Table 2: per (model, channel) the shares span 60.7–90.5%
  and four of 21 entries fall outside 70–89 (gpt2/V 90.5, pythia-2.8b/K 68.5,
  pythia-2.8b/Q 63.8, pythia-6.9b/Q 60.7). Pooled over a model's three
  channels they span 69.7% (Pythia-2.8B) to 89.0% (GPT-2), which is the
  reading under which the range holds, and its lower bound is a rounding
  edge: resampling pushes Pythia-2.8B below 69.5 in about half the draws,
  which is the whole claim flip rate, while the four exceptions themselves
  are stable (J 0.98).
- |z| ≥ 3, which upstream also tabulates, gives pooled shares 58.9–83.3% and
  six exceptions. Raw HuggingFace weights without LayerNorm folding and
  centring give 70.6–87.0% with a different exception set (pythia-2.8b/Q,
  pythia-6.9b/K at 90.4, pythia-6.9b/Q); single entries move 5–8 points.
  fp16 weights change nothing.
- Six further models (reported, not graded): Pythia-410m, Pythia-1.4B,
  GPT-Neo-1.3B and OPT-1.3B pool inside the range (77–86%); Pythia-1B (94.0%)
  and OPT-125m (91.4%) do not.
- Null: replacing each writer's output factor by an independent Haar
  rotation gives 4.2–4.7% far from chance (chance is about 5%) and every
  entry an exception; specificity is uninformative for range membership,
  since "all outside" is as stable a set as "these four outside".
- Deviations: no seeds axis (closed-form census, no randomness) and no
  templates axis (no text input); models loaded on CPU and the census run on
  GPU; Pythia-6.9B through the standard loader rather than upstream's
  streaming path; transformer-lens 3.8.1 against the lock's 3.7.0.

### Dissociating sycophancy representations (arXiv:2607.07003) — B, low confidence

Claim, byte-exact: "We find that different LLMs represent these subtypes
differently, with either more aligned or more distinct representations." For
Llama-3.1-8B-Instruct the paper quantifies "distinct" as a transfer gap:
probes trained on one sycophancy subtype reach 0.91 (factual) / 0.92
(opinion) ROC-AUC in domain and 0.70 / 0.61 across subtypes at the final
layer. Only the Llama half is audited (Gemma-3-12B does not fit next to
vLLM). Components are the eight decoder layers with the largest transfer
drop (universe 32); the claim is "distinct (drop ≥ 0.15) | shared" plus the
in-domain bucket at the paper's layer; the score is that drop. 48 real runs
(20 probe seeds, 20 conversation resamples, 7 variants), 41 null runs.

| check | value | 95% CI | state |
|---|---|---|---|
| structural stability (top-8 layers by drop) | J 0.509 | [0.472, 0.549] | ❌ |
| claim stability | 0.979 | [0.938, 1.000] | ✅ |
| score stability (transfer drop) | CV 0.225 | [0.186, 0.257] | ⚠️ undecided |
| beats random | 3.4x | [3.2, 3.7] | ✅ |
| specificity (permuted labels) | 3.4x | [3.0, 3.7] | ✅ |

- Reproduction: the released activation cache is private, so activations
  were regenerated with the upstream extractor from the committed
  GPT-5-labelled conversations. Mean of seeds 42–46 at the paper's "final
  layer": factual 0.928, opinion 0.938 in domain (paper 0.91 / 0.92),
  factual→opinion 0.738, opinion→factual 0.711 (paper 0.70 / 0.61).
- The released extractor stacks the hooked layers in lexicographic
  module-name order, so probe index 31, the paper's "final layer", is
  decoder layer 9; decoder layer 31 is index 25. At the true final layer the
  numbers are 0.910 / 0.916 in domain and 0.692 / 0.460 across (drop 0.30).
  The hook reads the token before the end-of-turn marker, not the marker.
- "Distinct" holds in 47 of 48 runs, but the drop is a layer choice: at the
  layer where the probes work best (decoder layer 12, in-domain 0.988 /
  0.969) the probes transfer (0.823 / 0.894, drop 0.12) and the claim
  flips to "shared". Which layers carry the largest drop is unstable
  (bootstrap J 0.44, seeds J 0.59): early layers L0–L6 and late layers
  L26–L31 trade places.
- Resampling the 1200-conversation pool moves the drop by a factor of two
  (score CV 0.23); the paper's five seeds share one conversation set.
- Null: upstream's shuffle-labels path drives both in-domain and transfer
  AUC to chance; the top-8 layer set scatters (null J 0.15).
- Deviations: the combined probe is not trained; transfer AUC computed
  inline rather than through upstream's path-bound evaluator; activations
  for the 1200-conversation pool per subtype in batches of 25; no templates
  axis (the conversations are fixed closed-model artifacts).

### Diff Mining, judge-free token-set battery (arXiv:2608.26462) — A, low confidence

Claim, byte-exact: "Empirically, Diff Mining succeeds across diverse
settings: on finetune domain detection, it significantly outperforms
state-of-the-art model diffing methods both in identifying relevant tokens
and in downstream performance when an interpretability agent is given access
to the extracted token set; on models with injected biases, it identifies
more than one third of the biases without targeted probing." The paper's
relevance metric is a gpt-5-mini judge and its bias number needs
Llama-3.3-70B-Instruct; neither is run. This card audits the judge-free part:
whether the top-100 token set Diff Mining returns for gemma-3-1b-it × the
cake_bake LoRA is a stable object, and how much of it is finetune-domain
vocabulary under a rule fixed before any run (a token occurs ≥ 10 times in
the finetune corpus, is not generic, and is ≥ 8x more frequent there than in
fineweb; 2851 tokens). Components are the top-100 token ids (universe = the
candidates the ordering ranks, about 146k); the claim buckets the domain
share; the score is the top-100 domain share. 51 real runs, 41 null runs.

| check | value | 95% CI | state |
|---|---|---|---|
| structural stability (top-100 tokens) | J 0.845 | [0.747, 0.926] | ⚠️ undecided |
| claim stability | 0.961 | [0.902, 1.000] | ✅ |
| score stability (domain share) | CV 0.099 | [0.028, 0.163] | ✅ |
| beats random | 2430x | [2149, 2664] | ✅ |
| specificity (scrambled adapter) | 12.8x | [9.0, 18.3] | ✅ |

- No shipped number to reproduce: the paper releases no token lists. Base
  run: domain share 0.65 of the top-100 and 0.95 of the top-20; top tokens
  Mediterranean, Professional, Cake, Baking, culinary, cookbook.
- Stable where the paper's protocol varies: 20 draw seeds J 0.97 (share
  0.63–0.66), 20 document resamples J 0.96, a disjoint fineweb slice and the
  Pile head both 0.64 (J 0.88). Top-K 20 or 500, 300 documents, 64 positions:
  0.56–0.68, same top-10.
- Not stable across the method's own switches: logit-lens extraction gives
  0.48 (top-20 0.70, a different vocabulary: flavorful, gastronomic,
  gourmet); the LoRA trained on a 1:1 mix with pretraining data gives 0.26
  with `<eos>`, `</i>` and "Medical" in its top-10; the full finetune gives
  0.78.
- Null: permuting the LoRA A-matrix input features (norms kept) returns
  garbage tokens with domain share 0.00–0.36 (null J 0.07).
- Deviations: vllm, dictionary-learning and the graders are not installed
  (placeholder module, packages registered without their `__init__`, two
  helpers executed from the pinned source); reference pool is the first
  40k qualifying fineweb documents rather than the full 1M sample.

## In progress and queued

| item | where | status |
|---|---|---|
| Expander SAE, 30 seeds and 30 bootstrap resamples | GPUs 0–2 | done, card updated above (A, high confidence) |
| CoAx, seed fix + 30 runs | GPUs 3–5, 6 shards | done, card updated above |
| AMS, 60 bootstrap resamples | GPU 5 | done, card updated above |
| FolkMotif, 20 seeds + scoring=raw | GPU 6 | done, card updated above |
| Dissociating sycophancy (arXiv:2607.07003), Llama-3.1-8B-Instruct half | GPU 7 | done, card above (B, low confidence) |
| Communication map census (arXiv:2608.22007) | GPU 6 / CPU | done, card above (B, low confidence) |
| Diff Mining (arXiv:2608.26462), judge-free top-K token-set battery on gemma-3-1b-it × cake_bake | GPUs 0–2 | done, card above (A, low confidence) |
| SWD fidelity (GPT-2) | GPUs 3–5, 6 shards | running; first run's circuit stage is the cost driver (greater-than validation loop), relaunch with 6 runs per axis on 9 shards planned |

Not auditable in this setting: Diff Mining's 70B "one third of 52 biases",
CTA's 10,400 attribution graphs, Future Localization (full 7B SFT),
Gemma-3-12B half of the sycophancy paper (does not fit next to vLLM).
