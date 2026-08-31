# Lens-transport comparison — Qwen3.5 family (run 2026-08-31)

**Question** (from the J-lens release): how much better is the Jacobian
transport than the standard linear transports — the logit lens and a tuned
lens — on the release's own evaluation items, under one pre-registered
battery? And how much of the previously-measured J-lens instability is the
*transport*, versus the *hit-based evaluation protocol*?

Setup: the released pre-fitted lens (`neuronpedia/jacobian-lens`), the repo's
own `lens-eval-multihop` (93 items) and `lens-eval-association` (51 items)
sets, and the upstream hit criterion (every latent intermediate at rank ≤ 5,
word-like mask, position −1). The tuned-lens baseline is trained by
`references/train_tuned_lens_qwen.py` on the same corpus family the released
lens was fitted on (Salesforce wikitext, 128-token sequences, 1000 sequences),
so the two fitted transports are matched on fitting data. Battery:
seeds / bootstrap / templates / hyperparams, 45 runs per lens, derangement
null (same prompts, rotated targets). Hardware: 8×H200, single day.

## Finding 1 — at ≤4B, the Jacobian transport is statistically
## indistinguishable from the free baseline on the release's own eval

![hit@5 by lens](figs/hit5_qwen3p5_4b.png)

Multihop hit@5: **jlens 0.31 · logit 0.27 · tuned 0.28**. Paired per-item
(93 items): 24 hit under both jlens and logit, **5 jlens-only vs 1
logit-only** — a sign test on 6 discordant items does not reject equality
(two-sided p ≈ 0.22). On the association set — the flagship "workspace"
demo class — every transport is at or near zero (jlens 2/51, others 0/51).

![scale](figs/hit5_scale.png)

At 0.8B the gap is the same size (0.26 vs 0.22). 27B results pending —
if the gap grows with scale, the defensible claim becomes "the Jacobian
transport's advantage is scale-emergent"; at ≤4B it is within noise.

## Finding 2 — the instability is the evaluation protocol's, not the transport's

![checks](figs/checks_qwen3p5_4b.png)

All three transports, under the identical battery, produce **the same check
profile**: structural stability of the hit-set ≈ 0.47 (far below the 0.8
bar) and specificity ≈ 0.81–0.89 — the derangement null (targets rotated to
the wrong items) yields hit-sets as stable as the real ones, for every
transport. The instability and non-specificity previously measured on the
J-lens card ([`../cards/jlens_qwen3p5_4b.md`](../cards/jlens_qwen3p5_4b.md))
are therefore properties of **hit@k evaluation on these item sets**, not of
the Jacobian transport: any linear readout inherits them. Hit-based lens
metrics reward frequency artifacts regardless of how the residual is
transported.

## Sanity checks performed

- **Leakage**: 5/93 multihop items contain an intermediate verbatim in the
  prompt (the `letterpos-*` class); results do not change materially with
  them excluded.
- **Tuned baseline is real**: translators trained to KL ≈ 30 nats/batch at
  late layers (from 257 at init); tuned and logit top-10 readouts differ at
  every (item, layer) — the near-tie with logit lens is not an untrained
  translator.
- **Raw readouts read by hand**: mid-layer raw top-10s are junk-dominated for
  every transport (underscores for jlens, multilingual glitch tokens for
  logit); hits come from late layers under the word-like mask. Random —
  not cherry-picked — examples are in the cached readouts
  (`jlens/logit/tuned .json.gz`, regenerable by the runner).

## Scope, honestly

One model family so far (Qwen3.5, 0.8B/4B; 27B pending), the released lens in
its released usage mode, the release's own item sets, pre-registered default
thresholds. Diagnostic grades on the cards are descriptive, low-confidence at
n_runs=20 where marked. None of this shows the Jacobian lens is "wrong" — the
workspace paper's qualitative claims are not tested here; what is tested is
the *measurement protocol* every lens comparison in the release relies on.

## Reproduce

```bash
python references/train_tuned_lens_qwen.py --layers 0-31 --out tuned.pt   # or shard per GPU
python references/run_lens_baselines_qwen.py precompute --lens {jlens,logit,tuned} ...
python references/run_lens_baselines_qwen.py battery --cache jlens=... --cache logit=... --cache tuned=...
python references/refit_jlens_qwen.py fit --shard {0..3} ...              # fit-reproducibility (pending)
```

Cards, comparison tables and figures in this directory; runner scripts in
`references/`.
