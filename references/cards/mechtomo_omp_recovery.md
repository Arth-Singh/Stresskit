# 🟠 Diagnostic Stability Card — descriptive grade **C** (high confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Claim:** Orthogonal matching pursuit recovers the 32-coordinate finite-effect map with Pearson r = 0.989 and held-out R-squared = 0.935.
> model: released HMM observer checkpoint experiments/hmm/frozen/model.pt (4-layer, d_model 96, seed 7; kwisatzh/mechanistic-tomography@5c097d2) · task: recover the 32-coordinate (layer x time-bin) finite-effect map on the implied belief z1 from 12 aggregate signed-mask interventions (epsilon 0.6, density 0.30) · method: orthogonal matching pursuit with validation-selected support size, upstream sparse_tomography_posthoc.py

Battery: `seeds, bootstrap, templates, hyperparams` — 57 runs (seed 7, 0.45s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.401 | [0.317, 0.511] | ≥ 0.800 | ❌ fail |
| claim stability | 0.684 | [0.561, 0.789] | ≥ 0.800 | ❌ fail |
| score stability | 1.685 | [0.900, 3.902] | ≤ 0.250 | ❌ fail |
| beats random | 5.751 | [4.539, 7.320] | ≥ 3.000 | ✅ pass |
| specificity | 3.036 | — | ≥ 1.500 | ⚠️ inconclusive |

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 57 |
| structured runs | 57 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.401 |
| Jaccard incl. size-mismatched runs | 0.266 |
| min pairwise Jaccard | 0.000 |
| random-null Jaccard | 0.070 |
| overlap vs random (×) | 5.751 |
| claim flip rate | 0.461 |
| modal claim share π* | 0.684 |
| distinct claims | 4 |
| score mean | 0.352 |
| score CV | 1.685 |
| median finding size | 3 |
| Jaccard 95% CI (bootstrap) | [0.317, 0.511] |
| flip rate 95% CI (bootstrap) | [0.346, 0.568] |
| null-control (specificity) | Jaccard 0.132 · flip 0.041 on 49 null runs |
| claim distribution | `not recovered; sparse support (k<=8)`×39, `recovered; sparse support (k<=8)`×16, `predictive-only; dense support (k>8)`×1, `not recovered; dense support (k>8)`×1 |
| score-variance shares (OAT) | bootstrap: 18%, hyperparams: 46%, seeds: 7%, templates: 28% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 25 | 0.263 | 0.497 | 0.680 | 1.748 |
| hyperparams | 5 | 0.510 | 0.600 | 0.600 | 2.223 |
| seeds | 25 | 0.253 | 0.420 | 0.720 | 0.948 |
| templates | 5 | 0.450 | 0.600 | 0.600 | 2.298 |

## Notes

- structural stability graded on 40 size-comparable runs; 17 run(s) with >2x size difference excluded (pooled-over-all value kept as mean_pairwise_jaccard_all_sizes)
- scope: graded artifact = the released measurement pool nt_mi_set1_v2 (256 signed-mask aggregate measurements on the released seed-7 HMM observer, epsilon 0.6, density 0.30) reduced by the upstream OMP post-hoc script at n_train=12 with validation-selected support size; usage mode = forward-only aggregate recovery on the fixed 4-layer x 8-bin coordinate basis, scored against upstream's own coordinate-patching reference on the same evaluation batch. The battery does NOT test the Section 5.2 attribution-patching calibration, the Qwen-2.5-7B experiment, designed (non-random) measurement optimality, other coordinate bases or intervention sizes, or any pretrained model
- data: kwisatzh/mechanistic-tomography@5c097d2 (Apache-2.0 code; the checkpoint and frozen numeric artifacts carry no explicit file-level license, per the intake inventory), every file SHA-256 verified against the frozen intake inventory; the checkpoint is loaded with weights_only=True after verification
- claim label thresholds pre-registered from upstream's own summary: held-out R^2 >= 0.9 and Pearson r >= 0.95 (sparse_recovery_summary.json threshold_crossings); sparse = k <= 8; score = held-out R^2
- upstream row: the base run (seed 7, released pool, n_train=12) selects k=4 {L0B7, L1B7, L2B7, L3B7} with Pearson r = 0.988606 and held-out R^2 = 0.934714; the released sparse_recovery_sample_efficiency.csv row has k=4, r = 0.988606, R^2 = 0.934714 -- reproduced to 1e-9
- null control: the released pool with responses permuted once (seed 0x5ec), so each response is paired with a design independent of the one that produced it -- the registry's declared null (random measurement designs at matched budget) realized as a re-pairing, run through the same finder on the same seeds/bootstrap axes. Direction: on permuted data OMP still returns a support of validation-selected size, so null Jaccard sits at the size-matched random level rather than zero; a real-data Jaccard near that level would mean the split, not the measurements, picks the coordinates
- templates: pools re-measured on CPU from the released checkpoint with the upstream generator at instrument seed 7 (same directions and evaluation batch for all four): the released design re-measured, two fresh signed designs (design seeds 11, 12), and one Bernoulli-mask design (seed 11; upstream's second mask family). Their reference map is the CPU re-measured coordinate-patching map
- budget: the reported 12-measurement model additionally uses 64 validation measurements to choose its support size (and 64 held-out to score it); the n_val=12 hyperparameter run tests selection at a budget matched to the training count, and n_train=8 lies below the budget the claim is made at (upstream's own curve fails there too), so read the hyperparams row of the per-axis table before the pooled one
- instrument transfer (not graded): the upstream generator re-run on CPU (torch 2.13.0) from the released checkpoint regenerates the released masks bit-identically from numpy seed 7; re-measuring the released design gives responses with Pearson r = 0.9974 to the released (MPS) responses, mean |diff| 0.079 (response sd 1.058), and a reference map with r = 0.9973 (same top-4 coordinates); the template pools inherit this CPU evaluation batch
- support identity (not graded): 9/56 perturbed real runs select exactly the base support {L0B7, L1B7, L2B7, L3B7} (bootstrap 1/24, hyperparams 2/4, seeds 4/24, templates 2/4); coordinates selected in at least half of the perturbed real runs: L0B7 (39/56), L2B7 (35/56), L1B7 (30/56)
- null-control supports: 30 distinct supports across 49 null runs (most common {L1B1} x5); median size 1; held-out R^2 mean -0.684, max -0.099; claims: `not recovered; sparse support (k<=8)` x48, `not recovered; dense support (k>8)` x1
- recovery by axis (label 'recovered' = held-out R^2 >= 0.9 and r >= 0.95): seeds 6/24 recovered (median held-out R^2 0.27); bootstrap 5/24 recovered (median held-out R^2 0.58); templates 2/4 recovered (median held-out R^2 0.29); hyperparams 2/4 recovered (median held-out R^2 0.91). Bootstrap resamples measurements with replacement, so a training measurement's duplicate can sit in the held-out set; that biases held-out R^2 upward, not downward
- specificity basis: the null base run selected k=5, so StressKit's 2x size guard grades null Jaccard on 3/49 null runs (sizes [5, 6, 10]) = 0.132, too few for a CI (state inconclusive); over all 49 null runs the null Jaccard is 0.037 and the real/null ratio would be 10.9x, so the graded 3.04x is the conservative (harsher) reading
- post-hoc regrade (verdict-trace mode, from_findings at bootstrap seed 0): grade C, low confidence vs the card's high; checks that differ -- claim_stability: card fail (CI [0.561, 0.789], value 0.684) vs post-hoc inconclusive (CI [0.544, 0.807], value 0.684) -- a check whose CI end sits at its bar is decided by the bootstrap seed, not by the data

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.3 · 2026-09-01T21:44:35+00:00*
