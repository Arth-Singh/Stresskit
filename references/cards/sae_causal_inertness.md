# 🟠 Diagnostic Stability Card — descriptive grade **C** (low confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Claim:** Our central result is causal: subjecting every recovered feature to ablation and steering, we find up to 77% of features passing a recovery bar (cosine >= 0.90) in a degraded SAE -- and 9% in a well-trained one -- are causally inert: the matched atom never fires when the feature is present, including matches at cosine ~1.000.
> model: Elhage et al. (2022) toy bottleneck model trained by the released harness (32 features, 8 hidden dims, sparsity 0.95, seed 0) with two TopK SAEs on its hidden activations (d_sae 128, k=4 well-trained and k=13 degraded); mohamed-bal/sae-causal-audit@3915d95 · task: census of causally inert pairs among cosine-recovered (ground-truth direction, decoder atom) matches · method: signed cosine matching against W_dec, then per-pair fired_frac / ablation / steering measurement through encode+decode, upstream sae_causal_audit.run_audit

Battery: `seeds, bootstrap, templates, hyperparams` — 33 runs (seed 0, 571.295s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.332 | [0.259, 0.423] | ≥ 0.800 | ❌ fail |
| claim stability | 0.848 | [0.727, 0.939] | ≥ 0.800 | ⚠️ inconclusive |
| score stability | 0.387 | [0.252, 0.502] | ≤ 0.250 | ❌ fail |
| beats random | 5.449 | [4.252, 6.938] | ≥ 3.000 | ✅ pass |
| specificity | 0.634 | [0.470, 0.856] | ≥ 1.500 | ❌ fail |

> ⚠️ **Underpowered:** the 95% CI straddles the bar for claim_stability — undecided in either direction. The grade is provisional; raise `n_runs` before reporting it.

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 33 |
| structured runs | 33 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.332 |
| Jaccard incl. size-mismatched runs | 0.279 |
| min pairwise Jaccard | 0.000 |
| random-null Jaccard | 0.061 |
| overlap vs random (×) | 5.449 |
| claim flip rate | 0.277 |
| modal claim share π* | 0.848 |
| distinct claims | 3 |
| score mean | 0.192 |
| score CV | 0.387 |
| median finding size | 6 |
| Jaccard 95% CI (bootstrap) | [0.259, 0.423] |
| flip rate 95% CI (bootstrap) | [0.119, 0.467] |
| null-control (specificity) | Jaccard 0.523 · flip 0.153 on 25 null runs |
| claim distribution | `inert in both; degraded >= well-trained`×28, `inert in degraded only; degraded >= well-trained`×3, `inert in both; well-trained > degraded`×2 |
| score-variance shares (OAT) | bootstrap: 32%, hyperparams: 2%, seeds: 14%, templates: 52% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 13 | 0.521 | 0.410 | 0.769 | 0.326 |
| hyperparams | 5 | 0.639 | 0.000 | 1.000 | 0.094 |
| seeds | 13 | 0.246 | 0.000 | 1.000 | 0.236 |
| templates | 5 | 0.356 | 0.600 | 0.600 | 0.864 |

## Notes

- structural stability graded on 29 size-comparable runs; 4 run(s) with >2x size difference excluded (pooled-over-all value kept as mean_pairwise_jaccard_all_sizes)
- pooled Jaccard (0.332) and axis-balanced Jaccard (0.441) diverge by more than 0.1 — per-axis run counts are shaping the pooled number; read the per-axis breakdown before citing it
- underpowered verdict: the 95% CI straddles the bar for claim_stability (pass) at n_runs=12 — these verdict components are not decided by the data. Treat the grade as provisional and raise n_runs (or widen the battery) before reporting it.
- scope: graded artifact = the released toy-regime pipeline (Elhage-style bottleneck model at ToyConfig(seed=0), two TopK SAEs at k=4 and k=13, d_sae 128) audited by upstream's own run_audit; usage mode = the known-ground-truth toy regime that the paper's Table 5 census is computed in. The battery does NOT test the real-model regime (GPT-2-small, 83 concepts, gpt2-small-res-jb), the superposition phase-diagram or TopK-versus-L1 reproductions, the ablation/steering specificity medians, or the read-inert versus write-inert taxonomy; and the 77% in the claim sentence is the originating write-up's figure, which the released instrument does not recompute -- Table 5's own degraded-SAE census is 17%
- data: no external dataset; every number is regenerated on CPU from mohamed-bal/sae-causal-audit@3915d95 (MIT), each imported source file SHA-256 verified against this runner before import. torch 2.13.0+cu130; upstream guarantees byte-exact results only inside its pinned CI environment (ubuntu-24.04, torch==2.13.0+cpu) and semantic agreement within rtol=1e-4 elsewhere
- finding representation pre-registered before any run: components = recovered-and-inert (SAE, feature) pairs over a universe of 64 pairs; score = pooled inert rate among recovered pairs; claim = presence of inertness in each SAE plus which SAE has the higher rate, both read directly off the claim sentence with no free threshold
- null control: the feature-to-probe pairing permuted once by a derangement (seed 0x5ec), so every matched atom is asked whether it fires for a ground-truth feature it was not matched to, while matching, the readout dimension and the recovery bar stay untouched. Direction: the null is strict rather than conservative. Inertness is the absence of an effect, so breaking the pairing pushes the census toward saturation -- a large, near-identical set every run, which inflates null Jaccard and therefore depresses the specificity ratio. Read a specificity failure here as 'the identity of the inert pairs is not more stable than a saturated census', not as evidence that the real census is random; StressKit's separate beats-random check already carries the size-matched random comparison
- inert rate by eligibility cohort (not graded): all-32-features (n=1): pooled inert rate 0.233 median, good 0.130, degraded 0.350; all-32-in-context (n=1): pooled inert rate 0.047 median, good 0.043, degraded 0.050; in-context-dense-background (n=1): pooled inert rate 0.024 median, good 0.000, degraded 0.053; in-context-presentation (n=1): pooled inert rate 0.024 median, good 0.000, degraded 0.053; released-22-well-represented (n=29): pooled inert rate 0.216 median, good 0.091, degraded 0.316
- presentation regime (not graded, and the largest single effect in this battery): with the feature presented in isolation as upstream does (n=30, census 7.0 pairs on average, pooled inert rate 0.208); with the same feature presented inside a sparse background, which is how upstream's own real-model regime defines feature-ON (n=3, census 1.3 pairs on average, pooled inert rate 0.032). Most of the census is specific to the isolation regime: an atom that never wins the TopK competition for a feature presented alone can still win it when the feature arrives in company. This is a statement about the scope of the measurement, not about whether the isolated-regime census is correct -- it is upstream's declared regime and it reproduces there
- per-SAE inert rates by axis (not graded; the claim sentence quotes 9% for the well-trained SAE and up to 77% for the degraded one): seeds (n=12): good 0.110+/-0.039, degraded 0.312+/-0.114, census size 5-11; bootstrap (n=12): good 0.108+/-0.073, degraded 0.345+/-0.131, census size 2-8; templates (n=4): good 0.043+/-0.053, degraded 0.126+/-0.129, census size 1-10; hyperparams (n=4): good 0.102+/-0.020, degraded 0.349+/-0.020, census size 6-10
- inert-pair geometry (not graded): across 33 real runs with a non-empty census, 29/33 contain at least one inert pair at cosine >= 0.999 (largest inert cosine 0.9921 on average, min over runs 0.9327), which is the existential reading the claim sentence makes; the census is not confined to those pairs, and its lowest inert cosine is 0.9353 on average (min 0.8881); 0 run(s) produced an empty census
- null-control census: over 25 null runs the census holds 28.4 pairs on average (range 16-40, real base 8), pooled inert rate 0.919+/-0.048, and shares 5.0 pairs with the real base census; claims: `inert in both; well-trained > degraded` x23, `inert in both; degraded >= well-trained` x2
- specificity basis: the null base run selected 32 pairs, so StressKit's 2x size guard grades null Jaccard on 25/25 null runs = 0.523; over all 25 null runs the null Jaccard is 0.523 and the real/null ratio would be 0.63x
- post-hoc regrade (verdict-trace mode, from_findings at bootstrap seed 0): grade C, low confidence vs the card's low; every check has the same state as on the card -- a check whose CI end sits at its bar is decided by the bootstrap seed, not by the data
- released census, run 1 (Linux 6.11 x86_64, CPython 3.12.3, torch 2.13.0+cu130 run CPU-only, numpy 2.5.2, outside Docker): good_k4 recovered 22, causally inert 2 -- both exactly the released values; bad_k13 recovered 19 (released 18 +/-1, within) and causally inert 6 (released 3 +/-1, OUTSIDE by 2). scripts/verify_results_tolerance.py exits 1 with DRIFT bad_k13.recovered_inert expected=3.0 got=6.0.
- released census, run 2 (macOS-15.5-arm64, CPython 3.12.12, torch 2.13.0 arm64 build, numpy 2.5.2, outside Docker): good_k4 recovered 22 and causally inert 3 against a released 2 (+1); bad_k13 recovered 17 and causally inert 3, i.e. exactly the value run 1 missed. Re-running in the same environment reproduces summary.json byte-for-byte, so the pipeline is deterministic within an environment and divergent across environments. NEITHER run used upstream's pinned wheel (torch==2.13.0+cpu on ubuntu-24.04); a run inside the published Docker image is the outstanding check and is not claimed here.
- the paper predicts this, names the mechanism, and bounds it -- credit where due. Section 8 documents that byte-exact cross-platform reproduction is unavailable because the torch 2.13.0+cpu wheels for different platforms are different binaries with different MKL versions; Table 5 annotates the degraded-SAE census as '3 (17%); 4 in +/-1 band'; Section 9 scopes the repository's two guarantees explicitly (byte-exact only inside the pinned CI environment, semantic -- continuous metrics within rtol 1e-4 and boundary-sensitive counts within +/-1 -- on any platform); and Section 9 reports the same flip we observed, one feature at cosine 0.924 sitting close enough to both the recovery bar and the TopK selection boundary that different BLAS builds resolve it differently.
- measured against that stated standard: run 2's deviation (+1 on good_k4.recovered_inert) is INSIDE the paper's +/-1 semantic guarantee and is therefore a reproduction success by the paper's own criterion. It trips the gate only because expected_results.json encodes atol=1 for the two bad_k13 counts and bare integers for good_k4, so the machine-readable file is stricter than the prose promises for that SAE. That is a spec/gate mismatch in the artifact, not a scientific failure, and the honest headline is that the released tolerance file does not implement the +/-1 band the paper documents. Run 1's deviation (+3 on bad_k13.recovered_inert) does exceed the band, but it used a CUDA-built wheel rather than the pinned CPU wheel, so it does not by itself establish that the band is too narrow.
- abstract versus Table 5, denominators: the abstract reads 'up to 77% of features passing a standard recovery bar (cosine >= 0.90) in a degraded SAE ... are causally inert'. The 77% is 17 inert of 22 MATCHED pairs, computed in the Section 6.3 pre-instrument experiment with no recovery bar applied; the bar-conditioned quantity is the Table 5 census, 3 inert of 18 recovered = 17%. So the abstract attaches the cosine >= 0.90 bar to a figure computed without it. The paper's body is not hiding this -- it states '17 of 22 pairs (77%)' with its denominator, and Table 5 prints 17% -- but the two numbers differ by a factor of four and only the smaller one is conditioned as the abstract describes. For the well-trained SAE the denominators coincide (recovered = matched = 22) and 9% is unchanged between the two sections.

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.4 · 2026-09-01T23:26:42+00:00*
