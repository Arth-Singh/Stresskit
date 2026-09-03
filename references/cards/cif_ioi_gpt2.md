# 🟠 Diagnostic Stability Card — descriptive grade **C** (low confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Grade rule v0.4:** a check counts as passed only when its whole 95% interval clears the bar; a decided specificity fail caps the grade at C; no null control caps it at B; overlap at or below the at-random floor is D.

> **Claim:** CIF certifies high-fidelity claims for GPT-2 Small IOI circuits, and its betting confidence sequence reduces certification cost 10-30x relative to the Hoeffding sequence
> model: gpt2 · task: IOI head-output patching, 200 prompts of one template, nested circuits of 3/7/9/11/13 heads (upstream E2) · method: CIF anytime-valid confidence sequences (Hoeffding and betting) on clipped normalised logit-difference recovery

Battery: `seeds, bootstrap, templates, hyperparams` — 36 runs (seed 0, 214.561s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.569 | [0.416, 0.715] | ≥ 0.800 | ❌ fail |
| claim stability | 0.611 | [0.472, 0.778] | ≥ 0.800 | ❌ fail |
| score stability | 0.045 | [0.013, 0.065] | ≤ 0.250 | ✅ pass |
| beats random | 3.680 | [2.688, 4.625] | ≥ 3.000 | ⚠️ inconclusive |

> ⚠️ **Underpowered:** the 95% CI straddles the bar for beats_random — undecided in either direction. The grade is provisional; raise `n_runs` before reporting it.

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 36 |
| structured runs | 36 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.569 |
| min pairwise Jaccard | 0.000 |
| random-null Jaccard | 0.155 |
| overlap vs random (×) | 3.680 |
| claim flip rate | 0.597 |
| modal claim share π* | 0.611 |
| distinct claims | 7 |
| score mean | 0.941 |
| score CV | 0.045 |
| median finding size | 5.000 |
| Jaccard 95% CI (bootstrap) | [0.416, 0.715] |
| flip rate 95% CI (bootstrap) | [0.404, 0.745] |
| claim distribution | `name-movers certified F0<=0.95; @0.8:<10x; @0.9:10-30x+`×22, `name-movers certified F0<=0.9; @0.8:<10x; @0.9:10-30x+`×7, `name-movers certified F0<=0.95; @0.8:<10x; @0.9:10-30x`×2, `name-movers certified F0<=none; @0.8:n/a; @0.9:n/a`×2, `name-movers certified F0<=0.8; @0.8:10-30x; @0.9:n/a`×1 |
| score-variance shares (OAT) | bootstrap: 0%, hyperparams: 62%, seeds: 0%, templates: 37% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 11 | 0.939 | 0.327 | 0.818 | 0.005 |
| hyperparams | 5 | 0.211 | 0.900 | 0.400 | 0.070 |
| seeds | 11 | 1.000 | 0.000 | 1.000 | 0.001 |
| templates | 12 | 0.443 | 0.727 | 0.500 | 0.054 |

## Notes

- underpowered verdict: the 95% CI straddles the bar for beats_random (pass) at n_runs=10 — these verdict components are not decided by the data. Treat the grade as provisional and raise n_runs (or widen the battery) before reporting it.
- upstream: AsiaeeLab/certified-interventional-fidelity@4b8359f (MIT); sampling code imported unmodified; file hashes e2_gpt2_patching.py 58397de92eeb, cif.py 4c9824c9d46b, e2_completeness.csv eb2dc5bf064f
- reproduction: the base run (seed 0, upstream template) reproduces 30 of 30 shipped i.i.d. certification rows exactly
- cost ratio (Hoeffding draws / betting draws to certify) in the base run: 3 heads @F0=0.8: 7.2x; 3 heads @F0=0.9: >=18.2x; 3 heads @F0=0.95: >=2.4x; 7 heads @F0=0.8: 6.6x; 7 heads @F0=0.9: >=16.9x; 7 heads @F0=0.95: >=1.3x; 9 heads @F0=0.8: 6.9x; 9 heads @F0=0.9: 19.0x; 9 heads @F0=0.95: >=4.8x; 11 heads @F0=0.8: 6.9x; 11 heads @F0=0.9: 18.8x; 11 heads @F0=0.95: >=5.4x; 13 heads @F0=0.8: 6.8x; 13 heads @F0=0.9: 18.4x; 13 heads @F0=0.95: >=5.6x. The abstract's 10-30x is reached only at F0=0.9, and only for the circuits Hoeffding certifies at all within 2000 draws; at F0=0.8 the ratio is 6.6-7.2x, and at F0=0.95 Hoeffding never certifies, so the ratio there is a censored lower bound.
- exact population means the sequences are estimating (the 200 prompts are the whole population, sampled with replacement): 3 heads 0.9628, 7 heads 0.9577, 9 heads 0.9690, 11 heads 0.9695, 13 heads 0.9707. E2 simulates a streaming certificate on a finite pool whose mean is computable exactly; the certificate demonstrates the machinery and adds no information about the circuit beyond those 200 effects.
- templates: the upstream repository ships one IOI template; the eleven alternatives are constructed here from the IOI paper's template family, with the same names, the same prompt seed, and the same IO-replacement corruption, and are labelled as such
- no null control: the null outcome of a certification procedure (nothing certified) is itself a stable profile, so the specificity check is undefined for this class of claim
- v0.3 grade: B; regraded 2026-09-03 under grade rule v0.4 from the recorded checks (schema 0.5)

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.5 · 2026-09-02T07:31:12+00:00*
