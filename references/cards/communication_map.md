# 🟠 Diagnostic Stability Card — descriptive grade **C** (low confidence)

> **Diagnostic OAT profile:** this localizes sensitivity; it does not issue a confirmatory verdict or certificate.

> **Grade rule v0.4:** a check counts as passed only when its whole 95% interval clears the bar; a decided specificity fail caps the grade at C; no null control caps it at B; overlap at or below the at-random floor is D.

> **Claim:** The census of all candidate channels, from 6.3x10^8 in GPT-2 to 1.3x10^11 in Pythia-6.9B, finds that 70-89% of head pairs are oriented far from chance, some coupled strongly and others actively avoiding each other
> model: gpt2, gpt2-medium, gpt2-large, gpt-neo-125m, pythia-160m, pythia-2.8b, pythia-6.9b · task: head-pair census: share of causally eligible head pairs whose coupling coefficient C^2 sits >= 2 SD from its rotation-null mean, per K/Q/V channel · method: upstream map_build.head_head + theory_census (closed-form Weingarten moments) at the pinned commit; weights extracted through the upstream loader

Battery: `bootstrap, hyperparams` — 24 runs (seed 0, 0.484s)

## Checks

| check | value | 95% CI | threshold | state |
|---|---|---|---|---|
| structural stability | 0.888 | [0.757, 1.000] | ≥ 0.800 | ⚠️ inconclusive |
| claim stability | 0.667 | [0.542, 0.833] | ≥ 0.800 | ⚠️ inconclusive |
| score stability | 0.021 | [0.002, 0.035] | ≤ 0.250 | ✅ pass |
| beats random | 7.460 | [6.360, 8.397] | ≥ 3.000 | ✅ pass |
| specificity | 0.888 | [0.755, 1.000] | ≥ 1.500 | ❌ fail |

> ⚠️ **Underpowered:** the 95% CI straddles the bar for structural_stability, claim_stability — undecided in either direction. The grade is provisional; raise `n_runs` before reporting it.

## Downstream utility

**NOT REPORTED** ⚠️ — no task outside interpretability, and no baseline that ignores model internals. A stable finding can still buy nothing.

## Pooled metrics

| metric | value |
|---|---|
| runs | 24 |
| structured runs | 24 |
| empty structural findings | 0 |
| empty structural finding rate | 0.000 |
| mean pairwise Jaccard | 0.888 |
| min pairwise Jaccard | 0.286 |
| random-null Jaccard | 0.119 |
| overlap vs random (×) | 7.460 |
| claim flip rate | 0.464 |
| modal claim share π* | 0.667 |
| distinct claims | 2 |
| score mean | 0.802 |
| score CV | 0.021 |
| median finding size | 4.000 |
| Jaccard 95% CI (bootstrap) | [0.757, 1.000] |
| flip rate 95% CI (bootstrap) | [0.303, 0.548] |
| null-control (specificity) | Jaccard 1.000 · flip 0.000 on 21 null runs |
| claim distribution | `per-model pooled shares inside 70-89%; per-channel entries not all inside`×16, `per-model pooled shares not inside 70-89%; per-channel entries not all inside`×8 |
| score-variance shares (OAT) | bootstrap: 0%, hyperparams: 100% |

## Per-axis breakdown

| axis | runs | Jaccard | flip rate | π* | score CV |
|---|---|---|---|---|---|
| bootstrap | 21 | 0.981 | 0.467 | 0.667 | 0.003 |
| hyperparams | 4 | 0.490 | 0.500 | 0.750 | 0.048 |

## Notes

- pooled Jaccard (0.888) and axis-balanced Jaccard (0.736) diverge by more than 0.1 — per-axis run counts are shaping the pooled number; read the per-axis breakdown before citing it
- underpowered verdict: the 95% CI straddles the bar for claim_stability (fail), structural_stability (pass) at n_runs=20 — these verdict components are not decided by the data. Treat the grade as provisional and raise n_runs (or widen the battery) before reporting it.
- upstream: richardzhewang/communication-map@c8ab3b0 (MIT); head_head, theory_census, load_tl and extract imported unmodified from experiments/map_build.py; file hashes map_build.py 559d516cdc86, gpt2/theory_census.json 0070c1ae3991, gpt2-medium/theory_census.json 0bb5d4e36cfb, gpt2-large/theory_census.json 2f5470989ec5, gpt-neo-125m/theory_census.json 221ea24411ad, pythia-160m/theory_census.json b10325c696fe, pythia-2.8b/theory_census.json 07f781f78dd2, pythia-6.9b/theory_census.json 9170d19eae3b
- environment: transformer-lens 3.8.1, transformers 5.16.1, torch 2.13.0+cu130 on an H200 (upstream lock: transformer-lens 3.7.0, transformers 5.14.1, torch 2.11+cu128 on an RTX 5090). Models are loaded on the CPU by switching map_build.DEV for the duration of load_tl (its GPU-resident fp32 copy of Pythia-2.8B/6.9B does not fit the GPU headroom here); the census runs on the GPU as upstream. Pythia-6.9B goes through load_tl rather than upstream's --stream-load path (validated bit-close upstream by verify_stream.py). Per-pair z is recomputed with the closed-form expression of theory_census, and its |z|>=2 and |z|>=3 shares are asserted equal to the upstream function's on every census
- reproduction of the released census, far-from-chance share per channel (released -> base run, %): gpt2 K 88.8->88.8 Q 87.6->87.6 V 90.5->90.5 (max |dshare| 0.00e+00); gpt2-medium K 81.7->81.7 Q 81.6->81.6 V 83.2->83.2 (max |dshare| 0.00e+00); gpt2-large K 85.1->85.1 Q 85.6->85.6 V 83.1->83.1 (max |dshare| 0.00e+00); gpt-neo-125m K 79.0->79.0 Q 81.3->81.3 V 81.4->81.4 (max |dshare| 0.00e+00); pythia-160m K 77.0->77.0 Q 80.8->80.8 V 84.2->84.2 (max |dshare| 0.00e+00); pythia-2.8b K 68.5->68.5 Q 63.8->63.8 V 77.0->77.0 (max |dshare| 0.00e+00); pythia-6.9b K 87.6->87.6 Q 60.7->60.7 V 81.3->81.3 (max |dshare| 1.97e-06)
- the abstract's 70-89% against the released Table 2 census: per (model, channel) the far-from-chance shares span 60.7-90.5% and 4 of 21 entries fall outside the range (gpt2/V 90.5%, pythia-2.8b/K 68.5%, pythia-2.8b/Q 63.8%, pythia-6.9b/Q 60.7%); pooled over the three channels of each model they span 69.7-89.0% (gpt2 89.0%, gpt2-medium 82.1%, gpt2-large 84.6%, gpt-neo-125m 80.6%, pythia-160m 80.7%, pythia-2.8b 69.7%, pythia-6.9b 76.5%), which is the reading under which the range holds. Base run: pooled 69.7-89.0%, exceptions ['gpt2/V', 'pythia-2.8b/K', 'pythia-2.8b/Q', 'pythia-6.9b/Q']
- threshold=3.0: pooled 58.9-83.3%, 6 exceptions ['gpt-neo-125m/K', 'pythia-160m/K', 'pythia-2.8b/K', 'pythia-2.8b/Q', 'pythia-2.8b/V', 'pythia-6.9b/Q']; per-channel far shares gpt2/K 83.3, gpt2/Q 81.5, gpt2/V 85.1, gpt2-medium/K 74.1, gpt2-medium/Q 74.3, gpt2-medium/V 76.5, gpt2-large/K 78.1, gpt2-large/Q 78.9, gpt2-large/V 75.2, gpt-neo-125m/K 69.5, gpt-neo-125m/Q 72.5, gpt-neo-125m/V 72.8, pythia-160m/K 66.7, pythia-160m/Q 72.1, pythia-160m/V 76.4, pythia-2.8b/K 57.9, pythia-2.8b/Q 51.6, pythia-2.8b/V 67.2, pythia-6.9b/K 81.6, pythia-6.9b/Q 47.4, pythia-6.9b/V 72.7
- processing=raw: pooled 70.6-87.0%, 3 exceptions ['pythia-2.8b/Q', 'pythia-6.9b/K', 'pythia-6.9b/Q']; per-channel far shares gpt2/K 87.3, gpt2/Q 85.5, gpt2/V 88.2, gpt2-medium/K 84.1, gpt2-medium/Q 83.7, gpt2-medium/V 82.5, gpt2-large/K 85.6, gpt2-large/Q 87.4, gpt2-large/V 82.9, gpt-neo-125m/K 85.8, gpt-neo-125m/Q 88.5, gpt-neo-125m/V 79.5, pythia-160m/K 78.7, pythia-160m/Q 80.8, pythia-160m/V 82.7, pythia-2.8b/K 70.5, pythia-2.8b/Q 64.5, pythia-2.8b/V 76.9, pythia-6.9b/K 90.4, pythia-6.9b/Q 62.1, pythia-6.9b/V 80.9
- precision=fp16: pooled 69.7-89.0%, 4 exceptions ['gpt2/V', 'pythia-2.8b/K', 'pythia-2.8b/Q', 'pythia-6.9b/Q']; per-channel far shares gpt2/K 88.8, gpt2/Q 87.6, gpt2/V 90.5, gpt2-medium/K 81.7, gpt2-medium/Q 81.6, gpt2-medium/V 83.2, gpt2-large/K 85.1, gpt2-large/Q 85.6, gpt2-large/V 83.1, gpt-neo-125m/K 79.0, gpt-neo-125m/Q 81.3, gpt-neo-125m/V 81.4, pythia-160m/K 76.9, pythia-160m/Q 80.8, pythia-160m/V 84.2, pythia-2.8b/K 68.5, pythia-2.8b/Q 63.8, pythia-2.8b/V 77.0, pythia-6.9b/K 87.6, pythia-6.9b/Q 60.7, pythia-6.9b/V 81.3
- extension census (not part of the finding): six further models under the upstream processing, far-from-chance shares at |z|>=2 (%): pythia-410m K 74.6 Q 80.7 V 76.7 pooled 77.4; pythia-1b K 93.1 Q 94.3 V 94.4 pooled 94.0 (outside); pythia-1.4b K 76.3 Q 70.7 V 86.5 pooled 77.8; gpt-neo-1.3B K 85.7 Q 85.8 V 87.6 pooled 86.4; opt-125m K 90.9 Q 91.1 V 92.3 pooled 91.4 (outside); opt-1.3b K 88.8 Q 81.9 V 87.0 pooled 85.9
- null control: each writer's output factor replaced by an independent Haar rotation of itself (seed 1516), the census otherwise unchanged; pooled far-from-chance shares 4.2-4.7% (chance is about 5%), every entry an exception. The specificity ratio is uninformative for a range-membership finding: with no signal every entry is outside the band, and 'all outside' is as stable a set as 'these four outside'
- seeds axis not run: the closed-form census has no randomness once the weights are fixed. Templates axis not run: the census takes no text or prompt input; the extension census above is the nearest substitute and is reported, not graded
- v0.3 grade: B; regraded 2026-09-03 under grade rule v0.4 from the recorded checks (schema 0.5)

*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) v1.0.0.dev0 · schema 0.5 · 2026-09-02T09:55:25+00:00*
