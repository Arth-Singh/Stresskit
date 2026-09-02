# Randomly selected raw records (base run, battery runs, null runs)

Selected with `random.Random(0)`, not cherry-picked. A pair is *recovered* when its unsigned cosine to the best decoder atom clears the bar, and *causally inert* when the matched atom's code is nonzero on exactly none of the feature-ON samples (`fired_frac = 0`).

## Base run (seed 0, upstream defaults, cohort `released-22-well-represented`)

Every inert pair first (deterministic, not sampled), then six randomly selected recovered-but-firing pairs.

| SAE | feature | cosine | fired_frac | causally inert |
|---|---|---|---|---|
| bad_k13 | f2 | 0.9911 | 0.0000 | yes |
| bad_k13 | f5 | 0.9983 | 0.0000 | yes |
| bad_k13 | f7 | 0.9327 | 0.0000 | yes |
| bad_k13 | f8 | 0.9998 | 0.0000 | yes |
| bad_k13 | f9 | 0.9588 | 0.0000 | yes |
| bad_k13 | f10 | 0.9452 | 0.0000 | yes |
| good_k4 | f5 | 0.9997 | 0.0000 | yes |
| good_k4 | f8 | 0.9998 | 0.0000 | yes |
| bad_k13 | f1 | 0.9999 | 1.0000 | no |
| bad_k13 | f17 | 0.9433 | 1.0000 | no |
| good_k4 | f2 | 0.9996 | 1.0000 | no |
| good_k4 | f3 | 0.9975 | 1.0000 | no |
| good_k4 | f13 | 0.9672 | 1.0000 | no |
| good_k4 | f15 | 0.9908 | 1.0000 | no |

## Five randomly selected perturbed runs

| axis | variant | census | pooled inert rate | good | degraded | claim |
|---|---|---|---|---|---|---|
| templates | template=in-context-dense-background | bad_k13:f7 | 0.0244 | 0/22 | 1/19 | inert in degraded only; degraded >= well-trained |
| bootstrap | resample=8 | bad_k13:f10, bad_k13:f2, bad_k13:f5, bad_k13:f7, bad_k13:f8, bad_k13:f9, good_k4:f5, good_k4:f8 | 0.2857 | 3/23 | 9/19 | inert in both; degraded >= well-trained |
| hyperparams | n_samples=200 | bad_k13:f10, bad_k13:f2, bad_k13:f5, bad_k13:f7, bad_k13:f8, bad_k13:f9, good_k4:f5, good_k4:f8 | 0.1951 | 2/22 | 6/19 | inert in both; degraded >= well-trained |
| bootstrap | resample=11 | bad_k13:f2, bad_k13:f9, good_k4:f5, good_k4:f8 | 0.1463 | 4/26 | 2/15 | inert in both; well-trained > degraded |
| bootstrap | resample=2 | bad_k13:f10, bad_k13:f2, bad_k13:f7, bad_k13:f8, bad_k13:f9, good_k4:f5 | 0.1500 | 1/24 | 5/16 | inert in both; degraded >= well-trained |

## Three randomly selected null-control runs (feature-to-probe pairing permuted)

| axis | variant | census size | pooled inert rate | good | degraded | claim |
|---|---|---|---|---|---|---|
| bootstrap | resample=4 | 16 | 0.7500 | 16/18 | 8/14 | inert in both; well-trained > degraded |
| seeds | seed=1520 | 34 | 0.9189 | 22/22 | 12/15 | inert in both; well-trained > degraded |
| seeds | seed=1525 | 38 | 0.9744 | 21/22 | 17/17 | inert in both; degraded >= well-trained |
