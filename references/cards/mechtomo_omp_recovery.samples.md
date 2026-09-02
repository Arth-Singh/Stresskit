# Randomly selected raw records (released pool and battery runs)

Selected with `random.Random(0)`, not cherry-picked. Masks are shown per layer (L0..L3), one character per time bin: `+` = +epsilon, `-` = -epsilon, `.` = untouched.

## Base run (seed 7, released pool): recovered coefficients vs the coordinate-patching reference

Selected coordinates first (deterministic, not sampled), then the four largest unselected reference effects.

| coordinate | OMP coefficient | reference effect | selected |
|---|---|---|---|
| L0B7 |  1.0983 |  1.1890 | yes |
| L2B7 |  0.8263 |  1.1404 | yes |
| L1B7 |  0.8953 |  1.1058 | yes |
| L3B7 |  0.6798 |  0.7319 | yes |
| L0B6 |  0.0000 |  0.1509 | no |
| L1B6 |  0.0000 |  0.1018 | no |
| L1B5 |  0.0000 |  0.0830 | no |
| L0B5 |  0.0000 |  0.0769 | no |

## Six randomly selected released measurements (base-run split role, base-fit prediction)

| # | mask (L0 L1 L2 L3) | response y | base fit | role in base run |
|---|---|---|---|---|
| 20 | `+....... +..-.+.. .-.+--++ .+..+..-` |  0.4738 |  0.0667 | held-out |
| 132 | `-....+.. -.+....- .-.-..-. .-..-...` | -1.0915 | -0.9751 | validation |
| 197 | `..-.-.+. .+..+... .++.-.-. ......-.` |  0.1322 | -0.0798 | unused (pool beyond the first 12) |
| 207 | `.+.+.-.. .-..+.+. .+.....- .......-` | -1.4227 | -1.5860 | train |
| 215 | `-+...-++ ++.-...- +.....-. .+...+++` |  0.8782 |  0.8030 | unused (pool beyond the first 12) |
| 248 | `-...++.. +..-...+ ...+..+. -.....++` |  1.8262 |  1.4952 | validation |

## Five randomly selected perturbed runs

| axis | variant | support | held-out R^2 | Pearson r | claim |
|---|---|---|---|---|---|
| templates | template=fresh-signed-design-12 | L1B7, L3B7 | -0.3647 |  0.5094 | not recovered; sparse support (k<=8) |
| hyperparams | n_train=16 | L0B7, L1B7, L2B7, L3B7 |  0.9571 |  0.9917 | recovered; sparse support (k<=8) |
| seeds | seed=27 | L3B7 |  0.0243 |  0.3008 | not recovered; sparse support (k<=8) |
| bootstrap | resample=7 | L0B4, L1B0, L1B5, L2B7, L3B6 | -1.5191 |  0.0853 | not recovered; sparse support (k<=8) |
| seeds | seed=30 | L2B7 |  0.2201 |  0.5161 | not recovered; sparse support (k<=8) |

## Three randomly selected null-control runs (responses permuted)

| axis | variant | support | held-out R^2 | Pearson r | claim |
|---|---|---|---|---|---|
| bootstrap | resample=13 | L3B5 | -0.1453 | -0.0848 | not recovered; sparse support (k<=8) |
| seeds | seed=1528 | L3B4 | -0.4526 | -0.0848 | not recovered; sparse support (k<=8) |
| bootstrap | resample=8 | L0B6 | -1.0892 | -0.0053 | not recovered; sparse support (k<=8) |
