"""Null-score leak: does a card's null control still score like its real data?

Every reference card with a null control records how stable the finder is on
the null (``metrics.null_control.mean_pairwise_jaccard`` or
``mean_pairwise_abs_cosine``) and, unchecked by any verdict, how well the null
SCORES (``score_mean``, ``score_cv``, ``n_runs``). The specificity check
compares stability only. When the null still scores as well as the real data
the null has not removed the signal, and a specificity failure on that card
means "the null is too soft", not "the method is non-specific". This module
quantifies that gap for one card at a time.

Inputs are the real per-run scores and either the null per-run scores
(:func:`leak_from_runs`) or the null's summary pair
(:func:`leak_from_summaries`). ``Finding.score`` declares no polarity, so the
caller states it:

- ``polarity`` is ``+1`` when a higher score is a stronger finding and ``-1``
  when a lower score is (a cross-entropy delta, a loss, a reconvergence
  ratio).
- ``scale`` is ``"ratio"`` for a non-negative score whose no-signal value is
  zero (a share, a rate, a fraction recovered), where the retention
  ``null_mean / real_mean`` is meaningful, and ``"signed"`` otherwise (a
  difference of two rates, a cosine, an R^2, an AUC or an accuracy with a
  chance floor). Retention is undefined for ``polarity=-1``, so that pairing
  is rejected: use ``"signed"``.

Statistics reported per card:

- ``d``: the signed standardized difference
  ``polarity * (real_mean - null_mean) / pooled_sd``, where ``pooled_sd`` is
  the run-count-weighted root mean square of the two population standard
  deviations, ``sqrt((n_real * sd_real^2 + n_null * sd_null^2) /
  (n_real + n_null))``. ``None`` when the pooled sd is zero.
- ``z``: the Welch z of the same signed difference, ``polarity *
  (real_mean - null_mean) / sqrt(sd_real^2 / n_real + sd_null^2 / n_null)``.
  ``None`` when the standard error is zero.
- ``retention``: ``null_mean / real_mean`` on the ratio scale when
  ``real_mean > 0``, else ``None``.
- ``ci_difference``: a percentile bootstrap interval on
  ``polarity * (mean(real) - mean(null))``, resampling the two groups
  independently; only available from per-run scores.

Leak classes (:func:`classify_leak`). The thresholds are choices, not
estimates, and deliberately coarse so that a class never hinges on the third
decimal of a card:

- ``null_matches_or_exceeds``: ``d <= 0.5`` (the null sits within half a
  pooled standard deviation of the real data, or beats it), or on the ratio
  scale ``retention >= 0.9`` (the null keeps at least 90% of the real score).
- ``null_degraded``: ``d >= 1.0`` and ``z >= 1.96`` (a full pooled standard
  deviation apart, decided at the two-sided 5% level) and, on the ratio scale,
  ``retention <= 0.5`` (the null keeps at most half the real score).
- ``partial``: everything in between.

When the pooled sd is zero, ``d`` and ``z`` are ``None`` and the class comes
from the retention alone (ratio scale) or from the sign of the signed
difference (signed scale): a null that scores no worse than the real data
matches it; one that scores strictly worse with zero spread is degraded.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

from .metrics import _percentile_ci, mean, std

POLARITIES = (1, -1)
SCALES = ("ratio", "signed")

CLASS_MATCHES = "null_matches_or_exceeds"
CLASS_DEGRADED = "null_degraded"
CLASS_PARTIAL = "partial"
CLASSES = (CLASS_MATCHES, CLASS_PARTIAL, CLASS_DEGRADED)

D_MATCHES_MAX = 0.5
D_DEGRADED_MIN = 1.0
Z_DEGRADED_MIN = 1.96
RETENTION_MATCHES_MIN = 0.9
RETENTION_DEGRADED_MAX = 0.5

THRESHOLDS = {
    "d_matches_max": D_MATCHES_MAX,
    "d_degraded_min": D_DEGRADED_MIN,
    "z_degraded_min": Z_DEGRADED_MIN,
    "retention_matches_min": RETENTION_MATCHES_MIN,
    "retention_degraded_max": RETENTION_DEGRADED_MAX,
}


def _check_polarity_scale(polarity: int, scale: str) -> None:
    if polarity not in POLARITIES:
        raise ValueError(f"polarity must be +1 or -1, got {polarity!r}")
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}, got {scale!r}")
    if scale == "ratio" and polarity == -1:
        raise ValueError(
            "retention null/real is undefined for a lower-is-stronger score; "
            "use scale='signed' with polarity=-1"
        )


def _check_group(label: str, sd: Optional[float], n: int) -> None:
    if n < 2:
        raise ValueError(f"{label}: need >= 2 runs for a standard deviation, got {n}")
    if sd is None or not math.isfinite(sd) or sd < 0:
        raise ValueError(
            f"{label}: standard deviation must be finite and >= 0, got {sd!r}"
        )


def pooled_sd(sd_a: float, n_a: int, sd_b: float, n_b: int) -> float:
    """Run-count-weighted root mean square of two population sds."""
    return math.sqrt((n_a * sd_a**2 + n_b * sd_b**2) / (n_a + n_b))


def bootstrap_ci_difference(
    a: Sequence[float],
    b: Sequence[float],
    *,
    seed: int = 0,
    n_boot: int = 500,
    alpha: float = 0.05,
) -> Optional[List[float]]:
    """Percentile bootstrap CI on ``mean(a) - mean(b)``.

    The two groups are resampled independently. Follows the ``_percentile_ci``
    convention of :mod:`stresskit.metrics`: at least 4 values per group, and at
    least ``max(20, n_boot // 10)`` replicates, else ``None``.
    """
    a = list(a)
    b = list(b)
    if len(a) < 4 or len(b) < 4:
        return None
    rng = random.Random(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        vals.append(mean(ra) - mean(rb))
    return _percentile_ci(vals, n_boot, alpha)


def classify_leak(stats: Dict[str, Any]) -> str:
    """One of ``null_matches_or_exceeds``, ``partial``, ``null_degraded``.

    ``stats`` is the dict returned by :func:`leak_from_summaries` or
    :func:`leak_from_runs`; the thresholds are the module constants.
    """
    scale = stats["scale"]
    retention = stats["retention"]
    d = stats["d"]
    z = stats["z"]
    on_ratio = scale == "ratio" and retention is not None
    if on_ratio and retention >= RETENTION_MATCHES_MIN:
        return CLASS_MATCHES
    if d is None:
        return _classify_without_spread(stats, on_ratio)
    if d <= D_MATCHES_MAX:
        return CLASS_MATCHES
    retention_degraded = scale != "ratio" or (
        retention is not None and retention <= RETENTION_DEGRADED_MAX
    )
    if (
        d >= D_DEGRADED_MIN
        and z is not None
        and z >= Z_DEGRADED_MIN
        and retention_degraded
    ):
        return CLASS_DEGRADED
    return CLASS_PARTIAL


def _classify_without_spread(stats: Dict[str, Any], on_ratio: bool) -> str:
    if on_ratio:
        if stats["retention"] <= RETENTION_DEGRADED_MAX:
            return CLASS_DEGRADED
        return CLASS_PARTIAL
    if stats["difference"] <= 0:
        return CLASS_MATCHES
    return CLASS_DEGRADED


def leak_from_summaries(
    real_mean: float,
    real_sd: float,
    n_real: int,
    null_mean: float,
    null_sd: float,
    n_null: int,
    *,
    polarity: int,
    scale: str,
) -> Dict[str, Any]:
    """Leak statistics from the two groups' means, population sds and sizes.

    This is the only path for a card whose null per-run scores were not kept;
    ``ci_difference`` is then ``None``.
    """
    _check_polarity_scale(polarity, scale)
    _check_group("real", real_sd, n_real)
    _check_group("null", null_sd, n_null)
    difference = polarity * (real_mean - null_mean)
    pooled = pooled_sd(real_sd, n_real, null_sd, n_null)
    se = math.sqrt(real_sd**2 / n_real + null_sd**2 / n_null)
    retention = null_mean / real_mean if scale == "ratio" and real_mean > 0 else None
    stats: Dict[str, Any] = {
        "polarity": polarity,
        "scale": scale,
        "real_mean": real_mean,
        "real_sd": real_sd,
        "n_real": n_real,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "n_null": n_null,
        "difference": difference,
        "pooled_sd": pooled,
        "d": difference / pooled if pooled > 0 else None,
        "z": difference / se if se > 0 else None,
        "retention": retention,
        "ci_difference": None,
    }
    stats["leak_class"] = classify_leak(stats)
    return stats


def leak_from_runs(
    real_scores: Sequence[float],
    null_scores: Sequence[float],
    *,
    polarity: int,
    scale: str,
    seed: int = 0,
    n_boot: int = 500,
) -> Dict[str, Any]:
    """Leak statistics from per-run scores, with the bootstrap CI on
    ``polarity * (mean(real) - mean(null))``."""
    _check_polarity_scale(polarity, scale)
    real = [float(x) for x in real_scores]
    null = [float(x) for x in null_scores]
    for label, xs in (("real_scores", real), ("null_scores", null)):
        if len(xs) < 2:
            raise ValueError(f"{label}: need >= 2 scores, got {len(xs)}")
        if not all(math.isfinite(x) for x in xs):
            raise ValueError(f"{label}: contains a non-finite score")
    stats = leak_from_summaries(
        mean(real),
        std(real),
        len(real),
        mean(null),
        std(null),
        len(null),
        polarity=polarity,
        scale=scale,
    )
    stats["ci_difference"] = bootstrap_ci_difference(
        [polarity * x for x in real],
        [polarity * x for x in null],
        seed=seed,
        n_boot=n_boot,
    )
    return stats
