"""Stability metrics and explicitly labeled inference candidates.

Core metrics draw on published protocols:

- pairwise Jaccard under resampling: Méloux, Portet & Peyrard,
  "Mechanistic Interpretability as Statistical Estimation" (arXiv:2510.00845).
- flip rate / modal share / filability: Mahale, "Explanation Multiplicity"
  (arXiv:2608.13754) — the unbiased Gini–Simpson form of the probability
  that two runs yield different claims.
- approximate random Jaccard k/(2N−k): same paper's size-matched null.
- pipeline variance shares: van der Ben et al., "Building Fast, Evaluating
  Slow" (arXiv:2607.19386) — here in a simple one-at-a-time (OAT) form.

Exact finite nulls and interval implementations added by StressKit are derived
and calibrated separately; their docstrings state assumptions and status.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections import Counter
from fractions import Fraction
from typing import Dict, List, Optional, Sequence


# --------------------------------------------------------------------------
# Structural stability
# --------------------------------------------------------------------------

def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity of two component sets. J(∅, ∅) is defined as 1.0."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def jaccard_fraction(a: frozenset, b: frozenset) -> Fraction:
    """Exact rational form of :func:`jaccard` for formal conformance."""
    if not a and not b:
        return Fraction(1, 1)
    union = len(a | b)
    return Fraction(len(a & b), union)


def pairwise_jaccard(sets: Sequence[frozenset]) -> List[float]:
    """All unordered pairwise Jaccard similarities."""
    return [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]


def mean_pairwise_jaccard(sets: Sequence[frozenset]) -> Optional[float]:
    """Mean pairwise Jaccard; None if fewer than two sets."""
    pj = pairwise_jaccard(sets)
    if not pj:
        return None
    return sum(pj) / len(pj)


def expected_random_jaccard(k: float, universe_size: int) -> Optional[float]:
    """Approximate E[J] for two uniform random k-subsets of an N-universe.

    E[|A∩B|] = k²/N, so J ≈ (k²/N) / (2k − k²/N) = k / (2N − k).
    This is the analytic size-matched null of arXiv:2608.13754.
    """
    if universe_size is None or universe_size <= 0 or k <= 0:
        return None
    k = min(k, universe_size)
    return k / (2 * universe_size - k)


def exact_expected_random_jaccard(
    k: int,
    universe_size: int,
    other_size: Optional[int] = None,
) -> Optional[float]:
    """Exact E[J] for independent uniform subsets of fixed sizes.

    If ``A`` and ``B`` are uniform ``k``- and ``l``-subsets of an
    ``N``-element universe, ``|A ∩ B|`` is hypergeometric.  This function
    averages ``x / (k + l - x)`` over that exact finite distribution rather
    than substituting the expected intersection into the nonlinear ratio.

    ``other_size`` defaults to ``k``.  Empty-set behavior follows
    :func:`jaccard`: two empty sets have agreement 1; one empty set and one
    nonempty set have agreement 0.  Returns ``None`` for a nonpositive
    universe and raises when a requested subset cannot exist.
    """
    value = exact_expected_random_jaccard_fraction(k, universe_size, other_size)
    return float(value) if value is not None else None


def exact_expected_random_jaccard_fraction(
    k: int,
    universe_size: int,
    other_size: Optional[int] = None,
) -> Optional[Fraction]:
    """Exact rational form of :func:`exact_expected_random_jaccard`."""
    n = int(universe_size)
    k = int(k)
    l = k if other_size is None else int(other_size)
    if n <= 0:
        return None
    if k < 0 or l < 0:
        raise ValueError(f"subset sizes must be nonnegative, got {k}, {l}")
    if k > n or l > n:
        raise ValueError(
            f"subset sizes k={k}, l={l} exceed universe_size={n}"
        )
    if k == 0 or l == 0:
        return Fraction(1 if k == l == 0 else 0, 1)

    support_lo = max(0, k + l - n)
    support_hi = min(k, l)
    normalizer = math.comb(n, l)
    expectation = Fraction(0, 1)
    for intersection in range(support_lo, support_hi + 1):
        if intersection == 0:
            continue
        probability = Fraction(
            math.comb(k, intersection) * math.comb(n - k, l - intersection),
            normalizer,
        )
        agreement = Fraction(intersection, k + l - intersection)
        expectation += agreement * probability
    return expectation


def exact_expected_random_jaccard_sizes(
    sizes: Sequence[int], universe_size: int
) -> Optional[float]:
    """Exact E[mean pairwise Jaccard] for heterogeneous uniform-set sizes.

    One independent uniform subset is drawn for each entry in ``sizes``.
    Linearity of expectation makes the target the mean of the exact
    pair-specific expectations.  Returns ``None`` with fewer than two sizes or
    a nonpositive universe.
    """
    if universe_size <= 0 or len(sizes) < 2:
        return None
    pair_expectations: List[float] = []
    for k, l in itertools.combinations((int(s) for s in sizes), 2):
        value = exact_expected_random_jaccard(k, universe_size, l)
        if value is None:  # guarded above; keeps the optional contract explicit
            return None
        pair_expectations.append(value)
    return sum(pair_expectations) / len(pair_expectations)


def exact_expected_core_noise_jaccard(
    core_size: int,
    noise_size: int,
    universe_size: int,
) -> Optional[float]:
    """Exact E[J] for a fixed core plus uniformly sampled nuisance items.

    Each finding contains the same ``core_size`` elements and an independent
    uniform ``noise_size``-subset of the remaining universe.  This provides a
    known-truth continuum between a deterministic finding and a wholly random
    fixed-size finding.
    """
    n = int(universe_size)
    core = int(core_size)
    noise = int(noise_size)
    if n <= 0:
        return None
    if core < 0 or noise < 0:
        raise ValueError(
            f"core_size and noise_size must be nonnegative, got {core}, {noise}"
        )
    available = n - core
    if core > n or noise > available:
        raise ValueError(
            "core and noise must fit within universe_size: "
            f"core_size={core}, noise_size={noise}, universe_size={n}"
        )
    if noise == 0:
        return 1.0

    normalizer = math.comb(available, noise)
    support_lo = max(0, 2 * noise - available)
    expectation = Fraction(0, 1)
    for intersection in range(support_lo, noise + 1):
        probability = Fraction(
            math.comb(noise, intersection)
            * math.comb(available - noise, noise - intersection),
            normalizer,
        )
        agreement = Fraction(core + intersection, core + 2 * noise - intersection)
        expectation += agreement * probability
    return float(expectation)


# --------------------------------------------------------------------------
# Claim stability
# --------------------------------------------------------------------------

def flip_rate(labels: Sequence[str]) -> Optional[float]:
    """Probability that two distinct runs yield different claims.

    Unbiased Gini–Simpson form: F = 1 − Σ n_c(n_c−1) / (N(N−1)).
    Returns None for fewer than two labels.
    """
    n = len(labels)
    if n < 2:
        return None
    counts = Counter(labels)
    same = sum(c * (c - 1) for c in counts.values())
    return 1.0 - same / (n * (n - 1))


def modal_share(labels: Sequence[str]) -> Optional[float]:
    """π* — the share of runs producing the most common claim."""
    if not labels:
        return None
    counts = Counter(labels)
    return max(counts.values()) / len(labels)


def filable(labels: Sequence[str], alpha: float = 0.2) -> Optional[bool]:
    """Filability criterion of arXiv:2608.13754: π* ≥ 1 − α."""
    ms = modal_share(labels)
    if ms is None:
        return None
    return ms >= 1.0 - alpha


def n_claim_classes(labels: Sequence[str]) -> int:
    return len(set(labels))


# --------------------------------------------------------------------------
# Score stability
# --------------------------------------------------------------------------

def mean(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def std(xs: Sequence[float]) -> Optional[float]:
    """Population standard deviation."""
    m = mean(xs)
    if m is None or len(xs) < 2:
        return None
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def coefficient_of_variation(xs: Sequence[float]) -> Optional[float]:
    """CV = σ / |μ|. None if undefined (|μ| ≈ 0 or fewer than 2 scores)."""
    m = mean(xs)
    s = std(xs)
    if m is None or s is None or abs(m) < 1e-12:
        return None
    return s / abs(m)


def score_variation_assessment(
    xs: Sequence[float],
    *,
    scale_type: str,
    minimum_abs_mean: float,
) -> Dict[str, object]:
    """Gate coefficient of variation behind explicit scale assumptions.

    Confirmatory CV is applicable only to finite, nonnegative ratio-scale
    scores with a mean above a preregistered domain-specific floor.  Signed,
    interval, ordinal, and near-zero scores return an explicit unsupported
    state instead of a misleading number.
    """
    values = [float(value) for value in xs]
    if minimum_abs_mean <= 0.0 or not math.isfinite(minimum_abs_mean):
        raise ValueError("minimum_abs_mean must be finite and positive")
    if not values:
        return {"applicable": False, "value": None, "reason": "no scores"}
    if any(not math.isfinite(value) for value in values):
        return {
            "applicable": False,
            "value": None,
            "reason": "scores contain non-finite values",
        }
    if scale_type != "ratio":
        return {
            "applicable": False,
            "value": None,
            "reason": f"CV requires ratio scale, got {scale_type!r}",
        }
    if any(value < 0.0 for value in values):
        return {
            "applicable": False,
            "value": None,
            "reason": "ratio-scale CV profile requires nonnegative scores",
        }
    score_mean = mean(values)
    assert score_mean is not None
    if abs(score_mean) < minimum_abs_mean:
        return {
            "applicable": False,
            "value": None,
            "reason": "score mean is below preregistered minimum_abs_mean",
        }
    if len(values) < 2:
        return {
            "applicable": False,
            "value": None,
            "reason": "at least two scores are required",
        }
    return {
        "applicable": True,
        "value": coefficient_of_variation(values),
        "reason": None,
        "scale_type": scale_type,
        "minimum_abs_mean": minimum_abs_mean,
    }


# --------------------------------------------------------------------------
# Semantic label handling (natural-language claims / oracle answers)
# --------------------------------------------------------------------------

def cluster_labels(labels: Sequence[str], equiv) -> List[int]:
    """Greedy single-link clustering of labels under an equivalence judge.

    ``equiv(a, b) -> bool`` decides whether two labels mean the same thing.
    Returns a cluster id per label; flip rate / modal share computed over
    these ids treat semantically equivalent phrasings as one claim. With a
    non-transitive judge the greedy assignment (first matching cluster
    representative wins) keeps results deterministic.
    """
    reps: List[str] = []
    ids: List[int] = []
    for lab in labels:
        for i, rep in enumerate(reps):
            if equiv(lab, rep):
                ids.append(i)
                break
        else:
            reps.append(lab)
            ids.append(len(reps) - 1)
    return ids


def rbo(list_a: Sequence, list_b: Sequence, p: float = 0.9) -> float:
    """Extrapolated Rank-Biased Overlap between two ranked lists.

    Webber, Moffat & Zobel (2010), eq. 32: the expected overlap seen by a
    reader who inspects rank d with probability p^(d-1). Unlike Jaccard on
    the top-k *set*, RBO weights the head of the list — which is what a
    lens readout or a ranked feature list actually claims. p=0.9 puts
    ~86% of the weight on the top 10 ranks.

    Handles uneven lengths per the paper's extrapolation; duplicate items
    within a list are an error (ranked lists are item-unique).
    """
    if not 0 < p < 1:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a, b = list(list_a), list(list_b)
    if len(set(a)) != len(a) or len(set(b)) != len(b):
        raise ValueError("rbo requires duplicate-free ranked lists")
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    s, l = len(a), len(b)  # noqa: E741
    seen_a, seen_b = set(), set()
    x_d = 0  # overlap at depth d
    a_sum = 0.0
    x_s = 0  # overlap at depth s (fixed once d > s)
    for d in range(1, l + 1):
        if d <= s:
            va, vb = a[d - 1], b[d - 1]
            if va == vb:
                x_d += 1
            else:
                if va in seen_b:
                    x_d += 1
                if vb in seen_a:
                    x_d += 1
            seen_a.add(va)
            seen_b.add(vb)
            if d == s:
                x_s = x_d
        else:
            if b[d - 1] in seen_a:
                x_d += 1
            seen_b.add(b[d - 1])
        weight = p ** (d - 1)
        contrib = x_d / d
        if d > s:
            contrib += x_s * (d - s) / (s * d)
        a_sum += contrib * weight
    x_l = x_d
    ext = ((x_l - x_s) / l + x_s / s) * p ** l
    return (1 - p) * a_sum + ext


def pairwise_rbo(lists: Sequence[Sequence], p: float = 0.9) -> Optional[float]:
    """Mean RBO over all unordered pairs of ranked lists; None if < 2."""
    pairs = list(itertools.combinations(range(len(lists)), 2))
    if not pairs:
        return None
    return sum(rbo(lists[i], lists[j], p) for i, j in pairs) / len(pairs)


def pairwise_agreement(answers: Sequence[str], judge) -> Optional[float]:
    """Mean pairwise agreement of answers under a judge; None if < 2."""
    pairs = list(itertools.combinations(answers, 2))
    if not pairs:
        return None
    return sum(1.0 for a, b in pairs if judge(a, b)) / len(pairs)


# --------------------------------------------------------------------------
# Uncertainty on the stability metrics themselves
# --------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    """Wilson score interval for a Bernoulli proportion k/n.

    Preferred over the normal approximation at the small n typical of
    oracle probes (n ≈ 9–36): it never leaves [0, 1] and stays honest
    at p near 0 or 1. Returns [lo, hi], or None when n == 0.
    """
    if n <= 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _pairwise_replicate(items: Sequence, idx: Sequence[int], pair_fn) -> Optional[float]:
    """Mean of ``pair_fn`` over one bootstrap replicate, self-pair free:
    only pairs whose members come from different original runs count."""
    total = count = 0
    n = len(idx)
    for a in range(n):
        for b in range(a + 1, n):
            if idx[a] == idx[b]:
                continue                         # self-pair: skip, don't fake
            total += pair_fn(items[idx[a]], items[idx[b]])
            count += 1
    return (total / count) if count else None


def _percentile_ci(vals: List[float], n_boot: int, alpha: float) -> Optional[List[float]]:
    if len(vals) < max(20, n_boot // 10):
        return None
    vals.sort()
    lo = vals[int((alpha / 2) * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return [lo, hi]


def bootstrap_ci_pairwise(
    items: Sequence,
    pair_fn,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> Optional[List[float]]:
    """Percentile bootstrap CI for a mean-pairwise statistic, self-pair free.

    Resampling runs with replacement duplicates runs; a naive pairwise mean
    then counts pairs of a run with its own copy (Jaccard 1.0, zero label
    flips), biasing the CI toward stability by roughly (1/n)·(1 − value).
    Here each replicate averages ``pair_fn`` only over pairs whose members
    come from *different original runs*, which removes that bias.

    Applies to any statistic that is a mean over distinct unordered pairs:
    mean pairwise Jaccard (``pair_fn=jaccard``) and the unbiased
    Gini–Simpson flip rate (``pair_fn=lambda a, b: a != b``).
    """
    import random as _random

    items = list(items)
    n = len(items)
    if n < 4:
        return None
    rng = _random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = _pairwise_replicate(items, idx, pair_fn)
        if v is not None:
            vals.append(v)
    return _percentile_ci(vals, n_boot, alpha)


def bootstrap_bca_ci_pairwise(
    items: Sequence,
    pair_fn,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    bounds: Optional[Sequence[float]] = (0.0, 1.0),
) -> Optional[List[float]]:
    """Bias-corrected and accelerated bootstrap CI for pairwise means.

    Run-level bootstrap replicates use the same self-pair-free statistic as
    :func:`bootstrap_ci_pairwise`. Bias correction uses the observed complete
    U-statistic; acceleration uses delete-one run estimates. BCa improves
    transformation invariance and first-order coverage in regular settings,
    but remains a calibration candidate for discrete or degenerate kernels.
    """
    import random as _random

    items = list(items)
    n = len(items)
    if n < 4:
        return None
    if n_boot < 20:
        raise ValueError("n_boot must be at least 20")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")

    incident = [0.0] * n
    total = 0.0
    for i, j in itertools.combinations(range(n), 2):
        value = float(pair_fn(items[i], items[j]))
        total += value
        incident[i] += value
        incident[j] += value
    point = total / math.comb(n, 2)
    leave_denominator = math.comb(n - 1, 2)
    leave_one = [
        (total - incident[i]) / leave_denominator
        for i in range(n)
    ]
    leave_mean = sum(leave_one) / n
    centered = [leave_mean - value for value in leave_one]
    square_sum = sum(value * value for value in centered)
    acceleration = (
        sum(value ** 3 for value in centered) / (6.0 * square_sum ** 1.5)
        if square_sum > 0.0
        else 0.0
    )

    rng = _random.Random(seed)
    bootstrap_values = []
    for _ in range(n_boot):
        indices = [rng.randrange(n) for _ in range(n)]
        value = _pairwise_replicate(items, indices, pair_fn)
        if value is not None:
            bootstrap_values.append(value)
    if len(bootstrap_values) < max(20, n_boot // 10):
        return None

    less = sum(value < point for value in bootstrap_values)
    equal = sum(value == point for value in bootstrap_values)
    proportion = (less + 0.5 * equal) / len(bootstrap_values)
    epsilon = 0.5 / len(bootstrap_values)
    proportion = min(1.0 - epsilon, max(epsilon, proportion))
    normal = statistics.NormalDist()
    bias_correction = normal.inv_cdf(proportion)

    adjusted = []
    for probability in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = normal.inv_cdf(probability)
        denominator = 1.0 - acceleration * (bias_correction + z)
        if abs(denominator) < 1e-12:
            adjusted_probability = 0.0 if probability < 0.5 else 1.0
        else:
            adjusted_probability = normal.cdf(
                bias_correction
                + (bias_correction + z) / denominator
            )
        adjusted.append(min(1.0, max(0.0, adjusted_probability)))

    bootstrap_values.sort()
    interval = [
        _linear_quantile(bootstrap_values, adjusted[0]),
        _linear_quantile(bootstrap_values, adjusted[1]),
    ]
    if interval[0] > interval[1]:
        if math.isclose(interval[0], interval[1], rel_tol=0.0, abs_tol=1e-12):
            midpoint = (interval[0] + interval[1]) / 2.0
            interval = [midpoint, midpoint]
        else:
            return None
    if bounds is not None:
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError(f"bounds must be ordered [lo, hi], got {bounds!r}")
        interval[0] = max(float(bounds[0]), interval[0])
        interval[1] = min(float(bounds[1]), interval[1])
    return interval


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Type-7 linear sample quantile for an already sorted nonempty sequence."""
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - weight)
        + float(sorted_values[upper]) * weight
    )


def jackknife_normal_ci_pairwise(
    items: Sequence,
    pair_fn,
    alpha: float = 0.05,
    bounds: Optional[Sequence[float]] = (0.0, 1.0),
) -> Optional[List[float]]:
    """Delete-one jackknife normal CI for a mean order-two U-statistic.

    This treats ``items`` as independent experimental units and accounts for
    dependence among pairs sharing an item.  It is a calibration candidate,
    not a universal answer for clustered or otherwise dependent runs.  The
    implementation uses incident pair sums, so it costs O(n²), not O(n³).

    ``bounds`` clips the interval for bounded metrics such as Jaccard and
    disagreement.  Pass ``None`` for an unbounded pair statistic.
    """
    items = list(items)
    n = len(items)
    if n < 4:
        return None
    incident = [0.0] * n
    total = 0.0
    for i, j in itertools.combinations(range(n), 2):
        value = float(pair_fn(items[i], items[j]))
        total += value
        incident[i] += value
        incident[j] += value
    point = total / math.comb(n, 2)
    leave_denominator = math.comb(n - 1, 2)
    pseudo = [
        n * point - (n - 1) * ((total - incident[i]) / leave_denominator)
        for i in range(n)
    ]
    pseudo_mean = sum(pseudo) / n
    variance = sum((x - pseudo_mean) ** 2 for x in pseudo) / (n - 1)
    standard_error = math.sqrt(variance / n)
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    lo, hi = point - z * standard_error, point + z * standard_error
    if bounds is not None:
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError(f"bounds must be ordered [lo, hi], got {bounds!r}")
        lo, hi = max(float(bounds[0]), lo), min(float(bounds[1]), hi)
    return [lo, hi]


def unbiased_variance_u_pairwise(items: Sequence, pair_fn) -> Optional[float]:
    """Unbiased variance estimator for a complete order-two U-statistic.

    The estimator combines three unbiased moments: a single-edge square,
    products of two edges sharing one observation, and products of disjoint
    edges.  It estimates finite-sample variance of the complete pairwise mean,
    including the second-order term omitted by first-order asymptotics.  Like
    many unbiased variance estimators, individual realizations can be negative.
    """
    items = list(items)
    n = len(items)
    if n < 4:
        return None
    edge_count = math.comb(n, 2)
    total = total_squares = 0.0
    incident = [0.0] * n
    incident_squares = [0.0] * n
    for i, j in itertools.combinations(range(n), 2):
        value = float(pair_fn(items[i], items[j]))
        if not math.isfinite(value):
            raise ValueError("pair_fn returned a non-finite value")
        total += value
        total_squares += value * value
        incident[i] += value
        incident[j] += value
        incident_squares[i] += value * value
        incident_squares[j] += value * value

    single_square = total_squares / edge_count
    incident_products_sum = sum(
        (subtotal * subtotal - squares) / 2.0
        for subtotal, squares in zip(incident, incident_squares)
    )
    incident_product = incident_products_sum / (3 * math.comb(n, 3))
    all_edge_products = (total * total - total_squares) / 2.0
    disjoint_products_sum = all_edge_products - incident_products_sum
    disjoint_product = disjoint_products_sum / (3 * math.comb(n, 4))
    numerator = edge_count * (single_square - disjoint_product)
    numerator += 6 * math.comb(n, 3) * (
        incident_product - disjoint_product
    )
    return numerator / (edge_count * edge_count)


def u_normal_ci_pairwise(
    items: Sequence,
    pair_fn,
    alpha: float = 0.05,
    bounds: Optional[Sequence[float]] = (0.0, 1.0),
) -> Optional[List[float]]:
    """Normal CI using unbiased finite-sample U-statistic variance estimate.

    Negative variance realizations are treated as unavailable rather than
    silently set to zero.  Normality remains an asymptotic assumption, so this
    interval requires calibration for discrete, skewed, or degenerate kernels.
    """
    items = list(items)
    if len(items) < 4:
        return None
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    point = mean(
        [float(pair_fn(items[i], items[j]))
         for i, j in itertools.combinations(range(len(items)), 2)]
    )
    variance = unbiased_variance_u_pairwise(items, pair_fn)
    assert point is not None and variance is not None
    if variance < 0.0 and not math.isclose(variance, 0.0, abs_tol=1e-15):
        return None
    standard_error = math.sqrt(max(0.0, variance))
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    interval = [point - z * standard_error, point + z * standard_error]
    if bounds is not None:
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError(f"bounds must be ordered [lo, hi], got {bounds!r}")
        interval[0] = max(float(bounds[0]), interval[0])
        interval[1] = min(float(bounds[1]), interval[1])
    return interval


def paired_mean_pairwise(
    items: Sequence,
    pair_fn,
    seed: int = 0,
) -> Optional[float]:
    """Mean kernel value over a seeded random partition into disjoint pairs.

    Under independent identically distributed runs, pair values are independent
    bounded observations of the same target as the complete order-two
    U-statistic.  One item is discarded when the run count is odd.  This loses
    efficiency but provides a simple reference whose uncertainty does not
    pretend that all overlapping pairs are independent.
    """
    values = _disjoint_pair_values(items, pair_fn, seed)
    return sum(values) / len(values) if values else None


def disjoint_pair_indices(n_items: int, seed: int = 0) -> List[List[int]]:
    """Seeded random partition indices used by paired finite-sample inference."""
    if n_items < 0:
        raise ValueError("n_items must be nonnegative")
    import random as _random

    indices = list(range(n_items))
    _random.Random(seed).shuffle(indices)
    return [indices[i:i + 2] for i in range(0, len(indices) - 1, 2)]


def hoeffding_ci_bounded(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
    bounds: Sequence[float] = (0.0, 1.0),
) -> Optional[List[float]]:
    """Two-sided finite-sample Hoeffding CI for IID bounded observations."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError(f"bounds must satisfy lo < hi, got {bounds!r}")
    values = [float(value) for value in values]
    if not values:
        return None
    lower, upper = float(bounds[0]), float(bounds[1])
    if any(
        not math.isfinite(value) or not lower <= value <= upper
        for value in values
    ):
        raise ValueError("observation lies outside declared bounds")
    point = sum(values) / len(values)
    half_width = (upper - lower) * math.sqrt(
        math.log(2.0 / alpha) / (2.0 * len(values))
    )
    return [max(lower, point - half_width), min(upper, point + half_width)]


def hoeffding_ci_pairwise(
    items: Sequence,
    pair_fn,
    seed: int = 0,
    alpha: float = 0.05,
    bounds: Sequence[float] = (0.0, 1.0),
) -> Optional[List[float]]:
    """Finite-sample Hoeffding CI from a random disjoint pairing of runs.

    For ``m = floor(n/2)`` independent pair values in known range ``[a, b]``,
    Hoeffding's inequality gives two-sided error at most ``alpha`` with
    half-width ``(b-a) * sqrt(log(2/alpha)/(2m))``.  Coverage is guaranteed
    under the independent-run assumption, though the interval can be much
    wider than asymptotic alternatives.  ``seed`` freezes the pairing.
    """
    values = _disjoint_pair_values(items, pair_fn, seed)
    try:
        return hoeffding_ci_bounded(values, alpha=alpha, bounds=bounds)
    except ValueError as error:
        if "outside declared bounds" in str(error):
            raise ValueError("pair_fn returned a value outside declared bounds") from error
        raise


def hoeffding_difference_pairwise(
    real_items: Sequence,
    null_items: Sequence,
    pair_fn,
    *,
    real_seed: int = 0,
    null_seed: int = 1,
    alpha: float = 0.05,
    bounds: Sequence[float] = (0.0, 1.0),
) -> Optional[Dict[str, object]]:
    """Finite-sample CI for real-minus-null pairwise agreement.

    Each group uses a disjoint-pair Hoeffding interval at error ``alpha/2``.
    A union bound then gives simultaneous coverage of at least ``1-alpha``;
    subtracting interval endpoints yields a valid interval for the difference.
    Difference is preferred over an unstable ratio when null agreement is near
    zero.
    """
    real_point = paired_mean_pairwise(real_items, pair_fn, seed=real_seed)
    null_point = paired_mean_pairwise(null_items, pair_fn, seed=null_seed)
    real_interval = hoeffding_ci_pairwise(
        real_items,
        pair_fn,
        seed=real_seed,
        alpha=alpha / 2.0,
        bounds=bounds,
    )
    null_interval = hoeffding_ci_pairwise(
        null_items,
        pair_fn,
        seed=null_seed,
        alpha=alpha / 2.0,
        bounds=bounds,
    )
    if (
        real_point is None
        or null_point is None
        or real_interval is None
        or null_interval is None
    ):
        return None
    span = float(bounds[1]) - float(bounds[0])
    difference_interval = [
        max(-span, real_interval[0] - null_interval[1]),
        min(span, real_interval[1] - null_interval[0]),
    ]
    return {
        "estimate": real_point - null_point,
        "ci": difference_interval,
        "real_estimate": real_point,
        "real_ci": real_interval,
        "null_estimate": null_point,
        "null_ci": null_interval,
    }


def cross_cluster_pair_mean(left_cluster: Sequence, right_cluster: Sequence, pair_fn) -> float:
    """Mean kernel value across all runs in two independent clusters."""
    if not left_cluster or not right_cluster:
        raise ValueError("clusters must be nonempty")
    values = [pair_fn(left, right) for left in left_cluster for right in right_cluster]
    return sum(values) / len(values)


def cluster_hoeffding_ci_pairwise(
    clusters: Sequence[Sequence],
    pair_fn,
    *,
    seed: int = 0,
    alpha: float = 0.05,
    bounds: Sequence[float] = (0.0, 1.0),
) -> Optional[Dict[str, object]]:
    """Cluster-aware disjoint-pair estimate and finite-sample interval.

    Clusters, not runs, are treated as IID units.  Kernel value for two
    clusters averages every cross-cluster run comparison, remaining bounded in
    the same range.  A seeded disjoint pairing of clusters then supports the
    Hoeffding guarantee without inflating sample size through dependent runs.
    """
    clusters = [tuple(cluster) for cluster in clusters]
    if any(not cluster for cluster in clusters):
        raise ValueError("clusters must be nonempty")

    def cluster_kernel(left, right):
        return cross_cluster_pair_mean(left, right, pair_fn)

    estimate = paired_mean_pairwise(clusters, cluster_kernel, seed=seed)
    interval = hoeffding_ci_pairwise(
        clusters,
        cluster_kernel,
        seed=seed,
        alpha=alpha,
        bounds=bounds,
    )
    if estimate is None or interval is None:
        return None
    return {
        "estimate": estimate,
        "ci": interval,
        "n_clusters": len(clusters),
        "n_runs": sum(len(cluster) for cluster in clusters),
    }


def _disjoint_pair_values(items: Sequence, pair_fn, seed: int) -> List[float]:
    """Seeded disjoint-pair kernel values shared by estimator and interval."""
    items = list(items)
    return [
        float(pair_fn(items[left], items[right]))
        for left, right in disjoint_pair_indices(len(items), seed)
    ]


def modal_share_hoeffding_ci(
    labels: Sequence[str],
    registered_classes: Sequence[str],
    *,
    alpha: float = 0.05,
) -> Optional[List[float]]:
    """Simultaneous finite-sample CI for population modal class probability.

    Labels must be IID draws from a preregistered finite class set. A union
    bound over per-class Hoeffding inequalities ensures every empirical class
    probability is within ``epsilon`` of its population value with probability
    at least ``1-alpha``. The maximum class probability is therefore within the
    same epsilon of empirical modal share.
    """
    labels = list(labels)
    classes = list(registered_classes)
    if not labels:
        return None
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("registered_classes must be nonempty and unique")
    unknown = sorted(set(labels) - set(classes))
    if unknown:
        raise ValueError(f"labels outside registered_classes: {unknown}")
    point = max(labels.count(label) for label in classes) / len(labels)
    epsilon = math.sqrt(
        math.log(2.0 * len(classes) / alpha) / (2.0 * len(labels))
    )
    return [max(0.0, point - epsilon), min(1.0, point + epsilon)]


def pairwise_kernel_variance_u(
    items: Sequence,
    pair_fn,
    bounds: Sequence[float] = (0.0, 1.0),
) -> Optional[float]:
    """Order-four U-statistic estimating variance of a pairwise kernel.

    Values are normalized to ``[0, 1]`` before computing Nguyen's symmetrized
    variance kernel.  The direct definition averages three disjoint pairings
    for every four observations and costs O(n⁴).  The identity used here
    subtracts incident-edge squared differences from all edge-pair squared
    differences, reducing cost to O(n²).
    """
    items = list(items)
    n = len(items)
    if n < 4:
        return None
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError(f"bounds must satisfy lo < hi, got {bounds!r}")
    edge_count, total, total_squares, incident, incident_squares = (
        _normalized_pairwise_summaries(items, pair_fn, bounds)
    )
    return _variance_u_from_pairwise_summaries(
        n, edge_count, total, total_squares, incident, incident_squares
    )


def _variance_u_from_pairwise_summaries(
    n: int,
    edge_count: int,
    total: float,
    total_squares: float,
    incident: Sequence[float],
    incident_squares: Sequence[float],
) -> float:
    all_edge_pair_differences = edge_count * total_squares - total * total
    incident_edge_pair_differences = sum(
        (n - 1) * squares - subtotal * subtotal
        for subtotal, squares in zip(incident, incident_squares)
    )
    disjoint_squared_differences = (
        all_edge_pair_differences - incident_edge_pair_differences
    )
    # Roundoff can make an algebraic zero slightly negative.
    disjoint_squared_differences = max(0.0, disjoint_squared_differences)
    disjoint_pairings = 3 * math.comb(n, 4)
    return disjoint_squared_differences / (2.0 * disjoint_pairings)


def _normalized_pairwise_summaries(
    items: Sequence,
    pair_fn,
    bounds: Sequence[float],
):
    """Complete normalized edge summaries for pairwise U-statistics."""
    n = len(items)
    lower, upper = float(bounds[0]), float(bounds[1])
    span = upper - lower
    total = total_squares = 0.0
    incident = [0.0] * n
    incident_squares = [0.0] * n
    for i, j in itertools.combinations(range(n), 2):
        raw = float(pair_fn(items[i], items[j]))
        if not math.isfinite(raw) or not lower <= raw <= upper:
            raise ValueError("pair_fn returned a value outside declared bounds")
        value = (raw - lower) / span
        total += value
        total_squares += value * value
        incident[i] += value
        incident[j] += value
        incident_squares[i] += value * value
        incident_squares[j] += value * value
    return (
        math.comb(n, 2),
        total,
        total_squares,
        incident,
        incident_squares,
    )


def nguyen_ci_pairwise(
    items: Sequence,
    pair_fn,
    alpha: float = 0.05,
    bounds: Sequence[float] = (0.0, 1.0),
) -> Optional[List[float]]:
    """Finite-sample empirical concentration CI for an order-two U-statistic.

    Implements equation (20) of Nguyen (2019), *Concentration-based
    confidence intervals for U-statistics* (arXiv:1903.01679), using the
    symmetrized order-four variance U-statistic.  Each one-sided bound receives
    error ``alpha / 2``; their union is therefore a two-sided ``1-alpha`` CI.

    Assumptions: runs are IID, ``pair_fn`` is symmetric, and its range is the
    declared finite interval.  The guarantee is distribution-free but can be
    conservative.  Values are internally normalized to ``[0, 1]``.
    """
    items = list(items)
    n = len(items)
    if n < 4:
        return None
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError(f"bounds must satisfy lo < hi, got {bounds!r}")
    lower, upper = float(bounds[0]), float(bounds[1])
    span = upper - lower
    edge_count, total, total_squares, incident, incident_squares = (
        _normalized_pairwise_summaries(items, pair_fn, bounds)
    )
    normalized_point = total / edge_count
    variance_u = _variance_u_from_pairwise_summaries(
        n, edge_count, total, total_squares, incident, incident_squares
    )
    effective_pairs = n // 2
    effective_variance_blocks = n // 4
    log_term = math.log(4.0 / alpha)
    radius = math.sqrt(2.0 * variance_u * log_term / effective_pairs)
    radius += math.sqrt(
        (1.0 / effective_pairs)
        * math.sqrt(1.0 / (2.0 * effective_variance_blocks))
        * log_term ** 1.5
    )
    radius += 4.0 * log_term / (3.0 * effective_pairs)
    return [
        max(lower, lower + span * (normalized_point - radius)),
        min(upper, lower + span * (normalized_point + radius)),
    ]


def bootstrap_ci_ratio_pairwise(
    num_items: Sequence,
    den_items: Sequence,
    pair_fn,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> Optional[List[float]]:
    """Percentile bootstrap CI for a ratio of two mean-pairwise statistics.

    For checks like specificity (stability on real runs ÷ stability on
    null-control runs) both sides are estimates, so the ratio's uncertainty
    must come from resampling *both* groups. Each replicate independently
    resamples the numerator and denominator runs, computes the self-pair-free
    mean-pairwise statistic on each (see ``bootstrap_ci_pairwise``), and
    records their ratio. Replicates whose denominator is (near-)zero are
    dropped. Returns None when either group has fewer than 4 runs or too few
    replicates were valid.
    """
    import random as _random

    num_items, den_items = list(num_items), list(den_items)
    if len(num_items) < 4 or len(den_items) < 4:
        return None
    rng = _random.Random(seed)
    vals = []
    for _ in range(n_boot):
        ni = [rng.randrange(len(num_items)) for _ in num_items]
        di = [rng.randrange(len(den_items)) for _ in den_items]
        num = _pairwise_replicate(num_items, ni, pair_fn)
        den = _pairwise_replicate(den_items, di, pair_fn)
        if num is None or den is None or den <= 1e-9:
            continue
        vals.append(num / den)
    return _percentile_ci(vals, n_boot, alpha)


def bootstrap_ci(
    items: Sequence,
    metric_fn,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> Optional[List[float]]:
    """Percentile bootstrap CI for a metric computed over runs.

    Resamples *runs* (not pairs) with replacement, following the resampling
    unit used in arXiv:2608.13754. Returns [lo, hi], or None when the metric
    is undefined or there are fewer than 4 items to resample.
    """
    import random as _random

    items = list(items)
    if len(items) < 4 or metric_fn(items) is None:
        return None
    rng = _random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [items[rng.randrange(len(items))] for _ in items]
        v = metric_fn(sample)
        if v is not None:
            vals.append(v)
    return _percentile_ci(vals, n_boot, alpha)


# --------------------------------------------------------------------------
# Variance attribution (one-at-a-time design)
# --------------------------------------------------------------------------

def variance_shares(groups: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """Share of score variance attributable to sweeping each axis.

    ``groups`` maps axis name -> scores observed while sweeping ONLY that
    axis (everything else held at the base configuration). This is an
    honest one-at-a-time attribution, not a crossed REML decomposition;
    shares are each axis's sweep variance normalized to sum to 1.
    """
    variances = {}
    for axis, xs in groups.items():
        s = std(list(xs))
        variances[axis] = (s ** 2) if s is not None else 0.0
    total = sum(variances.values())
    if total <= 0:
        return {axis: 0.0 for axis in variances}
    return {axis: v / total for axis, v in variances.items()}
