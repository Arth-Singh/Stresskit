"""Stability metrics.

Every metric here is lifted from a published protocol:

- pairwise Jaccard under resampling: Méloux, Portet & Peyrard,
  "Mechanistic Interpretability as Statistical Estimation" (arXiv:2510.00845).
- flip rate / modal share / filability: Mahale, "Explanation Multiplicity"
  (arXiv:2608.13754) — the unbiased Gini–Simpson form of the probability
  that two runs yield different claims.
- analytic expected random Jaccard k/(2N−k): same paper's size-matched null.
- pipeline variance shares: van der Ben et al., "Building Fast, Evaluating
  Slow" (arXiv:2607.19386) — here in a simple one-at-a-time (OAT) form.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence


# --------------------------------------------------------------------------
# Structural stability
# --------------------------------------------------------------------------

def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity of two component sets. J(∅, ∅) is defined as 1.0."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union


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
