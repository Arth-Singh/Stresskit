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


def pairwise_agreement(answers: Sequence[str], judge) -> Optional[float]:
    """Mean pairwise agreement of answers under a judge; None if < 2."""
    pairs = list(itertools.combinations(answers, 2))
    if not pairs:
        return None
    return sum(1.0 for a, b in pairs if judge(a, b)) / len(pairs)


# --------------------------------------------------------------------------
# Uncertainty on the stability metrics themselves
# --------------------------------------------------------------------------

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
    if len(vals) < max(20, n_boot // 10):
        return None
    vals.sort()
    lo = vals[int((alpha / 2) * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return [lo, hi]


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
