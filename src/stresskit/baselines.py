"""Null-model baselines.

The lesson of "Sanity Checks for Sparse Autoencoders" (arXiv:2602.14111)
and the size-matched null of "Explanation Multiplicity" (arXiv:2608.13754):
an interpretability result means little until it beats what randomness
produces at the same size. These helpers make that comparison one call.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from . import metrics as M
from .finding import Finding


def random_findings(
    k: int,
    universe_size: int,
    n: int = 20,
    seed: int = 0,
    claim_fn=None,
) -> List[Finding]:
    """n findings, each a uniform random k-subset of a universe of components.

    ``claim_fn(components) -> str`` optionally derives a claim label from each
    random subset, so claim flip rates can also be compared against chance.
    """
    if k > universe_size:
        raise ValueError(f"k={k} exceeds universe_size={universe_size}")
    rng = random.Random(seed)
    universe = range(universe_size)
    out = []
    for _ in range(n):
        comps = frozenset(rng.sample(universe, k))
        out.append(
            Finding(
                components=comps,
                claim=claim_fn(comps) if claim_fn else None,
                universe_size=universe_size,
                structure_present=True,
            )
        )
    return out


def random_jaccard_stats(
    k: int,
    universe_size: int,
    n: int = 20,
    seed: int = 0,
) -> Dict[str, Optional[float]]:
    """Empirical + analytic Jaccard stats for size-matched random subsets."""
    findings = random_findings(k, universe_size, n=n, seed=seed)
    sets = [f.components for f in findings]
    return {
        "empirical_mean": M.mean_pairwise_jaccard(sets),
        "analytic_expected": M.expected_random_jaccard(k, universe_size),
        "k": float(k),
        "universe_size": float(universe_size),
        "n": float(n),
    }


def empirical_random_jaccard(
    sizes: Sequence[int],
    universe_size: int,
    seed: int = 0,
    repeats: int = 200,
) -> Optional[float]:
    """Monte-Carlo E[mean pairwise Jaccard] for random subsets matching the
    *observed* size distribution.

    Unlike ``metrics.expected_random_jaccard`` (a ratio-of-expectations
    approximation that assumes one shared size k), this draws one random
    subset per observed size and averages the mean pairwise Jaccard over
    ``repeats`` draws — exact in expectation and honest about heterogeneous
    finding sizes, including the registered empty-set convention. Returns None
    if fewer than two sizes are supplied.
    """
    if any(int(s) < 0 for s in sizes):
        raise ValueError("finding sizes must be nonnegative")
    sizes = [min(int(s), int(universe_size)) for s in sizes]
    if len(sizes) < 2 or not universe_size or universe_size <= 0:
        return None
    rng = random.Random(seed)
    universe = range(int(universe_size))
    vals: List[float] = []
    for _ in range(repeats):
        sets = [frozenset(rng.sample(universe, s)) for s in sizes]
        v = M.mean_pairwise_jaccard(sets)
        if v is not None:
            vals.append(v)
    return sum(vals) / len(vals) if vals else None


def compare_to_random(
    sets: Sequence[frozenset],
    universe_size: int,
    seed: int = 0,
) -> Dict[str, Optional[float]]:
    """Compare observed structural stability against the size-matched null.

    Returns observed mean pairwise Jaccard, the null's, and the ratio.
    """
    sizes = sorted(len(s) for s in sets if s)
    if not sizes:
        return {"observed": None, "random": None, "ratio": None}
    k = sizes[len(sizes) // 2]  # median size
    observed = M.mean_pairwise_jaccard(list(sets))
    null = random_jaccard_stats(k, universe_size, n=max(20, len(sets)), seed=seed)
    rand = null["empirical_mean"]
    ratio = (observed / rand) if (observed is not None and rand) else None
    return {"observed": observed, "random": rand, "ratio": ratio}
