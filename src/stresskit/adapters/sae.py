"""SAE stability utilities.

Two checks the SAE literature keeps asking for but no library ships:

1. **Seed consistency** — SAEs trained on identical data with different seeds
   learn different features (Song et al., arXiv:2505.20254; flagged as an
   auditing pitfall in arXiv:2606.00033). ``seed_consistency`` measures it
   with the Mean Correlation Coefficient (MCC): optimal one-to-one matching
   of decoder rows across runs by |cosine|, as used in SynthSAEBench
   (arXiv:2602.14687).

2. **Redundancy audit** — near-duplicate features inside one SAE (e.g. the
   139 near-identical "math" transcoder features observed on Neuronpedia).
   ``redundancy_audit`` clusters features whose decoder directions exceed a
   cosine threshold.

Decoder convention: ``W_dec`` has shape ``(n_features, d_model)`` — rows are
feature directions (the SAELens convention). Pass ``W_dec.T`` if yours is
column-major.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _normalize_rows(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"decoder must be 2-D (n_features, d_model), got shape {W.shape}")
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return W / norms


def match_features(W_a: np.ndarray, W_b: np.ndarray) -> Dict[str, Any]:
    """Optimal one-to-one feature matching between two decoders.

    Returns the MCC (mean |cosine| of matched pairs) and the matching.
    Uses scipy's Hungarian algorithm when available; otherwise a greedy
    fallback (a lower bound on the optimal MCC).
    """
    A = _normalize_rows(W_a)
    B = _normalize_rows(W_b)
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"d_model mismatch: {A.shape[1]} vs {B.shape[1]} — are both decoders "
            f"(n_features, d_model)?"
        )
    sim = np.abs(A @ B.T)  # (n_a, n_b)

    method = "hungarian"
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-sim)
    except ImportError:  # pragma: no cover - exercised only without scipy
        method = "greedy"
        rows, cols = _greedy_match(sim)

    matched = sim[rows, cols]
    return {
        "mcc": float(matched.mean()) if matched.size else 0.0,
        "matched_similarities": matched,
        "row_indices": np.asarray(rows),
        "col_indices": np.asarray(cols),
        "method": method,
    }


def _greedy_match(sim: np.ndarray):
    sim = sim.copy()
    n = min(sim.shape)
    rows, cols = [], []
    for _ in range(n):
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        rows.append(int(i))
        cols.append(int(j))
        sim[i, :] = -1.0
        sim[:, j] = -1.0
    return np.asarray(rows), np.asarray(cols)


def seed_consistency(decoders: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Pairwise MCC across SAE training runs (different seeds, same data).

    An SAE release reporting a single run should treat mean MCC well below
    1.0 as a caveat: the features are partly an artifact of the seed.
    """
    if len(decoders) < 2:
        raise ValueError("need at least two decoders to measure consistency")
    pairs = {}
    values = []
    for (i, Wa), (j, Wb) in itertools.combinations(enumerate(decoders), 2):
        mcc = match_features(Wa, Wb)["mcc"]
        pairs[f"{i}-{j}"] = mcc
        values.append(mcc)
    return {
        "mean_mcc": float(np.mean(values)),
        "min_mcc": float(np.min(values)),
        "pairwise_mcc": pairs,
        "n_runs": len(decoders),
    }


def redundancy_audit(
    W_dec: np.ndarray,
    threshold: float = 0.9,
    batch_size: int = 2048,
) -> Dict[str, Any]:
    """Find near-duplicate feature directions inside one SAE.

    Two features are redundant if |cosine(d_i, d_j)| >= threshold. Redundant
    features are grouped into clusters with union-find. Suitable up to a few
    tens of thousands of features on CPU (similarity is computed in batches).
    """
    W = _normalize_rows(W_dec)
    n = W.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n_pairs = 0
    example_pairs: List[tuple] = []
    for start in range(0, n, batch_size):
        block = W[start : start + batch_size]
        sims = np.abs(block @ W.T)  # (b, n)
        bi, bj = np.nonzero(sims >= threshold)
        for i_local, j in zip(bi.tolist(), bj.tolist()):
            i = start + i_local
            if j <= i:  # dedupe: count each unordered pair once, skip diagonal
                continue
            union(i, j)
            n_pairs += 1
            if len(example_pairs) < 20:
                example_pairs.append((i, j, float(sims[i_local, j])))

    roots = [find(i) for i in range(n)]
    cluster_sizes: Dict[int, int] = {}
    for r in roots:
        cluster_sizes[r] = cluster_sizes.get(r, 0) + 1
    dup_clusters = {r: s for r, s in cluster_sizes.items() if s > 1}
    n_redundant = sum(s for s in dup_clusters.values())

    return {
        "n_features": n,
        "threshold": threshold,
        "n_duplicate_pairs": n_pairs,
        "n_redundant_features": n_redundant,
        "redundant_fraction": n_redundant / n if n else 0.0,
        "n_clusters": len(dup_clusters),
        "largest_cluster": max(dup_clusters.values()) if dup_clusters else 0,
        "example_pairs": example_pairs,
    }


def top_features_finding(
    activations: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    *,
    claim_fn=None,
    universe_size: Optional[int] = None,
):
    """Select the top-k features separating labeled examples and wrap as a
    Finding — a minimal reference selector for stress-testing "feature X
    encodes concept Y" claims.

    ``activations``: (n_examples, n_features); ``labels``: (n_examples,) in {0,1}.
    Selection: top-k |mean difference| between classes. Score: separation AUC
    proxy (mean difference of the top feature, normalized).
    """
    from ..finding import feature_set

    acts = np.asarray(activations, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    if acts.ndim != 2 or y.shape[0] != acts.shape[0]:
        raise ValueError("activations must be (n_examples, n_features) aligned with labels")
    if y.all() or (~y).all():
        raise ValueError("labels must contain both classes")
    diff = np.abs(acts[y].mean(axis=0) - acts[~y].mean(axis=0))
    order = np.argsort(-diff)
    top = order[:k]
    pooled_std = acts.std(axis=0)[top[0]] or 1.0
    effect = float(diff[top[0]] / pooled_std)
    comps = frozenset(int(i) for i in top)
    return feature_set(
        comps,
        claim=claim_fn(comps) if claim_fn else None,
        score=effect,
        universe_size=universe_size or acts.shape[1],
    )
