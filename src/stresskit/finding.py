"""The universal unit of StressKit: a Finding.

A Finding is the output of ONE run of a discovery method. It deliberately
captures the three things an interpretability claim can rest on:

- ``components``: the structural identity of the finding — a set of edges,
  attention heads, SAE features, neurons, or any hashable identifiers.
  Structural stability is measured with pairwise Jaccard overlap.
- ``claim``: a short qualitative label for what you would write in the paper
  ("early-layer, sparse", "refusal is single direction", ...). Claim stability
  is measured with the flip rate across the multiverse of runs.
- ``score``: a scalar quality metric (faithfulness, probe AUC, logit diff
  recovered, ...). Score stability is measured with the coefficient of
  variation.

Any subset of the three may be present; StressKit only grades the checks
that are applicable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass
class Finding:
    """The result of a single discovery run.

    Parameters
    ----------
    components:
        Hashable identifiers of the parts of the model this finding consists
        of (edges, heads, SAE feature ids, ...). May be empty for findings
        that are purely scalar (e.g. a probe accuracy claim).
    claim:
        Qualitative claim label. Two runs "agree" iff their labels are equal,
        so keep labels coarse and deterministic (derive them from the finding
        with a fixed function — never by hand, never with an LLM).
    score:
        Scalar quality metric for this run (higher = better is not assumed;
        only the variation matters).
    universe_size:
        Total number of candidate components the discovery searched over
        (e.g. number of edges in the computation graph). Enables the
        size-matched random baseline.
    meta:
        Free-form extras (kept out of all metrics).
    """

    components: frozenset = field(default_factory=frozenset)
    claim: Optional[str] = None
    score: Optional[float] = None
    universe_size: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.components, frozenset):
            self.components = frozenset(self.components)
        if self.score is not None:
            self.score = float(self.score)

    @property
    def size(self) -> int:
        return len(self.components)

    def has_structure(self) -> bool:
        return len(self.components) > 0


def circuit(
    edges: Iterable,
    *,
    claim: Optional[str] = None,
    score: Optional[float] = None,
    universe_size: Optional[int] = None,
    **meta: Any,
) -> Finding:
    """Build a Finding from a discovered circuit (a set of edges/nodes)."""
    return Finding(
        components=frozenset(edges),
        claim=claim,
        score=score,
        universe_size=universe_size,
        meta=dict(meta),
    )


def feature_set(
    features: Iterable,
    *,
    claim: Optional[str] = None,
    score: Optional[float] = None,
    universe_size: Optional[int] = None,
    **meta: Any,
) -> Finding:
    """Build a Finding from a set of SAE features / neurons / directions."""
    return Finding(
        components=frozenset(features),
        claim=claim,
        score=score,
        universe_size=universe_size,
        meta=dict(meta),
    )


def probe(
    score: float,
    *,
    claim: Optional[str] = None,
    components: Iterable = (),
    **meta: Any,
) -> Finding:
    """Build a Finding whose primary content is a scalar (probe acc, steering
    effect size, ...)."""
    return Finding(
        components=frozenset(components),
        claim=claim,
        score=score,
        meta=dict(meta),
    )
