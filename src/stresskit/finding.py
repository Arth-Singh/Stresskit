"""The universal unit of StressKit: a Finding.

A Finding is the output of ONE run of a discovery method. It deliberately
captures the three things an interpretability claim can rest on:

- ``components``: the structural identity of the finding — a set of edges,
  attention heads, SAE features, neurons, or any hashable identifiers.
  Structural stability is measured with pairwise Jaccard overlap.
- ``vector``: the structural identity of a *direction*-valued finding — a
  refusal direction, persona/steering vector, or probe weight vector.
  Structural stability is measured with mean pairwise |cosine| instead of
  Jaccard. A finding carries a set or a vector, never both.
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

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _norm(vector: Sequence[float]) -> float:
    """Euclidean norm, via math.hypot so that a direction whose coordinates
    are all tiny (or all huge) is not squared into a spurious zero norm."""
    return math.hypot(*vector)


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
    vector:
        Direction-valued structure: the coordinates of a refusal direction,
        steering/persona vector, or probe weight vector in some fixed basis.
        Mutually exclusive with ``components`` — a finding is set-valued or
        direction-valued, never both. Build one with
        :func:`stresskit.direction`, which unit-normalizes for you.
    """

    components: Optional[Iterable] = None
    claim: Optional[str] = None
    score: Optional[float] = None
    universe_size: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    structure_present: Optional[bool] = None
    vector: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        supplied_components = self.components is not None
        if self.components is None:
            self.components = frozenset()
        if not isinstance(self.components, frozenset):
            self.components = frozenset(self.components)
        if self.structure_present is None:
            self.structure_present = supplied_components
        else:
            self.structure_present = bool(self.structure_present)
        if self.score is not None:
            self.score = float(self.score)
        if self.vector is not None:
            if supplied_components and self.components:
                raise ValueError(
                    "a Finding carries either components= or vector=, not "
                    "both: a set is graded with Jaccard and a direction with "
                    "|cosine|, and the two are not comparable"
                )
            self.vector = tuple(float(x) for x in self.vector)
            if not self.vector:
                raise ValueError("vector= is empty; a direction needs at least one coordinate")
            if not all(math.isfinite(x) for x in self.vector):
                raise ValueError("vector= contains a NaN or infinite coordinate")
            if _norm(self.vector) == 0.0:
                raise ValueError("vector= has zero norm; the zero vector has no direction")

    @property
    def size(self) -> int:
        return len(self.components)

    @property
    def dim(self) -> Optional[int]:
        """Dimension of the direction, or None for a set-valued finding."""
        return len(self.vector) if self.vector is not None else None

    @property
    def kind(self) -> str:
        """Structural kind: ``'direction'`` when a vector rides on this
        finding, ``'set'`` when a component set does, ``'none'`` when the
        finding is purely claim- or score-valued."""
        if self.vector is not None:
            return "direction"
        if self.has_structure():
            return "set"
        return "none"

    def has_structure(self) -> bool:
        """Whether a component set was produced, including an empty set."""
        return bool(self.structure_present)

    def has_direction(self) -> bool:
        """Whether a direction vector was produced."""
        return self.vector is not None


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
        structure_present=True,
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
        structure_present=True,
    )


def direction(
    vector: Iterable,
    *,
    claim: Optional[str] = None,
    score: Optional[float] = None,
    **meta: Any,
) -> Finding:
    """Build a Finding from a direction — a refusal direction, a persona or
    steering vector, a probe weight vector.

    The vector is unit-normalized on construction, so what is stored is the
    direction and nothing else. Structural stability across runs is then mean
    pairwise |cosine| (:func:`stresskit.metrics.mean_pairwise_abs_cosine`),
    not Jaccard: a direction has no component set, and the set proxies people
    reach for instead — top-k logit-lens readout tokens, top-k coordinates —
    cap the structural check well below 1 even when two runs recovered the
    same direction.

    All coordinates must be finite and the vector must have nonzero norm;
    neither an empty, a non-finite, nor a zero vector has a direction.
    """
    values = tuple(float(x) for x in vector)
    if not values:
        raise ValueError("direction() needs at least one coordinate, got an empty vector")
    if not all(math.isfinite(x) for x in values):
        raise ValueError(
            "direction() got a vector with a NaN or infinite coordinate; fix "
            "the extraction before grading it"
        )
    norm = _norm(values)
    if norm == 0.0:
        raise ValueError(
            "direction() got the zero vector, which has no direction — a "
            "difference-in-means that came out exactly zero means the two "
            "classes did not separate, not that the direction is stable"
        )
    return Finding(
        vector=tuple(x / norm for x in values),
        claim=claim,
        score=score,
        meta=dict(meta),
        structure_present=False,
    )


def _hashable(c: Any) -> Any:
    """JSON arrays aren't hashable; components round-trip as tuples."""
    return tuple(_hashable(x) for x in c) if isinstance(c, list) else c


def findings_from_jsonl(
    path: str,
    *,
    components_key: str = "components",
    claim_key: str = "claim",
    score_key: str = "score",
    universe_size_key: str = "universe_size",
    axis_key: str = "axis",
) -> List[Finding]:
    """One Finding per line of a JSONL sweep log — the zero-wrapper entry.

    Each line is a JSON object; the ``*_key`` parameters name where your log
    keeps each field, so existing logs work unmodified::

        {"components": [[9, 6], [9, 9], [10, 0]], "claim": "late", "score": 3.1}
        {"components": [[9, 6], [10, 7]], "claim": "late", "score": 2.9, "axis": "seeds"}

    Missing fields are simply absent from the Finding (StressKit grades only
    the applicable checks). Nested arrays become tuples so components stay
    hashable. An ``axis`` field, when present, is kept in ``meta['axis']``
    for :func:`stresskit.from_findings`'s per-axis breakdown — or grade the
    file in one call with :func:`stresskit.from_jsonl`.
    """
    findings: List[Finding] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
            if not isinstance(rec, dict):
                raise ValueError(
                    f"{path}:{lineno}: expected a JSON object per line, "
                    f"got {type(rec).__name__}")
            meta: Dict[str, Any] = {}
            if rec.get(axis_key) is not None:
                meta["axis"] = str(rec[axis_key])
            if rec.get("universe") is not None:
                meta["universe"] = rec["universe"]
            findings.append(Finding(
                components=frozenset(
                    _hashable(c) for c in rec.get(components_key) or ()),
                claim=rec.get(claim_key),
                score=rec.get(score_key),
                universe_size=rec.get(universe_size_key),
                meta=meta,
                structure_present=(
                    components_key in rec and rec.get(components_key) is not None
                ),
            ))
    if not findings:
        raise ValueError(f"{path}: no findings found (file is empty)")
    return findings


def probe(
    score: float,
    *,
    claim: Optional[str] = None,
    components: Iterable = (),
    **meta: Any,
) -> Finding:
    """Build a Finding whose primary content is a scalar (probe acc, steering
    effect size, ...)."""
    component_set = frozenset(components)
    return Finding(
        components=component_set,
        claim=claim,
        score=score,
        meta=dict(meta),
        structure_present=bool(component_set),
    )
