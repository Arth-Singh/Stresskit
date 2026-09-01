"""Adapter for EAP-IG circuit graphs (hannamw/EAP-IG, the ``eap`` package).

Three entry points, in increasing effort:

1. **You have saved circuits** (``graph.to_json(...)`` files from past runs):

       from stresskit.adapters import eap
       import stresskit as sk

       findings = [eap.finding_from_json(p) for p in glob("runs/*.json")]
       print(sk.from_findings(findings).to_markdown())

2. **You have live scored graphs** — ``eap.graph_to_finding(g)``.

3. **You want the full battery** — wrap the attribute step:

       def graph_fn(data, seed, config):
           g = Graph.from_model(model)
           attribute(model, g, make_loader(data, seed), metric,
                     method=config["method"], ig_steps=config["ig_steps"])
           g.apply_topn(config["topn"], True)
           return g

       result = sk.stress(eap.finder_from_graph_fn(graph_fn), data,
                          hyperparams={"topn": [100, 400],
                                       "method": ["EAP", "EAP-IG-activations"]})

Never imports the eap package; graphs are duck-typed (``.edges`` mapping of
name -> object with ``.in_graph`` and ``.score``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from ..finding import Finding

_LAYER_RE = re.compile(r"^[am](\d+)")


def edge_layer(edge_name: str) -> Optional[int]:
    """Layer of an EAP edge's parent node ('a10.h7->m11' -> 10).

    Edges from 'input' have no layer; returns None for them.
    """
    m = _LAYER_RE.match(edge_name)
    return int(m.group(1)) if m else None


def layer_band_claim(edge_names, n_layers: int) -> str:
    """'early' / 'middle' / 'late' from the median layer of circuit edges."""
    layers = sorted(
        L for L in (edge_layer(e) for e in edge_names) if L is not None
    )
    if not layers:
        return "input-only"
    median = layers[len(layers) // 2]
    return ["early", "middle", "late"][min(2, 3 * median // n_layers)]


def graph_to_finding(
    graph: Any,
    *,
    score: Optional[float] = None,
    claim: Optional[str] = None,
    n_layers: Optional[int] = None,
    **meta: Any,
) -> Finding:
    """Finding from a scored eap.Graph: components are the in-graph edge
    names, universe is every candidate edge. If ``n_layers`` is given and
    ``claim`` is not, the claim defaults to the circuit's layer band."""
    edges = graph.edges
    in_edges = [name for name, e in edges.items() if e.in_graph]
    if claim is None and n_layers:
        claim = layer_band_claim(in_edges, n_layers)
    return Finding(
        components=frozenset(in_edges),
        claim=claim,
        score=score,
        universe_size=len(edges),
        meta=dict(meta),
        structure_present=True,
    )


def finding_from_json(
    path: str,
    *,
    score: Optional[float] = None,
    claim: Optional[str] = None,
    **meta: Any,
) -> Finding:
    """Finding from a ``Graph.to_json`` export — post-hoc, no model needed.

    If ``claim`` is omitted it defaults to the circuit's layer band
    (n_layers read from the exported cfg)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if "edges" not in d:
        raise ValueError(f"{path}: no 'edges' key — not an eap Graph export")
    in_edges = [name for name, e in d["edges"].items() if e.get("in_graph")]
    n_layers = (d.get("cfg") or {}).get("n_layers")
    if claim is None and n_layers:
        claim = layer_band_claim(in_edges, int(n_layers))
    return Finding(
        components=frozenset(in_edges),
        claim=claim,
        score=score,
        universe_size=len(d["edges"]),
        meta={"source": path, **meta},
        structure_present=True,
    )


def finder_from_graph_fn(
    graph_fn: Callable[[Any, int, dict], Any],
    *,
    score_fn: Optional[Callable[[Any], float]] = None,
    claim_fn: Optional[Callable[[Any], str]] = None,
    n_layers: Optional[int] = None,
) -> Callable[[Any, int, dict], Finding]:
    """Wrap ``graph_fn(data, seed, config) -> scored eap.Graph`` into a
    ``stress``-ready finding_fn. ``score_fn`` typically evaluates circuit
    faithfulness (``evaluate_graph`` vs baseline)."""

    def finding_fn(data, seed, config) -> Finding:
        g = graph_fn(data, seed, dict(config))
        return graph_to_finding(
            g,
            score=score_fn(g) if score_fn else None,
            claim=claim_fn(g) if claim_fn else None,
            n_layers=n_layers,
        )

    return finding_fn
