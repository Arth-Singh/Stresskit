"""TransformerLens / EAP-style adapters.

StressKit does not re-implement circuit discovery — it wraps yours. These
helpers convert the outputs that TransformerLens / EAP / EAP-IG pipelines
already produce (a dict of edge scores) into ``Finding`` objects, and derive
the coarse layer-band claim used throughout the stability literature.

Typical wiring::

    import stresskit as sk
    from stresskit.adapters.transformer_lens import edges_to_finding, layer_band_claim

    N_EDGES = 32491   # e.g. GPT-2 small factorized graph

    def finder(data, seed, config) -> sk.Finding:
        scores = run_eap(model, data, seed=seed,
                         metric=config["metric"],
                         ablation=config["ablation"])       # your code
        return edges_to_finding(
            scores,
            top_k=config["top_k"],
            universe_size=N_EDGES,
            layer_of=lambda e: int(e.split(".")[1]),        # "blocks.3.attn..." -> 3
            n_layers=12,
        )

    result = sk.stress(
        finder, dataset,
        battery=["seeds", "bootstrap", "templates", "hyperparams"],
        config={"metric": "logit_diff", "ablation": "patching", "top_k": 400},
        templates={"ABBA": abba_prompts, "BABA": baba_prompts},
        hyperparams={"metric": ["kl_div"], "ablation": ["mean"], "top_k": [200, 800]},
        model="gpt2-small", task="IOI", method="EAP",
    )
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Optional

from ..finding import Finding


def select_edges(
    edge_scores: Mapping[str, float],
    *,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    use_abs: bool = True,
) -> frozenset:
    """Turn a {edge: score} mapping into a component set, by top-k or
    absolute-score threshold (exactly one must be given)."""
    if (top_k is None) == (threshold is None):
        raise ValueError("pass exactly one of top_k= or threshold=")
    key = (lambda s: abs(s)) if use_abs else (lambda s: s)
    if top_k is not None:
        ranked = sorted(edge_scores.items(), key=lambda kv: -key(kv[1]))
        return frozenset(e for e, _ in ranked[:top_k])
    return frozenset(e for e, s in edge_scores.items() if key(s) >= threshold)


def layer_band_claim(
    components: Iterable[str],
    layer_of: Callable[[str], int],
    n_layers: int,
) -> str:
    """Coarse claim label: which third of the model holds the finding.

    This is the claim map whose instability arXiv:2608.13754 measured —
    deterministic, so flips reflect the finding, not the labeling.
    """
    layers = [layer_of(c) for c in components]
    if not layers:
        return "empty"
    thirds = [0, 0, 0]
    for layer in layers:
        band = min(2, int(3 * layer / max(1, n_layers)))
        thirds[band] += 1
    return ("early", "middle", "late")[thirds.index(max(thirds))]


def edges_to_finding(
    edge_scores: Mapping[str, float],
    *,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    universe_size: Optional[int] = None,
    score: Optional[float] = None,
    layer_of: Optional[Callable[[str], int]] = None,
    n_layers: Optional[int] = None,
    claim: Optional[str] = None,
) -> Finding:
    """Convert EAP-style edge scores into a Finding.

    If ``layer_of`` and ``n_layers`` are given and ``claim`` is not, the
    layer-band claim is derived automatically.
    """
    comps = select_edges(edge_scores, top_k=top_k, threshold=threshold)
    if claim is None and layer_of is not None and n_layers:
        claim = layer_band_claim(comps, layer_of, n_layers)
    return Finding(
        components=comps,
        claim=claim,
        score=score,
        universe_size=universe_size or len(edge_scores),
        structure_present=True,
    )
