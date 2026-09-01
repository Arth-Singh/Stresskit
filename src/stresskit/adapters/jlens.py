"""Adapter for Jacobian-lens readouts (anthropics/jacobian-lens, the
companion code to "Verbalizable Representations Form a Global Workspace
in Language Models", July 2026).

A J-lens readout at one (layer, position) is a *ranked* list of vocabulary
tokens. This adapter turns readouts into StressKit findings and provides
the two checks the community flagged within days of release:

- **rank sensitivity** — reported workspace readings depend on analytic
  choices (top-k cutoff, workspace band, lens rank); stress them.
- **junk contamination** — raw top-K readouts are dominated by punctuation,
  fragments and glitch tokens on some models (the repo's own
  ``mask_display`` flag works around this silently). ``junk_share``
  measures it instead of hiding it.

Comparison of ranked readouts should use rank-biased overlap
(``stresskit.metrics.rbo``), not set Jaccard — the head of the list is
the claim.

The adapter is import-free of torch/jlens: pass decoded token lists.
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Sequence

from ..finding import Finding
from .. import metrics as M

# A "word-like" token: optional leading space then >=2 letters (any case).
# Everything else — punctuation, digits, byte fragments, CJK single chars,
# glitch tokens — counts as junk for English-concept readouts.
_WORDLIKE = re.compile(r"^[\s▁Ġ]?[A-Za-z]{2,}$")


def is_wordlike(token: str) -> bool:
    return bool(_WORDLIKE.match(token))


def junk_share(tokens: Sequence[str]) -> float:
    """Fraction of a readout that is not word-like.

    On a healthy concept readout this is low; readouts "infested with
    glitch tokens" (reported for GPT-2 J-space within days of release)
    score high. Empty input is an error — a readout always has tokens.
    """
    if not tokens:
        raise ValueError("junk_share of an empty readout")
    return sum(0 if is_wordlike(t) else 1 for t in tokens) / len(tokens)


def readout_finding(
    ranked_tokens: Sequence[str],
    *,
    k: Optional[int] = None,
    universe_size: Optional[int] = None,
    score: Optional[float] = None,
    claim: Optional[str] = None,
    **meta,
) -> Finding:
    """Finding from one ranked readout: components = top-k token strings,
    claim defaults to the top-1 token, and the full ranking is kept in
    ``meta['ranked']`` so ``pairwise_readout_rbo`` can weight by rank."""
    ranked = [str(t) for t in ranked_tokens]
    top = ranked[:k] if k is not None else ranked
    return Finding(
        components=frozenset(top),
        claim=claim if claim is not None else (top[0] if top else None),
        score=score,
        universe_size=universe_size,
        meta={"ranked": ranked, "junk_share": junk_share(ranked), **meta},
        structure_present=True,
    )


def pairwise_readout_rbo(findings: Sequence[Finding], p: float = 0.9) -> Optional[float]:
    """Mean rank-biased overlap across findings built by readout_finding."""
    lists = [f.meta["ranked"] for f in findings]
    return M.pairwise_rbo(lists, p=p)


def min_rank(
    ranked_by_layer: Mapping[int, Sequence[str]],
    target: str,
    *,
    layers: Optional[Sequence[int]] = None,
    normalize: bool = True,
) -> Optional[int]:
    """Best (minimum) rank of ``target`` across the given layers' readouts.

    This is the upstream hit criterion: a target is a hit at pass@k when
    its min-over-layers rank <= k inside the workspace band. Returns a
    1-based rank, or None when the target never appears (which callers
    must treat as a miss, not rank infinity minus one).
    """
    def norm(s: str) -> str:
        return s.strip().lower() if normalize else s

    want = norm(target)
    best: Optional[int] = None
    for layer, ranked in ranked_by_layer.items():
        if layers is not None and layer not in layers:
            continue
        for i, tok in enumerate(ranked):
            if norm(tok) == want:
                if best is None or i + 1 < best:
                    best = i + 1
                break
    return best


def band_layers(n_layers: int, band: str) -> List[int]:
    """Layer sets for the workspace-band hyperparameter.

    ``band``: "mid-third" (the generic workspace band), "mid-half", or
    "all". The upstream paper reports over a model-specific band; which
    band is a reportable analytic choice, so stress it.
    """
    if band == "mid-third":
        return list(range(n_layers // 3, 2 * n_layers // 3))
    if band == "mid-half":
        return list(range(n_layers // 4, 3 * n_layers // 4))
    if band == "all":
        return list(range(n_layers))
    raise ValueError(f"unknown band {band!r}: use mid-third|mid-half|all")
