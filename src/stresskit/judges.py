"""Answer-equivalence judges.

2026 interpretability outputs are increasingly natural language (Activation
Oracles, verbalizers, introspection reports, J-lens readouts). Comparing
them with exact string equality is meaningless, so every StressKit metric
that compares answers or claims accepts a *judge*: a callable
``judge(a, b) -> bool`` deciding whether two answers say the same thing.

This module ships dependency-free reference judges. For serious work plug
in your own — an embedding-similarity judge or an LLM judge:

    def llm_judge(a: str, b: str) -> bool:
        return client.ask(f"Do these two answers assert the same fact? ...") == "yes"

    sk.stress(..., claim_equiv=llm_judge)

An *abstain detector* is the other primitive: ``abstain(answer) -> bool``,
True when the answer declines / expresses uncertainty rather than asserting
content. It powers the null-hallucination check in ``stresskit.oracle``
(Activation Oracles "will frequently produce an answer even when confidence
is low" — arXiv:2512.15674's own limitations section).
"""

from __future__ import annotations

import re
import string
from typing import Callable

Judge = Callable[[str, str], bool]

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, drop articles."""
    text = str(text).lower().translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact(a: str, b: str) -> bool:
    """Strict string equality (the v0.1 behavior)."""
    return str(a) == str(b)


def normalized(a: str, b: str) -> bool:
    """Equality after normalization — the sane default for short labels."""
    return normalize(a) == normalize(b)


def contains(answer: str, expected: str) -> bool:
    """Normalized containment: does the answer contain the expected string?

    This is the "exact target recovery" metric of the Activation Oracle
    literature (arXiv:2512.15674 checks whether the ground-truth attribute
    appears in the response, case-insensitive).
    """
    na, ne = normalize(answer), normalize(expected)
    return bool(ne) and ne in na


def token_f1(threshold: float = 0.6) -> Judge:
    """Judge factory: token-level F1 overlap ≥ threshold.

    A tolerant lexical judge for free-text answers when you have no model
    to call. SQuAD-style.
    """

    def judge(a: str, b: str) -> bool:
        ta, tb = normalize(a).split(), normalize(b).split()
        if not ta or not tb:
            return ta == tb
        common = 0
        pool = list(tb)
        for tok in ta:
            if tok in pool:
                pool.remove(tok)
                common += 1
        if common == 0:
            return False
        precision = common / len(ta)
        recall = common / len(tb)
        return 2 * precision * recall / (precision + recall) >= threshold

    return judge


_ABSTAIN_PATTERNS = re.compile(
    r"(i\s+(do\s*n[o']t|cannot|can\s*not|can't)\s+(know|tell|say|determine|answer)"
    r"|do\s*n[o']t\s+know"
    r"|not\s+(sure|certain|clear|able|enough\s+information)"
    r"|\bunclear\b|\buncertain\b|\bunknown\b|\bunable\b"
    r"|no\s+(information|idea|evidence|discernible|apparent)"
    r"|cannot\s+(be\s+)?(determined|answered|identified)"
    r"|\bn\s*/\s*a\b|there\s+is\s+no\b|does\s+not\s+(appear|contain|seem))",
    re.IGNORECASE,
)


def default_abstain(answer: str) -> bool:
    """Heuristic refusal/uncertainty detector for null-control probes.

    Deliberately conservative regexes; for real audits, supply an LLM-based
    ``abstain`` callable instead.
    """
    return bool(_ABSTAIN_PATTERNS.search(str(answer)))
