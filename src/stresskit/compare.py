"""Stability regression testing: compare two StressKit artifacts.

The "codecov" move, made literal: a pipeline that produces a Stability Card
per release can diff the new card against the last one and fail CI when
stability regresses — a check flipping from pass to fail, or the grade
dropping. ``stresskit compare old.json new.json --fail-on-regression`` is
the whole integration.

Comparison is deliberately conservative about what it calls a change:

- **regressed / improved** are verdict-level (a check's pass flag flipped,
  or the letter grade moved). Point-value drift that doesn't flip a verdict
  is reported but not judged.
- a value delta is only called **decisive** when both cards carry a 95% CI
  for that check and the intervals do not overlap; overlapping CIs mean the
  two runs are statistically indistinguishable on that check.
- checks whose thresholds differ between the cards are flagged and excluded
  from regression verdicts — a "regression" against a moved goalpost is
  meaningless.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .card import GRADE_ORDER, classify_artifact_dict, verify_artifact_dict

_METADATA_KEYS = ("model", "task", "method")


def _grade_rank(g: Optional[str]) -> Optional[int]:
    return GRADE_ORDER.index(g) if g in GRADE_ORDER else None


def _cis_disjoint(ci_a: Optional[Sequence[float]],
                  ci_b: Optional[Sequence[float]]) -> Optional[bool]:
    if not ci_a or not ci_b:
        return None
    return ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]


def _checks_of(d: Dict[str, Any], kind: str) -> Dict[str, Any]:
    return (d.get("verdict", {}).get("checks")
            if kind == "stability_card" else d.get("checks")) or {}


def _identity_of(d: Dict[str, Any], kind: str) -> Dict[str, Any]:
    if kind == "stability_card":
        claim = d.get("claim", {})
        return {k: claim.get(k) for k in _METADATA_KEYS}
    return {"oracle_name": d.get("oracle_name")}


def compare_cards(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two artifacts of the same kind; ``a`` is the baseline.

    Returns a dict with per-check rows, the regression/improvement lists,
    ``grade_regressed``, an overall ``regressed`` flag (any check regression
    or a grade drop, over comparable checks only), and ``caveats``. Both
    artifacts are re-verified first; a card that does not re-derive from its
    own metrics poisons any comparison, so that is a hard error.
    """
    kind_a, kind_b = classify_artifact_dict(a), classify_artifact_dict(b)
    if "unknown" in (kind_a, kind_b):
        raise ValueError("both inputs must be stability cards or oracle reports")
    if kind_a != kind_b:
        raise ValueError(
            f"cannot compare a {kind_a} against a {kind_b} — "
            "the checks measure different things")
    for label, d in (("baseline", a), ("candidate", b)):
        result = verify_artifact_dict(d)
        if not result["ok"]:
            raise ValueError(
                f"{label} does not verify (recomputed grade "
                f"{result['recomputed_grade']}): {result['problems'][:3]}")

    caveats: List[str] = []
    ident_a, ident_b = _identity_of(a, kind_a), _identity_of(b, kind_b)
    if ident_a != ident_b:
        caveats.append(
            f"the cards describe different findings ({ident_a} vs {ident_b}) "
            "— this is a cross-finding comparison, not a regression test on "
            "one pipeline")

    checks_a, checks_b = _checks_of(a, kind_a), _checks_of(b, kind_b)
    names = sorted(set(checks_a) | set(checks_b))
    rows: Dict[str, Dict[str, Any]] = {}
    regressions: List[str] = []
    improvements: List[str] = []
    for name in names:
        ca, cb = checks_a.get(name), checks_b.get(name)
        if ca is None or cb is None:
            missing_side = "baseline" if ca is None else "candidate"
            caveats.append(f"check {name!r} exists only on one card "
                           f"(missing from the {missing_side}) — not compared")
            rows[name] = {"comparable": False,
                          "value_a": ca and ca.get("value"),
                          "value_b": cb and cb.get("value")}
            continue
        comparable = ca.get("threshold") == cb.get("threshold") \
            and ca.get("op") == cb.get("op")
        if not comparable:
            caveats.append(
                f"check {name!r} has different thresholds "
                f"({ca.get('op')} {ca.get('threshold')} vs "
                f"{cb.get('op')} {cb.get('threshold')}) — excluded from "
                "regression verdicts")
        va, vb = ca.get("value"), cb.get("value")
        delta = (vb - va) if va is not None and vb is not None else None
        op = ca.get("op") or ">="
        # positive `better` means the candidate moved in the passing direction
        better = None if delta is None else (delta if op == ">=" else -delta)
        regressed = comparable and bool(ca.get("passed")) and not cb.get("passed")
        improved = comparable and not ca.get("passed") and bool(cb.get("passed"))
        rows[name] = {
            "comparable": comparable,
            "value_a": va, "value_b": vb, "delta": delta,
            "passed_a": ca.get("passed"), "passed_b": cb.get("passed"),
            "op": op, "threshold": ca.get("threshold"),
            "ci_a": ca.get("ci"), "ci_b": cb.get("ci"),
            "moved_toward_passing": None if better is None else better > 0,
            "decisive": _cis_disjoint(ca.get("ci"), cb.get("ci")),
            "regressed": regressed,
            "improved": improved,
        }
        if regressed:
            regressions.append(name)
        if improved:
            improvements.append(name)

    grade_a = a.get("verdict", {}).get("grade")
    grade_b = b.get("verdict", {}).get("grade")
    ra, rb = _grade_rank(grade_a), _grade_rank(grade_b)
    grade_regressed = ra is not None and rb is not None and rb > ra

    conf_a = (a.get("metrics", {}).get("pooled", {}).get("confidence")
              if kind_a == "stability_card"
              else a.get("metrics", {}).get("confidence"))
    conf_b = (b.get("metrics", {}).get("pooled", {}).get("confidence")
              if kind_b == "stability_card"
              else b.get("metrics", {}).get("confidence"))
    if "low" in (conf_a, conf_b):
        caveats.append(
            "at least one card is low-confidence (a CI straddles its bar) — "
            "verdict flips involving an undecided check may be sampling "
            "noise, not a real change")

    return {
        "kind": kind_a,
        "identity_a": ident_a, "identity_b": ident_b,
        "grade_a": grade_a, "grade_b": grade_b,
        "confidence_a": conf_a, "confidence_b": conf_b,
        "grade_regressed": grade_regressed,
        "grade_improved": ra is not None and rb is not None and rb < ra,
        "checks": rows,
        "regressions": regressions,
        "improvements": improvements,
        "regressed": grade_regressed or bool(regressions),
        "caveats": caveats,
    }


def compare_markdown(cmp: Dict[str, Any],
                     labels: Tuple[str, str] = ("baseline", "candidate")) -> str:
    """Human-readable render of a :func:`compare_cards` result."""
    la, lb = labels
    verdict = ("⛔ REGRESSED" if cmp["regressed"]
               else "✅ no regression"
               + (" · improvements!" if cmp["improvements"]
                  or cmp["grade_improved"] else ""))
    lines = [
        f"# Stability comparison — {verdict}",
        "",
        f"> {la}: grade **{cmp['grade_a']}** "
        f"({cmp['confidence_a'] or 'unknown'} confidence) · "
        f"{lb}: grade **{cmp['grade_b']}** "
        f"({cmp['confidence_b'] or 'unknown'} confidence)",
        "",
        f"| check | {la} | {lb} | Δ | verdict |",
        "|---|---|---|---|---|",
    ]

    def fmt(x):
        return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

    for name, r in cmp["checks"].items():
        if not r.get("comparable"):
            status = "⚠️ not comparable"
        elif r["regressed"]:
            status = "⛔ pass → fail"
        elif r["improved"]:
            status = "✅ fail → pass"
        else:
            status = "pass" if r["passed_b"] else "fail"
            status += " (unchanged"
            if r.get("decisive"):
                status += ", CIs disjoint — real movement"
            status += ")"
        delta = r.get("delta")
        arrow = "" if delta is None else (" ↑" if delta > 0 else " ↓" if delta < 0 else "")
        lines.append(
            f"| {name.replace('_', ' ')} | {fmt(r.get('value_a'))} "
            f"| {fmt(r.get('value_b'))} | {fmt(delta)}{arrow} | {status} |")
    if cmp["caveats"]:
        lines += ["", "**Caveats**", ""]
        lines += [f"- {c}" for c in cmp["caveats"]]
    return "\n".join(lines)


def compare_paths(path_a: str, path_b: str) -> Dict[str, Any]:
    """Load two artifact JSONs and compare them (baseline first)."""
    with open(path_a, encoding="utf-8") as f:
        a = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        b = json.load(f)
    return compare_cards(a, b)
