"""Rebuild the findings behind a committed Stability Card and regrade it.

A card written since schema 0.2 carries its per-run records, so the real
runs can be turned back into :class:`~stresskit.finding.Finding` objects and
fed to :func:`stresskit.from_findings` at the recorded seed; the result must
reproduce every check value, interval and state on the card. Null-control
runs are not on the card; they live in the ``<stem>.runs.json`` manifest the
card scripts write next to it, and are rebuilt from there when the manifest
embeds their components.

:func:`relabel_grade` is the migration path from grade rule v0.3 to v0.4: it
re-derives the letter from the recorded checks alone, exactly as
:func:`stresskit.verify_card_dict` does, so a relabelled card verifies without
touching any evidence.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import metrics as M
from .battery import (
    StressResult,
    Thresholds,
    from_findings,
    grade_checks,
    make_check,
)
from .card import (
    GRADE_RULES,
    SCHEMA_VERSION,
    _CHECK_SOURCES,
    _DIRECTION_CHECK_SOURCES,
    _card_structure_kind,
    _components_digest,
    classify_artifact_dict,
)
from .finding import Finding

_THRESHOLD_ORDER = (
    "jaccard",
    "modal_share",
    "score_cv",
    "random_margin",
    "specificity_ratio",
    "random_floor",
    "cosine",
)

_HOMONYM_STEM = re.compile(r"^homonym_reconvergence_(?P<slug>.+)$")
_HOMONYM_NULL_KEY = "null_runs_permuted"


def card_name(card: Mapping[str, Any]) -> str:
    """Human label for error messages: ``task / model`` as the scoreboard
    prints it, or the claim statement when neither is recorded."""
    claim = card.get("claim") or {}
    parts = [str(claim[k]) for k in ("task", "model") if claim.get(k)]
    if parts:
        return " / ".join(parts)
    return str(claim.get("statement") or "(unnamed card)")


def is_diagnostic_stability_card(card: Any) -> bool:
    """True for a diagnostic stability card; False for confirmatory cards,
    oracle reports, confirmatory-profile cards and anything else."""
    if classify_artifact_dict(card) != "stability_card":
        return False
    profile = (card.get("verdict") or {}).get("profile")
    return profile in (None, "diagnostic")


def _row_has_structure(row: Mapping[str, Any]) -> bool:
    present = row.get("structure_present")
    if present is not None:
        return bool(present)
    if row.get("components") is not None:
        return True
    return int(row.get("size") or 0) > 0


def _universe_meta(row: Mapping[str, Any]) -> Dict[str, Any]:
    universe = row.get("universe")
    if universe is None:
        universe = (row.get("meta") or {}).get("universe")
    return {"universe": universe} if universe is not None else {}


def _finding_from_row(
    row: Mapping[str, Any],
    *,
    universe_size: Optional[int],
    what: str,
) -> Finding:
    structured = _row_has_structure(row)
    components = row.get("components")
    if structured and components is None:
        raise ValueError(
            f"{what}: run {row.get('variant')!r} is structured (size "
            f"{row.get('size')}) but carries no components — a hash-only "
            "record cannot be regraded"
        )
    return Finding(
        components=(
            frozenset(str(c) for c in components) if components is not None else None
        ),
        claim=row.get("claim"),
        score=row.get("score"),
        universe_size=universe_size,
        meta=_universe_meta(row),
        structure_present=structured if components is not None else None,
    )


def findings_from_card_dict(
    card: Mapping[str, Any], *, name: Optional[str] = None
) -> Tuple[List[Finding], List[str]]:
    """The card's real runs as findings (base first) and the axis label of
    every non-base run, ready for :func:`stresskit.from_findings`."""
    what = f"card {name or card_name(card)}"
    rows = card.get("runs") or []
    if len(rows) < 2:
        raise ValueError(f"{what}: carries {len(rows)} run records, need >= 2")
    if rows[0].get("axis") != "base":
        raise ValueError(
            f"{what}: first run record is axis {rows[0].get('axis')!r}, "
            "expected the base run first"
        )
    if _card_structure_kind(card) == "direction":
        raise ValueError(
            f"{what}: direction card — its runs carry digests of their "
            "vectors, not the vectors; regrade it through "
            "regrade_direction_card"
        )
    universe_size = (card.get("claim") or {}).get("universe_size")
    findings = [
        _finding_from_row(row, universe_size=universe_size, what=what) for row in rows
    ]
    axes = [str(row["axis"]) for row in rows[1:]]
    return findings, axes


def findings_from_manifest_real_rows(
    card: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    name: Optional[str] = None,
) -> Tuple[List[Finding], List[str]]:
    """Real runs of a hash-only card, rebuilt from its manifest.

    Every manifest row must stand in the card's run order and hash to the
    ``components_sha256`` the card recorded for it; the card's digests are
    what make the manifest trustworthy."""
    what = f"card {name or card_name(card)}"
    real_rows = [
        r for r in manifest.get("runs") or [] if r.get("group", "real") == "real"
    ]
    card_rows = card.get("runs") or []
    if len(real_rows) != len(card_rows):
        raise ValueError(
            f"{what}: manifest has {len(real_rows)} real rows for "
            f"{len(card_rows)} card runs"
        )
    if not card_rows or card_rows[0].get("axis") != "base":
        raise ValueError(f"{what}: card run records do not start with the base run")
    universe_size = (card.get("claim") or {}).get("universe_size")
    findings: List[Finding] = []
    for card_row, row in zip(card_rows, real_rows):
        if card_row.get("variant") != row.get("variant"):
            raise ValueError(
                f"{what}: manifest row {row.get('variant')!r} does not match "
                f"card run {card_row.get('variant')!r} in order"
            )
        digest = card_row.get("components_sha256")
        components = row.get("components")
        if digest is not None and (
            components is None or _components_digest(components) != digest
        ):
            raise ValueError(
                f"{what}: manifest components for run {row.get('variant')!r} "
                "do not hash to the card's components_sha256"
            )
        merged = dict(card_row)
        if components is not None:
            merged["components"] = components
        findings.append(
            _finding_from_row(merged, universe_size=universe_size, what=what)
        )
    axes = [str(row["axis"]) for row in card_rows[1:]]
    return findings, axes


def null_findings_from_manifest(
    manifest: Mapping[str, Any],
    *,
    universe_size: Optional[int],
    key: Optional[str] = None,
) -> List[Finding]:
    """Null-control findings from a card manifest, base first.

    Rows come from ``manifest[key]`` when ``key`` is given, else from the
    ``group == "null"`` rows of ``manifest["runs"]``, else from
    ``manifest["null"]``. A structured row without components is an error:
    the specificity ratio cannot be re-derived from sizes alone."""
    if key is not None:
        rows = manifest.get(key)
        if rows is None:
            raise ValueError(f"manifest has no {key!r} block")
    elif "runs" in manifest:
        rows = [r for r in manifest["runs"] if r.get("group") == "null"]
    else:
        rows = manifest.get("null")
    if not rows:
        raise ValueError("manifest carries no null-control rows")
    return [
        _finding_from_row(row, universe_size=universe_size, what="null manifest")
        for row in rows
    ]


def resolve_manifest(card_path: str) -> Tuple[Optional[str], Optional[str]]:
    """(manifest path, null-rows key) for a card path, or (None, None).

    The regular layout is ``<stem>.runs.json``; the homonym cards keep their
    runs under ``raw/homonym_<slug>/runs_<slug>.json`` with the graded null
    in its own block."""
    stem, _ = os.path.splitext(card_path)
    sidecar = stem + ".runs.json"
    if os.path.exists(sidecar):
        return sidecar, None
    directory, base = os.path.split(stem)
    m = _HOMONYM_STEM.match(base)
    if m:
        slug = m.group("slug")
        raw = os.path.join(directory, "raw", f"homonym_{slug}", f"runs_{slug}.json")
        if os.path.exists(raw):
            return raw, _HOMONYM_NULL_KEY
    return None, None


def load_manifest(card_path: str) -> Optional[Tuple[Dict[str, Any], Optional[str]]]:
    """The parsed manifest and its null-rows key, or None without one."""
    path, key = resolve_manifest(card_path)
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f), key


def load_null_findings(card_path: str) -> Optional[List[Finding]]:
    """Null-control findings for the card at ``card_path``, or None when no
    manifest exists or the one that does carries no usable null rows."""
    loaded = load_manifest(card_path)
    if loaded is None:
        return None
    manifest, key = loaded
    with open(card_path, encoding="utf-8") as f:
        card = json.load(f)
    try:
        return null_findings_from_manifest(
            manifest,
            universe_size=(card.get("claim") or {}).get("universe_size"),
            key=key,
        )
    except ValueError:
        return None


def card_thresholds(card: Mapping[str, Any]) -> Thresholds:
    """The card's pass bars as a :class:`Thresholds`.

    Every bar the card records must equal the registered default — the
    committed cards were all graded under the defaults, and a card that was
    not needs a human, not a silent override. Bars the card predates
    (``specificity_ratio``, ``random_floor``) take their defaults."""
    stored = (card.get("verdict") or {}).get("thresholds") or {}
    defaults = asdict(Thresholds())
    unknown = sorted(set(stored) - set(defaults))
    if unknown:
        raise ValueError(
            f"card {card_name(card)}: unknown thresholds {unknown}; "
            f"known: {sorted(defaults)}"
        )
    off = {
        k: (stored[k], defaults[k]) for k in stored if float(stored[k]) != defaults[k]
    }
    if off:
        detail = ", ".join(
            f"{k}={stored_v} (default {default_v})"
            for k, (stored_v, default_v) in sorted(off.items())
        )
        raise ValueError(
            f"card {card_name(card)}: stored thresholds differ from the "
            f"registered defaults: {detail}"
        )
    merged = dict(defaults)
    merged.update({k: float(v) for k, v in stored.items()})
    return Thresholds(**merged)


def regrade_card(
    card: Mapping[str, Any],
    *,
    seed: int,
    null_findings: Optional[Sequence[Finding]] = None,
    name: Optional[str] = None,
) -> StressResult:
    """Post-hoc regrade of a set-valued card from its own run records."""
    findings, axes = findings_from_card_dict(card, name=name)
    return regrade_findings(
        card, findings, axes, seed=seed, null_findings=null_findings
    )


def regrade_findings(
    card: Mapping[str, Any],
    findings: Sequence[Finding],
    axes: Sequence[str],
    *,
    seed: int,
    null_findings: Optional[Sequence[Finding]] = None,
) -> StressResult:
    """Post-hoc regrade of ``findings`` under the card's thresholds and
    claim metadata — the findings having come from the card itself or,
    for a hash-only card, from its digest-checked manifest."""
    claim = card.get("claim") or {}
    return from_findings(
        findings,
        axes=axes,
        null_findings=null_findings,
        thresholds=card_thresholds(card),
        seed=seed,
        claim_statement=claim.get("statement"),
        model=claim.get("model"),
        task=claim.get("task"),
        method=claim.get("method"),
    )


def _upper_pairs(matrix: Sequence[Sequence[float]]) -> List[float]:
    n = len(matrix)
    return [matrix[i][j] for i in range(n) for j in range(i + 1, n)]


def regrade_direction_card(card: Mapping[str, Any], *, seed: int) -> Dict[str, Any]:
    """Recompute a direction card's checks from its embedded |cosine|
    matrices and grade them under both rules.

    Mirrors the direction branch of ``battery._analyze``: the structural
    metric, its interval, the random-null ratio and the specificity ratio
    are all functions of the pairwise matrices and the seed; claim and score
    intervals resample the card's runs."""
    what = f"card {card_name(card)}"
    if _card_structure_kind(card) != "direction":
        raise ValueError(f"{what}: not a direction card")
    block = card.get("directions") or {}
    matrix = block.get("abs_cosine")
    if not matrix:
        raise ValueError(
            f"{what}: no embedded directions.abs_cosine matrix; the structural "
            "metric cannot be recomputed"
        )
    thresholds = card_thresholds(card)
    checks: Dict[str, Any] = {}

    pairs = _upper_pairs(matrix)
    if not pairs:
        raise ValueError(f"{what}: directions matrix covers fewer than 2 runs")
    abs_cosine = sum(pairs) / len(pairs)
    abs_cosine_ci = M.bootstrap_ci_pairwise(
        range(len(matrix)), lambda a, b: matrix[a][b], seed=seed
    )
    checks["structural_stability"] = make_check(
        abs_cosine,
        thresholds.cosine,
        ">=",
        "mean pairwise |cosine| across all perturbed runs",
        ci=abs_cosine_ci,
    )

    rows = card.get("runs") or []
    labels = [r["claim"] for r in rows if r.get("claim") is not None]
    if len(rows) >= 2 and labels:
        checks["claim_stability"] = make_check(
            M.modal_share(labels),
            thresholds.modal_share,
            ">=",
            "modal claim share π* (filability at α=0.2)",
            ci=M.bootstrap_ci(labels, M.modal_share, seed=seed),
        )
    scores = [r["score"] for r in rows if r.get("score") is not None]
    cv = M.coefficient_of_variation(scores)
    if cv is not None:
        checks["score_stability"] = make_check(
            cv,
            thresholds.score_cv,
            "<=",
            "coefficient of variation of the quality score",
            ci=M.bootstrap_ci(scores, M.coefficient_of_variation, seed=seed),
        )

    random_cos = M.expected_random_abs_cosine(block["dim"])
    if random_cos:
        checks["beats_random"] = make_check(
            abs_cosine / random_cos,
            thresholds.random_margin,
            ">=",
            "direction overlap vs random unit vectors in R^d (×)",
            ci=(
                [abs_cosine_ci[0] / random_cos, abs_cosine_ci[1] / random_cos]
                if abs_cosine_ci
                else None
            ),
        )

    null_matrix = block.get("null_abs_cosine")
    if null_matrix:
        null_pairs = _upper_pairs(null_matrix)
        null_cosine = sum(null_pairs) / len(null_pairs)
        ratio = abs_cosine / null_cosine if null_cosine > 1e-9 else float("inf")

        def pair(a: Tuple[str, int], b: Tuple[str, int]) -> float:
            grid = matrix if a[0] == "real" else null_matrix
            return grid[a[1]][b[1]]

        checks["specificity"] = make_check(
            ratio,
            thresholds.specificity_ratio,
            ">=",
            "direction stability on real vs null-control data (×)",
            ci=M.bootstrap_ci_ratio_pairwise(
                [("real", i) for i in range(len(matrix))],
                [("null", i) for i in range(len(null_matrix))],
                pair,
                seed=seed,
            ),
        )

    borderline = [name for name, c in checks.items() if c.get("robust") is False]
    resolvable = [c for c in checks.values() if c.get("robust") is not None]
    if not resolvable:
        confidence = "unknown"
    elif borderline:
        confidence = "low"
    else:
        confidence = "high"
    return {
        "checks": checks,
        "grade_v03": grade_checks(
            checks, rule="v0.3", random_floor=thresholds.random_floor
        ),
        "grade_v04": grade_checks(
            checks, rule="v0.4", random_floor=thresholds.random_floor
        ),
        "confidence": confidence,
    }


def recomputed_checks(card: Mapping[str, Any]) -> Dict[str, Any]:
    """Each recorded check rebuilt from its (value, threshold, op, ci) —
    the same reconstruction ``verify_card_dict`` grades from."""
    sources = (
        _DIRECTION_CHECK_SOURCES
        if _card_structure_kind(card) == "direction"
        else _CHECK_SOURCES
    )
    out: Dict[str, Any] = {}
    for name, c in ((card.get("verdict") or {}).get("checks") or {}).items():
        op = c.get("op") or sources.get(name, (None, None))[1]
        if op is None:
            raise ValueError(f"card {card_name(card)}: unknown check {name!r}")
        if c.get("value") is None or c.get("threshold") is None:
            raise ValueError(
                f"card {card_name(card)}: check {name!r} lacks value/threshold"
            )
        out[name] = make_check(
            c["value"], c["threshold"], op, c.get("description") or "", ci=c.get("ci")
        )
    return out


def grade_reasons(checks: Mapping[str, Any], *, random_floor: float) -> Dict[str, Any]:
    """Why the decided rule lands where it does: decided passes and fails,
    undecided checks, and whichever cap or floor applies."""
    br = checks.get("beats_random")
    at_random = (
        br is not None and br.get("value") is not None and br["value"] <= random_floor
    )
    specificity = checks.get("specificity")
    if specificity is None:
        cap = "B"
        cap_reason = "no null control caps at B"
    elif specificity.get("state") == "fail":
        cap = "C"
        cap_reason = "decided specificity fail caps at C"
    else:
        cap = "A"
        cap_reason = None
    return {
        "decided_pass": [n for n, c in checks.items() if c.get("state") == "pass"],
        "decided_fail": [n for n, c in checks.items() if c.get("state") == "fail"],
        "undecided": [n for n, c in checks.items() if c.get("state") == "inconclusive"],
        "point_pass": [n for n, c in checks.items() if c.get("passed")],
        "cap": cap,
        "cap_reason": cap_reason,
        "at_random_floor": at_random,
    }


def relabel_grade(
    card: Mapping[str, Any], *, rule: str = "v0.4", note_date: str
) -> Dict[str, Any]:
    """A copy of a diagnostic stability card graded under ``rule`` from its
    recorded checks, at schema 0.5.

    The evidence is untouched: values, intervals, ``passed`` and ``robust``
    stay as recorded. Cards from before schema 0.3 gain the fields 0.5
    requires (a diagnostic profile and each check's interval ``state``, the
    state being what ``verify_card_dict`` recomputes anyway). The recorded
    grade must re-derive under the rule the card was graded with, else the
    card is inconsistent and relabelling it would bury that."""
    if rule not in GRADE_RULES:
        raise ValueError(f"grade rule must be one of {GRADE_RULES}, got {rule!r}")
    if not is_diagnostic_stability_card(card):
        raise ValueError(
            "relabel_grade takes a diagnostic stability card; got "
            f"{classify_artifact_dict(card)} with profile "
            f"{(card.get('verdict') or {}).get('profile')!r}"
        )
    new = copy.deepcopy(dict(card))
    verdict = new["verdict"]
    thresholds = card_thresholds(new)
    checks = recomputed_checks(new)
    if not checks:
        raise ValueError(f"card {card_name(card)}: no checks to grade")
    old_rule = verdict.get("grade_rule", "v0.3")
    old_grade = verdict["grade"]
    rederived_old = grade_checks(
        checks, rule=old_rule, random_floor=thresholds.random_floor
    )
    if rederived_old != old_grade:
        raise ValueError(
            f"card {card_name(card)}: recorded grade {old_grade!r} does not "
            f"re-derive under its own rule {old_rule} ({rederived_old!r}); "
            "fix the card before relabelling it"
        )
    new_grade = grade_checks(checks, rule=rule, random_floor=thresholds.random_floor)

    for name, fresh in checks.items():
        verdict["checks"][name].setdefault("state", fresh["state"])
    stored = dict(verdict.get("thresholds") or {})
    stored.setdefault("specificity_ratio", thresholds.specificity_ratio)
    stored.setdefault("random_floor", thresholds.random_floor)
    relabelled = {
        "grade": new_grade,
        "grade_rule": rule,
        "profile": verdict.get("profile", "diagnostic"),
        "confirmatory_state": verdict.get("confirmatory_state", "not_applicable"),
        "required_checks": verdict.get("required_checks", []),
        "checks": verdict["checks"],
        "thresholds": {k: stored[k] for k in _THRESHOLD_ORDER if k in stored},
    }
    for key, value in verdict.items():
        relabelled.setdefault(key, value)
    new["verdict"] = relabelled
    new["schema_version"] = SCHEMA_VERSION
    if old_rule != rule:
        new["notes"] = list(new.get("notes") or []) + [
            f"{old_rule} grade: {old_grade}; regraded {note_date} under grade "
            f"rule {rule} from the recorded checks (schema {SCHEMA_VERSION})"
        ]
    return new
