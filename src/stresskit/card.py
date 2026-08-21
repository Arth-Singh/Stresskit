"""The Stability Card — StressKit's shareable artifact.

A Stability Card is a machine-readable record of how an interpretability
claim held up under a perturbation battery: what was varied, what stayed
the same, and a letter grade. It is designed to be attached to papers,
README files, and model/SAE releases, and rendered as a shields.io badge.

Schema: src/stresskit/schemas/stability_card_v0.json (version 0.2).

Since schema 0.2 a card carries its per-run records (sizes, claims,
scores, components or their SHA-256 digests), making it a self-contained,
recomputable artifact: ``stresskit verify`` re-derives the pooled metrics,
checks, grade, and confidence from the card alone.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "0.2"
GRADE_ORDER = ("A", "B", "C", "D")

# Per-run components are embedded on the card (making it a self-contained,
# recomputable artifact) up to this many total component entries; beyond
# that only per-run SHA-256 digests are kept.
MAX_EMBED_COMPONENTS = 20_000


def _components_digest(components) -> str:
    payload = json.dumps(sorted(str(c) for c in components))
    return hashlib.sha256(payload.encode()).hexdigest()

_GRADE_COLORS = {
    "A": "brightgreen",
    "B": "yellowgreen",
    "C": "orange",
    "D": "red",
}
_GRADE_EMOJI = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}

_REQUIRED_TOP_LEVEL = (
    "schema_version",
    "stresskit_version",
    "created_at",
    "claim",
    "battery",
    "metrics",
    "verdict",
    "provenance",
)


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "✅" if x else "❌"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


@dataclass
class StabilityCard:
    claim: Dict[str, Any]
    battery: Dict[str, Any]
    metrics: Dict[str, Any]
    verdict: Dict[str, Any]
    provenance: Dict[str, Any]
    schema_version: str = SCHEMA_VERSION
    stresskit_version: str = ""
    created_at: str = ""
    notes: List[str] = field(default_factory=list)
    runs: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_stress(
        cls,
        result: "StressResult",  # noqa: F821 - avoid circular import at type time
        *,
        battery: List[str],
        n_runs: int,
        seed: int,
        base_config: Dict[str, Any],
        thresholds: "Thresholds",  # noqa: F821
        claim_statement: Optional[str],
        model: Optional[str],
        task: Optional[str],
        method: Optional[str],
        notes: List[str],
        wall_seconds: float,
        claim_equiv_used: bool = False,
    ) -> "StabilityCard":
        from . import __version__

        # Per-run record: enough to recompute the pooled metrics from the
        # card alone. Components are embedded when the total stays small;
        # otherwise only their digest is kept (structure still tamper-
        # evident, no longer recomputable offline).
        total_components = sum(r.finding.size for r in result.runs)
        embed = total_components <= MAX_EMBED_COMPONENTS
        run_rows: List[Dict[str, Any]] = []
        for r in result.runs:
            f = r.finding
            row: Dict[str, Any] = {
                "axis": r.axis,
                "variant": r.variant,
                "seed": r.seed,
                "size": f.size,
                "claim": f.claim,
                "score": f.score,
                "universe": f.meta.get("universe"),
            }
            if f.has_structure():
                row["components_sha256"] = _components_digest(f.components)
                if embed:
                    row["components"] = sorted(str(c) for c in f.components)
            run_rows.append(row)

        return cls(
            runs=run_rows,
            stresskit_version=__version__,
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            claim={
                "statement": claim_statement or result.base.claim or "(unstated)",
                "base_claim_label": result.base.claim,
                "model": model,
                "task": task,
                "method": method,
                "base_size": result.base.size or None,
                "universe_size": result.base.universe_size,
            },
            battery={
                "axes": battery,
                "n_runs_total": n_runs,
                "seed": seed,
                "base_config": base_config,
                "claim_equiv_used": claim_equiv_used,
                "components_embedded": embed,
            },
            metrics={
                "pooled": result.pooled,
                "per_axis": result.axis_metrics,
                **(
                    {"null_control": result.null_summary}
                    if getattr(result, "null_summary", None) is not None
                    else {}
                ),
            },
            verdict={
                "grade": result.grade,
                "checks": result.checks,
                "thresholds": {
                    "jaccard": thresholds.jaccard,
                    "modal_share": thresholds.modal_share,
                    "score_cv": thresholds.score_cv,
                    "random_margin": thresholds.random_margin,
                },
            },
            provenance={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "wall_seconds": wall_seconds,
            },
            notes=list(notes),
        )

    # ------------------------------------------------------------- (de)serialize
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stresskit_version": self.stresskit_version,
            "created_at": self.created_at,
            "claim": self.claim,
            "battery": self.battery,
            "metrics": self.metrics,
            "verdict": self.verdict,
            "provenance": self.provenance,
            "notes": self.notes,
            "runs": self.runs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StabilityCard":
        validate_card_dict(d)
        return cls(
            schema_version=d["schema_version"],
            stresskit_version=d["stresskit_version"],
            created_at=d["created_at"],
            claim=d["claim"],
            battery=d["battery"],
            metrics=d["metrics"],
            verdict=d["verdict"],
            provenance=d["provenance"],
            notes=list(d.get("notes", [])),
            runs=list(d.get("runs", [])),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> "StabilityCard":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ----------------------------------------------------------------- renders
    @property
    def grade(self) -> str:
        return self.verdict.get("grade", "D")

    def badge_dict(self) -> Dict[str, Any]:
        """shields.io endpoint JSON (https://shields.io/badges/endpoint-badge).

        Host this JSON anywhere public, then embed:
        https://img.shields.io/endpoint?url=<raw-url-of-this-json>
        """
        pooled = self.metrics.get("pooled", {})
        j = pooled.get("mean_pairwise_jaccard")
        ms = pooled.get("modal_share")
        cv = pooled.get("score_cv")
        if j is not None:
            detail = f"J={j:.2f}"
        elif ms is not None:
            detail = f"π*={ms:.2f}"
        elif cv is not None:
            detail = f"CV={cv:.2f}"
        else:
            detail = "n/a"
        return {
            "schemaVersion": 1,
            "label": "stability",
            "message": f"{self.grade} · {detail}",
            "color": _GRADE_COLORS.get(self.grade, "lightgrey"),
        }

    def to_markdown(self) -> str:
        pooled = self.metrics.get("pooled", {})
        checks = self.verdict.get("checks", {})
        emoji = _GRADE_EMOJI.get(self.grade, "")
        lines: List[str] = []
        confidence = pooled.get("confidence")
        conf_str = f" ({confidence} confidence)" if confidence else ""
        lines.append(f"# {emoji} Stability Card — grade **{self.grade}**{conf_str}")
        lines.append("")
        lines.append(f"> **Claim:** {self.claim.get('statement')}")
        ctx = " · ".join(
            f"{k}: {v}" for k, v in (
                ("model", self.claim.get("model")),
                ("task", self.claim.get("task")),
                ("method", self.claim.get("method")),
            ) if v
        )
        if ctx:
            lines.append(f"> {ctx}")
        lines.append("")
        lines.append(
            f"Battery: `{', '.join(self.battery.get('axes', []))}` — "
            f"{self.battery.get('n_runs_total')} runs "
            f"(seed {self.battery.get('seed')}, "
            f"{self.provenance.get('wall_seconds', '?')}s)"
        )
        lines.append("")

        lines.append("## Checks")
        lines.append("")
        lines.append("| check | value | 95% CI | threshold | pass |")
        lines.append("|---|---|---|---|---|")
        for name, c in checks.items():
            op = c.get("op") or ("≤" if name == "score_stability" else "≥")
            op = {">=": "≥", "<=": "≤"}.get(op, op)
            ci = c.get("ci")
            ci_str = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "—"
            # ⚠ marks a verdict the CI does not actually resolve
            straddle = c.get("robust") is False
            if c.get("passed"):
                mark = "⚠️" if straddle else "✅"
            else:
                mark = "❌⚠️" if straddle else "❌"
            lines.append(
                f"| {name.replace('_', ' ')} | {_fmt(c.get('value'))} | {ci_str} | "
                f"{op} {_fmt(c.get('threshold'))} | {mark} |"
            )
        lines.append("")
        if confidence == "low":
            bl = ", ".join(pooled.get("borderline_checks", []))
            lines.append(
                f"> ⚠️ **Underpowered:** the 95% CI straddles the bar for {bl} — "
                f"undecided in either direction. The grade is provisional; "
                f"raise `n_runs` before reporting it."
            )
            lines.append("")

        lines.append("## Pooled metrics")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for key, label in (
            ("n_runs", "runs"),
            ("mean_pairwise_jaccard", "mean pairwise Jaccard"),
            ("mean_pairwise_jaccard_all_sizes", "Jaccard incl. size-mismatched runs"),
            ("min_pairwise_jaccard", "min pairwise Jaccard"),
            ("expected_random_jaccard", "random-null Jaccard"),
            ("jaccard_vs_random", "overlap vs random (×)"),
            ("flip_rate", "claim flip rate"),
            ("modal_share", "modal claim share π*"),
            ("n_claim_classes", "distinct claims"),
            ("score_mean", "score mean"),
            ("score_cv", "score CV"),
            ("median_size", "median finding size"),
        ):
            if key in pooled and pooled[key] is not None:
                lines.append(f"| {label} | {_fmt(pooled[key])} |")
        for key, label in (
            ("mean_pairwise_jaccard_ci95", "Jaccard 95% CI (bootstrap)"),
            ("flip_rate_ci95", "flip rate 95% CI (bootstrap)"),
        ):
            ci = pooled.get(key)
            if ci:
                lines.append(f"| {label} | [{_fmt(ci[0])}, {_fmt(ci[1])}] |")
        null_control = self.metrics.get("null_control")
        if null_control:
            nj = null_control.get("mean_pairwise_jaccard")
            nf = null_control.get("flip_rate")
            lines.append(
                f"| null-control (specificity) | Jaccard {_fmt(nj)} · "
                f"flip {_fmt(nf)} on {null_control.get('n_runs')} null runs |"
            )
        claim_counts = pooled.get("claim_counts") or {}
        if len(claim_counts) > 1:
            top = ", ".join(f"`{k}`×{v}" for k, v in list(claim_counts.items())[:5])
            lines.append(f"| claim distribution | {top} |")
        shares = pooled.get("variance_shares") or {}
        if shares:
            share_txt = ", ".join(f"{k}: {v:.0%}" for k, v in shares.items())
            lines.append(f"| score-variance shares (OAT) | {share_txt} |")
        lines.append("")

        per_axis = self.metrics.get("per_axis", {})
        if per_axis:
            lines.append("## Per-axis breakdown")
            lines.append("")
            lines.append("| axis | runs | Jaccard | flip rate | π* | score CV |")
            lines.append("|---|---|---|---|---|---|")
            for axis, m in per_axis.items():
                lines.append(
                    f"| {axis} | {m.get('n_runs')} | "
                    f"{_fmt(m.get('mean_pairwise_jaccard'))} | "
                    f"{_fmt(m.get('flip_rate'))} | "
                    f"{_fmt(m.get('modal_share'))} | "
                    f"{_fmt(m.get('score_cv'))} |"
                )
            lines.append("")

        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")

        lines.append(
            f"*Generated by [StressKit](https://github.com/Arth-Singh/Stresskit) "
            f"v{self.stresskit_version} · schema {self.schema_version} · "
            f"{self.created_at}*"
        )
        return "\n".join(lines)


def validate_card_dict(d: Dict[str, Any]) -> None:
    """Minimal structural validation (no external jsonschema dependency)."""
    if not isinstance(d, dict):
        raise ValueError("Stability Card must be a JSON object")
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in d]
    if missing:
        raise ValueError(f"Stability Card missing required fields: {missing}")
    grade = d.get("verdict", {}).get("grade")
    if grade not in GRADE_ORDER:
        raise ValueError(f"Stability Card verdict.grade must be one of {GRADE_ORDER}, got {grade!r}")


# check name -> (pooled metric it must equal, comparison direction)
_CHECK_SOURCES = {
    "structural_stability": ("mean_pairwise_jaccard", ">="),
    "claim_stability": ("modal_share", ">="),
    "score_stability": ("score_cv", "<="),
    "beats_random": ("jaccard_vs_random", ">="),
    "specificity": (None, ">="),
}


def _verify_runs(d: Dict[str, Any], pooled: Dict[str, Any],
                 problems: List[str]) -> None:
    """Recompute pooled metrics from the card's own per-run records.

    Only possible when components are embedded (small batteries) — digests
    alone prove integrity of nothing further. Claims are recomputed only
    when no claim_equiv judge was used (a judge cannot be re-run offline).
    Component identity is compared via their string forms, which is exact
    for homogeneous component types.
    """
    from . import metrics as M

    runs = d.get("runs") or []
    if not runs:
        return
    base = next((r for r in runs if r.get("axis") == "base"), runs[0])

    # hash consistency for every structured run
    for r in runs:
        if r.get("components") is not None:
            if _components_digest(r["components"]) != r.get("components_sha256"):
                problems.append(
                    f"run {r.get('variant')!r}: components do not match "
                    "their recorded sha256"
                )

    structured = [
        r for r in runs
        if r.get("components") is not None
        and r.get("universe") == base.get("universe")
    ]
    if len(structured) >= 2 and base.get("size"):
        sets = [frozenset(r["components"]) for r in structured]
        base_size = base["size"]
        comparable = [
            s for s in sets if base_size / 2 <= len(s) <= base_size * 2
        ]
        graded = comparable if len(comparable) < len(sets) else sets
        expected_j = M.mean_pairwise_jaccard(graded)
        stored_j = pooled.get("mean_pairwise_jaccard")
        if expected_j is not None and stored_j is not None \
                and abs(expected_j - stored_j) > 1e-9:
            problems.append(
                f"pooled mean_pairwise_jaccard {stored_j} does not recompute "
                f"from the card's runs ({expected_j})"
            )

    if not d.get("battery", {}).get("claim_equiv_used"):
        labels = [r["claim"] for r in runs if r.get("claim") is not None]
        stored_ms = pooled.get("modal_share")
        if len(labels) >= 2 and stored_ms is not None:
            expected_ms = M.modal_share(labels)
            if abs(expected_ms - stored_ms) > 1e-9:
                problems.append(
                    f"pooled modal_share {stored_ms} does not recompute "
                    f"from the card's runs ({expected_ms})"
                )

    scores = [r["score"] for r in runs if r.get("score") is not None]
    stored_cv = pooled.get("score_cv")
    if len(scores) >= 2 and stored_cv is not None:
        expected_cv = M.coefficient_of_variation(scores)
        if expected_cv is not None and abs(expected_cv - stored_cv) > 1e-9:
            problems.append(
                f"pooled score_cv {stored_cv} does not recompute from the "
                f"card's runs ({expected_cv})"
            )


def verify_card_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Auditor mode: re-derive a card's verdict from its own contents.

    Three layers, each catching a different class of tampering:

    1. **checks** — pass/fail and the CI ``robust`` flag re-derive from each
       check's (value, threshold, op, ci); check values must equal the
       pooled metrics they summarize; the grade and confidence re-derive
       from the checks.
    2. **runs** (schema >= 0.2, components embedded) — the pooled Jaccard,
       modal share, and score CV recompute from the card's own per-run
       records, and every run's components match their SHA-256 digest.
    3. **specificity** — the ratio re-derives from the null-control block.

    Returns ``{"ok": bool, "problems": [str], "recomputed_grade": str}``.
    """
    from .battery import grade_checks, make_check  # deferred: circular import

    validate_card_dict(d)
    checks = d.get("verdict", {}).get("checks", {})
    pooled = d.get("metrics", {}).get("pooled", {})
    problems: List[str] = []
    recomputed: Dict[str, Any] = {}
    # schema 0.1 cards predate the symmetric robust semantics and carry no
    # runs; only their point-estimate layer is verifiable
    strict = str(d.get("schema_version", "0.1")) >= "0.2"

    for name, c in checks.items():
        src, default_op = _CHECK_SOURCES.get(name, (None, None))
        op = c.get("op") or default_op
        if op is None:
            problems.append(f"unknown check {name!r}")
            continue
        value, threshold = c.get("value"), c.get("threshold")
        if value is None or threshold is None:
            problems.append(f"{name}: missing value/threshold")
            continue
        fresh = make_check(value, threshold, op, "", ci=c.get("ci"))
        recomputed[name] = fresh
        if bool(c.get("passed")) != fresh["passed"]:
            problems.append(
                f"{name}: stored passed={c.get('passed')} but "
                f"{value} {op} {threshold} is {fresh['passed']}"
            )
        if strict and c.get("ci") is not None and c.get("robust") != fresh["robust"]:
            problems.append(
                f"{name}: stored robust={c.get('robust')} but the CI "
                f"{c.get('ci')} implies {fresh['robust']}"
            )
        if src is not None:
            pv = pooled.get(src)
            if pv is not None and abs(pv - value) > 1e-9:
                problems.append(f"{name}: value {value} != pooled {src} {pv}")
        elif name == "specificity":
            null_control = d.get("metrics", {}).get("null_control") or {}
            nj = null_control.get("mean_pairwise_jaccard")
            j = pooled.get("mean_pairwise_jaccard")
            if nj is not None and j is not None:
                expected = j / nj if nj > 1e-9 else float("inf")
                if abs(expected - value) > 1e-9:
                    problems.append(
                        f"specificity: value {value} != pooled/null ratio {expected}"
                    )
            else:
                problems.append(
                    "specificity check present but metrics.null_control is missing"
                )

    if recomputed:
        regrade = grade_checks(recomputed)
        stored = d["verdict"]["grade"]
        if regrade != stored:
            problems.append(f"grade: stored {stored!r}, recomputed {regrade!r}")
        stored_conf = pooled.get("confidence")
        if strict and stored_conf is not None:
            straddling = [c for c in recomputed.values()
                          if c.get("robust") is False]
            resolvable = [c for c in recomputed.values()
                          if c.get("robust") is not None]
            expected_conf = ("unknown" if not resolvable
                             else "low" if straddling else "high")
            if stored_conf != expected_conf:
                problems.append(
                    f"confidence: stored {stored_conf!r}, "
                    f"recomputed {expected_conf!r}"
                )
    else:
        regrade = "D"
        problems.append("no recomputable checks on the card")

    _verify_runs(d, pooled, problems)

    return {"ok": not problems, "problems": problems, "recomputed_grade": regrade}


def load_card(path: str) -> StabilityCard:
    return StabilityCard.load(path)
