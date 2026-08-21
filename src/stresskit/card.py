"""The Stability Card — StressKit's shareable artifact.

A Stability Card is a machine-readable record of how an interpretability
claim held up under a perturbation battery: what was varied, what stayed
the same, and a letter grade. It is designed to be attached to papers,
README files, and model/SAE releases, and rendered as a shields.io badge.

Schema: src/stresskit/schemas/stability_card_v0.json (version 0.1).
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "0.1"
GRADE_ORDER = ("A", "B", "C", "D")

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
    ) -> "StabilityCard":
        from . import __version__

        return cls(
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
            if c.get("passed"):
                # ⚠ marks a pass the CI does not actually resolve
                mark = "✅" if c.get("robust") is not False else "⚠️"
            else:
                mark = "❌"
            lines.append(
                f"| {name.replace('_', ' ')} | {_fmt(c.get('value'))} | {ci_str} | "
                f"{op} {_fmt(c.get('threshold'))} | {mark} |"
            )
        lines.append("")
        if confidence == "low":
            bl = ", ".join(pooled.get("borderline_checks", []))
            lines.append(
                f"> ⚠️ **Underpowered:** {bl} pass on the point estimate but the "
                f"95% CI straddles the bar. The grade is provisional — raise "
                f"`n_runs` before reporting it."
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


def verify_card_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Auditor mode: re-derive a card's verdict from its own recorded metrics.

    Recomputes every check's pass/fail from (value, threshold), cross-checks
    each check value against the pooled metrics it must equal, re-derives
    the specificity ratio from the null-control block, and regrades. Any
    disagreement means the card was edited after the fact or produced by a
    non-conforming implementation.

    Returns ``{"ok": bool, "problems": [str], "recomputed_grade": str}``.
    """
    from .battery import grade_checks  # deferred: battery imports this module

    validate_card_dict(d)
    checks = d.get("verdict", {}).get("checks", {})
    pooled = d.get("metrics", {}).get("pooled", {})
    problems: List[str] = []
    recomputed: Dict[str, Any] = {}

    for name, c in checks.items():
        src, op = _CHECK_SOURCES.get(name, (None, None))
        if op is None:
            problems.append(f"unknown check {name!r}")
            continue
        value, threshold = c.get("value"), c.get("threshold")
        if value is None or threshold is None:
            problems.append(f"{name}: missing value/threshold")
            continue
        passed = value >= threshold if op == ">=" else value <= threshold
        recomputed[name] = {"value": value, "passed": passed}
        if bool(c.get("passed")) != passed:
            problems.append(
                f"{name}: stored passed={c.get('passed')} but "
                f"{value} {op} {threshold} is {passed}"
            )
        if src is not None:
            pv = pooled.get(src)
            if pv is not None and abs(pv - value) > 1e-9:
                problems.append(
                    f"{name}: value {value} != pooled {src} {pv}"
                )
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
    else:
        regrade = "D"
        problems.append("no recomputable checks on the card")

    return {"ok": not problems, "problems": problems, "recomputed_grade": regrade}


def load_card(path: str) -> StabilityCard:
    return StabilityCard.load(path)
