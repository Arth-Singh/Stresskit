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
        lines.append(f"# {emoji} Stability Card — grade **{self.grade}**")
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
        lines.append("| check | value | threshold | pass |")
        lines.append("|---|---|---|---|")
        for name, c in checks.items():
            op = "≤" if name == "score_stability" else "≥"
            lines.append(
                f"| {name.replace('_', ' ')} | {_fmt(c.get('value'))} | "
                f"{op} {_fmt(c.get('threshold'))} | {_fmt(c.get('passed'))} |"
            )
        lines.append("")

        lines.append("## Pooled metrics")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for key, label in (
            ("n_runs", "runs"),
            ("mean_pairwise_jaccard", "mean pairwise Jaccard"),
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


def load_card(path: str) -> StabilityCard:
    return StabilityCard.load(path)
