"""The Stability Card — StressKit's shareable artifact.

A Stability Card is a machine-readable record of how an interpretability
claim held up under a perturbation battery: what was varied, what stayed
the same, and a letter grade. It is designed to be attached to papers,
README files, and model/SAE releases, and rendered as a shields.io badge.

Schema: src/stresskit/schemas/stability_card_v0.json (version 0.3).

Since schema 0.2 a card carries its per-run records (sizes, claims,
scores, components or their SHA-256 digests), making it a self-contained,
recomputable artifact: ``stresskit verify`` re-derives the pooled metrics,
checks, grade, and confidence from the card alone.

Schema 0.3 records interval-derived three-state checks and labels cards from
the current one-at-a-time battery as diagnostic. Diagnostic grades are
descriptive summaries, not confirmatory certificates.

Schema 0.4 adds direction-valued findings: a card whose runs produced
directions instead of component sets carries ``battery.structure_kind ==
"direction"`` and a ``directions`` block holding the pairwise |cosine| matrix
over the graded runs, so the structural metric, its bootstrap CI, and the
grade still re-derive from the card alone without embedding raw
high-dimensional vectors. Cards at 0.1-0.3 load and verify unchanged.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "0.4"
SUPPORTED_SCHEMA_VERSIONS = ("0.1", "0.2", "0.3", "0.4")
GRADE_ORDER = ("A", "B", "C", "D")

# Per-run components are embedded on the card (making it a self-contained,
# recomputable artifact) up to this many total component entries; beyond
# that only per-run SHA-256 digests are kept.
MAX_EMBED_COMPONENTS = 20_000


def _components_digest(components) -> str:
    payload = json.dumps(sorted(str(c) for c in components))
    return hashlib.sha256(payload.encode()).hexdigest()


def _vector_digest(vector) -> str:
    """SHA-256 of a direction's stored (unit-normalized) coordinates.

    The card embeds the pairwise |cosine| matrix, not the vectors, so this is
    what ties the matrix to the actual directions: an auditor holding the raw
    vectors re-normalizes them, hashes, and gets these digests back.
    """
    payload = json.dumps([float(x) for x in vector])
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
    utility: Optional[Dict[str, Any]] = None
    directions: Optional[Dict[str, Any]] = None

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
        utility: Optional[Dict[str, Any]] = None,
        directions: Optional[Dict[str, Any]] = None,
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
                "structure_present": f.has_structure(),
            }
            if f.has_structure():
                row["components_sha256"] = _components_digest(f.components)
                if embed:
                    row["components"] = sorted(str(c) for c in f.components)
            if f.has_direction():
                row["direction_dim"] = f.dim
                row["direction_sha256"] = _vector_digest(f.vector)
            run_rows.append(row)

        structure_kind = getattr(result, "structure_kind", "set")
        return cls(
            runs=run_rows,
            utility=utility,
            directions=directions,
            stresskit_version=__version__,
            created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            claim={
                "statement": claim_statement or result.base.claim or "(unstated)",
                "base_claim_label": result.base.claim,
                "model": model,
                "task": task,
                "method": method,
                "base_size": (
                    result.base.size if result.base.has_structure() else None
                ),
                "universe_size": result.base.universe_size,
            },
            battery={
                "axes": battery,
                "n_runs_total": n_runs,
                "seed": seed,
                "base_config": base_config,
                "claim_equiv_used": claim_equiv_used,
                "components_embedded": embed,
                **({"structure_kind": structure_kind}
                   if structure_kind == "direction" else {}),
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
                "profile": "diagnostic",
                "confirmatory_state": "not_applicable",
                "required_checks": [],
                "checks": result.checks,
                "thresholds": {
                    "jaccard": thresholds.jaccard,
                    "modal_share": thresholds.modal_share,
                    "score_cv": thresholds.score_cv,
                    "random_margin": thresholds.random_margin,
                    **({"cosine": thresholds.cosine}
                       if structure_kind == "direction" else {}),
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
            **({"utility": self.utility} if self.utility is not None else {}),
            **({"directions": self.directions}
               if self.directions is not None else {}),
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
            utility=d.get("utility"),
            directions=d.get("directions"),
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
        label = (
            "diagnostic stability"
            if self.verdict.get("profile") == "diagnostic"
            else "stability"
        )
        return {
            "schemaVersion": 1,
            "label": label,
            "message": f"{self.grade} · {detail}",
            "color": _GRADE_COLORS.get(self.grade, "lightgrey"),
        }

    def _utility_lines(self) -> List[str]:
        """Render the downstream-utility axis, or say it was never answered.

        Stability says a finding is reproducible under other defensible
        analyses; it never says the finding is worth anything. That question
        gets its own section so an unanswered one is visible.
        """
        from .utility import utility_check

        lines = ["## Downstream utility", ""]
        if self.utility is None:
            lines += [
                "**NOT REPORTED** ⚠️ — no task outside interpretability, and no "
                "baseline that ignores model internals. A stable finding can "
                "still buy nothing.",
                "",
            ]
            return lines
        u = self.utility
        result = utility_check(u)
        mark = {"pass": "✅ pass", "fail": "❌ fail",
                "inconclusive": "⚠️ inconclusive"}.get(result["state"], "—")
        lines.append(f"> **Task:** {u.get('task')}")
        lines.append(f"> metric: {u.get('metric')} · n = {u.get('n')}")
        lines.append("")
        lines.append("| approach | uses internals | value |")
        lines.append("|---|---|---|")
        lines.append(f"| **this finding** | yes | {_fmt(u.get('with_method'))} |")
        for b in u.get("baselines", []):
            lines.append(
                f"| {b.get('name')} | {'yes' if b.get('uses_internals') else 'no'} | "
                f"{_fmt(b.get('value'))} |"
            )
        lines.append("")
        ci = u.get("delta_ci95")
        ci_str = f" (95% CI [{_fmt(ci[0])}, {_fmt(ci[1])}])" if ci else ""
        lines.append(
            f"Margin over the best non-internals baseline "
            f"(*{u.get('reference_baseline')}*): "
            f"**{_fmt(u.get('delta_vs_non_internals'))}**{ci_str} — {mark}"
        )
        for key in ("reason", "task_phrasing_warning"):
            note = result.get(key) if key == "reason" else u.get(key)
            if note:
                lines.append("")
                lines.append(f"> ⚠️ {note}")
        lines.append("")
        return lines

    def to_markdown(self) -> str:
        pooled = self.metrics.get("pooled", {})
        checks = self.verdict.get("checks", {})
        direction_valued = (
            self.battery.get("structure_kind", "set") == "direction"
        )
        emoji = _GRADE_EMOJI.get(self.grade, "")
        lines: List[str] = []
        confidence = pooled.get("confidence")
        conf_str = f" ({confidence} confidence)" if confidence else ""
        profile = self.verdict.get("profile")
        diagnostic = profile == "diagnostic"
        title_prefix = "Diagnostic " if diagnostic else ""
        lines.append(
            f"# {emoji} {title_prefix}Stability Card — descriptive grade "
            f"**{self.grade}**{conf_str}"
        )
        lines.append("")
        if diagnostic:
            lines.append(
                "> **Diagnostic OAT profile:** this localizes sensitivity; it "
                "does not issue a confirmatory verdict or certificate."
            )
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
        lines.append("| check | value | 95% CI | threshold | state |")
        lines.append("|---|---|---|---|---|")
        for name, c in checks.items():
            op = c.get("op") or ("≤" if name == "score_stability" else "≥")
            op = {">=": "≥", "<=": "≤"}.get(op, op)
            ci = c.get("ci")
            ci_str = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "—"
            state = c.get("state")
            if state is None:  # legacy card rendering
                straddle = c.get("robust") is False
                state = (
                    "inconclusive" if straddle
                    else "pass" if c.get("passed") else "fail"
                )
            mark = {
                "pass": "✅ pass",
                "fail": "❌ fail",
                "inconclusive": "⚠️ inconclusive",
            }.get(state, "—")
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

        lines.extend(self._utility_lines())

        lines.append("## Pooled metrics")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for key, label in (
            ("n_runs", "runs"),
            ("n_structured_runs", "structured runs"),
            ("n_empty_findings", "empty structural findings"),
            ("empty_finding_rate", "empty structural finding rate"),
            ("mean_pairwise_jaccard", "mean pairwise Jaccard"),
            ("mean_pairwise_jaccard_all_sizes", "Jaccard incl. size-mismatched runs"),
            ("min_pairwise_jaccard", "min pairwise Jaccard"),
            ("expected_random_jaccard", "random-null Jaccard"),
            ("jaccard_vs_random", "overlap vs random (×)"),
            ("n_direction_runs", "direction runs"),
            ("direction_dim", "direction dimension d"),
            ("mean_pairwise_abs_cosine", "mean pairwise \\|cos\\|"),
            ("min_pairwise_abs_cosine", "min pairwise \\|cos\\|"),
            ("mean_pairwise_abs_cosine_axis_balanced", "\\|cos\\| axis-balanced"),
            ("expected_random_abs_cosine", "random-null \\|cos\\| in R^d"),
            ("abs_cosine_vs_random", "direction overlap vs random (×)"),
            ("flip_rate", "claim flip rate"),
            ("modal_share", "modal claim share π*"),
            ("n_claim_classes", "distinct claims"),
            ("score_mean", "score mean"),
            ("score_cv", "score CV"),
            ("median_size", "median finding size"),
        ):
            if direction_valued and key in ("n_structured_runs",
                                            "n_empty_findings"):
                continue
            if key in pooled and pooled[key] is not None:
                lines.append(f"| {label} | {_fmt(pooled[key])} |")
        for key, label in (
            ("mean_pairwise_jaccard_ci95", "Jaccard 95% CI (bootstrap)"),
            ("mean_pairwise_abs_cosine_ci95", "\\|cos\\| 95% CI (bootstrap)"),
            ("flip_rate_ci95", "flip rate 95% CI (bootstrap)"),
        ):
            ci = pooled.get(key)
            if ci:
                lines.append(f"| {label} | [{_fmt(ci[0])}, {_fmt(ci[1])}] |")
        null_control = self.metrics.get("null_control")
        if null_control:
            if direction_valued:
                struct_label = "\\|cos\\|"
                nj = null_control.get("mean_pairwise_abs_cosine")
            else:
                struct_label = "Jaccard"
                nj = null_control.get("mean_pairwise_jaccard")
            nf = null_control.get("flip_rate")
            lines.append(
                f"| null-control (specificity) | {struct_label} {_fmt(nj)} · "
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
            struct_col = "\\|cos\\|" if direction_valued else "Jaccard"
            struct_key = ("mean_pairwise_abs_cosine" if direction_valued
                          else "mean_pairwise_jaccard")
            lines.append(
                f"| axis | runs | {struct_col} | flip rate | π* | score CV |")
            lines.append("|---|---|---|---|---|---|")
            for axis, m in per_axis.items():
                lines.append(
                    f"| {axis} | {m.get('n_runs')} | "
                    f"{_fmt(m.get(struct_key))} | "
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
    version = str(d.get("schema_version"))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            "Stability Card schema_version must be one of "
            f"{SUPPORTED_SCHEMA_VERSIONS}, got {version!r}"
        )
    grade = d.get("verdict", {}).get("grade")
    if grade not in GRADE_ORDER:
        raise ValueError(f"Stability Card verdict.grade must be one of {GRADE_ORDER}, got {grade!r}")
    if "utility" in d:
        from .utility import validate_utility_block

        validate_utility_block(d["utility"])
    if "directions" in d:
        _validate_directions_block(d["directions"])
    if _schema_at_least(version, "0.3"):
        verdict = d.get("verdict", {})
        profile = verdict.get("profile")
        if profile not in ("diagnostic", "confirmatory"):
            raise ValueError(
                "schema 0.3 verdict.profile must be 'diagnostic' or "
                f"'confirmatory', got {profile!r}"
            )
        overall = verdict.get("confirmatory_state")
        if overall not in ("pass", "fail", "inconclusive", "not_applicable"):
            raise ValueError(
                "schema 0.3 verdict.confirmatory_state has invalid value "
                f"{overall!r}"
            )
        required = verdict.get("required_checks")
        if not isinstance(required, list) or not all(
                isinstance(name, str) for name in required):
            raise ValueError(
                "schema 0.3 verdict.required_checks must be a list of strings"
            )
        checks = verdict.get("checks")
        if not isinstance(checks, dict):
            raise ValueError("Stability Card verdict.checks must be an object")
        for name, check in checks.items():
            state = check.get("state") if isinstance(check, dict) else None
            if state not in ("pass", "fail", "inconclusive"):
                raise ValueError(
                    f"schema 0.3 check {name!r} has invalid or missing state "
                    f"{state!r}"
                )


def _validate_directions_block(block: Any) -> None:
    """Structural validation of the direction-valued extension (schema 0.4)."""
    if not isinstance(block, dict):
        raise ValueError("Stability Card 'directions' must be an object")
    dim = block.get("dim")
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        raise ValueError(
            f"directions.dim must be a positive integer, got {dim!r}"
        )
    for key in ("abs_cosine", "null_abs_cosine"):
        matrix = block.get(key)
        if matrix is None:
            continue
        if not isinstance(matrix, list) or not matrix:
            raise ValueError(f"directions.{key} must be a nonempty matrix")
        n = len(matrix)
        for row in matrix:
            if not isinstance(row, list) or len(row) != n:
                raise ValueError(
                    f"directions.{key} must be square, got a {n}-row matrix "
                    "with a mismatched row"
                )
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       for v in row):
                raise ValueError(f"directions.{key} must contain only numbers")
    order = block.get("order")
    if order is not None and (
        not isinstance(order, list)
        or not all(isinstance(x, str) for x in order)
    ):
        raise ValueError("directions.order must be a list of strings")


def _card_structure_kind(d: Dict[str, Any]) -> str:
    """``"direction"`` for a schema-0.4 direction card, ``"set"`` otherwise.

    Cards written before 0.4 carry no marker and are set-valued by
    construction, so the absent key is the answer, not a missing field.
    """
    return d.get("battery", {}).get("structure_kind", "set")


def _schema_at_least(version: str, minimum: str) -> bool:
    """Compare supported ``major.minor`` schema versions numerically."""
    return tuple(int(part) for part in version.split(".")) >= tuple(
        int(part) for part in minimum.split(".")
    )


# check name -> (pooled metric it must equal, comparison direction)
_CHECK_SOURCES = {
    "structural_stability": ("mean_pairwise_jaccard", ">="),
    "claim_stability": ("modal_share", ">="),
    "score_stability": ("score_cv", "<="),
    "beats_random": ("jaccard_vs_random", ">="),
    "specificity": (None, ">="),
}

# same, for direction-valued cards: |cosine| replaces Jaccard everywhere the
# structural metric appears, claims and scores are graded identically
_DIRECTION_CHECK_SOURCES = {
    "structural_stability": ("mean_pairwise_abs_cosine", ">="),
    "claim_stability": ("modal_share", ">="),
    "score_stability": ("score_cv", "<="),
    "beats_random": ("abs_cosine_vs_random", ">="),
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
        and r.get("structure_present", r.get("size", 0) > 0)
        and r.get("universe") == base.get("universe")
    ]
    base_has_structure = base.get(
        "structure_present", base.get("size", 0) > 0
    )
    if len(structured) >= 2 and base_has_structure:
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


def _verify_directions(d: Dict[str, Any], pooled: Dict[str, Any],
                       problems: List[str]) -> None:
    """Recompute a direction card's structural evidence from its own matrix.

    The card embeds the pairwise |cosine| matrix over exactly the runs that
    were graded, so everything the structural verdict rests on is
    recomputable offline: the mean, the minimum, and the bootstrap CI, which
    resamples runs and is therefore a function of the matrix and the recorded
    seed alone. When a null control was run its matrix is embedded too, so
    the specificity ratio and its interval recompute as well.

    The matrix is not self-authenticating — it is a summary, not the
    directions. What ties it to real vectors is the per-run SHA-256 of each
    unit direction, which an auditor holding the raw vectors re-derives.
    """
    from . import metrics as M

    block = d.get("directions")
    runs = d.get("runs") or []
    direction_runs = [r for r in runs if r.get("direction_sha256")]
    if _card_structure_kind(d) == "direction":
        if "structural_stability" in d.get("verdict", {}).get("checks", {}) \
                and not block:
            problems.append(
                "direction card grades structural_stability but carries no "
                "'directions' block to recompute it from"
            )
        if not direction_runs:
            problems.append(
                "direction card carries no per-run direction digests"
            )
    if not block:
        return

    dim = block.get("dim")
    bad_dim = [r.get("variant") for r in direction_runs
               if r.get("direction_dim") != dim]
    if bad_dim:
        problems.append(
            f"runs {bad_dim} record a direction dimension other than "
            f"directions.dim ({dim})"
        )

    matrix = block.get("abs_cosine")
    if matrix is None:
        return
    n = len(matrix)
    order = block.get("order")
    if order is not None and len(order) != n:
        problems.append(
            f"directions.order names {len(order)} runs but the matrix is {n}x{n}"
        )
    if len(direction_runs) < n:
        problems.append(
            f"directions matrix covers {n} runs but only "
            f"{len(direction_runs)} runs carry a direction digest"
        )
    for i in range(n):
        if abs(matrix[i][i] - 1.0) > 1e-9:
            problems.append(
                f"directions matrix diagonal [{i}][{i}] is {matrix[i][i]}, "
                "not 1.0 — a direction is not parallel to itself"
            )
        for j in range(n):
            if not (-1e-9 <= matrix[i][j] <= 1.0 + 1e-9):
                problems.append(
                    f"directions matrix [{i}][{j}] = {matrix[i][j]} is "
                    "outside [0, 1]; |cosine| cannot be"
                )
            elif abs(matrix[i][j] - matrix[j][i]) > 1e-9:
                problems.append(
                    f"directions matrix is not symmetric at [{i}][{j}]"
                )

    pairs = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return
    for key, expected in (
        ("mean_pairwise_abs_cosine", sum(pairs) / len(pairs)),
        ("min_pairwise_abs_cosine", min(pairs)),
    ):
        stored = pooled.get(key)
        if stored is not None and abs(expected - stored) > 1e-9:
            problems.append(
                f"pooled {key} {stored} does not recompute from the card's "
                f"direction matrix ({expected})"
            )

    boot = block.get("bootstrap") or {}
    if not all(k in boot for k in ("n_boot", "alpha", "seed")):
        return
    stored_ci = pooled.get("mean_pairwise_abs_cosine_ci95")
    if stored_ci is not None:
        fresh = M.bootstrap_ci_pairwise(
            range(n), lambda a, b: matrix[a][b],
            n_boot=boot["n_boot"], seed=boot["seed"], alpha=boot["alpha"],
        )
        if fresh is None or any(
                abs(a - b) > 1e-9 for a, b in zip(fresh, stored_ci)):
            problems.append(
                f"pooled mean_pairwise_abs_cosine_ci95 {stored_ci} does not "
                f"recompute from the direction matrix at seed "
                f"{boot['seed']} ({fresh})"
            )

    null_matrix = block.get("null_abs_cosine")
    if not null_matrix:
        return
    m = len(null_matrix)
    null_pairs = [null_matrix[i][j] for i in range(m) for j in range(i + 1, m)]
    null_control = d.get("metrics", {}).get("null_control") or {}
    stored_null = null_control.get("mean_pairwise_abs_cosine")
    if null_pairs and stored_null is not None:
        expected_null = sum(null_pairs) / len(null_pairs)
        if abs(expected_null - stored_null) > 1e-9:
            problems.append(
                f"null-control mean_pairwise_abs_cosine {stored_null} does "
                f"not recompute from the card's null matrix ({expected_null})"
            )
    stored_spec_ci = pooled.get("specificity_ci95")
    if stored_spec_ci is not None:
        def pair(a, b):
            grid = matrix if a[0] == "real" else null_matrix
            return grid[a[1]][b[1]]

        fresh_spec = M.bootstrap_ci_ratio_pairwise(
            [("real", i) for i in range(n)],
            [("null", i) for i in range(m)],
            pair, n_boot=boot["n_boot"], seed=boot["seed"],
            alpha=boot["alpha"],
        )
        if fresh_spec is None or any(
                abs(a - b) > 1e-9 for a, b in zip(fresh_spec, stored_spec_ci)):
            problems.append(
                f"pooled specificity_ci95 {stored_spec_ci} does not recompute "
                f"from the card's direction matrices ({fresh_spec})"
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
    4. **directions** (schema >= 0.4) — for a direction-valued card, the
       pooled |cosine|, its bootstrap CI, and the specificity interval
       recompute from the embedded pairwise |cosine| matrices.

    Returns the legacy descriptive grade plus any recomputable normative
    confirmatory state. A diagnostic card's confirmatory state is always
    ``not_applicable``.
    """
    from .battery import (  # deferred: circular import
        confirmatory_verdict,
        grade_checks,
        make_check,
    )

    validate_card_dict(d)
    checks = d.get("verdict", {}).get("checks", {})
    pooled = d.get("metrics", {}).get("pooled", {})
    kind = _card_structure_kind(d)
    sources = (_DIRECTION_CHECK_SOURCES if kind == "direction"
               else _CHECK_SOURCES)
    structural_metric = ("mean_pairwise_abs_cosine" if kind == "direction"
                         else "mean_pairwise_jaccard")
    problems: List[str] = []
    recomputed: Dict[str, Any] = {}
    # schema 0.1 cards predate the symmetric robust semantics and carry no
    # runs; only their point-estimate layer is verifiable
    version = str(d.get("schema_version", "0.1"))
    strict = _schema_at_least(version, "0.2")
    stateful = _schema_at_least(version, "0.3")

    for name, c in checks.items():
        src, default_op = sources.get(name, (None, None))
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
        if (stateful or "state" in c) and c.get("state") != fresh["state"]:
            problems.append(
                f"{name}: stored state={c.get('state')!r} but the CI "
                f"{c.get('ci')} implies {fresh['state']!r}"
            )
        if src is not None:
            pv = pooled.get(src)
            if pv is not None and abs(pv - value) > 1e-9:
                problems.append(f"{name}: value {value} != pooled {src} {pv}")
        elif name == "specificity":
            null_control = d.get("metrics", {}).get("null_control") or {}
            nj = null_control.get(structural_metric)
            j = pooled.get(structural_metric)
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

    profile = d.get("verdict", {}).get("profile")
    stored_overall = d.get("verdict", {}).get("confirmatory_state")
    if stateful and profile == "diagnostic":
        recomputed_overall: Optional[str] = "not_applicable"
        if stored_overall != recomputed_overall:
            problems.append(
                "confirmatory_state: diagnostic profile must store "
                "'not_applicable'"
            )
        if d.get("verdict", {}).get("required_checks"):
            problems.append("diagnostic profile cannot declare required_checks")
    elif stateful and profile == "confirmatory":
        required = d.get("verdict", {}).get("required_checks") or []
        missing_required = [name for name in required if name not in recomputed]
        if not required:
            problems.append("confirmatory profile requires required_checks")
            recomputed_overall = "inconclusive"
        elif missing_required:
            problems.append(
                f"confirmatory required checks missing: {missing_required}"
            )
            recomputed_overall = "inconclusive"
        else:
            recomputed_overall = confirmatory_verdict(
                recomputed, required=required
            )
        if stored_overall != recomputed_overall:
            problems.append(
                f"confirmatory_state: stored {stored_overall!r}, "
                f"recomputed {recomputed_overall!r}"
            )
    else:
        recomputed_overall = None

    _verify_runs(d, pooled, problems)
    _verify_directions(d, pooled, problems)

    return {
        "ok": not problems,
        "problems": problems,
        "recomputed_grade": regrade,
        "recomputed_confirmatory_state": recomputed_overall,
    }


# check name -> the metrics key it must equal (oracle reliability reports)
_ORACLE_CHECK_SOURCES = {
    "answer_consistency": ("answer_consistency", ">="),
    "known_accuracy": ("known_accuracy", ">="),
    "prompt_sensitivity": ("prompt_spread", "<="),
    "null_hallucination": ("null_hallucination_rate", "<="),
}

ORACLE_ARTIFACT = "stresskit_oracle_report"


def _verify_oracle_probes(d: Dict[str, Any], metrics: Dict[str, Any],
                          problems: List[str]) -> None:
    """Recompute the report's pooled metrics from its own per-probe rows.

    Mirrors the pooling in ``stress_oracle``: headline rates are per-probe
    (macro) means; the Wilson CIs are on the pooled (micro) counts.
    """
    from . import metrics as M

    per_probe = d.get("per_probe") or []
    if not per_probe:
        return
    known = [p for p in per_probe if p.get("kind") == "known"]
    nulls = [p for p in per_probe if p.get("kind") == "null"]
    non_null = [p for p in per_probe if p.get("kind") != "null"]

    def _avg(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    for metric_key, rows, row_key in (
        ("answer_consistency", non_null, "consistency"),
        ("known_accuracy", known, "accuracy"),
        ("prompt_spread", known, "prompt_spread"),
        ("null_hallucination_rate", nulls, "hallucination_rate"),
    ):
        stored = metrics.get(metric_key)
        expected = _avg(rows, row_key)
        if stored is not None and expected is not None \
                and abs(stored - expected) > 1e-9:
            problems.append(
                f"{metric_key} {stored} does not recompute from the report's "
                f"per-probe rows ({expected})"
            )

    for ci_key, rows, count_key in (
        ("known_accuracy_ci95", known, "n_correct"),
        ("null_hallucination_ci95", nulls, "n_asserted"),
    ):
        stored_ci = metrics.get(ci_key)
        if stored_ci is None or not rows:
            continue
        if any(count_key not in p or "n_answers" not in p for p in rows):
            continue  # counts not recorded on this report; CI unverifiable
        expected_ci = M.wilson_ci(sum(p[count_key] for p in rows),
                                  sum(p["n_answers"] for p in rows))
        if expected_ci is not None and any(
                abs(a - b) > 1e-9 for a, b in zip(stored_ci, expected_ci)):
            problems.append(
                f"{ci_key} {stored_ci} does not recompute from the report's "
                f"per-probe counts ({expected_ci})"
            )


def verify_oracle_report_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Auditor mode for oracle reliability reports (``stress_oracle``).

    Same contract as :func:`verify_card_dict`, adapted to the report
    artifact:

    1. **checks** — pass/fail and the CI ``robust`` flag re-derive from each
       check's (value, threshold, op, ci); check values must equal the
       pooled metrics they summarize; the grade and confidence re-derive
       from the checks.
    2. **probes** — the pooled consistency / accuracy / spread /
       hallucination metrics recompute from the report's own per-probe
       rows, and the Wilson CIs from the recorded counts.

    Returns ``{"ok": bool, "problems": [str], "recomputed_grade": str}``.
    """
    from .battery import grade_checks, make_check  # deferred: circular import

    if d.get("artifact") != ORACLE_ARTIFACT:
        raise ValueError(
            f"not an oracle report: artifact != {ORACLE_ARTIFACT!r}")
    checks = d.get("checks", {})
    metrics = d.get("metrics", {})
    problems: List[str] = []
    recomputed: Dict[str, Any] = {}

    for name, c in checks.items():
        src, default_op = _ORACLE_CHECK_SOURCES.get(name, (None, None))
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
        if c.get("ci") is not None and c.get("robust") != fresh["robust"]:
            problems.append(
                f"{name}: stored robust={c.get('robust')} but the CI "
                f"{c.get('ci')} implies {fresh['robust']}"
            )
        if "state" in c and c.get("state") != fresh["state"]:
            problems.append(
                f"{name}: stored state={c.get('state')!r} but the CI "
                f"{c.get('ci')} implies {fresh['state']!r}"
            )
        if src is not None:
            mv = metrics.get(src)
            if mv is not None and abs(mv - value) > 1e-9:
                problems.append(f"{name}: value {value} != metrics {src} {mv}")

    if recomputed:
        regrade = grade_checks(recomputed)
        stored = d.get("verdict", {}).get("grade")
        if regrade != stored:
            problems.append(f"grade: stored {stored!r}, recomputed {regrade!r}")
        stored_conf = metrics.get("confidence")
        if stored_conf is not None:
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
        problems.append("no recomputable checks on the report")

    _verify_oracle_probes(d, metrics, problems)

    return {"ok": not problems, "problems": problems, "recomputed_grade": regrade}


def classify_artifact_dict(d: Any) -> str:
    """Identify a loaded JSON object as a verifiable StressKit artifact.

    Returns ``"stability_card"``, ``"confirmatory_card"``,
    ``"oracle_report"``, or ``"unknown"``
    (badges, traces, raw dumps, and anything not produced by StressKit).
    """
    if not isinstance(d, dict):
        return "unknown"
    if d.get("artifact") == ORACLE_ARTIFACT:
        return "oracle_report"
    from .confirmatory import CONFIRMATORY_ARTIFACT
    if d.get("artifact") == CONFIRMATORY_ARTIFACT:
        return "confirmatory_card"
    if all(k in d for k in ("schema_version", "claim", "battery",
                            "metrics", "verdict")):
        return "stability_card"
    return "unknown"


def verify_artifact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Verify any StressKit artifact, dispatching on its kind.

    Returns the :func:`verify_card_dict` / :func:`verify_oracle_report_dict`
    result with an added ``"kind"`` key. Raises ``ValueError`` for objects
    that are not verifiable StressKit artifacts.
    """
    kind = classify_artifact_dict(d)
    if kind == "stability_card":
        result = verify_card_dict(d)
    elif kind == "oracle_report":
        result = verify_oracle_report_dict(d)
    elif kind == "confirmatory_card":
        from .confirmatory import verify_confirmatory_card_dict
        result = verify_confirmatory_card_dict(d)
    else:
        raise ValueError(
            "not a verifiable StressKit artifact (expected a stability card "
            "confirmatory card, or oracle reliability report)")
    result["kind"] = kind
    return result


def load_card(path: str) -> StabilityCard:
    return StabilityCard.load(path)
