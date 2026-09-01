"""Legacy diagnostic stability inventory — every v0.x card in one table.

Scans directories of StressKit artifacts (stability cards and oracle
reliability reports), extracts each verdict, and renders a deterministic
markdown scoreboard. The repo's own SCOREBOARD.md is generated from
``references/`` by ``stresskit scoreboard`` and kept fresh by CI, so the
table can never drift from the cards it summarizes.

This compatibility view is not StressKit v1 publication evidence. Determinism
matters: no timestamps are emitted and rows sort on stable identity keys, never
grade, so regenerating from unchanged cards is byte-identical.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from .card import _GRADE_EMOJI, classify_artifact_dict

_CONF_LABEL = {"high": "high", "low": "**low**", "unknown": "unknown"}


def _cell(text: str) -> str:
    """Make free text safe inside a markdown table cell.

    Card fields are arbitrary strings (externally submitted cards
    included), so markdown-active characters are neutralized too — a model
    name must never smuggle a link, image, or raw HTML into the rendered
    scoreboard.
    """
    s = str(text).replace("\n", " ")
    for ch in "\\|[]<>`*_":
        s = s.replace(ch, "\\" + ch)
    return s


def _fmt_ratio(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.2f}×"


def _fmt(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.2f}"


def _checks_summary(checks: Dict[str, Any]) -> str:
    passed = sum(1 for c in checks.values() if c.get("passed"))
    undecided = sum(1 for c in checks.values() if c.get("robust") is False)
    s = f"{passed}/{len(checks)}"
    if undecided:
        s += f" ({undecided} undecided)"
    return s


def _row_from_stability_card(d: Dict[str, Any], path: str) -> Dict[str, Any]:
    claim = d.get("claim", {})
    pooled = d.get("metrics", {}).get("pooled", {})
    checks = d.get("verdict", {}).get("checks", {})
    spec = checks.get("specificity", {}).get("value")
    return {
        "path": path,
        "kind": "stability card",
        "finding": " / ".join(
            str(claim.get(k)) for k in ("task", "model") if claim.get(k)
        ) or os.path.basename(path),
        "method": claim.get("method") or "—",
        "grade": d["verdict"]["grade"],
        "confidence": pooled.get("confidence") or "unknown",
        "n": d.get("battery", {}).get("n_runs_total"),
        "checks": _checks_summary(checks),
        "headline": (
            f"J={_fmt(pooled.get('mean_pairwise_jaccard'))}, "
            f"specificity {_fmt_ratio(spec)}"
        ),
    }


def _row_from_oracle_report(d: Dict[str, Any], path: str) -> Dict[str, Any]:
    metrics = d.get("metrics", {})
    checks = d.get("checks", {})
    return {
        "path": path,
        "kind": "oracle report",
        "finding": d.get("oracle_name") or os.path.basename(path),
        "method": "activation reader",
        "grade": d["verdict"]["grade"],
        "confidence": metrics.get("confidence") or "unknown",
        "n": metrics.get("n_answers"),
        "checks": _checks_summary(checks),
        "headline": (
            f"accuracy {_fmt(metrics.get('known_accuracy'))}, "
            f"null hallucination {_fmt(metrics.get('null_hallucination_rate'))}"
        ),
    }


def collect_rows(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Collect scoreboard rows from files and (recursively) directories.

    Non-artifact JSONs are skipped silently — badges, traces, and raw dumps
    legitimately live next to cards.
    """
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(
                glob.glob(os.path.join(p, "**", "*.json"), recursive=True)))
        else:
            files.append(p)

    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        kind = classify_artifact_dict(d)
        if kind == "stability_card":
            rows.append(_row_from_stability_card(d, fp))
        elif kind == "oracle_report":
            rows.append(_row_from_oracle_report(d, fp))
    rows.sort(key=lambda r: (
        r["kind"] != "stability card", r["finding"], r["path"]
    ))
    return rows


def _link(row: Dict[str, Any], relative_to: Optional[str]) -> str:
    """Link a row to its markdown render when one exists beside the JSON."""
    target = row["path"]
    md = os.path.splitext(target)[0] + ".md"
    if os.path.exists(md):
        target = md
    if relative_to:
        target = os.path.relpath(target, relative_to)
    return target.replace(os.sep, "/")


def scoreboard_markdown(rows: Sequence[Dict[str, Any]],
                        relative_to: Optional[str] = None) -> str:
    """Render collected rows as the scoreboard markdown document."""
    lines = [
        "# Stability Scoreboard",
        "",
        "> **Legacy diagnostic inventory.** Not a StressKit v1 evidence board;",
        "> use `stresskit audit publish` for verified claim-level publication.",
        "",
        "Every finding graded by StressKit's reference batteries, under the",
        "default thresholds and the protocol in",
        "[`references/PROTOCOL.md`](references/PROTOCOL.md). Each row links",
        "to the full card; every card re-derives from its own recorded",
        "metrics via `stresskit verify` (CI enforces this on every push).",
        "",
        "A grade is a reliability measurement under pre-registered checks —",
        "**not** a judgment of a paper's value, and never a claim of",
        "misconduct. Undecided checks (95% CI straddling its bar) lower a",
        "verdict's confidence; a low-confidence grade is provisional.",
        "",
        "| finding | method | grade | confidence | checks passed | runs/answers | headline | card |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        emoji = _GRADE_EMOJI.get(r["grade"], "")
        finding, method = (_cell(r["finding"]), _cell(r["method"]))
        lines.append(
            f"| {finding} | {method} | {emoji} **{r['grade']}** "
            f"| {_CONF_LABEL.get(r['confidence'], r['confidence'])} "
            f"| {r['checks']} | {r['n'] if r['n'] is not None else '—'} "
            f"| {r['headline']} | [{r['kind']}]({_link(r, relative_to)}) |"
        )
    lines += [
        "",
        "**Grades**: A — all applicable checks pass · B — at least half ·",
        "C — at least one · D — none, or indistinguishable from random.",
        "",
        "Want a finding on this board? See",
        "[CONTRIBUTING.md](CONTRIBUTING.md) — submissions arrive as PRs",
        "carrying the card JSON, the runner script, and a `stresskit verify`",
        "pass.",
        "",
        "*Generated by `stresskit scoreboard` — do not edit by hand.*",
    ]
    return "\n".join(lines)


def write_scoreboard(paths: Sequence[str], output: str) -> int:
    """Collect rows and write the scoreboard; returns the row count."""
    rows = collect_rows(paths)
    out_dir = os.path.dirname(os.path.abspath(output))
    md = scoreboard_markdown(rows, relative_to=out_dir)
    with open(output, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    return len(rows)
