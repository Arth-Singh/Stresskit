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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _checks_counts(checks: Dict[str, Any]) -> Tuple[int, int, int]:
    """(passed, total, undecided) over a verdict's checks."""
    passed = sum(1 for c in checks.values() if c.get("passed"))
    undecided = sum(1 for c in checks.values() if c.get("robust") is False)
    return passed, len(checks), undecided


def _checks_summary(checks: Dict[str, Any]) -> str:
    passed, total, undecided = _checks_counts(checks)
    s = f"{passed}/{total}"
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
        "checks_passed": _checks_counts(checks)[0],
        "checks_total": _checks_counts(checks)[1],
        "headline": (
            # pipes are escaped here, as _cell does for free-text columns
            (f"\\|cos\\|={_fmt(pooled.get('mean_pairwise_abs_cosine'))}, "
             if d.get("battery", {}).get("structure_kind") == "direction"
             else f"J={_fmt(pooled.get('mean_pairwise_jaccard'))}, ")
            + f"specificity {_fmt_ratio(spec)}"
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
        "checks_passed": _checks_counts(checks)[0],
        "checks_total": _checks_counts(checks)[1],
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


PAPERS_FILENAME = "papers.json"
_PAPER_KEYS = ("title", "models", "cards", "reproduced", "result", "audited")


def find_papers_registry(paths: Sequence[str]) -> Optional[str]:
    """The first ``papers.json`` sitting directly inside one of ``paths``."""
    for p in paths:
        if os.path.isdir(p):
            candidate = os.path.join(p, PAPERS_FILENAME)
            if os.path.exists(candidate):
                return candidate
    return None


def load_papers(registry_path: str) -> List[Dict[str, Any]]:
    """Load the hand-maintained paper registry behind the leaderboard.

    Each entry names one audited paper and the cards graded for it. Card
    paths are relative to the registry's directory and must exist: a
    registry that points at a missing card is an error, not a blank row.
    Entries keep their file order, which is the leaderboard order.
    """
    with open(registry_path, encoding="utf-8") as f:
        d = json.load(f)
    papers = d.get("papers") if isinstance(d, dict) else None
    if not isinstance(papers, list) or not papers:
        raise ValueError(
            f"{registry_path}: expected {{\"papers\": [...]}} with at least one entry")
    base = os.path.dirname(os.path.abspath(registry_path))
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(papers):
        missing = [k for k in _PAPER_KEYS if k not in p]
        if missing:
            raise ValueError(f"{registry_path}: paper #{i} lacks {missing}")
        if not isinstance(p["cards"], list) or not p["cards"]:
            raise ValueError(f"{registry_path}: paper {p['title']!r} lists no cards")
        cards = []
        for c in p["cards"]:
            full = os.path.normpath(os.path.join(base, c))
            if not os.path.exists(full):
                raise FileNotFoundError(
                    f"{registry_path}: paper {p['title']!r} lists missing card {c}")
            cards.append(full)
        out.append({**p, "cards": cards})
    return out


def _norm(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def paper_rows(papers: Sequence[Dict[str, Any]],
               rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Join registry entries with collected card rows, in registry order.

    Every graded card must belong to exactly one paper: a registered path
    that is not a graded artifact, or a graded artifact no entry claims,
    is an error, so the leaderboard can never silently drop a card.
    """
    by_path = {_norm(r["path"]): r for r in rows}
    claimed: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []
    for p in papers:
        card_rows = []
        for c in p["cards"]:
            key = _norm(c)
            if key in claimed:
                raise ValueError(
                    f"{c} is listed under both {claimed[key]!r} and {p['title']!r}")
            claimed[key] = p["title"]
            r = by_path.get(key)
            if r is None:
                raise ValueError(
                    f"paper {p['title']!r}: {c} is not a graded card or report")
            card_rows.append(r)
        out.append({
            **p,
            "rows": card_rows,
            "checks_passed": sum(r["checks_passed"] for r in card_rows),
            "checks_total": sum(r["checks_total"] for r in card_rows),
            "n_low_confidence": sum(1 for r in card_rows if r["confidence"] == "low"),
            "n_runs": sum(r["n"] or 0 for r in card_rows),
        })
    stray = sorted(r["path"] for r in rows if _norm(r["path"]) not in claimed)
    if stray:
        raise ValueError(
            "graded artifacts missing from the paper registry: "
            + ", ".join(stray) + f" — add them to {PAPERS_FILENAME}")
    return out


def registered_paper_rows(paths: Sequence[str], rows: Sequence[Dict[str, Any]],
                          papers_path: Optional[str] = None,
                          ) -> Optional[List[Dict[str, Any]]]:
    """Paper rows for ``paths`` when a registry exists (given or found)."""
    registry = papers_path or find_papers_registry(paths)
    if registry is None:
        return None
    return paper_rows(load_papers(registry), rows)


def arxiv_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def papers_markdown_lines(prows: Sequence[Dict[str, Any]],
                          relative_to: Optional[str] = None) -> List[str]:
    """The paper leaderboard table: one row per paper, one grade per card."""
    lines = [
        "| paper | models | grades (one per card) | checks passed | runs | "
        "reproduced the released number? | result | audited |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in prows:
        title = _cell(p["title"])
        if p.get("arxiv"):
            title = (f"[{title}]({arxiv_url(p['arxiv'])}) "
                     f"(arXiv:{_cell(p['arxiv'])})")
        grades = " ".join(
            f"[{_GRADE_EMOJI.get(r['grade'], '')} **{r['grade']}**]"
            f"({_link(r, relative_to)})" + ("†" if r["confidence"] == "low" else "")
            for r in p["rows"])
        lines.append(
            f"| {title} | {_cell(p['models'])} | {grades} "
            f"| {p['checks_passed']}/{p['checks_total']} | {p['n_runs']} "
            f"| {_cell(p['reproduced'])} | {_cell(p['result'])} "
            f"| {_cell(p['audited'])} |")
    return lines


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
                        relative_to: Optional[str] = None,
                        papers: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    """Render collected rows as the scoreboard markdown document.

    With ``papers`` (see :func:`paper_rows`) the document opens with the
    paper leaderboard, one row per audited paper, before the per-finding
    table.
    """
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
    ]
    if papers is not None:
        lines += [
            "## Papers",
            "",
            "One row per audited paper, in the order of "
            f"`references/{PAPERS_FILENAME}` (newest audit first); one grade per",
            "card, each linking to its card. † marks a low-confidence grade.",
            "",
            *papers_markdown_lines(papers, relative_to),
            "",
            "## All graded findings",
            "",
        ]
    lines += [
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


def write_scoreboard(paths: Sequence[str], output: str,
                     papers_path: Optional[str] = None) -> int:
    """Collect rows and write the scoreboard; returns the row count.

    A ``papers.json`` inside one of ``paths`` (or ``papers_path``) adds the
    paper leaderboard and makes an unregistered card an error.
    """
    rows = collect_rows(paths)
    out_dir = os.path.dirname(os.path.abspath(output))
    prows = registered_paper_rows(paths, rows, papers_path)
    md = scoreboard_markdown(rows, relative_to=out_dir, papers=prows)
    with open(output, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    return len(rows)
