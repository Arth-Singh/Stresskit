"""``stresskit site`` — a static site from a directory of reference cards.

Builds the shareable face of a card collection: an index with the headline
numbers, the most dramatic verdict-trace chart as the hero figure, and the
full scoreboard; plus one page per card (the HTML render with its trace
chart embedded, when a ``<stem>.trace.json`` sits next to the card JSON).

Everything is static, self-contained, and deterministic — built for
GitHub Pages (see ``.github/workflows/pages.yml``) but hostable anywhere.
"""

from __future__ import annotations

import html
import json
import os
import shutil
from typing import Any, Dict, List, Optional

from .card import _GRADE_EMOJI
from .htmlcard import _GRADE_HEX, card_html
from .scoreboard import arxiv_url, collect_rows, registered_paper_rows
from .tracechart import trace_svg

_INDEX_CSS = """
body { font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
       sans-serif; color: #1f2328; background: #f6f8fa; margin: 0; }
.wrap { max-width: 920px; margin: 2.5rem auto; padding: 0 1rem; }
.panel { background: #fff; border: 1px solid #d1d9e0; border-radius: 12px;
         padding: 1.8rem 2rem; box-shadow: 0 1px 3px rgba(31,35,40,.06);
         margin-bottom: 1.4rem; }
h1 { font-size: 1.7rem; margin: 0 0 .3rem; }
.tag { color: #59636e; font-size: 1.02rem; margin: 0 0 1rem; }
.kpis { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.2rem 0 .2rem; }
.kpi { flex: 1 1 150px; border: 1px solid #d1d9e0; border-radius: 10px;
       padding: .8rem 1rem; background: #f6f8fa; }
.kpi b { display: block; font-size: 1.6rem; }
.kpi span { color: #59636e; font-size: .85rem; }
.hero svg { max-width: 100%; height: auto; }
table { border-collapse: collapse; width: 100%; }
th { text-align: left; font-size: .78rem; text-transform: uppercase;
     letter-spacing: .04em; color: #59636e; padding: .45rem .55rem;
     border-bottom: 2px solid #d1d9e0; }
td { padding: .55rem .55rem; border-bottom: 1px solid #eaeef2;
     font-size: .92rem; }
.g { font-weight: 700; padding: .1rem .55rem; border-radius: 6px;
     color: #fff; display: inline-block; }
.low { font-size: .78rem; color: #d4770c; }
.badges { white-space: nowrap; }
.badges a { text-decoration: none; }
.note { color: #59636e; font-size: .88rem; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: #f6f8fa; border: 1px solid #d1d9e0; border-radius: 6px;
       padding: .1rem .35rem; font-size: .85em; }
"""


def _slug(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _trace_for(card_path: str) -> Optional[Dict[str, Any]]:
    trace_path = os.path.splitext(card_path)[0] + ".trace.json"
    if os.path.exists(trace_path):
        return _load(trace_path)
    return None


def _hero_trace(rows: List[Dict[str, Any]]):
    """The most dramatic run-count story: the trace with the largest
    settled_n (ties broken by path, for determinism)."""
    best = None
    for r in rows:
        trace = _trace_for(r["path"])
        if trace and trace.get("settled_n"):
            key = (trace["settled_n"], r["path"])
            if best is None or key > best[0]:
                best = (key, r, trace)
    return (best[1], best[2]) if best else (None, None)


def _papers_panel(prows, e) -> str:
    """The paper leaderboard: one row per audited paper, one badge per card."""
    body = []
    for p in prows:
        title = e(p["title"])
        if p.get("arxiv"):
            title = (f'<a href="{e(arxiv_url(p["arxiv"]))}">{title}</a> '
                     f'<span class="note">arXiv:{e(p["arxiv"])}</span>')
        badges = " ".join(
            f'<a href="{e(_slug(r["path"]))}.html" title="{e(r["finding"])}">'
            f'<span class="g" style="background:{_GRADE_HEX.get(r["grade"], "#59636e")}">'
            f'{e(r["grade"])}</span></a>' + ("<sup>†</sup>" if r["confidence"] == "low" else "")
            for r in p["rows"])
        body.append(
            "<tr>"
            f"<td>{title}</td>"
            f'<td>{e(p["models"])}</td>'
            f'<td class="badges">{badges}</td>'
            f'<td>{p["checks_passed"]}/{p["checks_total"]}</td>'
            f'<td>{e(p["reproduced"])}</td>'
            f'<td>{e(p["result"])}</td>'
            f'<td>{e(p["audited"])}</td>'
            "</tr>")
    return (
        '<div class="panel">'
        '<h2 style="font-size:1.1rem;margin-top:0">Papers</h2>'
        '<p class="note">One row per audited paper, newest first; one grade per '
        'card, each linking to its card. † marks a low-confidence grade (a '
        "check's 95% CI straddles its bar).</p>"
        "<table><thead><tr><th>paper</th><th>models</th><th>grades</th>"
        "<th>checks passed</th><th>reproduced the released number?</th>"
        "<th>result</th><th>audited</th></tr></thead>"
        f"<tbody>{chr(10).join(body)}</tbody></table></div>")


def build_site(paths: List[str], outdir: str,
               repo_url: str = "https://github.com/Arth-Singh/Stresskit",
               papers_path: Optional[str] = None) -> int:
    """Build the static site; returns the number of card pages written.

    A ``papers.json`` inside one of ``paths`` (or ``papers_path``) adds the
    paper leaderboard panel to the index and makes an unregistered card an
    error.
    """
    rows = collect_rows(paths)
    if not rows:
        raise ValueError("no cards or reports found under the given paths")
    prows = registered_paper_rows(paths, rows, papers_path)
    os.makedirs(outdir, exist_ok=True)
    e = html.escape

    # ---- per-card pages (and copy each card JSON next to its page) --------
    table_rows = []
    for r in rows:
        d = _load(r["path"])
        slug = _slug(r["path"])
        json_name = f"{slug}.json"
        shutil.copyfile(r["path"], os.path.join(outdir, json_name))
        extras = []
        trace = _trace_for(r["path"])
        if trace:
            extras.append(
                '<div class="trace"><h2>Verdict-stability trace</h2>'
                + trace_svg(trace) + "</div>")
        extras.append(
            f'<div class="trace"><h2>Audit this card</h2>'
            f'<p class="legend">Every number above re-derives from the '
            f'<a href="{e(json_name)}">card JSON</a>: '
            f"<code>pip install stress-kit && stresskit verify "
            f"{e(json_name)}</code></p></div>")
        nav = '<p class="nav"><a href="index.html">← all graded findings</a></p>'
        page = card_html(d, extra_sections=extras, nav_html=nav)
        with open(os.path.join(outdir, f"{slug}.html"), "w",
                  encoding="utf-8") as f:
            f.write(page)

        color = _GRADE_HEX.get(r["grade"], "#59636e")
        conf = ('<div class="low">low confidence — provisional</div>'
                if r["confidence"] == "low" else "")
        table_rows.append(
            "<tr>"
            f'<td><a href="{e(slug)}.html">{e(r["finding"])}</a></td>'
            f'<td>{e(r["method"])}</td>'
            f'<td><span class="g" style="background:{color}">'
            f'{e(r["grade"])}</span>{conf}</td>'
            f'<td>{e(r["checks"])}</td>'
            f'<td>{e(r["headline"])}</td>'
            "</tr>")

    # ---- index -------------------------------------------------------------
    n_cards = len(rows)
    n_certified = sum(1 for r in rows
                      if r["grade"] == "A" and r["confidence"] == "high")
    n_undecided = sum(1 for r in rows if r["confidence"] == "low")
    n_failing = sum(1 for r in rows if r["grade"] in {"C", "D"})
    papers_kpi = (f'<div class="kpi"><b>{len(prows)}</b><span>papers audited</span></div>'
                  if prows is not None else "")
    papers_html = _papers_panel(prows, e) if prows is not None else ""

    hero_row, hero = _hero_trace(rows)
    hero_html = ""
    if hero is not None:
        hero_html = (
            '<div class="panel hero">'
            + trace_svg(hero, title=f"{hero_row['finding']}: which grade "
                                    "would your paper report?")
            + f'<p class="note">Grade distribution over random size-n subsets '
              f'of the full battery. Full card: '
              f'<a href="{e(_slug(hero_row["path"]))}.html">'
              f'{e(hero_row["finding"])}</a>.</p></div>')

    index = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StressKit — graded interpretability findings</title>
<style>{_INDEX_CSS}</style>
</head>
<body>
<div class="wrap">
<div class="panel">
<h1>Do interpretability findings survive re-running?</h1>
<p class="tag">Published findings and instruments, stress-tested under a
pre-registered battery — seeds, resampling, prompt templates, hyperparameters,
and null controls — and graded A–D.
<a href="{e(repo_url)}">StressKit</a> is the harness; every verdict below
re-derives from its own card via <code>stresskit verify</code>.</p>
<div class="kpis">
{papers_kpi}
<div class="kpi"><b>{n_cards}</b><span>findings graded</span></div>
<div class="kpi"><b>{n_certified}</b><span>certified A (every CI clears its bar)</span></div>
<div class="kpi"><b>{n_undecided}</b><span>statistically undecided at full battery</span></div>
<div class="kpi"><b>{n_failing}</b><span>graded C/D</span></div>
</div>
</div>
{papers_html}
{hero_html}
<div class="panel">
<h2 style="font-size:1.1rem;margin-top:0">All graded findings</h2>
<table>
<thead><tr><th>finding</th><th>method</th><th>grade</th>
<th>checks passed</th><th>headline</th></tr></thead>
<tbody>
{chr(10).join(table_rows)}
</tbody>
</table>
<p class="note" style="margin-top:1rem">
A grade is a reliability measurement under stated thresholds — not a judgment
of a paper's value, never a claim of misconduct
(<a href="{e(repo_url)}/blob/main/references/PROTOCOL.md">the evidence
standard</a>). Grades: {_GRADE_EMOJI['A']} A all checks pass ·
{_GRADE_EMOJI['B']} B at least half · {_GRADE_EMOJI['C']} C at least one ·
{_GRADE_EMOJI['D']} D none, or indistinguishable from random.
Want a finding on this board? <a
href="{e(repo_url)}/blob/main/CONTRIBUTING.md">Submit a card</a>.</p>
</div>
</div>
</body>
</html>
"""
    with open(os.path.join(outdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(index)
    return n_cards
