"""Self-contained HTML render of a Stability Card or oracle report.

One file, no scripts, no external assets — safe to attach to a paper's
project page, drop into a repo, or serve from anywhere. Each check row
draws its 95% CI as a bar against the threshold tick, which makes the
"undecided" story visible at a glance: a band crossing the tick IS the
low-confidence verdict.

Deterministic given the card (no generation timestamps), so re-rendering
an unchanged card is byte-identical.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional

from .card import classify_artifact_dict

_GRADE_HEX = {"A": "#2da44e", "B": "#9a9f2c", "C": "#d4770c", "D": "#cf222e"}
_CONF_TEXT = {
    "high": "high confidence — every CI decides its check",
    "low": "LOW CONFIDENCE — at least one CI straddles its bar; "
           "the grade is provisional",
    "unknown": "confidence unknown — no CIs available",
}

_CSS = """
:root { color-scheme: light; }
body { font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
       sans-serif; color: #1f2328; background: #f6f8fa; margin: 0; }
.wrap { max-width: 860px; margin: 2.5rem auto; padding: 0 1rem; }
.card { background: #fff; border: 1px solid #d1d9e0; border-radius: 12px;
        padding: 2rem 2.25rem; box-shadow: 0 1px 3px rgba(31,35,40,.06); }
.head { display: flex; gap: 1.5rem; align-items: center; }
.grade { flex: 0 0 auto; width: 84px; height: 84px; border-radius: 14px;
         color: #fff; display: flex; align-items: center; justify-content:
         center; font-size: 46px; font-weight: 700; }
h1 { font-size: 1.35rem; margin: 0 0 .3rem; }
.sub { color: #59636e; font-size: .92rem; }
.conf { margin: 1rem 0 0; font-size: .9rem; padding: .5rem .8rem;
        border-radius: 8px; background: #f6f8fa; border: 1px solid #d1d9e0; }
.conf.low { background: #fff8f0; border-color: #d4770c; }
table { border-collapse: collapse; width: 100%; margin-top: 1.4rem; }
th { text-align: left; font-size: .8rem; text-transform: uppercase;
     letter-spacing: .04em; color: #59636e; padding: .45rem .6rem;
     border-bottom: 2px solid #d1d9e0; }
td { padding: .55rem .6rem; border-bottom: 1px solid #eaeef2;
     vertical-align: middle; }
.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.mark { font-size: 1.05rem; white-space: nowrap; }
.bar { position: relative; height: 14px; background: #eaeef2;
       border-radius: 7px; min-width: 160px; }
.ci { position: absolute; top: 3px; height: 8px; border-radius: 4px;
      background: #9ec5fe; }
.pt { position: absolute; top: 1px; width: 4px; height: 12px;
      border-radius: 2px; background: #0550ae; }
.thr { position: absolute; top: -3px; width: 2px; height: 20px;
       background: #cf222e; }
.legend { color: #59636e; font-size: .8rem; margin-top: .5rem; }
.notes { margin-top: 1.6rem; }
.notes h2 { font-size: 1rem; margin-bottom: .4rem; }
.notes li { margin: .3rem 0; font-size: .92rem; }
.foot { margin-top: 1.8rem; color: #59636e; font-size: .82rem; }
a { color: #0969da; text-decoration: none; }
"""


def _fmtv(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return html.escape(str(x))


def _ci_bar(value: Optional[float], ci: Optional[List[float]],
            threshold: Optional[float]) -> str:
    """The CI-vs-threshold bar: band, point, and the red threshold tick."""
    pts = [p for p in [value, threshold] + list(ci or []) if p is not None]
    if not pts or threshold is None or value is None:
        return ""
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    lo -= 0.15 * span
    hi += 0.15 * span

    def pct(x: float) -> float:
        return round(100 * (x - lo) / (hi - lo), 2)

    parts = ['<div class="bar">']
    if ci:
        left, width = pct(ci[0]), max(pct(ci[1]) - pct(ci[0]), 0.5)
        parts.append(f'<span class="ci" style="left:{left}%;'
                     f'width:{width}%"></span>')
    parts.append(f'<span class="pt" style="left:{pct(value)}%"></span>')
    parts.append(f'<span class="thr" style="left:{pct(threshold)}%"></span>')
    parts.append("</div>")
    return "".join(parts)


def _check_rows(checks: Dict[str, Any]) -> str:
    rows = []
    for name, c in checks.items():
        passed, robust = c.get("passed"), c.get("robust")
        if passed:
            mark = "⚠️ pass, undecided" if robust is False else "✅ pass"
        else:
            mark = "⚠️ fail, undecided" if robust is False else "❌ fail"
        ci = c.get("ci")
        ci_str = f"[{_fmtv(ci[0])}, {_fmtv(ci[1])}]" if ci else "—"
        op = html.escape({">=": "≥", "<=": "≤"}.get(c.get("op"), c.get("op") or ""))
        rows.append(
            "<tr>"
            f"<td>{html.escape(name.replace('_', ' '))}</td>"
            f'<td class="num">{_fmtv(c.get("value"))}</td>'
            f'<td class="num">{ci_str}</td>'
            f'<td class="num">{op} {_fmtv(c.get("threshold"))}</td>'
            f"<td>{_ci_bar(c.get('value'), ci, c.get('threshold'))}</td>"
            f'<td class="mark">{mark}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def card_html(d: Dict[str, Any]) -> str:
    """Render any StressKit artifact dict as a standalone HTML page."""
    kind = classify_artifact_dict(d)
    if kind == "stability_card":
        claim = d.get("claim", {})
        title = claim.get("statement") or "Stability Card"
        sub_bits = [claim.get("model"), claim.get("task"), claim.get("method"),
                    f"{d.get('battery', {}).get('n_runs_total')} runs"]
        checks = d.get("verdict", {}).get("checks", {})
        confidence = d.get("metrics", {}).get("pooled", {}).get("confidence")
        label = "Stability Card"
    elif kind == "oracle_report":
        title = d.get("oracle_name") or "Oracle Reliability Report"
        m = d.get("metrics", {})
        sub_bits = [f"{m.get('n_probes')} probes", f"{m.get('n_answers')} answers"]
        checks = d.get("checks", {})
        confidence = m.get("confidence")
        label = "Oracle Reliability Report"
    else:
        raise ValueError("not a renderable StressKit artifact")

    grade = d.get("verdict", {}).get("grade", "?")
    color = _GRADE_HEX.get(grade, "#59636e")
    sub = " · ".join(html.escape(str(b)) for b in sub_bits if b)
    conf_cls = ' low' if confidence == "low" else ""
    conf_html = (f'<p class="conf{conf_cls}">'
                 f'{html.escape(_CONF_TEXT.get(confidence, ""))}</p>'
                 if confidence in _CONF_TEXT else "")
    notes = d.get("notes") or []
    notes_html = ""
    if notes:
        items = "\n".join(f"<li>{html.escape(str(n))}</li>" for n in notes)
        notes_html = f'<div class="notes"><h2>Notes</h2><ul>{items}</ul></div>'
    version = html.escape(str(d.get("stresskit_version", "")))
    created = html.escape(str(d.get("created_at", "")))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — grade {html.escape(grade)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap"><div class="card">
<div class="head">
  <div class="grade" style="background:{color}">{html.escape(grade)}</div>
  <div>
    <h1>{html.escape(title)}</h1>
    <div class="sub">{label} · {sub}</div>
  </div>
</div>
{conf_html}
<table>
<thead><tr><th>check</th><th>value</th><th>95% CI</th><th>bar</th>
<th>CI vs bar</th><th>verdict</th></tr></thead>
<tbody>
{_check_rows(checks)}
</tbody>
</table>
<p class="legend">Blue band: 95% CI · blue tick: point estimate · red line:
the pass bar. A band crossing the red line means the check is undecided in
either direction.</p>
{notes_html}
<p class="foot">Generated by <a href="https://github.com/Arth-Singh/Stresskit">StressKit</a>
{version} · battery run {created} · verify this card:
<code>stresskit verify card.json</code></p>
</div></div>
</body>
</html>
"""


def render_html_path(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return card_html(json.load(f))
