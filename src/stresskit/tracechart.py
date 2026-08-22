"""The verdict-trace chart: how the grade distribution changes with run count.

Renders a ``verdict_trace`` result (see ``stresskit.verdict_trace``) as a
self-contained SVG — 100% stacked columns of grade shares per run count,
with the coin-flip region and ``settled_n`` annotated. This is the figure
that makes the run-count problem visible: a column split ~50/50 at n = 6
IS the claim that a 6-run stability report is a coin toss.

Design notes (deliberate, please keep):

- Grade colors are a CVD-validated 4-step scale (adjacent-pair ΔE ≥ 15.3
  under CVD simulation, ≥ 17.5 normal vision, on white). The B amber sits
  below 3:1 contrast on white, so shares are always also written as text
  (direct labels + native ``<title>`` tooltips) — color never carries the
  value alone.
- Deterministic: no timestamps, no randomness — same trace, same bytes.
- Stdlib only; the SVG embeds anywhere (GitHub README, the site, a paper).
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional

GRADE_COLORS = {"A": "#008300", "B": "#eda100", "C": "#e34948", "D": "#9a2028"}
_LABEL_INK = {"A": "#ffffff", "B": "#3b2a00", "C": "#ffffff", "D": "#ffffff"}
_INK = "#1f2328"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_SURFACE = "#ffffff"

_FONT = ('font-family="system-ui,-apple-system,\'Segoe UI\',sans-serif"')


def _fmt_pct(x: float) -> str:
    return f"{round(x * 100)}%"


def trace_svg(trace: Dict[str, Any], *, title: Optional[str] = None,
              width: int = 760) -> str:
    """Render a verdict trace dict as a standalone SVG string."""
    sizes: List[int] = list(trace["sizes"])
    per_size = trace["per_size"]
    settled_n = trace.get("settled_n")
    n_total = trace.get("n_total")
    n_sub = next(iter(per_size.values()), {}).get("n_subsamples")

    # the most coin-flip column: lowest modal-grade share
    flip_size, flip_share = None, 1.0
    for s in sizes:
        share = per_size[str(s)].get("modal_grade_share", 1.0)
        if share < flip_share:
            flip_size, flip_share = s, share

    top, bottom, left, right = 96, 76, 56, 16
    plot_h = 240
    height = top + plot_h + bottom
    plot_w = width - left - right
    n_cols = len(sizes)
    slot = plot_w / n_cols
    col_w = min(48, slot * 0.62)

    e = html.escape
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Grade distribution vs run count">',
        f'<rect width="{width}" height="{height}" fill="{_SURFACE}"/>',
    ]

    chart_title = title or "Which grade would this battery report at n runs?"
    parts.append(
        f'<text x="{left}" y="26" {_FONT} font-size="15" font-weight="600" '
        f'fill="{_INK}">{e(chart_title)}</text>')
    sub = (f"{n_sub} random size-n subsets of the {n_total} runs, each "
           f"regraded with the full analysis")
    parts.append(
        f'<text x="{left}" y="44" {_FONT} font-size="11.5" '
        f'fill="{_MUTED}">{e(sub)}</text>')

    # legend: its own band between the title block and the plot, right-aligned
    lx = width - right - 4 * 40
    for g in "ABCD":
        parts.append(
            f'<rect x="{lx}" y="{top - 30}" width="10" height="10" rx="2" '
            f'fill="{GRADE_COLORS[g]}"/>')
        parts.append(
            f'<text x="{lx + 14}" y="{top - 21}" {_FONT} font-size="11.5" '
            f'fill="{_INK}">{g}</text>')
        lx += 40

    # y gridlines at 0 / 50 / 100%
    for frac, lab in ((0.0, "0%"), (0.5, "50%"), (1.0, "100%")):
        y = top + plot_h * (1 - frac)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" '
            f'y2="{y:.1f}" stroke="{_GRID}" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" {_FONT} font-size="11" '
            f'fill="{_MUTED}" text-anchor="end">{lab}</text>')

    # columns
    for idx, s in enumerate(sizes):
        entry = per_size[str(s)]
        dist = entry.get("grade_dist", {})
        x = left + slot * idx + (slot - col_w) / 2
        y_cursor = top
        for g in "ABCD":
            share = float(dist.get(g, 0.0))
            if share <= 0:
                continue
            seg_h = plot_h * share
            gap = 1 if y_cursor > top else 0  # 2px surface gap between fills
            tip = (f"n={s}: grade {g} in {_fmt_pct(share)} of "
                   f"{entry.get('n_subsamples')} subsets")
            parts.append(
                f'<rect x="{x:.1f}" y="{y_cursor + gap:.1f}" '
                f'width="{col_w:.1f}" height="{max(seg_h - gap, 0.5):.1f}" '
                f'rx="2" fill="{GRADE_COLORS[g]}"><title>{e(tip)}</title></rect>')
            if seg_h >= 15:  # direct label — the relief for the amber WARN
                parts.append(
                    f'<text x="{x + col_w / 2:.1f}" '
                    f'y="{y_cursor + seg_h / 2 + 4:.1f}" {_FONT} '
                    f'font-size="10.5" fill="{_LABEL_INK[g]}" '
                    f'text-anchor="middle">{_fmt_pct(share)}</text>')
            y_cursor += seg_h
        # x tick label; the settled column is emphasized
        is_settled = settled_n is not None and s == settled_n
        weight = ' font-weight="700"' if is_settled else ""
        fill = _INK if is_settled else _MUTED
        parts.append(
            f'<text x="{x + col_w / 2:.1f}" y="{top + plot_h + 18}" {_FONT} '
            f'font-size="11.5" fill="{fill}"{weight} '
            f'text-anchor="middle">{s}</text>')

    # baseline + x-axis title
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" '
        f'y2="{top + plot_h}" stroke="#c3c2b7" stroke-width="1"/>')
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{top + plot_h + 34}" {_FONT} '
        f'font-size="11.5" fill="{_MUTED}" text-anchor="middle">'
        "runs included (n)</text>")

    # annotations: the coin-flip column and where the verdict settles
    notes = []
    if flip_size is not None and flip_share < 0.67:
        dist = per_size[str(flip_size)].get("grade_dist", {})
        split = " / ".join(
            f"{g} {_fmt_pct(sh)}" for g, sh in
            sorted(dist.items(), key=lambda kv: -kv[1]) if sh >= 0.05)
        notes.append(f"⚠ at n = {flip_size} the verdict is a coin flip: {split}")
    if settled_n is not None:
        notes.append(f"verdict settles at n = {settled_n} "
                     f"(grade {trace.get('full_grade')})")
    elif trace.get("full_grade"):
        notes.append(
            f"never settles: grade {trace.get('full_grade')} at all "
            f"{n_total} runs, still contested below that")
    if notes:
        parts.append(
            f'<text x="{left}" y="{height - 14}" {_FONT} font-size="11.5" '
            f'fill="{_INK}">{e("   ·   ".join(notes))}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def trace_svg_path(path: str, **kwargs: Any) -> str:
    with open(path, encoding="utf-8") as f:
        return trace_svg(json.load(f), **kwargs)
