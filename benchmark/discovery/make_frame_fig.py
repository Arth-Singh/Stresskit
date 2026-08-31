#!/usr/bin/env python3
"""Render the August-2026 code-availability census as a figure.

One measure (paper counts) over five mutually exclusive statuses, so the chart
is a single-series bar with direct value labels and no legend.

    python benchmark/discovery/make_frame_fig.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FRAME = HERE / "august-2026-frame.json"
FIGS = HERE / "figs"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
BAR = "#0173B2"
BAR_EMPHASIS = "#029E73"

LABELS = {
    "public_repo": "Public repository\nreleased by the authors",
    "dependency_links_only": "Links only to\ndependency repositories",
    "no_repo_hf_links_only": "HuggingFace references,\nno repository",
    "code_promised_not_released": "Code promised,\nnot released",
    "no_public_code_found": "No public code found",
}
ORDER = [
    "public_repo",
    "no_public_code_found",
    "dependency_links_only",
    "no_repo_hf_links_only",
    "code_promised_not_released",
]


def main() -> int:
    frame = json.loads(FRAME.read_text())
    counts = frame["counts"]
    by_status = counts["by_code_status"]
    total = counts["tier_a"]
    licensed = counts["public_repo_with_spdx_license"]

    labels = [LABELS[k] for k in ORDER]
    values = [by_status.get(k, 0) for k in ORDER]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ypos = range(len(ORDER))
    ax.barh(list(ypos), values, height=0.62, color=BAR, zorder=3)

    # the registry-eligible slice: released and carrying a license
    ax.barh([0], [licensed], height=0.62, color=BAR_EMPHASIS, zorder=4)

    for y, value in zip(ypos, values):
        ax.text(value + total * 0.012, y, f"{value}  ({value / total:.0%})",
                va="center", ha="left", fontsize=10, color=INK, zorder=5)
    ax.text(licensed / 2, 0, f"{licensed} licensed", va="center", ha="center",
            fontsize=9.5, color="white", weight="bold", zorder=6)

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, total * 0.62)
    ax.set_xlabel("papers", fontsize=10, color=MUTED)
    ax.tick_params(axis="x", labelsize=9, colors=MUTED)
    ax.grid(axis="x", color="#e4e4e4", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d0d0d0")

    ax.set_title(
        f"Released code among {total} August 2026 mechanistic-interpretability papers",
        fontsize=12.5, color=INK, pad=14, loc="left")
    fig.text(0.005, 0.005,
             "arXiv submissions 2026-08-01 to 2026-08-31, narrow-term stratum. "
             "Green marks the released repositories that carry a license.",
             fontsize=8.5, color=MUTED)

    FIGS.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out = FIGS / "august_2026_code_availability.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
