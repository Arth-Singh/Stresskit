"""Figures for the lens-transport comparison findings (README embeds these).

Reads the comparison JSONs and stability-card JSONs produced by
run_lens_baselines_qwen.py and renders PNGs. Rerun after new models land.
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"jlens": "#0173B2", "logit": "#DE8F05", "tuned": "#029E73"}
LABELS = {"jlens": "Jacobian lens (released)", "logit": "logit lens (identity)",
          "tuned": "tuned lens (matched corpus)"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#cccccc",
                     "axes.linewidth": 0.8, "figure.facecolor": "white"})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def fig_hit5(comparison_path, out_png, title):
    comp = json.load(open(comparison_path))
    sets = list(comp["sets"])
    lenses = [ln for ln in ("jlens", "logit", "tuned")
              if ln in comp["sets"][sets[0]]["summary"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.24
    for i, ln in enumerate(lenses):
        xs = [j + (i - (len(lenses) - 1) / 2) * width for j in range(len(sets))]
        ys = [comp["sets"][s]["summary"][ln]["hit@5"] for s in sets]
        ax.bar(xs, ys, width * 0.92, color=COLORS[ln], label=LABELS[ln])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.008, f"{y:.2f}", ha="center", va="bottom",
                    fontsize=9, color="#333333")
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([s.replace("lens-eval-", "") for s in sets])
    ax.set_ylabel("hit@5 (all intermediates rank ≤ 5)")
    ax.set_ylim(0, max(0.45, ax.get_ylim()[1]))
    ax.set_title(title, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print("->", out_png)


def fig_checks(card_glob, out_png, title):
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, card_glob))):
        if "comparison" in os.path.basename(path):
            continue
        card = json.load(open(path))
        ln = card["claim"]["method"].split()[0]
        checks = card["verdict"]["checks"]
        rows.append((ln, checks["structural_stability"], checks["specificity"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    panels = [("structural stability (Jaccard of hit-sets)", 1, 0.8),
              ("specificity (real vs shuffled-target null)", 2, 1.5)]
    for ax, (name, idx, bar) in zip(axes, panels):
        for y, row in enumerate(rows):
            ln, *checks = row
            c = checks[idx - 1]
            lo, hi = c["ci"]
            ax.plot([lo, hi], [y, y], color=COLORS[ln], lw=2)
            ax.plot([c["value"]], [y], "o", color=COLORS[ln], ms=7)
            ax.text(c["value"], y - 0.32, f'{c["value"]:.2f}', ha="center",
                    va="top", fontsize=9, color="#333333")
        ax.axvline(bar, color="#999999", lw=1, ls="--")
        ax.text(bar, -0.75, f" bar {bar}", color="#777777", fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_ylim(-0.9, len(rows) - 0.3)
        ax.set_title(name, loc="left", fontsize=10, pad=14)
        style(ax)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_png, dpi=200)
    print("->", out_png)


def fig_scale(out_png):
    """jlens vs logit hit@5 (multihop) across model scale, if caches exist."""
    points = []
    for path in sorted(glob.glob(os.path.join(HERE, "lens_baseline_comparison_*.json"))):
        comp = json.load(open(path))
        s = comp["sets"].get("lens-eval-multihop", {}).get("summary", {})
        if "jlens" in s and "logit" in s:
            points.append((comp["model"].split("/")[-1], s))
    if len(points) < 2:
        print("scale figure skipped: fewer than 2 models")
        return
    order = {"Qwen3.5-0.8B": 0, "Qwen3.5-4B": 1, "Qwen3.5-27B": 2, "Qwen3.6-27B": 3}
    points.sort(key=lambda p: order.get(p[0], 99))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    xs = range(len(points))
    for ln in ("jlens", "logit"):
        ys = [p[1][ln]["hit@5"] for p in points]
        ax.plot(xs, ys, "-o", color=COLORS[ln], label=LABELS[ln], ms=7, lw=2)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, f"{y:.2f}", ha="center", fontsize=9, color="#333333")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([p[0] for p in points])
    ax.set_ylabel("hit@5, lens-eval-multihop")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.05 * (hi - lo) + 0.02)
    ax.set_title("Does the Jacobian transport's edge over the logit lens grow with scale?",
                 loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print("->", out_png)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig_hit5(os.path.join(HERE, "lens_baseline_comparison_qwen3p5_4b.json"),
             os.path.join(HERE, "figs", "hit5_qwen3p5_4b.png"),
             "Qwen3.5-4B — three linear transports, upstream hit criterion")
    fig_checks("lens_baseline_*_qwen3p5_4b.json",
               os.path.join(HERE, "figs", "checks_qwen3p5_4b.png"),
               "Same battery, same derangement null — every transport inherits the same failures")
    fig_scale(os.path.join(HERE, "figs", "hit5_scale.png"))
