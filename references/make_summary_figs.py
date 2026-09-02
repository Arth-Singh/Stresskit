"""Summary figures over every graded reference card.

Reads the cards, their verdict traces and the paper registry under
``references/`` and writes three figures to ``references/figs/``:

- ``checks_by_card.png``: for each of the five diagnostic checks, how many of
  the graded cards pass, are undecided (95% CI straddles the bar) or fail.
- ``verdict_settle_n.png``: the run count at which each card's verdict
  settles (modal grade of 30 subsamples matches the full-sample grade with
  at least 90% agreement), against the seed counts papers typically report.
- ``threshold_sensitivity.png``: pass / undecided / fail counts for the
  structural-stability and specificity checks when their bars are moved,
  recomputed from the values and intervals recorded on the cards.

No model is run; everything is derived from the stored artifacts, so the
figures cannot disagree with the cards.

Usage:
    PYTHONPATH=src python references/make_summary_figs.py [--out references/figs]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stresskit.scoreboard import collect_rows, registered_paper_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATUS = {"pass": "#0ca30c", "incon": "#fab219", "fail": "#d03b3b"}
LABEL = {"pass": "pass", "incon": "undecided", "fail": "fail"}
CHECK_LABEL = {
    "structural_stability": "structural stability\n(which components, J ≥ 0.8)",
    "claim_stability": "claim stability\n(the sentence, π* ≥ 0.8)",
    "score_stability": "score stability\n(the number, CV ≤ 0.25)",
    "beats_random": "beats random\n(≥ 3× size-matched null)",
    "specificity": "specificity\n(≥ 1.5× a null that cannot be real)",
}
CHECK_ORDER = list(CHECK_LABEL)
INK = "#222222"
MUTED = "#6b6b6b"


def state_of(value, threshold, op, ci):
    """The harness's rule (battery.py): a check is decided only when its
    whole 95% interval sits on one side of the bar."""
    passed = value >= threshold if op == ">=" else value <= threshold
    lo, hi = ci
    if op == ">=":
        robust = lo >= threshold if passed else hi < threshold
    else:
        robust = hi <= threshold if passed else lo > threshold
    if not robust:
        return "incon"
    return "pass" if passed else "fail"


def recorded_state(check):
    """State as the card records it; schema 0.2 cards carry ``passed`` and
    ``robust`` but no ``state`` field."""
    state = check.get("state")
    if state == "inconclusive":
        return "incon"
    if state in ("pass", "fail"):
        return state
    if check.get("robust") is False:
        return "incon"
    return "pass" if check.get("passed") else "fail"


def load_cards(references_dir):
    rows = collect_rows([references_dir])
    papers = registered_paper_rows([references_dir], rows)
    paper_of = {}
    for paper in papers:
        for row in paper["rows"]:
            paper_of[os.path.normpath(row["path"])] = paper["title"]
    cards = []
    for row in rows:
        with open(row["path"]) as handle:
            card = json.load(handle)
        checks = card.get("verdict", {}).get("checks", {})
        if not isinstance(checks, dict):
            checks = {}
        trace_path = row["path"][: -len(".json")] + ".trace.json"
        settled = None
        if os.path.exists(trace_path):
            with open(trace_path) as handle:
                settled = json.load(handle).get("settled_n")
        cards.append(
            {
                "name": os.path.basename(row["path"])[: -len(".json")],
                "paper": paper_of.get(os.path.normpath(row["path"]), "?"),
                "grade": row["grade"],
                "confidence": row["confidence"],
                "checks": checks,
                "settled_n": settled,
                "n_total": card.get("battery", {}).get("n_runs_total"),
            }
        )
    return cards


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cfcfcf")
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def fig_checks(cards, out):
    counts = {name: Counter() for name in CHECK_ORDER}
    for card in cards:
        for name in CHECK_ORDER:
            check = card["checks"].get(name)
            if check is None:
                continue
            counts[name][recorded_state(check)] += 1
    print("checks_by_card:", {n: dict(c) for n, c in counts.items()})
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    y = list(range(len(CHECK_ORDER)))[::-1]
    for idx, name in zip(y, CHECK_ORDER):
        left = 0
        total = sum(counts[name].values())
        for state in ("pass", "incon", "fail"):
            n = counts[name][state]
            if n == 0:
                continue
            ax.barh(idx, n, left=left, color=STATUS[state], height=0.62, edgecolor="white", linewidth=2)
            if n >= 3:
                ax.text(left + n / 2, idx, f"{LABEL[state]} {n}", ha="center", va="center", fontsize=8.5, color="white" if state != "incon" else INK)
            else:
                ax.text(left + n / 2, idx + 0.42, f"{LABEL[state]} {n}", ha="center", va="bottom", fontsize=7.5, color=INK)
            left += n
        ax.text(total + 0.6, idx, f"n = {total} cards", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([CHECK_LABEL[n] for n in CHECK_ORDER], fontsize=9)
    ax.set_xlim(0, 50)
    n_papers = len({card["paper"] for card in cards})
    ax.set_xlabel(f"graded reference cards ({len(cards)} cards, {n_papers} papers)")
    ax.set_title(f"Which checks the {len(cards)} graded cards pass (undecided: the 95% CI straddles the bar)", fontsize=10, loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_settle(cards, out):
    rows = [c for c in cards if c["settled_n"] is not None]
    rows.sort(key=lambda c: (c["settled_n"], c["name"]))
    fig, ax = plt.subplots(figsize=(8.4, 0.28 * len(rows) + 1.9))
    y = list(range(len(rows)))[::-1]
    for idx, card in zip(y, rows):
        n = card["settled_n"]
        colour = STATUS["pass"] if card["confidence"] == "high" else STATUS["incon"]
        ax.plot([0, n], [idx, idx], color="#dddddd", linewidth=1.2, zorder=1)
        ax.scatter([n], [idx], s=34, color=colour, zorder=2, edgecolor="white", linewidth=0.8)
        tag = "" if n < card["n_total"] else " (not settled by the full battery)"
        ax.text(n + 1.2, idx, f"{n}{tag}", va="center", fontsize=8, color=INK)
    ax.axvspan(3, 5, color="#efefef", zorder=0)
    ax.text(6.5, -0.9, "shaded: 3–5 seeds, the run count papers typically report", ha="left", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([c["name"].replace("_", " ") for c in rows], fontsize=7.5)
    ax.set_xlabel("runs before the modal grade of 30 subsamples\nmatches the full battery at least 90% of the time")
    ax.set_xlim(0, max(c["n_total"] for c in rows) + 75)
    ax.set_ylim(-1.6, len(rows) - 0.4)
    ax.set_title("Runs needed before a verdict settles\n(green: high-confidence card; amber: a CI still straddles a bar)", fontsize=9.5, loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_thresholds(cards, out):
    grids = {
        "structural_stability": [0.6, 0.7, 0.8, 0.9],
        "specificity": [1.2, 1.5, 2.0, 3.0],
    }
    titles = {
        "structural_stability": "structural stability: Jaccard bar moved (default 0.8)",
        "specificity": "specificity: real-vs-null ratio bar moved (default 1.5×)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, (name, bars) in zip(axes, grids.items()):
        with_ci = [c for c in cards if c["checks"].get(name) and c["checks"][name].get("ci")]
        for pos, bar in enumerate(bars):
            counts = Counter()
            for card in with_ci:
                check = card["checks"][name]
                counts[state_of(check["value"], bar, check["op"], check["ci"])] += 1
            print(f"threshold_sensitivity {name} bar {bar}: {dict(counts)} (n={len(with_ci)} cards with a CI)")
            bottom = 0
            for state in ("pass", "incon", "fail"):
                n = counts[state]
                if n == 0:
                    continue
                ax.bar(pos, n, bottom=bottom, color=STATUS[state], width=0.62, edgecolor="white", linewidth=2)
                ax.text(pos, bottom + n / 2, f"{LABEL[state]}\n{n}", ha="center", va="center", fontsize=7.5, color="white" if state != "incon" else INK)
                bottom += n
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([("%g" % b) + ("  (default)" if b in (0.8, 1.5) else "") for b in bars], fontsize=8)
        ax.set_title(titles[name] + f"\n{len(with_ci)} cards with a recorded interval", fontsize=9, loc="left", color=INK)
        ax.set_ylabel("cards")
        style(ax)
    fig.suptitle("Do the counts depend on where the bars sit? Recomputed from each card's recorded value and 95% interval", fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--references", default=HERE)
    parser.add_argument("--out", default=os.path.join(HERE, "figs"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cards = load_cards(args.references)
    fig_checks(cards, os.path.join(args.out, "checks_by_card.png"))
    fig_settle(cards, os.path.join(args.out, "verdict_settle_n.png"))
    fig_thresholds(cards, os.path.join(args.out, "threshold_sensitivity.png"))
    print(f"{len(cards)} cards; figures written to {args.out}")


if __name__ == "__main__":
    main()
