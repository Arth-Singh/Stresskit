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
- ``specificity_by_null.png``: the specificity outcome of every card that has
  a null control, split by how the null was built: a signal-destroying null
  (labels permuted, adapter scrambled, calibration data replaced by noise)
  against a structure-preserving null (the task corrupted, items re-paired
  or deranged, weights rotated) that leaves the finder's output size intact.
- ``grade_migration.png``: how every card's letter moved from grade rule v0.3
  (point estimates vote) to v0.4 (only decided checks count), read from the
  v0.3 grade each regraded card keeps in its notes.
- ``battery_calibration.png``: from the frozen planted-truth study
  (``artifacts/calibration/battery-known-truth-primary.json``): how often each
  check's 95% interval covers the exact truth at 6 to 100 runs, and how often
  the letter grade is wrong under each rule, split by the card's confidence.
- ``null_score_leak.png``: from ``artifacts/self_audit/null-score-leak.json``:
  for every battery with a null control, how far the null's score sits below
  the real score, coloured by the specificity outcome.

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

# How each card's null control is built, keyed by card-name prefix. Kept
# explicit so the split can be audited against the card notes.
NULL_FAMILY = {
    "folkmotif": "signal", "diff_mining": "signal", "sycophancy": "signal",
    "refusal_direction": "signal", "impossibility_truth": "signal", "harc": "signal",
    "reins": "signal", "faithfulness": "signal", "ams": "signal", "swd": "signal",
    "ioi": "structure", "greater_than": "structure", "coax": "structure",
    "communication_map": "structure", "sae_causal": "structure", "homonym": "structure",
    "jlens": "structure", "lens_baseline": "structure", "mechtomo": "structure",
}
FAMILY_LABEL = {
    "signal": "signal-destroying null\n(labels permuted, adapter scrambled,\ncalibration replaced by noise)",
    "structure": "structure-preserving null\n(task corrupted, items re-paired,\nweights rotated; output size kept)",
}


def null_family(name):
    for prefix, family in NULL_FAMILY.items():
        if name.startswith(prefix):
            return family
    return None
MUTED = "#6b6b6b"
GRADE_COLOUR = {"A": "#0ca30c", "B": "#8cbf26", "C": "#fab219", "D": "#d03b3b"}
CHECK_SHORT = {
    "structural_stability": "structural",
    "claim_stability": "claim",
    "score_stability": "score",
    "beats_random": "beats random",
    "specificity": "specificity",
}
CHECK_COLOUR = {
    "structural_stability": "#1f5fbf",
    "claim_stability": "#7a3db8",
    "score_stability": "#c2571a",
    "beats_random": "#3a8c3a",
    "specificity": "#9a1f4a",
}


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
        v03 = None
        for note in card.get("notes", []):
            if note.startswith("v0.3 grade: "):
                v03 = note[len("v0.3 grade: ")]
        rule = card.get("verdict", {}).get("grade_rule", "v0.3")
        cards.append(
            {
                "grade_v03": v03 if rule == "v0.4" else row["grade"],
                "grade_rule": rule,
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


def fig_null_family(cards, out):
    counts = {"signal": Counter(), "structure": Counter()}
    unmapped = []
    for card in cards:
        check = card["checks"].get("specificity")
        if check is None:
            continue
        family = null_family(card["name"])
        if family is None:
            unmapped.append(card["name"])
            continue
        counts[family][recorded_state(check)] += 1
    if unmapped:
        raise SystemExit(f"cards with a specificity check but no null family: {unmapped}")
    print("specificity_by_null:", {k: dict(v) for k, v in counts.items()})
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    y = [1, 0]
    for idx, family in zip(y, ("signal", "structure")):
        left = 0
        total = sum(counts[family].values())
        small = []
        for state in ("pass", "incon", "fail"):
            n = counts[family][state]
            if n == 0:
                continue
            ax.barh(idx, n, left=left, color=STATUS[state], height=0.62, edgecolor="white", linewidth=2)
            if n >= 4:
                ax.text(left + n / 2, idx, f"{LABEL[state]} {n}", ha="center", va="center", fontsize=8.5, color="white" if state != "incon" else INK)
            else:
                small.append(f"{LABEL[state]} {n}")
            left += n
        tail = f"n = {total}" + (f"  ({', '.join(small)})" if small else "")
        ax.text(total + 0.5, idx, tail, va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([FAMILY_LABEL[f] for f in ("signal", "structure")], fontsize=8.5)
    ax.set_xlim(0, 33)
    ax.set_xlabel("cards with a specificity check")
    ax.set_title("Whether a finding beats its null depends on how the null was built", fontsize=10, loc="left", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_grade_migration(cards, out):
    """v0.3 -> v0.4 transitions over every card that carries both grades."""
    graded = [c for c in cards if c["grade_rule"] == "v0.4" and c["grade_v03"] in GRADE_COLOUR]
    if not graded:
        print("grade_migration: no regraded card carries a v0.3 grade note; skipped")
        return
    matrix = Counter((c["grade_v03"], c["grade"]) for c in graded)
    print("grade_migration:", {f"{a}->{b}": n for (a, b), n in sorted(matrix.items())})
    letters = list(GRADE_COLOUR)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for i, old in enumerate(letters):
        for j, new in enumerate(letters):
            n = matrix.get((old, new), 0)
            face = GRADE_COLOUR[new] if n and old != new else ("#e9e9e9" if n else "white")
            ax.add_patch(plt.Rectangle((j, len(letters) - 1 - i), 1, 1, facecolor=face, edgecolor="white", linewidth=2))
            if n:
                colour = "white" if (old != new and new != "B") else INK
                ax.text(j + 0.5, len(letters) - 1 - i + 0.5, str(n), ha="center", va="center", fontsize=13, color=colour, fontweight="bold")
    ax.set_xlim(0, len(letters))
    ax.set_ylim(0, len(letters))
    ax.set_xticks([k + 0.5 for k in range(len(letters))])
    ax.set_xticklabels(letters)
    ax.set_yticks([k + 0.5 for k in range(len(letters))])
    ax.set_yticklabels(letters[::-1])
    ax.set_xlabel("grade under rule v0.4 (only decided checks count)")
    ax.set_ylabel("grade under rule v0.3 (point estimates vote)")
    moved = sum(n for (a, b), n in matrix.items() if a != b)
    ax.set_title(f"How the letter moved when the rule changed\n{len(graded)} cards, {moved} changed letter, none rose", fontsize=10, loc="left", color=INK)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, colors=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _pooled_rate(results, numerator, denominator):
    num = sum(r["counts"].get(numerator, 0) for r in results)
    den = sum(r["counts"].get(denominator, 0) for r in results)
    return (num / den if den else None), den


def _log_axis(ax, run_counts):
    ax.set_xscale("log")
    ax.set_xticks(run_counts)
    ax.set_xticklabels([str(n) for n in run_counts])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("runs in the battery")


def fig_battery_calibration(payload_path, out):
    if not os.path.exists(payload_path):
        print(f"battery_calibration: {payload_path} not found; skipped")
        return
    with open(payload_path) as handle:
        results = json.load(handle)["results"]
    run_counts = sorted({r["n_runs"] for r in results})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for name in CHECK_ORDER:
        xs, ys, errs = [], [], []
        for n in run_counts:
            cell = [r for r in results if r["n_runs"] == n]
            rate, den = _pooled_rate(cell, f"{name}:covered", f"{name}:ci_available")
            if rate is None:
                continue
            xs.append(n)
            ys.append(rate)
            errs.append((rate * (1 - rate) / den) ** 0.5 if den else 0)
        print(f"battery_calibration coverage {name}: " + ", ".join(f"n={x} {y:.3f}" for x, y in zip(xs, ys)))
        ax1.errorbar(xs, ys, yerr=errs, marker="o", markersize=4, linewidth=1.4, capsize=2, color=CHECK_COLOUR[name], label=CHECK_SHORT[name])
    ax1.axhline(0.95, color=MUTED, linestyle="--", linewidth=1)
    ax1.text(run_counts[-1], 0.953, "nominal 95%", ha="right", va="bottom", fontsize=8, color=MUTED)
    _log_axis(ax1, run_counts)
    ax1.set_ylabel("share of intervals covering the exact truth")
    ax1.set_title("Does the shipped 95% interval cover the truth?\n(pooled over the planted-truth cells, MCSE bars)", fontsize=9.5, loc="left", color=INK)
    ax1.legend(fontsize=8, frameon=False, loc="lower right")
    style(ax1)
    for tag, colour, label in (("v03", "#9a9a9a", "rule v0.3"), ("v04", "#1f5fbf", "rule v0.4")):
        for conf, dash in (("high", "-"), ("low", ":")):
            xs, ys = [], []
            for n in run_counts:
                cell = [r for r in results if r["n_runs"] == n]
                rate, den = _pooled_rate(cell, f"wrong_{tag}&conf:{conf}", f"conf:{conf}")
                if rate is None:
                    continue
                xs.append(n)
                ys.append(rate)
            print(f"battery_calibration P(grade wrong | {conf}) {tag}: " + ", ".join(f"n={x} {y:.3f}" for x, y in zip(xs, ys)))
            ax2.plot(xs, ys, marker="o", markersize=4, linewidth=1.4, linestyle=dash, color=colour, label=f"{label}, {conf} confidence")
    _log_axis(ax2, run_counts)
    ax2.set_ylabel("share of trials with the wrong letter")
    ax2.set_title("How often is the letter wrong, given the card's confidence?\n(pooled over cells; truth grade from the exact truths)", fontsize=9.5, loc="left", color=INK)
    ax2.legend(fontsize=7.5, frameon=False)
    style(ax2)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def fig_null_score_leak(payload_path, out):
    if not os.path.exists(payload_path):
        print(f"null_score_leak: {payload_path} not found; skipped")
        return
    with open(payload_path) as handle:
        payload = json.load(handle)
    rows = []
    for card in payload["cards"]:
        d = card["leak"].get("d")
        if d is None:
            continue
        state = {"pass": "pass", "fail": "fail"}.get(card["specificity"]["state"], "incon")
        rows.append((card["family"], d, state, card["card"], card["leak"]["leak_class"]))
    rows.sort(key=lambda r: (r[0], r[1]))
    print("null_score_leak:", Counter((r[0], r[4]) for r in rows))
    fig, ax = plt.subplots(figsize=(8.4, 0.24 * len(rows) + 1.8))
    limit = 12.0
    y = list(range(len(rows)))[::-1]
    for idx, (family, d, state, name, cls) in zip(y, rows):
        x = max(-limit, min(limit, d))
        marker = "o" if family == "signal" else "s"
        ax.plot([0, x], [idx, idx], color="#dddddd", linewidth=1.0, zorder=1)
        ax.scatter([x], [idx], s=30, marker=marker, color=STATUS[state], zorder=2, edgecolor="white", linewidth=0.6)
        if abs(d) > limit:
            ax.text(x + (0.25 if d > 0 else -0.25), idx, f"{d:.0f}", ha="left" if d > 0 else "right", va="center", fontsize=6.5, color=MUTED)
    ax.axvspan(-limit - 1.5, 0.5, color="#f6f6f6", zorder=0)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
    ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(0.4, len(rows) - 0.2, "d ≤ 0.5: null matches or exceeds", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.text(1.1, len(rows) - 0.2, "d ≥ 1: null degraded", ha="left", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[3].replace('_', ' ')}  ({r[0]} null)" for r in rows], fontsize=6.8)
    ax.set_xlim(-limit - 1.5, limit + 1.5)
    ax.set_ylim(-1, len(rows) + 0.8)
    ax.set_xlabel("standardized gap d, real score minus null score (clipped at ±12)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=STATUS[s], label=f"specificity {LABEL[s]}") for s in ("pass", "incon", "fail")]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.94))
    fig.suptitle(
        "Does the null control still score like the real data?\n"
        "One row per battery; circles: signal-destroying null, squares: structure-preserving null",
        fontsize=9.5, x=0.01, ha="left", color=INK,
    )
    style(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
    fig_null_family(cards, os.path.join(args.out, "specificity_by_null.png"))
    fig_grade_migration(cards, os.path.join(args.out, "grade_migration.png"))
    root = os.path.dirname(os.path.abspath(args.references))
    fig_battery_calibration(
        os.path.join(root, "artifacts", "calibration", "battery-known-truth-primary.json"),
        os.path.join(args.out, "battery_calibration.png"),
    )
    fig_null_score_leak(
        os.path.join(root, "artifacts", "self_audit", "null-score-leak.json"),
        os.path.join(args.out, "null_score_leak.png"),
    )
    print(f"{len(cards)} cards; figures written to {args.out}")


if __name__ == "__main__":
    main()
