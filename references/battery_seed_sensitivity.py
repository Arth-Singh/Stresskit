"""Battery-seed sensitivity of every regradable reference card.

The top-level battery seed drives the Monte-Carlo random null and every
bootstrap interval, so it can move a check's state and with it the letter
grade. Every card was graded at one seed. This script rebuilds each card's
findings from the card (and its null runs from the sidecar manifest where one
exists), regrades them at ``--seeds`` fresh battery seeds with
``stresskit.from_findings`` under the card's own thresholds, and records how
often the grade, the confidence and each check's state change. With
``--traces`` it also rebuilds the verdict trace at every seed and records the
range of ``settled_n`` and of the six-run modal grade; traces are rebuilt only where the
card's whole battery is recoverable (group A, and cards without a null control),
because a trace without the null runs would drop the specificity check.

Cards fall into three groups: A, real and null runs recoverable (the full
battery is regraded); B, real runs only (specificity is fixed at the card's
value and gets no fresh interval, so it is excluded from the flip counts);
C, direction-valued cards, regraded from the embedded |cosine| matrices.
Hash-only cards cannot be regraded and are listed with the reason.

Usage:
    PYTHONPATH=src python references/battery_seed_sensitivity.py \
        --root references --out artifacts/self_audit/seed-sensitivity.json \
        --seeds 20 --workers 8 [--traces] [--only NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter
from multiprocessing import Pool

import stresskit as sk
from stresskit.card_findings import (
    card_thresholds,
    findings_from_card_dict,
    load_null_findings,
    regrade_card,
    regrade_direction_card,
)
from stresskit.scoreboard import collect_rows

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST_SEED = 20260904


def diagnostic_cards(root, only):
    cards = []
    for row in collect_rows([root]):
        with open(row["path"]) as handle:
            card = json.load(handle)
        if card.get("artifact") is not None or "checks" not in card.get("verdict", {}):
            continue
        if card.get("verdict", {}).get("profile") == "confirmatory":
            continue
        name = os.path.basename(row["path"])[: -len(".json")]
        if only and name != only:
            continue
        cards.append((name, row["path"], card))
    if only and not cards:
        raise SystemExit(f"no diagnostic stability card named {only!r} under {root}")
    return cards


def classify(path, card):
    if card.get("battery", {}).get("structure_kind") == "direction":
        return "C", None, None, None
    try:
        findings, axes = findings_from_card_dict(card)
    except ValueError as exc:
        return "skip", None, None, str(exc)
    nulls = (
        load_null_findings(path) if "null_control" in card.get("metrics", {}) else None
    )
    return ("A" if nulls is not None else "B"), (findings, axes), nulls, None


def one_seed(task):
    path, seed, with_trace = task
    with open(path) as handle:
        card = json.load(handle)
    group, real, nulls, _ = classify(path, card)
    thresholds = card_thresholds(card)
    out = {"seed": seed}
    if group == "C":
        res = regrade_direction_card(card, seed=seed)
        checks = res["checks"]
        out["grade"] = res["grade_v04"]
        out["confidence"] = res["confidence"]
    else:
        findings, axes = real
        res = regrade_card(card, seed=seed, null_findings=nulls)
        checks = res.checks
        out["grade"] = res.grade
        out["confidence"] = res.pooled["confidence"]
        traceable = nulls is not None or "null_control" not in card.get("metrics", {})
        if with_trace and traceable and len(findings) >= 5:
            trace = sk.verdict_trace(
                findings, null_findings=nulls, seed=seed, thresholds=thresholds
            )
            six = trace["per_size"].get(6)
            out["trace"] = {
                "settled_n": trace["settled_n"],
                "six_modal": six["modal_grade"] if six else None,
                "six_share": six["modal_grade_share"] if six else None,
            }
    out["states"] = {name: c.get("state") for name, c in checks.items()}
    out["values"] = {
        name: checks[name]["value"]
        for name in ("beats_random", "specificity")
        if name in checks
    }
    return out


def summarize(name, path, card, group, reason, per_seed):
    if group == "skip":
        return {"card": name, "path": path, "group": group, "reason": reason}
    grades = [r["grade"] for r in per_seed]
    confidences = [r["confidence"] for r in per_seed]
    check_names = sorted({n for r in per_seed for n in r["states"]})
    flips = {}
    for check in check_names:
        if group == "B" and check == "specificity":
            continue
        states = Counter(r["states"].get(check) for r in per_seed)
        flips[check] = {"states": dict(states), "distinct": len(states)}
    ranges = {}
    for key in ("beats_random", "specificity"):
        vals = [
            r["values"][key]
            for r in per_seed
            if key in r["values"] and r["values"][key] is not None
        ]
        if vals and not (group == "B" and key == "specificity"):
            ranges[key] = {"min": min(vals), "max": max(vals)}
    reference = grades[0]
    row = {
        "card": name,
        "path": path,
        "group": group,
        "n_runs": card["battery"]["n_runs_total"],
        "recorded_grade": card["verdict"]["grade"],
        "recorded_rule": card["verdict"].get("grade_rule", "v0.3"),
        "seeds": [r["seed"] for r in per_seed],
        "grade_at_first_seed": reference,
        "grades": dict(Counter(grades)),
        "grade_agreement": sum(1 for g in grades if g == reference) / len(grades),
        "confidence": dict(Counter(confidences)),
        "check_state_distinct": {k: v["distinct"] for k, v in flips.items()},
        "check_states": {k: v["states"] for k, v in flips.items()},
        "value_ranges": ranges,
    }
    traces = [r["trace"] for r in per_seed if "trace" in r]
    if traces:
        settled = [t["settled_n"] for t in traces]
        finite = [s for s in settled if s is not None]
        row["trace"] = {
            "settled_n_min": min(finite) if finite else None,
            "settled_n_max": max(finite) if finite else None,
            "settled_n_mode": statistics.mode(finite) if finite else None,
            "never_settles": sum(1 for s in settled if s is None),
            "six_modal": dict(Counter(t["six_modal"] for t in traces)),
            "six_share_min": min(
                t["six_share"] for t in traces if t["six_share"] is not None
            ),
            "six_share_max": max(
                t["six_share"] for t in traces if t["six_share"] is not None
            ),
        }
    return row


def cards_digest(paths):
    parts = []
    for path in sorted(paths):
        with open(path, "rb") as handle:
            parts.append(
                f"{os.path.relpath(path)}:{hashlib.sha256(handle.read()).hexdigest()}"
            )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def markdown(rows):
    lines = [
        "| card | group | n | grade at first seed | grade agreement | grades over seeds | confidence | checks with >1 state | beats random range | specificity range | settled n (min/mode/max) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["group"] == "skip":
            lines.append(
                f"| {row['card']} | skipped | | | | | | {row['reason']} | | | |"
            )
            continue
        moving = (
            ", ".join(k for k, v in row["check_state_distinct"].items() if v > 1)
            or "none"
        )
        br = row["value_ranges"].get("beats_random")
        sp = row["value_ranges"].get("specificity")
        tr = row.get("trace")
        br_text = "{:.2f}–{:.2f}".format(br["min"], br["max"]) if br else "—"
        sp_text = "{:.2f}–{:.2f}".format(sp["min"], sp["max"]) if sp else "—"
        if tr:
            tr_text = "{}/{}/{}".format(
                tr["settled_n_min"], tr["settled_n_mode"], tr["settled_n_max"]
            )
            if tr["never_settles"]:
                tr_text += " ({} never)".format(tr["never_settles"])
        else:
            tr_text = "—"
        grades = " ".join("{}{}".format(g, n) for g, n in sorted(row["grades"].items()))
        confidence = " ".join(
            "{}{}".format(c, n) for c, n in sorted(row["confidence"].items())
        )
        lines.append(
            "| {} | {} | {} | {} | {:.0%} | {} | {} | {} | {} | {} | {} |".format(
                row["card"],
                row["group"],
                row["n_runs"],
                row["grade_at_first_seed"],
                row["grade_agreement"],
                grades,
                confidence,
                moving,
                br_text,
                sp_text,
                tr_text,
            )
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=HERE)
    parser.add_argument(
        "--out",
        default=os.path.join(
            HERE, "..", "artifacts", "self_audit", "seed-sensitivity.json"
        ),
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    parser.add_argument("--traces", action="store_true")
    parser.add_argument("--only", default=None)
    args = parser.parse_args(argv)

    cards = diagnostic_cards(args.root, args.only)
    seeds = [FIRST_SEED + i for i in range(args.seeds)]
    plan = []
    groups = {}
    for name, path, card in cards:
        group, _, _, reason = classify(path, card)
        groups[name] = (group, reason)
        if group != "skip":
            plan.extend((path, seed, args.traces) for seed in seeds)
    print(
        f"{len(cards)} cards: "
        + ", ".join(
            f"{g}={sum(1 for v in groups.values() if v[0] == g)}"
            for g in ("A", "B", "C", "skip")
        ),
        file=sys.stderr,
    )
    print(f"{len(plan)} regrades on {args.workers} workers", file=sys.stderr)
    with Pool(args.workers) as pool:
        results = pool.map(one_seed, plan, chunksize=1)
    by_path = {}
    for task, result in zip(plan, results):
        by_path.setdefault(task[0], []).append(result)
    rows = []
    for name, path, card in cards:
        group, reason = groups[name]
        rows.append(summarize(name, path, card, group, reason, by_path.get(path, [])))
    payload = {
        "schema_version": "0.1",
        "seeds": seeds,
        "traces": args.traces,
        "grade_rule": sk.card.GRADE_RULE,
        "cards_digest": cards_digest([path for _, path, _ in cards]),
        "cards": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    with open(args.out[: -len(".json")] + ".md", "w") as handle:
        handle.write(markdown(rows) + "\n")
    graded = [r for r in rows if r["group"] != "skip"]
    print(markdown(rows))
    print()
    print(
        f"cards regraded: {len(graded)}; grade moves with the seed on "
        f"{sum(1 for r in graded if len(r['grades']) > 1)}; confidence moves on "
        f"{sum(1 for r in graded if len(r['confidence']) > 1)}; at least one check state moves on "
        f"{sum(1 for r in graded if any(v > 1 for v in r['check_state_distinct'].values()))}"
    )
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
