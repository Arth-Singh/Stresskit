"""Relabel every committed diagnostic stability card under grade rule v0.4.

The grade rule changed (v0.3 point rule -> v0.4 decided rule) without any
evidence changing, so the migration is a pure function of each card's
recorded checks: ``relabel_grade`` re-derives the letter exactly as
``stresskit verify`` does, stamps the rule and the two thresholds v0.4
registered, bumps the schema to 0.5 and appends a note naming the old grade.
The re-rendered ``.md`` and ``.badge.json`` sidecars follow the card wherever
the repository already has them.

Dry run by default: everything lands under ``--out`` (mirroring the paths
under ``--root``) together with ``grade-migration.json``, the table this
script prints. ``--apply`` writes the cards and sidecars in place instead;
``grade-migration.json`` still goes to ``--out``.

Usage:
    PYTHONPATH=src python3 references/regrade_cards.py --root references --out /tmp/regrade_dry
    PYTHONPATH=src python3 references/regrade_cards.py --root references --out /tmp/regrade --apply
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from stresskit.card import GRADE_ORDER, GRADE_RULE, StabilityCard, verify_card_dict
from stresskit.card_findings import (
    card_name,
    card_thresholds,
    grade_reasons,
    is_diagnostic_stability_card,
    recomputed_checks,
    relabel_grade,
)
from stresskit.scoreboard import collect_rows


def diagnostic_cards(
    root: str, only: Optional[str]
) -> List[Tuple[str, Dict[str, Any]]]:
    """(path, card) for every diagnostic stability card under ``root``, in
    scoreboard order; confirmatory certificates and oracle reports are not
    cards and never come back from here."""
    out = []
    for row in collect_rows([root]):
        if row["kind"] != "stability card":
            continue
        path = row["path"]
        if only and only not in os.path.relpath(path, root):
            continue
        with open(path, encoding="utf-8") as f:
            card = json.load(f)
        if is_diagnostic_stability_card(card):
            out.append((path, card))
    return out


def reason_text(reasons: Dict[str, Any], n_checks: int) -> str:
    if reasons["at_random_floor"]:
        return "beats_random at or below the at-random floor: D outright"
    parts = [
        f"decided passes {len(reasons['decided_pass'])}/{n_checks} "
        f"(point passes {len(reasons['point_pass'])})"
    ]
    if reasons["undecided"]:
        parts.append("undecided: " + ", ".join(reasons["undecided"]))
    if reasons["decided_fail"]:
        parts.append("decided fails: " + ", ".join(reasons["decided_fail"]))
    if reasons["cap_reason"]:
        parts.append(reasons["cap_reason"])
    return "; ".join(parts)


def unified(a: str, b: str, a_name: str, b_name: str) -> List[str]:
    return list(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            fromfile=a_name,
            tofile=b_name,
            lineterm="",
            n=0,
        )
    )


def changed_lines(diff: List[str]) -> List[str]:
    return [
        line
        for line in diff
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith(("+++", "---"))
    ]


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def migrate_card(
    path: str,
    card: Dict[str, Any],
    *,
    root: str,
    out: str,
    apply: bool,
    note_date: str,
) -> Dict[str, Any]:
    rel = os.path.relpath(path, root)
    stem = os.path.splitext(path)[0]
    dest_stem = stem if apply else os.path.splitext(os.path.join(out, rel))[0]
    verdict = card["verdict"]
    old_grade = verdict["grade"]
    row: Dict[str, Any] = {
        "card": os.path.splitext(rel)[0],
        "path": path,
        "name": card_name(card),
        "schema_before": card["schema_version"],
        "structure_kind": card["battery"].get("structure_kind", "set"),
        "grade_v03": old_grade,
        "rule_before": verdict.get("grade_rule", "v0.3"),
    }

    relabelled = relabel_grade(card, rule=GRADE_RULE, note_date=note_date)
    checks = recomputed_checks(card)
    reasons = grade_reasons(checks, random_floor=card_thresholds(card).random_floor)
    report = verify_card_dict(relabelled)
    already = (
        f"already graded under {GRADE_RULE} (fresh card; relabel is a no-op); "
        if row["rule_before"] == GRADE_RULE
        else ""
    )
    row.update(
        {
            "grade_v04": relabelled["verdict"]["grade"],
            "changed": relabelled["verdict"]["grade"] != old_grade,
            "reason": already + reason_text(reasons, len(checks)),
            **{
                k: reasons[k]
                for k in (
                    "decided_pass",
                    "decided_fail",
                    "undecided",
                    "point_pass",
                    "cap",
                    "cap_reason",
                    "at_random_floor",
                )
            },
            "verify_ok": report["ok"],
            "verify_problems": report["problems"],
        }
    )

    new_card = StabilityCard.from_dict(relabelled)
    untouched_md = StabilityCard.from_dict(card).to_markdown() + "\n"
    relabelled_md = new_card.to_markdown() + "\n"
    row["md_relabel_changed_lines"] = changed_lines(
        unified(untouched_md, relabelled_md, "untouched render", "relabelled render")
    )
    committed_md_path = stem + ".md"
    if os.path.exists(committed_md_path):
        with open(committed_md_path, encoding="utf-8") as f:
            committed_md = f.read()
        row["md_untouched_rerender_identical"] = committed_md == untouched_md
        row["md_untouched_rerender_diff"] = unified(
            committed_md, untouched_md, "committed .md", "untouched render"
        )
        row["md_relabelled_vs_committed_diff"] = unified(
            committed_md, relabelled_md, "committed .md", "relabelled render"
        )
    else:
        row["md_untouched_rerender_identical"] = None

    badge = new_card.badge_dict()
    committed_badge_path = stem + ".badge.json"
    if os.path.exists(committed_badge_path):
        with open(committed_badge_path, encoding="utf-8") as f:
            row["badge_changed"] = json.load(f) != badge
    else:
        row["badge_changed"] = None

    written = []
    os.makedirs(os.path.dirname(dest_stem) or ".", exist_ok=True)
    new_card.save(dest_stem + ".json")
    written.append(dest_stem + ".json")
    if os.path.exists(committed_md_path):
        write_text(dest_stem + ".md", relabelled_md)
        written.append(dest_stem + ".md")
    if os.path.exists(committed_badge_path):
        write_json(dest_stem + ".badge.json", badge)
        written.append(dest_stem + ".badge.json")
    row["written"] = written
    return row


def print_table(rows: List[Dict[str, Any]]) -> None:
    width = max(len(r["card"]) for r in rows)
    print(f"{'card':<{width}}  v0.3  v0.4  reason")
    for r in rows:
        mark = "*" if r.get("changed") else " "
        print(
            f"{r['card']:<{width}}  {r['grade_v03']:<4}  "
            f"{r.get('grade_v04', '?'):<3}{mark} {r.get('reason') or r.get('error')}"
        )
    print("(* = grade changed)")


def print_counts(rows: List[Dict[str, Any]]) -> None:
    before = {g: sum(1 for r in rows if r["grade_v03"] == g) for g in GRADE_ORDER}
    after = {g: sum(1 for r in rows if r.get("grade_v04") == g) for g in GRADE_ORDER}
    print(
        "grades before (as recorded): "
        + "  ".join(f"{g} {n}" for g, n in before.items())
    )
    print(
        "grades after (v0.4):         "
        + "  ".join(f"{g} {n}" for g, n in after.items())
    )
    transitions: Dict[str, int] = {}
    for r in rows:
        if r.get("changed"):
            key = f"{r['grade_v03']}->{r['grade_v04']}"
            transitions[key] = transitions.get(key, 0) + 1
    if transitions:
        print(
            "transitions: "
            + ", ".join(f"{k} {v}" for k, v in sorted(transitions.items()))
        )
    print(f"changed: {sum(1 for r in rows if r.get('changed'))}/{len(rows)}")


def print_md_check(rows: List[Dict[str, Any]]) -> None:
    first = next(
        (r for r in rows if r.get("md_untouched_rerender_identical") is not None),
        None,
    )
    if first is None:
        print("no committed .md found next to any card; re-render check skipped")
        return
    print(f"re-render check on {first['card']} (first card with a committed .md):")
    if first["md_untouched_rerender_identical"]:
        print("  untouched card re-renders byte-identical to the committed .md")
    else:
        print(
            "  untouched card does NOT re-render identically "
            "(renderer drift since the card was written):"
        )
        for line in first["md_untouched_rerender_diff"]:
            print("    " + line)
    print("  relabelled card vs committed .md:")
    for line in first["md_relabelled_vs_committed_diff"]:
        print("    " + line)
    print()
    print("re-render of the untouched card vs committed .md, all cards:")
    for r in rows:
        status = r.get("md_untouched_rerender_identical")
        if status is None:
            note = "no committed .md"
        elif status:
            note = "identical"
        else:
            n = len(changed_lines(r["md_untouched_rerender_diff"]))
            note = (
                f"differs ({n} changed lines: renderer drift, "
                "the .md will be rewritten)"
            )
        print(
            f"  {r['card']}: {note}; relabel changes "
            f"{len(r.get('md_relabel_changed_lines', []))} lines"
        )
    badge_changes = [r["card"] for r in rows if r.get("badge_changed")]
    if badge_changes:
        print("badges that change: " + ", ".join(badge_changes))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default="references")
    parser.add_argument(
        "--out",
        required=True,
        help="output directory for the dry run and grade-migration.json",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write cards and sidecars in place"
    )
    parser.add_argument(
        "--only", default=None, help="substring filter on the card path"
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="date written into the regrade note (ISO, default today)",
    )
    args = parser.parse_args(argv)

    cards = diagnostic_cards(args.root, args.only)
    if not cards:
        print(f"no diagnostic stability cards under {args.root!r}", file=sys.stderr)
        return 1
    rows: List[Dict[str, Any]] = []
    failures = 0
    for path, card in cards:
        try:
            row = migrate_card(
                path,
                card,
                root=args.root,
                out=args.out,
                apply=args.apply,
                note_date=args.date,
            )
        except ValueError as e:
            failures += 1
            row = {
                "card": os.path.splitext(os.path.relpath(path, args.root))[0],
                "path": path,
                "grade_v03": card["verdict"]["grade"],
                "error": f"ERROR: {e}",
            }
        rows.append(row)
        if not row.get("verify_ok", True):
            failures += 1

    print_table(rows)
    print()
    print_counts(rows)
    print()
    print_md_check(rows)
    print()
    bad = [r for r in rows if not r.get("verify_ok", False)]
    for r in bad:
        print(
            f"VERIFY FAILED {r['card']}: {r.get('verify_problems') or r.get('error')}"
        )
    print(f"{len(rows) - len(bad)}/{len(rows)} relabelled cards verify")
    print(f"cards written {'in place' if args.apply else 'to ' + args.out}")

    migration = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "root": args.root,
        "rule": GRADE_RULE,
        "note_date": args.date,
        "applied": args.apply,
        "cards": rows,
    }
    write_json(os.path.join(args.out, "grade-migration.json"), migration)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
