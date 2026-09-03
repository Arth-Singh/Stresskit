"""Recover check intervals that an older card schema never wrote.

Schema 0.2 predates the bootstrap interval on ``beats_random``, so a card
written then records that check with ``ci: null``. Grade rule v0.4 counts a
check only when its whole interval clears the bar, and a check with no
interval is undecided by definition — so such a card is graded as though the
evidence were missing when in fact the card carries the runs the interval is
computed from.

This recomputes those intervals from the card's own embedded runs, at the
card's own battery seed, exactly as ``from_findings`` would, and writes them
onto the checks together with the interval's state and a note. Only checks
that are a function of the real runs alone are recovered; specificity needs
the null runs, which are not on the card, and stays undecided.

Usage:
    PYTHONPATH=src python references/recover_missing_intervals.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from stresskit.battery import grade_checks, make_check
from stresskit.card import SCHEMA_VERSION, StabilityCard, verify_card_dict
from stresskit.card_findings import (
    card_name,
    card_thresholds,
    is_diagnostic_stability_card,
    regrade_card,
    recomputed_checks,
)
from stresskit.scoreboard import collect_rows

HERE = os.path.dirname(os.path.abspath(__file__))
# checks whose interval is a function of the real runs alone
RECOVERABLE = (
    "structural_stability",
    "claim_stability",
    "score_stability",
    "beats_random",
)


def recoverable_cards(root: str) -> List[Dict[str, Any]]:
    out = []
    for row in collect_rows([root]):
        with open(row["path"]) as handle:
            card = json.load(handle)
        if not is_diagnostic_stability_card(card):
            continue
        missing = [
            name
            for name, check in card["verdict"]["checks"].items()
            if check.get("ci") is None and name in RECOVERABLE
        ]
        if missing and card.get("runs"):
            out.append({"path": row["path"], "card": card, "missing": missing})
    return out


def recover(entry: Dict[str, Any], *, note_date: str) -> Dict[str, Any]:
    card = entry["card"]
    thresholds = card_thresholds(card)
    fresh = regrade_card(card, seed=card["battery"]["seed"])
    recovered = []
    for name in entry["missing"]:
        new = fresh.checks.get(name)
        recorded = card["verdict"]["checks"][name]
        if new is None or new.get("ci") is None:
            continue
        if abs(new["value"] - recorded["value"]) > 1e-9:
            raise ValueError(
                f"card {card_name(card)}: {name} value {recorded['value']} does not "
                f"reproduce from the card's runs ({new['value']})"
            )
        rebuilt = make_check(
            recorded["value"],
            recorded["threshold"],
            recorded["op"],
            recorded.get("description", ""),
            ci=new["ci"],
        )
        recorded["ci"] = rebuilt["ci"]
        recorded["robust"] = rebuilt["robust"]
        recorded["state"] = rebuilt["state"]
        recovered.append((name, rebuilt["ci"], rebuilt["state"]))
    if not recovered:
        return {
            "path": entry["path"],
            "recovered": [],
            "grade": card["verdict"]["grade"],
        }
    old_grade = card["verdict"]["grade"]
    checks = recomputed_checks(card)
    card["verdict"]["grade"] = grade_checks(
        checks, rule=card["verdict"]["grade_rule"], random_floor=thresholds.random_floor
    )
    card["notes"] = list(card.get("notes") or []) + [
        "recovered on {} the bootstrap interval for {} from the card's own runs at "
        "seed {}: the schema this card was written under ({}) did not store it, so the "
        "check read as undecided and the grade was {} instead of {}".format(
            note_date,
            ", ".join(name for name, _, _ in recovered),
            card["battery"]["seed"],
            "0.2",
            old_grade,
            card["verdict"]["grade"],
        )
    ]
    return {
        "path": entry["path"],
        "recovered": recovered,
        "old_grade": old_grade,
        "grade": card["verdict"]["grade"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=HERE)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--date", default="2026-09-03")
    args = parser.parse_args(argv)

    entries = recoverable_cards(args.root)
    if not entries:
        print("no card has a recoverable missing interval")
        return 0
    failures = 0
    for entry in entries:
        report = recover(entry, note_date=args.date)
        name = os.path.basename(entry["path"])[: -len(".json")]
        if not report["recovered"]:
            print(f"{name}: nothing recovered (interval still unavailable)")
            continue
        for check, ci, state in report["recovered"]:
            print(f"{name}: {check} interval [{ci[0]:.4g}, {ci[1]:.4g}] -> {state}")
        print(f"{name}: grade {report['old_grade']} -> {report['grade']}")
        card = entry["card"]
        verdict = verify_card_dict(card)
        if not verdict["ok"]:
            failures += 1
            print(f"{name}: FAILS verification: {verdict['problems']}")
            continue
        if args.apply:
            stem = entry["path"][: -len(".json")]
            obj = StabilityCard.from_dict(card)
            obj.save(entry["path"])
            with open(stem + ".md", "w", encoding="utf-8") as handle:
                handle.write(obj.to_markdown() + "\n")
            with open(stem + ".badge.json", "w", encoding="utf-8") as handle:
                json.dump(obj.badge_dict(), handle, indent=2)
                handle.write("\n")
            print(f"{name}: written (schema {SCHEMA_VERSION})")
    if failures:
        print(f"{failures} card(s) failed verification; nothing written for them")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
