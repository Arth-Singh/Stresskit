"""StressKit command line.

Commands
--------
stresskit render <card.json>          render a Stability Card as markdown
stresskit badge  <card.json> [-o f]   emit shields.io endpoint JSON for the badge
stresskit report [--field value ...]  generate the Minimum Reporting Checklist
stresskit verify <cards-or-dirs ...>  auditor mode: re-derive every artifact's
                                      checks and grade from its own metrics
stresskit version
"""

from __future__ import annotations

import argparse
import json
import sys

from .card import StabilityCard
from .report import CHECKLIST_FIELDS, generate_checklist


def _cmd_render(args: argparse.Namespace) -> int:
    card = StabilityCard.load(args.card)
    print(card.to_markdown())
    return 0


def _cmd_badge(args: argparse.Namespace) -> int:
    card = StabilityCard.load(args.card)
    payload = json.dumps(card.badge_dict(), indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"badge JSON written to {args.output}")
        print(
            "embed with: https://img.shields.io/endpoint?url=<public raw URL of that file>"
        )
    else:
        print(payload)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    answers = {key: getattr(args, key) for key, _, _ in CHECKLIST_FIELDS}
    md = generate_checklist({k: v for k, v in answers.items() if v})
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"checklist written to {args.output}")
    else:
        print(md)
    return 0


def _iter_artifact_paths(paths):
    """Yield (path, from_directory) for every JSON file named by ``paths``.

    Directories are walked recursively; files are passed through verbatim
    so a non-artifact named explicitly is an error, not a silent skip.
    """
    import glob as _glob
    import os as _os

    for p in paths:
        if _os.path.isdir(p):
            pattern = _os.path.join(p, "**", "*.json")
            for fp in sorted(_glob.glob(pattern, recursive=True)):
                yield fp, True
        else:
            yield p, False


def _cmd_verify(args: argparse.Namespace) -> int:
    from .card import classify_artifact_dict, verify_artifact_dict

    n_ok = n_fail = n_skip = 0
    for path, from_dir in _iter_artifact_paths(args.cards):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"FAILED: {path} — unreadable ({e})")
            n_fail += 1
            continue
        kind = classify_artifact_dict(d)
        if kind == "unknown":
            if from_dir:  # badges, traces, raw dumps living next to cards
                n_skip += 1
                continue
            print(f"FAILED: {path} — not a verifiable StressKit artifact")
            n_fail += 1
            continue
        result = verify_artifact_dict(d)
        checks = (d.get("verdict", {}).get("checks")
                  if kind == "stability_card" else d.get("checks")) or {}
        if result["ok"]:
            print(f"OK: {path} — verdict {d['verdict']['grade']} re-derives "
                  f"from the {kind.replace('_', ' ')}'s own metrics "
                  f"({len(checks)} checks)")
            n_ok += 1
        else:
            print(f"FAILED: {path} — does not verify "
                  f"(recomputed grade {result['recomputed_grade']})")
            for problem in result["problems"]:
                print(f"  - {problem}")
            n_fail += 1

    total = n_ok + n_fail + n_skip
    if total == 0:
        print("FAILED: no JSON artifacts found")
        return 1
    if total > 1 or n_skip:
        print(f"\n{n_ok} verified, {n_fail} failed, "
              f"{n_skip} skipped (not cards/reports)")
    return 1 if n_fail else 0


def _cmd_scoreboard(args: argparse.Namespace) -> int:
    from .scoreboard import collect_rows, scoreboard_markdown, write_scoreboard

    if args.output:
        n = write_scoreboard(args.paths, args.output)
        print(f"scoreboard with {n} findings written to {args.output}")
    else:
        print(scoreboard_markdown(collect_rows(args.paths)))
    return 0


def _cmd_version(_: argparse.Namespace) -> int:
    from . import __version__

    print(f"stresskit {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stresskit",
        description="Stability harness for mechanistic interpretability claims.",
    )
    sub = p.add_subparsers(dest="command")

    pr = sub.add_parser("render", help="render a Stability Card as markdown")
    pr.add_argument("card", help="path to a stability card .json")
    pr.set_defaults(func=_cmd_render)

    pb = sub.add_parser("badge", help="emit shields.io endpoint JSON for a card")
    pb.add_argument("card", help="path to a stability card .json")
    pb.add_argument("-o", "--output", help="write JSON here instead of stdout")
    pb.set_defaults(func=_cmd_badge)

    pp = sub.add_parser("report", help="generate the Minimum Reporting Checklist")
    for key, title, why in CHECKLIST_FIELDS:
        pp.add_argument(f"--{key.replace('_', '-')}", dest=key, help=f"{title}: {why}")
    pp.add_argument("-o", "--output", help="write markdown here instead of stdout")
    pp.set_defaults(func=_cmd_report)

    pf = sub.add_parser(
        "verify",
        help="auditor mode: re-derive checks and grades from artifacts' own metrics",
    )
    pf.add_argument(
        "cards", nargs="+", metavar="card",
        help="stability card / oracle report .json files, or directories "
             "to scan recursively (non-artifact JSONs in directories are skipped)",
    )
    pf.set_defaults(func=_cmd_verify)

    ps = sub.add_parser(
        "scoreboard",
        help="render a markdown scoreboard of every card/report found",
    )
    ps.add_argument(
        "paths", nargs="+",
        help="card/report .json files or directories to scan recursively",
    )
    ps.add_argument("-o", "--output", help="write markdown here instead of stdout")
    ps.set_defaults(func=_cmd_scoreboard)

    pv = sub.add_parser("version", help="print version")
    pv.set_defaults(func=_cmd_version)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
