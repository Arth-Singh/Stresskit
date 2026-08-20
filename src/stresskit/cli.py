"""StressKit command line.

Commands
--------
stresskit render <card.json>          render a Stability Card as markdown
stresskit badge  <card.json> [-o f]   emit shields.io endpoint JSON for the badge
stresskit report [--field value ...]  generate the Minimum Reporting Checklist
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
