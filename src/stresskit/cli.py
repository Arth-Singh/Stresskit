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
    if args.html:
        from .htmlcard import render_html_path

        page = render_html_path(args.card)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(page)
            print(f"HTML card written to {args.output}")
        else:
            print(page)
        return 0
    card = StabilityCard.load(args.card)
    md = card.to_markdown()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"markdown written to {args.output}")
    else:
        print(md)
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
        try:
            result = verify_artifact_dict(d)
        except ValueError as e:
            # One unverifiable artifact must not abort the batch: an auditor
            # running `stresskit verify` over a directory needs a verdict for
            # every other card in it. Counted as a failure, never skipped.
            print(f"FAILED: {path} — cannot be verified ({e})")
            n_fail += 1
            continue
        checks = (d.get("verdict", {}).get("checks")
                  if kind == "stability_card" else d.get("checks")) or {}
        if result["ok"]:
            verdict = (
                d["verdict"].get("grade")
                if kind != "confirmatory_card"
                else d["verdict"].get("state")
            )
            print(f"OK: {path} — verdict {verdict} re-derives "
                  f"from the {kind.replace('_', ' ')}'s own metrics "
                  f"({len(checks)} checks)")
            n_ok += 1
        else:
            recomputed = result.get(
                "recomputed_grade", result.get("recomputed_state", "unknown")
            )
            print(f"FAILED: {path} — does not verify "
                  f"(recomputed verdict {recomputed})")
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


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    run_demo(html_dir=args.html)
    return 0


def _cmd_trace(args: argparse.Namespace) -> int:
    from .tracechart import trace_svg_path

    svg = trace_svg_path(args.trace, title=args.title)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(svg + "\n")
        print(f"trace chart written to {args.output}")
    else:
        print(svg)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare_markdown, compare_paths

    try:
        cmp = compare_paths(args.baseline, args.candidate)
    except ValueError as e:
        print(f"FAILED: {e}")
        return 1
    print(compare_markdown(cmp))
    if args.fail_on_regression and cmp["regressed"]:
        return 1
    return 0


def _cmd_scoreboard(args: argparse.Namespace) -> int:
    from .scoreboard import (collect_rows, registered_paper_rows,
                             scoreboard_markdown, write_scoreboard)

    if args.output:
        n = write_scoreboard(args.paths, args.output, papers_path=args.papers)
        print(f"scoreboard with {n} findings written to {args.output}")
    else:
        rows = collect_rows(args.paths)
        papers = registered_paper_rows(args.paths, rows, args.papers)
        print(scoreboard_markdown(rows, papers=papers))
    return 0


def _cmd_site(args: argparse.Namespace) -> int:
    from .site import build_site

    n = build_site(args.paths, args.output, repo_url=args.repo_url,
                   papers_path=args.papers)
    print(f"site with {n} card pages written to {args.output}/")
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

    from .audit_cli import add_audit_parser

    add_audit_parser(sub)

    pr = sub.add_parser(
        "render", help="render a Stability Card as markdown or HTML")
    pr.add_argument("card", help="path to a stability card .json")
    pr.add_argument(
        "--html", action="store_true",
        help="emit a self-contained shareable HTML page instead of markdown")
    pr.add_argument("-o", "--output", help="write here instead of stdout")
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
        help="auditor mode: re-derive metrics, intervals, checks, and verdicts",
    )
    pf.add_argument(
        "cards", nargs="+", metavar="card",
        help="stability / confirmatory card or oracle report .json files, or directories "
             "to scan recursively (non-artifact JSONs in directories are skipped)",
    )
    pf.set_defaults(func=_cmd_verify)

    pd = sub.add_parser(
        "demo",
        help="30-second demo: one method graded on a real effect vs pure noise",
    )
    pd.add_argument(
        "--html", metavar="DIR",
        help="also write both stability cards as HTML pages into DIR")
    pd.set_defaults(func=_cmd_demo)

    pt = sub.add_parser(
        "trace",
        help="render a verdict trace as an SVG chart (grade shares vs n)",
    )
    pt.add_argument("trace", help="path to a verdict trace .json")
    pt.add_argument("--title", help="chart title override")
    pt.add_argument("-o", "--output", help="write SVG here instead of stdout")
    pt.set_defaults(func=_cmd_trace)

    pc = sub.add_parser(
        "compare",
        help="stability regression test: diff two cards (baseline first)",
    )
    pc.add_argument("baseline", help="baseline card/report .json")
    pc.add_argument("candidate", help="candidate card/report .json")
    pc.add_argument(
        "--fail-on-regression", action="store_true",
        help="exit 1 when a check flips pass→fail or the grade drops "
             "(for CI gates)",
    )
    pc.set_defaults(func=_cmd_compare)

    ps = sub.add_parser(
        "scoreboard",
        help="render legacy diagnostic card inventory (not v1 evidence)",
    )
    ps.add_argument(
        "paths", nargs="+",
        help="card/report .json files or directories to scan recursively",
    )
    ps.add_argument("-o", "--output", help="write markdown here instead of stdout")
    ps.add_argument("--papers", metavar="JSON",
                    help="paper registry for the leaderboard (default: a "
                         "papers.json inside one of the given directories)")
    ps.set_defaults(func=_cmd_scoreboard)

    pw = sub.add_parser(
        "site",
        help="build a static site (index + card pages + trace charts) "
             "from directories of cards",
    )
    pw.add_argument("paths", nargs="+",
                    help="card/report .json files or directories")
    pw.add_argument("-o", "--output", default="_site",
                    help="output directory (default _site)")
    pw.add_argument("--repo-url",
                    default="https://github.com/Arth-Singh/Stresskit",
                    help="repository URL used in links")
    pw.add_argument("--papers", metavar="JSON",
                    help="paper registry for the leaderboard panel (default: a "
                         "papers.json inside one of the given directories)")
    pw.set_defaults(func=_cmd_site)

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
