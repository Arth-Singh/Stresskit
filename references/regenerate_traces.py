"""Rebuild the verdict-stability traces of the committed diagnostic cards.

A trace (``<stem>.trace.json`` / ``.trace.md``) regrades random subsets of a
card's runs and reports how the grade settles with run count. The committed
traces were graded under rule v0.3; this rebuilds them under the current rule
from the runs the repository still holds: the real runs on the card (or, for
a hash-only card, in its manifest, checked against the card's digests) and
the null-control runs in the card's ``.runs.json`` manifest.

Step 0, before anything is written: every card whose runs are recoverable is
regraded at its recorded seed and every check value, interval and state is
compared with what the card recorded. Direction cards are reproduced through
their embedded |cosine| matrices instead. A card is only traced when the
reproduction is exact for every check the card records.

Cards named with ``--rerun-fresh`` are being re-run from scratch elsewhere;
they are listed as such and neither reproduced nor traced here.

Dry run by default: traces land under ``--out`` mirroring the paths under
``--root``; ``--apply`` writes them in place.

Usage:
    PYTHONPATH=src python3 references/regenerate_traces.py --root references --out /tmp/traces_dry
    PYTHONPATH=src python3 references/regenerate_traces.py --root references --out /tmp/traces --apply
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from stresskit import verdict_trace, verdict_trace_markdown
from stresskit.battery import grade_checks, make_check
from stresskit.card import _card_structure_kind
from stresskit.card_findings import (
    card_name,
    card_thresholds,
    findings_from_card_dict,
    findings_from_manifest_real_rows,
    is_diagnostic_stability_card,
    load_manifest,
    null_findings_from_manifest,
    regrade_direction_card,
    regrade_findings,
)
from stresskit.finding import Finding
from stresskit.scoreboard import collect_rows


def diagnostic_cards(root: str, only: Optional[str]) -> List[Tuple[str, int]]:
    """(path, n_runs) of every diagnostic stability card under ``root``,
    largest battery first so the slow traces start early."""
    cards = []
    for row in collect_rows([root]):
        if row["kind"] != "stability card":
            continue
        if only and only not in os.path.relpath(row["path"], root):
            continue
        with open(row["path"], encoding="utf-8") as f:
            card = json.load(f)
        if is_diagnostic_stability_card(card):
            cards.append((row["path"], len(card.get("runs") or [])))
    cards.sort(key=lambda pc: -pc[1])
    return cards


def recover_real(
    path: str, card: Dict[str, Any]
) -> Tuple[Optional[List[Finding]], Optional[List[str]], Optional[str]]:
    """(findings, axes, note) with the findings None when unrecoverable; the
    note says why, or how a hash-only card's runs were recovered."""
    name = os.path.basename(path)
    if card["battery"].get("components_embedded", True):
        try:
            findings, axes = findings_from_card_dict(card, name=name)
        except ValueError as e:
            return None, None, str(e)
        return findings, axes, None
    loaded = load_manifest(path)
    if loaded is None:
        return None, None, "hash-only card and no manifest to take the components from"
    try:
        findings, axes = findings_from_manifest_real_rows(card, loaded[0], name=name)
    except ValueError as e:
        return None, None, f"hash-only card; manifest rejected: {e}"
    return (
        findings,
        axes,
        "hash-only card: real runs taken from the manifest, digest-checked "
        "against the card",
    )


def recover_null(
    path: str, card: Dict[str, Any]
) -> Tuple[Optional[List[Finding]], Optional[str]]:
    """(null findings, None) or (None, reason); (None, None) when the card
    never had a null control."""
    null_block = card["metrics"].get("null_control")
    if null_block is None:
        return None, None
    loaded = load_manifest(path)
    if loaded is None:
        return None, "null runs missing: no .runs.json manifest next to the card"
    manifest, key = loaded
    try:
        nulls = null_findings_from_manifest(
            manifest, universe_size=card["claim"].get("universe_size"), key=key
        )
    except ValueError as e:
        return None, f"null runs missing: {e}"
    expected = null_block.get("n_runs")
    if expected is not None and len(nulls) != expected:
        return None, (
            f"null runs missing: manifest has {len(nulls)} null rows, the card "
            f"pooled {expected}"
        )
    return nulls, None


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def compare_checks(
    card: Dict[str, Any], fresh: Dict[str, Any], *, missing_reason: Optional[str]
) -> Dict[str, Any]:
    """Recorded checks vs freshly computed ones, field by field.

    A card older than schema 0.3 records no ``state``; the comparison then
    uses the state ``verify`` derives from the recorded interval."""
    recorded = card["verdict"]["checks"]
    per_check: Dict[str, str] = {}
    exact = 0
    for name, c in recorded.items():
        if name not in fresh:
            per_check[name] = (
                f"not recomputable ({missing_reason})"
                if missing_reason and name == "specificity"
                else "MISSING from the regrade"
            )
            continue
        f = fresh[name]
        state = (
            c.get("state")
            or make_check(c["value"], c["threshold"], c["op"], "", ci=c.get("ci"))[
                "state"
            ]
        )
        problems = []
        if not _same(f["value"], c["value"]):
            problems.append(f"value {c['value']!r} -> {f['value']!r}")
        if not _same(f["ci"], c.get("ci")):
            problems.append(f"ci {c.get('ci')!r} -> {f['ci']!r}")
        if f["state"] != state:
            problems.append(f"state {state!r} -> {f['state']!r}")
        if problems:
            per_check[name] = "MISMATCH: " + "; ".join(problems)
        else:
            per_check[name] = "exact"
            exact += 1
    extra = sorted(set(fresh) - set(recorded))
    return {
        "per_check": per_check,
        "n_exact": exact,
        "n_recorded": len(recorded),
        "extra_checks": extra,
        "all_exact": exact == len(recorded) and not extra,
    }


def finish(report: Dict[str, Any], t0: float, **fields: Any) -> Dict[str, Any]:
    report.update(fields)
    report["seconds"] = round(time.time() - t0, 1)
    return report


def trace_digest(trace: Dict[str, Any]) -> Dict[str, Any]:
    """settled_n and the modal grade at n = 6; sizes are keyed by int in a
    fresh trace and by string once it has been through JSON."""
    per_size = {str(k): v for k, v in trace.get("per_size", {}).items()}
    return {
        "full_grade": trace.get("full_grade"),
        "settled_n": trace.get("settled_n"),
        "modal_grade_at_6": (per_size.get("6") or {}).get("modal_grade"),
    }


def reproduce_and_trace(
    job: Dict[str, Any], card: Dict[str, Any], report: Dict[str, Any], t0: float
) -> Dict[str, Any]:
    path, stem, dest_stem = job["path"], job["stem"], job["dest_stem"]
    battery_seed = card["battery"]["seed"]
    thresholds = card_thresholds(card)

    if report["kind"] == "direction":
        fresh = regrade_direction_card(card, seed=battery_seed)
        return finish(
            report,
            t0,
            reproduction=compare_checks(card, fresh["checks"], missing_reason=None),
            grade_reproduced=(
                fresh["grade_v04"]
                if report["rule_recorded"] == "v0.4"
                else fresh["grade_v03"]
            )
            == card["verdict"]["grade"],
            grade_v04_from_matrix=fresh["grade_v04"],
            recoverable=False,
            reason=(
                "direction card: the runs carry digests of their vectors, not "
                "the vectors, so subsets cannot be regraded; checks reproduced "
                "from the embedded |cosine| matrices instead"
            ),
        )

    findings, axes, real_note = recover_real(path, card)
    if findings is None:
        return finish(report, t0, recoverable=False, reason=real_note)
    nulls, null_reason = recover_null(path, card)
    notes = [n for n in (real_note, null_reason) if n]

    result = regrade_findings(
        card, findings, axes, seed=battery_seed, null_findings=nulls
    )
    report["reproduction"] = compare_checks(
        card, result.checks, missing_reason=null_reason
    )
    report["n_null_runs"] = len(nulls) if nulls else 0
    if not null_reason:
        rederived = grade_checks(
            result.checks,
            rule=report["rule_recorded"],
            random_floor=thresholds.random_floor,
        )
        report["grade_reproduced"] = rederived == card["verdict"]["grade"]
        report["grade_v04_from_runs"] = result.grade

    if null_reason:
        return finish(report, t0, recoverable=False, reason="; ".join(notes))
    if not report["reproduction"]["all_exact"]:
        return finish(
            report,
            t0,
            recoverable=False,
            reason="regrade does not reproduce the recorded checks; not traced",
        )

    trace = verdict_trace(
        findings, null_findings=nulls, seed=job["seed"], thresholds=thresholds
    )
    os.makedirs(os.path.dirname(dest_stem) or ".", exist_ok=True)
    with open(dest_stem + ".trace.json", "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
        f.write("\n")
    with open(dest_stem + ".trace.md", "w", encoding="utf-8") as f:
        f.write(verdict_trace_markdown(trace) + "\n")
    old = None
    if os.path.exists(stem + ".trace.json"):
        with open(stem + ".trace.json", encoding="utf-8") as f:
            old = trace_digest(json.load(f))
    return finish(
        report,
        t0,
        recoverable=True,
        reason="; ".join(notes)
        or (
            "runs on the card, null runs in the manifest"
            if nulls
            else "runs on the card, no null control"
        ),
        written=[dest_stem + ".trace.json", dest_stem + ".trace.md"],
        new=trace_digest(trace),
        old=old,
    )


def process_card(job: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    with open(job["path"], encoding="utf-8") as f:
        card = json.load(f)
    report: Dict[str, Any] = {
        "card": job["card"],
        "path": job["path"],
        "name": card_name(card),
        "schema": card["schema_version"],
        "grade_recorded": card["verdict"]["grade"],
        "rule_recorded": card["verdict"].get("grade_rule", "v0.3"),
        "kind": _card_structure_kind(card),
        "n_runs": len(card.get("runs") or []),
        "has_null_block": card["metrics"].get("null_control") is not None,
        "committed_trace": os.path.exists(job["stem"] + ".trace.json"),
    }
    if job["rerun_fresh"]:
        return finish(
            report,
            t0,
            recoverable=False,
            reason="re-run fresh: being re-run from scratch under the current "
            "rule elsewhere; not reproduced or traced from the committed runs",
        )
    try:
        return reproduce_and_trace(job, card, report, t0)
    except ValueError as e:
        return finish(report, t0, recoverable=False, error=True, reason=f"ERROR: {e}")


def print_reports(reports: List[Dict[str, Any]]) -> None:
    width = max(len(r["card"]) for r in reports)
    print("step 0 — regrade at the recorded seed vs the recorded checks")
    for r in reports:
        rep = r.get("reproduction")
        if rep is None:
            print(f"  {r['card']:<{width}}  not attempted: {r['reason']}")
            continue
        summary = f"{rep['n_exact']}/{rep['n_recorded']} checks exact"
        if rep["extra_checks"]:
            summary += f"; extra checks in the regrade: {rep['extra_checks']}"
        if "grade_reproduced" in r:
            summary += (
                "; recorded grade re-derives"
                if r["grade_reproduced"]
                else "; RECORDED GRADE DOES NOT RE-DERIVE"
            )
        print(f"  {r['card']:<{width}}  {summary}")
        for name, status in rep["per_check"].items():
            if status != "exact":
                print(f"  {'':<{width}}    {name}: {status}")
    print()
    recoverable = [r for r in reports if r.get("recoverable")]
    print(f"recoverable ({len(recoverable)}):")
    for r in recoverable:
        old, new = r.get("old"), r["new"]
        line = (
            f"  {r['card']:<{width}}  n={r['n_runs']} null={r.get('n_null_runs', 0)}  "
            f"grade {r['grade_recorded']} ({r['rule_recorded']}) -> {new['full_grade']}  "
        )
        if old:
            line += (
                f"settled_n {old['settled_n']} -> {new['settled_n']}  "
                f"modal@6 {old['modal_grade_at_6']} -> {new['modal_grade_at_6']}"
            )
        else:
            line += (
                f"settled_n {new['settled_n']} modal@6 {new['modal_grade_at_6']} "
                "(no committed trace)"
            )
        print(line + f"  [{r['seconds']}s] {r['reason']}")
    print()
    skipped = [r for r in reports if not r.get("recoverable")]
    print(f"not recoverable ({len(skipped)}):")
    for r in skipped:
        trace_note = " (committed trace stays)" if r["committed_trace"] else ""
        print(f"  {r['card']:<{width}}  {r['reason']}{trace_note}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default="references")
    parser.add_argument("--out", required=True, help="output directory for the dry run")
    parser.add_argument("--apply", action="store_true", help="write traces in place")
    parser.add_argument(
        "--only", default=None, help="substring filter on the card path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="verdict_trace seed (the committed traces used 0)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="cards traced in parallel",
    )
    parser.add_argument(
        "--rerun-fresh",
        default="",
        help="comma-separated card path substrings being re-run from scratch "
        "elsewhere; listed but neither reproduced nor traced",
    )
    args = parser.parse_args(argv)
    fresh = [s for s in args.rerun_fresh.split(",") if s]

    t0 = time.time()
    cards = diagnostic_cards(args.root, args.only)
    if not cards:
        print(f"no diagnostic stability cards under {args.root!r}", file=sys.stderr)
        return 1
    jobs = []
    for path, _ in cards:
        rel = os.path.relpath(path, args.root)
        stem = os.path.splitext(path)[0]
        jobs.append(
            {
                "path": path,
                "card": os.path.splitext(rel)[0],
                "stem": stem,
                "dest_stem": stem
                if args.apply
                else os.path.splitext(os.path.join(args.out, rel))[0],
                "seed": args.seed,
                "rerun_fresh": any(s in rel for s in fresh),
            }
        )
    if args.jobs > 1 and len(jobs) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            reports = list(pool.map(process_card, jobs))
    else:
        reports = [process_card(j) for j in jobs]
    reports.sort(key=lambda r: r["card"])

    print_reports(reports)
    os.makedirs(args.out, exist_ok=True)
    with open(
        os.path.join(args.out, "trace-regeneration.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "root": args.root,
                "seed": args.seed,
                "applied": args.apply,
                "rerun_fresh": fresh,
                "cards": reports,
                "wall_seconds": round(time.time() - t0, 1),
            },
            f,
            indent=2,
        )
        f.write("\n")
    n_written = sum(1 for r in reports if r.get("written"))
    print()
    print(
        f"{n_written} trace(s) written {'in place' if args.apply else 'to ' + args.out} "
        f"in {time.time() - t0:.0f}s with {args.jobs} job(s)"
    )
    errors = [r["card"] for r in reports if r.get("error")]
    mismatched = [
        r["card"]
        for r in reports
        if r.get("reproduction")
        and any(
            s.startswith("MISMATCH") or s.startswith("MISSING")
            for s in r["reproduction"]["per_check"].values()
        )
    ]
    if errors:
        print(f"errors on: {', '.join(errors)}")
    if mismatched:
        print(f"step 0 mismatches on: {', '.join(mismatched)}")
    return 1 if errors or mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
