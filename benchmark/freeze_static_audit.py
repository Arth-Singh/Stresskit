#!/usr/bin/env python3
"""Run the static upstream audit over every pinned checkout and freeze the result.

Wraps ``audit_upstreams.audit_manifest`` into the artifact format stored under
``artifacts/benchmark/upstream-static-audit-<date>.json``: per-upstream row
counts, a summary block, the SHA-256 of the manifest and auditor that produced
it, and the interpreter/platform used. The artifact is bound to the manifest by
hash, so it must be regenerated whenever ``upstream_sources.json`` changes.

Provenance only. It does not install dependencies, load models, run upstream
commands, or reproduce any claim.

Usage::

    python benchmark/freeze_static_audit.py --clone-root /path/to/checkouts \
        --out artifacts/benchmark/upstream-static-audit-20260831.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_upstreams import audit_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("benchmark/upstream_sources.json")
AUDITOR = Path("benchmark/audit_upstreams.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(clone_root: Path) -> dict:
    manifest = json.loads((REPO_ROOT / MANIFEST).read_text())
    result = audit_manifest(clone_root, manifest)
    rows = result["upstreams"]
    if not result["all_ok"]:
        failing = {k: v["errors"] for k, v in rows.items() if not v["ok"]}
        raise SystemExit(f"static audit failed; not freezing: {json.dumps(failing, indent=2)}")

    warnings = [w for row in rows.values() for w in row["python_syntax_warnings"]]
    licenses = [manifest["upstreams"][k]["license"]["status"] for k in rows]
    missing = sorted(k for k in rows if manifest["upstreams"][k]["license"]["status"] != "found")
    return {
        "artifact_type": "stresskit_upstream_static_audit_result",
        "schema_version": manifest["schema_version"],
        "checked_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(),
                        "platform": f"{platform.system()} {platform.release()} {platform.machine()}"},
        "inputs": {"manifest": str(MANIFEST), "manifest_sha256": _sha256(REPO_ROOT / MANIFEST),
                   "auditor": str(AUDITOR), "auditor_sha256": _sha256(REPO_ROOT / AUDITOR)},
        "not_execution_smoke": True,
        "all_ok": True,
        "summary": {
            "upstreams": len(rows),
            "pinned_commits_verified": len(rows),
            "git_trees_verified": len(rows),
            "entrypoint_paths_verified": sum(r["entrypoints_checked"] for r in rows.values()),
            "tracked_python_files_parsed": sum(r["python_files_parsed"] for r in rows.values()),
            "syntax_errors": sum(len(r["python_syntax_errors"]) for r in rows.values()),
            "syntax_warnings": len(warnings),
            "source_licenses_found": licenses.count("found"),
            "source_licenses_missing": len(missing),
        },
        "upstreams": {k: {"ok": r["ok"], "entrypoints": r["entrypoints_checked"],
                          "python_files": r["python_files_parsed"],
                          "syntax_warnings": len(r["python_syntax_warnings"])}
                      for k, r in sorted(rows.items())},
        "warnings": warnings,
        "interpretation": (
            "All pinned source trees were present, their registered entrypoint paths existed, and every "
            "tracked Python file parsed. This is not dependency-install, model-load, command-execution, or "
            "claim-reproduction evidence. Upstreams without a repository-level source license remain "
            f"excluded before freeze: {', '.join(missing) or 'none'}."),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    artifact = freeze(args.clone_root)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    s = artifact["summary"]
    print(f"{args.out}: {s['upstreams']} upstreams, {s['entrypoint_paths_verified']} entrypoints, "
          f"{s['tracked_python_files_parsed']} py files, {s['syntax_warnings']} warnings, "
          f"{s['source_licenses_missing']} missing licenses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
