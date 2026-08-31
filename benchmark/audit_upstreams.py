#!/usr/bin/env python3
"""Verify pinned upstream source without importing or executing it.

This is a provenance/static-syntax audit, not an upstream reproduction smoke.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(__file__).with_name("upstream_sources.json")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_python_files(repo: Path) -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "*.py"],
        check=True,
        capture_output=True,
    ).stdout
    return [repo / item.decode() for item in output.split(b"\0") if item]


def audit_upstream(repo: Path, expected: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if commit != expected["commit"]:
        errors.append(f"commit: expected {expected['commit']}, got {commit}")
    if tree != expected["tree"]:
        errors.append(f"tree: expected {expected['tree']}, got {tree}")

    missing_entrypoints = [
        path for path in expected["entrypoint_paths"] if not (repo / path).exists()
    ]
    if missing_entrypoints:
        errors.append(f"missing entrypoints: {missing_entrypoints}")

    license_record = expected["license"]
    license_path = license_record.get("path")
    if license_path is not None:
        resolved = repo / license_path
        if not resolved.is_file():
            errors.append(f"missing license file: {license_path}")
        else:
            observed_hash = _sha256(resolved)
            if observed_hash != license_record["sha256"]:
                errors.append(
                    f"license hash: expected {license_record['sha256']}, "
                    f"got {observed_hash}"
                )
    elif license_record["status"] != "not_found":
        errors.append("null license path requires status=not_found")

    python_files = _tracked_python_files(repo)
    syntax_errors: list[str] = []
    syntax_warnings: list[str] = []
    for path in python_files:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", SyntaxWarning)
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            syntax_warnings.extend(
                f"{path.relative_to(repo)}:{warning.lineno}: {warning.message}"
                for warning in caught
                if issubclass(warning.category, SyntaxWarning)
            )
        except (SyntaxError, UnicodeDecodeError) as exc:
            syntax_errors.append(f"{path.relative_to(repo)}: {exc}")
    expected_count = expected["static_python_syntax"]["tracked_files"]
    if len(python_files) != expected_count:
        errors.append(
            f"tracked Python count: expected {expected_count}, got {len(python_files)}"
        )
    if syntax_errors:
        errors.append(f"Python syntax errors: {syntax_errors}")

    return {
        "commit": commit,
        "tree": tree,
        "entrypoints_checked": len(expected["entrypoint_paths"]),
        "missing_entrypoints": missing_entrypoints,
        "python_files_parsed": len(python_files),
        "python_syntax_errors": syntax_errors,
        "python_syntax_warnings": syntax_warnings,
        "ok": not errors,
        "errors": errors,
    }


def audit_manifest(
    clone_root: Path,
    manifest: dict[str, Any],
    upstreams: list[str] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    selected = upstreams or list(manifest["upstreams"])
    unknown = sorted(set(selected) - set(manifest["upstreams"]))
    if unknown:
        raise ValueError(f"unknown upstreams: {unknown}")
    for name in selected:
        expected = manifest["upstreams"][name]
        repo = clone_root / name
        if not repo.is_dir():
            results[name] = {
                "ok": False,
                "errors": [f"missing checkout: {repo}"],
            }
            continue
        results[name] = audit_upstream(repo, expected)
    return {
        "artifact_type": "stresskit_upstream_static_audit_result",
        "manifest_schema_version": manifest["schema_version"],
        "not_execution_smoke": True,
        "all_ok": all(row["ok"] for row in results.values()),
        "upstreams": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--upstream", action="append", dest="upstreams")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = audit_manifest(args.clone_root, manifest, args.upstreams)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
