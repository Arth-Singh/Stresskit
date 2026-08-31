#!/usr/bin/env python3
"""Pin one upstream checkout into an ``upstream_sources.json`` manifest row.

Records commit, tree, license file (path, SPDX, SHA-256), registered
entrypoint paths, and static Python-syntax status for a local git checkout,
so that ``audit_upstreams.py`` can later re-verify the identical tree from an
independent clone. This is provenance evidence only: it does not install
dependencies, load models, or reproduce any result.

Usage::

    python benchmark/pin_upstream.py --name persona_vectors \
        --checkout /path/to/clone \
        --repository https://github.com/safety-research/persona_vectors \
        --entrypoint generate_vec.py --entrypoint activation_steer.py \
        [--spdx Apache-2.0] [--merge-into benchmark/upstream_sources.json]

Without ``--merge-into`` the row is printed as JSON. With it, the row is
inserted (or replaced) under ``upstreams`` and the manifest is rewritten with
sorted keys, two-space indentation, and a trailing newline.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

LICENSE_CANDIDATES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE.rst", "COPYING")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guess_spdx(text: str) -> Optional[str]:
    """Map common license texts to SPDX ids; return None when unsure."""
    head = text[:4000]
    if "Apache License" in head and "2.0" in head:
        return "Apache-2.0"
    if "MIT License" in head or "Permission is hereby granted, free of charge" in head:
        return "MIT"
    if "BSD" in head and "Redistribution and use in source and binary forms" in head:
        return "BSD-3-Clause" if "Neither the name" in head else "BSD-2-Clause"
    if "GNU GENERAL PUBLIC LICENSE" in head:
        return "GPL-3.0-only" if "Version 3" in head else "GPL-2.0-only"
    return None


def find_license(repo: Path) -> Optional[Path]:
    lowered = {p.name.lower(): p for p in repo.iterdir() if p.is_file()}
    for name in LICENSE_CANDIDATES:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def pin(repo: Path, repository: str, entrypoints: List[str], spdx: Optional[str]) -> Dict[str, Any]:
    missing = [p for p in entrypoints if not (repo / p).exists()]
    if missing:
        raise SystemExit(f"entrypoint paths do not exist in checkout: {missing}")
    if _git(repo, "status", "--porcelain"):
        raise SystemExit(f"checkout is dirty; pin a clean tree: {repo}")

    license_path = find_license(repo)
    if license_path is None:
        license_row: Dict[str, Any] = {
            "status": "not_found", "spdx": None, "path": None, "sha256": None,
        }
    else:
        text = license_path.read_text(encoding="utf-8", errors="replace")
        resolved = spdx or guess_spdx(text)
        if resolved is None:
            raise SystemExit(
                f"could not identify SPDX id for {license_path}; pass --spdx explicitly"
            )
        license_row = {
            "status": "found",
            "spdx": resolved,
            "path": license_path.name,
            "sha256": _sha256(license_path),
        }

    listed = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "*.py"], check=True, capture_output=True
    ).stdout
    py_files = [repo / item.decode() for item in listed.split(b"\0") if item]
    errors = []
    for path in py_files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"{path.relative_to(repo)}: {exc}")
    if not py_files:
        raise SystemExit(f"no tracked Python files in {repo}; manifest requires > 0")

    return {
        "repository": repository,
        "commit": _git(repo, "rev-parse", "HEAD"),
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "license": license_row,
        "entrypoint_paths": list(entrypoints),
        "static_python_syntax": {
            "status": "pass" if not errors else "fail",
            "tracked_files": len(py_files),
            **({"errors": errors} if errors else {}),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True, help="manifest key, e.g. persona_vectors")
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--repository", required=True, help="canonical https clone URL")
    parser.add_argument("--entrypoint", action="append", required=True, dest="entrypoints")
    parser.add_argument("--spdx", default=None, help="override SPDX id when detection fails")
    parser.add_argument("--merge-into", type=Path, default=None)
    args = parser.parse_args()

    row = pin(args.checkout, args.repository, args.entrypoints, args.spdx)
    if args.merge_into is None:
        print(json.dumps({args.name: row}, indent=2))
        return 0
    manifest = json.loads(args.merge_into.read_text())
    manifest["upstreams"][args.name] = row
    manifest["upstreams"] = dict(sorted(manifest["upstreams"].items()))
    args.merge_into.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.name}: {row['commit']} license={row['license']['spdx']} "
          f"py={row['static_python_syntax']['tracked_files']} -> {args.merge_into}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
