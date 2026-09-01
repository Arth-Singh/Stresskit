"""Outcome-blind August 2026 pass 3b: omitted terms plus Tier-B code audit.

Reads only discovery frame and public source metadata.  It never opens
StressKit result, card, score, or verdict artifacts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


OMITTED_TERMS = (
    "chain of thought",
    "reasoning trace",
    "persona",
    "introspection",
    "evaluation awareness",
    "model organism",
)

ALLOWED_CATEGORIES = {
    "cs.LG", "cs.CL", "cs.AI", "cs.CV", "cs.NE", "stat.ML",
    "cs.CY", "cs.CR", "cs.SE", "cs.IR", "cs.MA",
}

_ATOM = "{http://www.w3.org/2005/Atom}"
_GITHUB = re.compile(r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")
_HF = re.compile(r"https?://huggingface\.co/[A-Za-z0-9_.~/-]+")
_TAG = re.compile(r"<[^>]+>")
_RELEASE_LANGUAGE = re.compile(
    r"\b(our\s+(code|implementation|repository)|code\s+is\s+(publicly\s+)?available|"
    r"implementation\s+is\s+available|project\s+(page|repository)|github\s+repository)\b",
    re.IGNORECASE,
)
_DEPENDENCY_LANGUAGE = re.compile(
    r"\b(we\s+(use|build\s+on|adopt)|based\s+on|dependency|library|framework|"
    r"transformer.?lens|sae.?lens|pyvene)\b",
    re.IGNORECASE,
)


def _request(url: str, *, accept: str = "*/*", timeout: float = 30.0) -> bytes:
    headers = {
        "User-Agent": "StressKit/1.0 outcome-blind-discovery contact=repository-maintainer",
        "Accept": accept,
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def query_arxiv(term: str) -> List[Dict[str, Any]]:
    """Query official Atom API for one exact term and fixed submission window."""
    query = f'all:"{term}" AND submittedDate:[202608010000 TO 202608312359]'
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": 500,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    root = ET.fromstring(_request(url, accept="application/atom+xml"))
    rows = []
    for entry in root.findall(_ATOM + "entry"):
        identifier = (entry.findtext(_ATOM + "id") or "").rsplit("/", 1)[-1]
        identifier = re.sub(r"v\d+$", "", identifier)
        categories = [node.attrib.get("term", "")
                      for node in entry.findall(_ATOM + "category")]
        if not (set(categories) & ALLOWED_CATEGORIES):
            continue
        rows.append({
            "arxiv_id": identifier,
            "title": " ".join((entry.findtext(_ATOM + "title") or "").split()),
            "published": (entry.findtext(_ATOM + "published") or "")[:10],
            "categories": categories,
        })
    return rows


def _clean_repo(url_match: re.Match) -> str:
    owner, repository = url_match.group(1), url_match.group(2)
    repository = repository.rstrip(".,);]}'\"")
    if repository.endswith(".git"):
        repository = repository[:-4]
    return f"https://github.com/{owner}/{repository}"


def _github_metadata(repository_url: str) -> Dict[str, Any]:
    owner_repo = repository_url.split("github.com/", 1)[1]
    url = "https://api.github.com/repos/" + owner_repo
    try:
        payload = json.loads(_request(url, accept="application/vnd.github+json"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
            json.JSONDecodeError) as exc:
        return _github_html_fallback(repository_url, type(exc).__name__)
    license_row = payload.get("license") or {}
    return {
        "repository": repository_url,
        "status": "public",
        "default_branch": payload.get("default_branch"),
        "head_commit_at_audit": payload.get("pushed_at"),
        "license_spdx": license_row.get("spdx_id"),
        "license_status": (
            "found" if license_row.get("spdx_id") not in (None, "NOASSERTION")
            else "not_found"
        ),
        "created_at": payload.get("created_at"),
        "pushed_at": payload.get("pushed_at"),
        "archived": payload.get("archived") is True,
    }


def _license_from_text(text: str) -> Optional[str]:
    lowered = text.casefold()
    if "apache license" in lowered and "version 2.0" in lowered:
        return "Apache-2.0"
    if "permission is hereby granted, free of charge" in lowered:
        return "MIT"
    if "gnu general public license" in lowered:
        return "GPL"
    if "mozilla public license" in lowered:
        return "MPL"
    if "redistribution and use in source and binary forms" in lowered:
        return "BSD"
    if "isc license" in lowered:
        return "ISC"
    return None


def _github_html_fallback(repository_url: str, api_error: str) -> Dict[str, Any]:
    """Audit existence/license without consuming GitHub API quota."""
    try:
        page = _request(repository_url, accept="text/html").decode("utf-8", "replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return {"repository": repository_url, "status": "github_fetch_error",
                "api_error": api_error, "error": type(exc).__name__}
    branch_match = re.search(r'"defaultBranch":"([^"]+)"', page)
    branch = branch_match.group(1) if branch_match else "main"
    owner_repo = repository_url.split("github.com/", 1)[1]
    license_spdx = None
    license_path = None
    for candidate_branch in dict.fromkeys((branch, "main", "master")):
        for filename in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
            raw_url = (
                f"https://raw.githubusercontent.com/{owner_repo}/"
                f"{candidate_branch}/{filename}"
            )
            try:
                license_text = _request(raw_url, accept="text/plain").decode(
                    "utf-8", "replace"
                )
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                continue
            license_spdx = _license_from_text(license_text)
            license_path = filename
            branch = candidate_branch
            break
        if license_path:
            break
    return {
        "repository": repository_url,
        "status": "public",
        "default_branch": branch,
        "license_spdx": license_spdx,
        "license_path": license_path,
        "license_status": "found" if license_spdx else "not_found",
        "metadata_source": "github_html_and_raw_license_fallback",
        "api_error": api_error,
    }


def audit_code(arxiv_id: str) -> Dict[str, Any]:
    """Audit repository links and licenses without reading StressKit outcomes."""
    source_kind = "html"
    try:
        raw = _request(f"https://arxiv.org/html/{arxiv_id}", accept="text/html")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        source_kind = "abstract"
        try:
            raw = _request(f"https://arxiv.org/abs/{arxiv_id}", accept="text/html")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            return {"status": "manual_review_fetch_failure", "source": source_kind,
                    "error": type(exc).__name__,
                    "disposition": "manual_review_before_freeze"}
    page = raw.decode("utf-8", "replace")
    repositories = sorted({_clean_repo(match) for match in _GITHUB.finditer(page)})
    authored = []
    dependencies = []
    for repository in repositories:
        location = page.lower().find(repository.lower())
        context_html = page[max(0, location - 500): location + len(repository) + 500]
        context = html.unescape(_TAG.sub(" ", context_html))
        if _RELEASE_LANGUAGE.search(context) and not _DEPENDENCY_LANGUAGE.search(context):
            authored.append(repository)
        else:
            dependencies.append(repository)
    huggingface = sorted(set(_HF.findall(page)))
    if not authored:
        if repositories:
            return {
                "status": "dependency_links_only",
                "source": source_kind,
                "cited_repositories": repositories,
                "huggingface_refs": huggingface,
                "disposition": "excluded_dependency_links_only",
            }
        if huggingface:
            return {"status": "huggingface_refs_only", "source": source_kind,
                    "huggingface_refs": huggingface,
                    "disposition": "excluded_no_authored_source_repository"}
        return {"status": "no_public_code_found", "source": source_kind,
                "disposition": "excluded_no_public_code_found"}
    audits = [_github_metadata(repository) for repository in authored]
    licensed = [row for row in audits if row.get("license_status") == "found"]
    errors = [row for row in audits if row.get("status") != "public"]
    if licensed:
        disposition = "eligible_for_claim_mapping_and_smoke"
    elif errors:
        disposition = "manual_review_before_freeze"
    else:
        disposition = "excluded_repository_license_unresolved"
    return {
        "status": "public_repo",
        "source": source_kind,
        "authored_repositories": audits,
        "cited_repositories": dependencies,
        "huggingface_refs": huggingface,
        "disposition": disposition,
    }


def compact_code_audit(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep exact disposition while digesting potentially huge cited-link lists."""
    output = {
        key: row[key] for key in ("status", "source", "disposition", "error")
        if key in row
    }
    authored = row.get("authored_repositories")
    if isinstance(authored, list):
        output["authored_repositories"] = authored
    for key in ("cited_repositories", "huggingface_refs"):
        values = row.get(key)
        if isinstance(values, list):
            output[key + "_count"] = len(values)
            output[key + "_sha256"] = __import__("hashlib").sha256(
                json.dumps(sorted(values), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
    return output


def _ledger_code(row: Mapping[str, Any]) -> List[Any]:
    """Encode one complete disposition compactly; full link sets stay digested."""
    public = row.get("authored_repositories", [])
    public_summary = [
        [value.get("repository"), value.get("license_spdx"),
         value.get("license_status"), value.get("status")]
        for value in public if isinstance(value, Mapping)
    ] if isinstance(public, list) else []
    return [
        row.get("status"), row.get("disposition"),
        row.get("cited_repositories_count", 0),
        row.get("cited_repositories_sha256"),
        row.get("huggingface_refs_count", 0),
        row.get("huggingface_refs_sha256"),
        public_summary,
        __import__("hashlib").sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        ).hexdigest(),
    ]


def run_pass3b(base_frame: Mapping[str, Any], *, request_delay: float = 3.0,
               workers: int = 6) -> Dict[str, Any]:
    """Run omitted-term search and code-audit every retained Tier-B row."""
    base_tier_a = {row["arxiv_id"]: row for row in base_frame["papers"]}
    base_tier_b = {row["arxiv_id"]: row for row in base_frame["tier_b_papers"]}
    term_rows: Dict[str, Dict[str, Any]] = {}
    term_counts = {}
    for index, term in enumerate(OMITTED_TERMS):
        if index and request_delay:
            time.sleep(request_delay)
        rows = query_arxiv(term)
        term_counts[term] = len(rows)
        for row in rows:
            existing = term_rows.setdefault(row["arxiv_id"], dict(row))
            existing.setdefault("matched_omitted_terms", []).append(term)
    promoted = set(term_rows) & set(base_tier_b)
    new_tier_a = set(term_rows) - set(base_tier_a) - set(base_tier_b)
    retained_tier_b = set(base_tier_b) - promoted
    audit_ids = sorted(retained_tier_b | promoted | new_tier_a)
    code_by_id = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(audit_code, identifier): identifier
                   for identifier in audit_ids}
        for future in concurrent.futures.as_completed(futures):
            identifier = futures[future]
            try:
                code_by_id[identifier] = future.result()
            except Exception as exc:  # preserve row for manual audit
                code_by_id[identifier] = {
                    "status": "manual_review_fetch_failure",
                    "error": type(exc).__name__,
                    "disposition": "manual_review_before_freeze",
                }
    omitted_rows = []
    for identifier, row in sorted(term_rows.items()):
        if identifier in base_tier_a:
            relation = "already_tier_a"
            code = base_tier_a[identifier].get("code")
        elif identifier in promoted:
            relation = "promoted_from_tier_b"
            code = compact_code_audit(code_by_id[identifier])
        else:
            relation = "new_tier_a_candidate"
            code = compact_code_audit(code_by_id[identifier])
        omitted_rows.append({**row, "frame_relation": relation, "code": code})
    tier_b_rows = []
    for identifier in sorted(retained_tier_b):
        tier_b_rows.append({
            **base_tier_b[identifier],
            "matched_terms": ["linear probe or superposition only"],
            "code": compact_code_audit(code_by_id[identifier]),
        })
    dispositions: Dict[str, int] = {}
    for row in omitted_rows + tier_b_rows:
        code = row.get("code") or {}
        disposition = code.get("disposition", "recorded_in_base_tier_a")
        dispositions[disposition] = dispositions.get(disposition, 0) + 1
    omitted_ledger = [
        [row["arxiv_id"], row["published"], row["matched_omitted_terms"],
         row["frame_relation"], *_ledger_code(row.get("code") or {})]
        for row in omitted_rows
    ]
    tier_b_ledger = [
        [row["arxiv_id"], row["published"], *_ledger_code(row.get("code") or {})]
        for row in tier_b_rows
    ]
    return {
        "schema_version": "1.0",
        "frame_id": "august-2026-pass3b",
        "status": "candidate_frame_not_frozen",
        "outcome_blind": True,
        "base_frame": "august-2026-frame.json",
        "base_frame_sha256": __import__("hashlib").sha256(
            json.dumps(base_frame, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        ).hexdigest(),
        "window": dict(base_frame["window"]),
        "omitted_terms": list(OMITTED_TERMS),
        "term_counts_after_category_filter": term_counts,
        "rules": {
            "tier_a": "any omitted narrow term promotes or adds a Tier-A candidate",
            "tier_b": "original linear-probe/superposition-only row retained unless promoted",
            "code_audit": "all retained Tier-B, promoted, and new rows receive HTML repository-context and GitHub license audit",
            "no_outcomes_read": True,
        },
        "counts": {
            "omitted_term_unique": len(term_rows),
            "already_tier_a": len(set(term_rows) & set(base_tier_a)),
            "promoted_from_tier_b": len(promoted),
            "new_tier_a_candidates": len(new_tier_a),
            "retained_tier_b_code_audited": len(tier_b_rows),
            "dispositions": dispositions,
        },
        "ledger_formats": {
            "omitted_term_ledger": [
                "arxiv_id", "published", "matched_terms", "frame_relation",
                "code_status", "disposition", "cited_repo_count",
                "cited_repo_set_sha256", "huggingface_count",
                "huggingface_set_sha256", "public_repo_license_rows",
                "code_audit_sha256"
            ],
            "retained_tier_b_ledger": [
                "arxiv_id", "published", "code_status", "disposition",
                "cited_repo_count", "cited_repo_set_sha256", "huggingface_count",
                "huggingface_set_sha256", "public_repo_license_rows",
                "code_audit_sha256"
            ],
            "public_repo_license_row": [
                "repository", "license_spdx", "license_status", "fetch_status"
            ],
        },
        "omitted_term_ledger": omitted_ledger,
        "retained_tier_b_ledger": tier_b_ledger,
        "limitations": [
            "Repository authorship uses sentence-context heuristics and requires manual review before freeze.",
            "GitHub unauthenticated rate limits or arXiv HTML absence remain explicit manual-review rows.",
            "Code availability is not execution, reproduction, audit, or outcome evidence.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).with_name(
        "august-2026-frame.json"))
    parser.add_argument("--request-delay", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    base = json.loads(args.base.read_text())
    result = run_pass3b(base, request_delay=args.request_delay, workers=args.workers)
    print(json.dumps(
        result, indent=None if args.compact else 2, sort_keys=True,
        separators=(",", ":") if args.compact else None,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
