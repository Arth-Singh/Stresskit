"""Outcome-blind arXiv interpretability code census (discovery pass 3).

Reconstructs the pass-3 methodology recorded in
``august-2026-frame.json`` and ``pass3b.py``, parameterised by submission window
and term set, and renders the result into the same frame schema.  It reads only
public arXiv and GitHub metadata: it never opens StressKit result, card, score,
or verdict artifacts.

``july-2026-frame.json`` was produced with::

    GITHUB_TOKEN=... python benchmark/discovery/july_2026_census.py \
        --from 202607010000 --to 202607312359 --frame-id july-2026 \
        --out july-2026-raw.json --frame-out july-2026-frame.json

``--render-from`` re-renders a frame from an existing raw dump without touching
the network; passing ``--generated`` with the recorded timestamp then reproduces
a committed frame byte for byte.
"""

from __future__ import annotations

import argparse
import datetime as dt
import concurrent.futures
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# The fifteen pass-3 terms, in the order recorded in august-2026-frame.json.
PASS3_TERMS = (
    "mechanistic interpretability",
    "sparse autoencoder",
    "activation patching",
    "activation steering",
    "steering vector",
    "circuit discovery",
    "logit lens",
    "linear probe",
    "attribution graph",
    "transcoder",
    "crosscoder",
    "causal tracing",
    "refusal direction",
    "model diffing",
    "superposition",
)

# The six pass-3b terms (benchmark/discovery/pass3b.py OMITTED_TERMS).
PASS3B_TERMS = (
    "chain of thought",
    "reasoning trace",
    "persona",
    "introspection",
    "evaluation awareness",
    "model organism",
)

# Tier B rule from august-2026-frame.json query.tier_b_rule.
BROAD_PASS3_TERMS = {"linear probe", "superposition"}
NARROW_PASS3_TERMS = tuple(t for t in PASS3_TERMS if t not in BROAD_PASS3_TERMS)
# Stratification treats every pass-3b term as broad: none of the six names a
# mechanistic-interpretability method, and pass 3b itself filed their hits as
# "new broad candidates ... subject to manual interpretability-scope review".
BROAD_TERMS = BROAD_PASS3_TERMS | set(PASS3B_TERMS)

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
# Detects "code will be released" style promises when no repository URL is present.
_PROMISE_LANGUAGE = re.compile(
    r"\b(code\s+(and\s+\w+\s+)?(will\s+be|is\s+to\s+be)\s+(made\s+)?(publicly\s+)?"
    r"(available|released|open[- ]sourced)|"
    r"we\s+will\s+(publicly\s+)?release|will\s+be\s+released\s+upon|"
    r"code\s+(will\s+be\s+)?released\s+upon\s+(acceptance|publication)|"
    r"upon\s+acceptance[^.]{0,40}(code|repository)|"
    r"(code|implementation)[^.]{0,60}available\s+upon\s+(request|acceptance))\b",
    re.IGNORECASE,
)

# arXiv's own HTML chrome links these on every rendered page; they are page
# furniture, not paper citations, and must not be read as repository links.
BOILERPLATE_REPOS = {
    "https://github.com/arXiv/html_feedback",
    "https://github.com/brucemiller/LaTeXML",
    "https://github.com/arXiv/arxiv-browse",
    "https://github.com/arXiv/arxiv-base",
}

_RATE_LIMIT_EVENTS: List[Dict[str, Any]] = []


def _request(url: str, *, accept: str = "*/*", timeout: float = 40.0) -> bytes:
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


def _request_retry(url: str, *, accept: str = "*/*", attempts: int = 4,
                   backoff: float = 5.0) -> bytes:
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return _request(url, accept=accept)
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (403, 429):
                _RATE_LIMIT_EVENTS.append({"url": url, "code": exc.code})
                time.sleep(backoff * (attempt + 1) * 3)
                continue
            if exc.code in (500, 502, 503, 504):
                time.sleep(backoff * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(backoff * (attempt + 1))
    raise last  # type: ignore[misc]


API_MAX_RESULTS = 100  # this endpoint reports at most 100 totalResults per query
_TRUNCATION_EVENTS: List[Dict[str, Any]] = []


def _query_window(term: str, window_from: str, window_to: str) -> List[Dict[str, Any]]:
    """One arXiv query. Returns every entry the endpoint reports for the window."""
    query = f'all:"{term}" AND submittedDate:[{window_from} TO {window_to}]'
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        "search_query": query, "start": 0, "max_results": API_MAX_RESULTS,
        "sortBy": "submittedDate", "sortOrder": "descending",
    })
    root = ET.fromstring(_request_retry(url, accept="application/atom+xml"))
    rows = []
    for entry in root.findall(_ATOM + "entry"):
        identifier = (entry.findtext(_ATOM + "id") or "").rsplit("/", 1)[-1]
        identifier = re.sub(r"v\d+$", "", identifier)
        rows.append({
            "arxiv_id": identifier,
            "title": " ".join((entry.findtext(_ATOM + "title") or "").split()),
            "abstract": " ".join((entry.findtext(_ATOM + "summary") or "").split()),
            "published": (entry.findtext(_ATOM + "published") or "")[:10],
            "categories": [n.attrib.get("term", "")
                           for n in entry.findall(_ATOM + "category")],
        })
    return rows


def _split_window(window_from: str, window_to: str):
    """Halve a [YYYYMMDDHHMM, YYYYMMDDHHMM] window on a day boundary."""
    import datetime as _dt
    start = _dt.date(int(window_from[:4]), int(window_from[4:6]), int(window_from[6:8]))
    end = _dt.date(int(window_to[:4]), int(window_to[4:6]), int(window_to[6:8]))
    if start >= end:
        return None
    mid = start + (end - start) / 2
    return ((window_from, mid.strftime("%Y%m%d") + "2359"),
            ((mid + _dt.timedelta(days=1)).strftime("%Y%m%d") + "0000", window_to))


def query_arxiv(term: str, window_from: str, window_to: str,
                delay: float = 3.0) -> List[Dict[str, Any]]:
    """Query the official Atom API for one exact term inside the submission window.

    The endpoint caps a single query at API_MAX_RESULTS rows.  When a window
    returns the cap the window is halved and each half re-queried, so the census
    is complete rather than silently truncated at the most recent 100 rows.
    """
    rows = _query_window(term, window_from, window_to)
    if len(rows) < API_MAX_RESULTS:
        return rows
    halves = _split_window(window_from, window_to)
    if halves is None:
        _TRUNCATION_EVENTS.append({"term": term, "window": [window_from, window_to],
                                   "returned": len(rows)})
        return rows
    merged: Dict[str, Dict[str, Any]] = {r["arxiv_id"]: r for r in rows}
    for lo, hi in halves:
        time.sleep(delay)
        for row in query_arxiv(term, lo, hi, delay):
            merged[row["arxiv_id"]] = row
    return list(merged.values())


def _clean_repo(url_match: "re.Match[str]") -> str:
    owner, repository = url_match.group(1), url_match.group(2)
    repository = repository.rstrip(".,);]}'\"")
    if repository.endswith(".git"):
        repository = repository[:-4]
    return f"https://github.com/{owner}/{repository}"


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
    try:
        page = _request(repository_url, accept="text/html").decode("utf-8", "replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        code = getattr(exc, "code", None)
        return {"repository": repository_url,
                "status": "not_found" if code == 404 else "github_fetch_error",
                "api_error": api_error, "error": type(exc).__name__,
                "http_code": code}
    branch_match = re.search(r'"defaultBranch":"([^"]+)"', page)
    branch = branch_match.group(1) if branch_match else "main"
    owner_repo = repository_url.split("github.com/", 1)[1]
    license_spdx = None
    license_path = None
    for candidate_branch in dict.fromkeys((branch, "main", "master")):
        for filename in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
            raw_url = (f"https://raw.githubusercontent.com/{owner_repo}/"
                       f"{candidate_branch}/{filename}")
            try:
                license_text = _request(raw_url, accept="text/plain").decode(
                    "utf-8", "replace")
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                continue
            license_spdx = _license_from_text(license_text)
            license_path = filename
            branch = candidate_branch
            break
        if license_path:
            break
    return {
        "repository": repository_url, "status": "public", "default_branch": branch,
        "license_spdx": license_spdx, "license_path": license_path,
        "license_status": "found" if license_spdx else "not_found",
        "metadata_source": "github_html_and_raw_license_fallback",
        "api_error": api_error,
    }


def _github_metadata(repository_url: str) -> Dict[str, Any]:
    owner_repo = repository_url.split("github.com/", 1)[1]
    url = "https://api.github.com/repos/" + owner_repo
    try:
        payload = json.loads(_request_retry(url, accept="application/vnd.github+json"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"repository": repository_url, "status": "not_found",
                    "http_code": 404}
        return _github_html_fallback(repository_url, f"HTTPError{exc.code}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return _github_html_fallback(repository_url, type(exc).__name__)
    license_row = payload.get("license") or {}
    head = None
    try:
        commits = json.loads(_request_retry(
            f"https://api.github.com/repos/{owner_repo}/commits?per_page=1",
            accept="application/vnd.github+json"))
        if isinstance(commits, list) and commits:
            head = commits[0].get("sha")
    except Exception:  # noqa: BLE001 - head sha is best-effort provenance
        head = None
    return {
        "repository": repository_url,
        "status": "public",
        "default_branch": payload.get("default_branch"),
        "head_commit_at_discovery": head,
        "license_spdx": license_row.get("spdx_id"),
        "license_status": ("found"
                           if license_row.get("spdx_id") not in (None, "NOASSERTION")
                           else "not_found"),
        "repo_created_at": (payload.get("created_at") or "")[:10] or None,
        "last_push_at_discovery": (payload.get("pushed_at") or "")[:10] or None,
        "stargazers": payload.get("stargazers_count"),
        "archived": payload.get("archived") is True,
    }


def audit_code(arxiv_id: str, title: str) -> Dict[str, Any]:
    """Scan the arXiv HTML rendering for repository URLs and audit each on GitHub."""
    source_kind = "html"
    try:
        raw = _request_retry(f"https://arxiv.org/html/{arxiv_id}", accept="text/html",
                             attempts=2)
    except Exception:  # noqa: BLE001
        source_kind = "abs"
        try:
            raw = _request_retry(f"https://arxiv.org/abs/{arxiv_id}",
                                 accept="text/html", attempts=3)
        except Exception as exc:  # noqa: BLE001
            return {"html_source": "fetch_failed",
                    "code": {"status": "manual_review_fetch_failure",
                             "error": type(exc).__name__}}
    page = raw.decode("utf-8", "replace")
    plain = html.unescape(_TAG.sub(" ", page))
    repositories = sorted({_clean_repo(m) for m in _GITHUB.finditer(page)}
                          - BOILERPLATE_REPOS)
    authored: List[str] = []
    dependencies: List[str] = []
    title_tokens = {w for w in re.split(r"[^a-z0-9]+", title.lower())
                    if len(w) > 3}
    for repository in repositories:
        location = page.lower().find(repository.lower())
        context_html = page[max(0, location - 500): location + len(repository) + 500]
        context = html.unescape(_TAG.sub(" ", context_html))
        release = bool(_RELEASE_LANGUAGE.search(context))
        dependency = bool(_DEPENDENCY_LANGUAGE.search(context))
        if release and not dependency:
            authored.append(repository)
            continue
        # Tie-break only: repo name echoes the paper title and no dependency language.
        name = repository.rsplit("/", 1)[-1].lower()
        name_tokens = {w for w in re.split(r"[^a-z0-9]+", name) if len(w) > 3}
        if not dependency and name_tokens and (name_tokens & title_tokens):
            authored.append(repository)
        else:
            dependencies.append(repository)
    huggingface = sorted({u.split("huggingface.co/", 1)[1]
                          for u in _HF.findall(page)})
    if not authored:
        if repositories:
            return {"html_source": source_kind,
                    "code": {"status": "dependency_links_only",
                             "cited_repositories": dependencies}}
        if _PROMISE_LANGUAGE.search(plain):
            return {"html_source": source_kind,
                    "code": {"status": "code_promised_not_released"}}
        if huggingface:
            return {"html_source": source_kind,
                    "code": {"status": "no_repo_hf_links_only",
                             "huggingface_refs": huggingface}}
        return {"html_source": source_kind,
                "code": {"status": "no_public_code_found"}}
    audits = [_github_metadata(r) for r in authored]
    live = [a for a in audits if a.get("status") == "public"]
    if not live:
        return {"html_source": source_kind,
                "code": {"status": "authored_repo_not_found",
                         "authored_repositories": audits}}
    licensed = [a for a in live if a.get("license_status") == "found"]
    primary = (licensed or live)[0]
    others = [a["repository"] for a in live if a["repository"] != primary["repository"]]
    return {
        "html_source": source_kind,
        "code": {
            "status": "public_repo",
            "repository": primary["repository"],
            "head_commit_at_discovery": primary.get("head_commit_at_discovery"),
            "license_spdx": primary.get("license_spdx"),
            "license_status": primary.get("license_status"),
            "repo_created_at": primary.get("repo_created_at"),
            "last_push_at_discovery": primary.get("last_push_at_discovery"),
            "other_authored_repos": others,
        },
        "_all_authored": audits,
        "_cited": dependencies,
    }


def build_frame(window_from: str, window_to: str, frame_id: str, *,
                terms: List[str], request_delay: float = 3.0,
                workers: int = 4, audit: bool = True) -> Dict[str, Any]:
    term_rows: Dict[str, Dict[str, Any]] = {}
    raw_term_counts: Dict[str, int] = {}
    for index, term in enumerate(terms):
        if index and request_delay:
            time.sleep(request_delay)
        rows = query_arxiv(term, window_from, window_to, request_delay)
        raw_term_counts[term] = len(rows)
        print(f"  [{term}] {len(rows)} raw hits", file=sys.stderr, flush=True)
        for row in rows:
            existing = term_rows.setdefault(row["arxiv_id"], dict(row))
            existing.setdefault("matched_terms", [])
            if term not in existing["matched_terms"]:
                existing["matched_terms"].append(term)

    unique_hits = len(term_rows)
    after_category = {i: r for i, r in term_rows.items()
                      if set(r["categories"]) & ALLOWED_CATEGORIES}

    def has_signal(row: Mapping[str, Any]) -> List[str]:
        blob = (row["title"] + " " + row["abstract"]).lower()
        return [t for t in row["matched_terms"] if t.lower() in blob]

    with_signal: Dict[str, Dict[str, Any]] = {}
    for identifier, row in after_category.items():
        confirmed = has_signal(row)
        if confirmed:
            with_signal[identifier] = {**row, "matched_terms": confirmed}

    tier_a: Dict[str, Dict[str, Any]] = {}
    tier_b: Dict[str, Dict[str, Any]] = {}
    tier_c: Dict[str, Dict[str, Any]] = {}
    for identifier, row in with_signal.items():
        matched = set(row["matched_terms"])
        if matched & set(NARROW_PASS3_TERMS):
            tier_a[identifier] = row
        elif matched & BROAD_TERMS:
            tier_b[identifier] = row
        else:
            tier_c[identifier] = row

    counted_terms = {t: sum(1 for r in after_category.values()
                            if t in r["matched_terms"]) for t in terms}

    result: Dict[str, Any] = {
        "raw_term_counts": raw_term_counts,
        "term_counts_after_category_filter": counted_terms,
        "unique_hits": unique_hits,
        "after_category_filter": len(after_category),
        "with_interpretability_signal": len(with_signal),
        "tier_a": tier_a,
        "tier_b": tier_b,
        "tier_c": tier_c,
        "truncation_events": list(_TRUNCATION_EVENTS),
    }
    if not audit:
        return result

    audits: Dict[str, Dict[str, Any]] = {}
    ids = sorted(tier_a)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(audit_code, i, tier_a[i]["title"]): i for i in ids}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            identifier = futures[future]
            done += 1
            try:
                audits[identifier] = future.result()
            except Exception as exc:  # noqa: BLE001
                audits[identifier] = {"html_source": "fetch_failed",
                                      "code": {"status": "manual_review_fetch_failure",
                                               "error": type(exc).__name__}}
            if done % 10 == 0:
                print(f"  audited {done}/{len(ids)}", file=sys.stderr, flush=True)
    result["audits"] = audits
    result["rate_limit_events"] = list(_RATE_LIMIT_EVENTS)
    result["truncation_events"] = list(_TRUNCATION_EVENTS)
    return result


def _iso_day(stamp):
    """202607010000 -> 2026-07-01."""
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"


def render_frame(raw, frame_id, window_from, window_to, generated=None):
    """Render a raw census dump into the frame schema of august-2026-frame.json.

    window_from / window_to are the arXiv API stamps the census was run with.
    """
    TERM_ORDER = list(PASS3_TERMS) + list(PASS3B_TERMS)
    generated = generated or dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0).isoformat()

    def order(terms):
        return [t for t in TERM_ORDER if t in set(terms)]

    def code_block(audit):
        code = dict(audit["code"])
        if code["status"] == "public_repo" and code.get("license_status") != "found":
            code["license_spdx"] = None
        return code

    papers = []
    for arxiv_id in sorted(raw["tier_a"], reverse=True):
        row = raw["tier_a"][arxiv_id]
        audit = raw["audits"][arxiv_id]
        papers.append({
            "arxiv_id": arxiv_id,
            "title": row["title"],
            "published": row["published"],
            "matched_terms": order(row["matched_terms"]),
            "html_source": audit["html_source"],
            "code": code_block(audit),
        })
    papers.sort(key=lambda p: (p["published"], p["arxiv_id"]), reverse=True)

    tier_b = [{"arxiv_id": i, "title": raw["tier_b"][i]["title"],
               "published": raw["tier_b"][i]["published"]}
              for i in sorted(raw["tier_b"])]
    tier_b.sort(key=lambda p: (p["published"], p["arxiv_id"]), reverse=True)

    by_status: dict[str, int] = {}
    for paper in papers:
        by_status[paper["code"]["status"]] = by_status.get(paper["code"]["status"], 0) + 1
    released = [p for p in papers if p["code"]["status"] == "public_repo"]
    licensed = [p for p in released if p["code"]["license_spdx"]]

    pass3b_only = sum(
        1 for i in raw["tier_b"]
        if not (set(raw["tier_b"][i]["matched_terms"]) & BROAD_PASS3_TERMS))

    frame = {
        "schema_version": "0.1",
        "frame_id": frame_id,
        "status": "candidate_frame_not_frozen",
        "outcome_blind": True,
        "generated": generated,
        "window": {"submitted_from": _iso_day(window_from),
                   "submitted_to": _iso_day(window_to),
                   "source": "arXiv API"},
        "query": {
            "endpoint": "http://export.arxiv.org/api/query",
            "field": "all",
            "terms": {t: raw["term_counts_after_category_filter"][t] for t in TERM_ORDER},
            "category_filter": sorted(ALLOWED_CATEGORIES),
            "tier_a_rule": "matched >=1 narrow mechanistic-interpretability term",
            "tier_b_rule": ("matched only broad terms ('linear probe', 'superposition', "
                            "or one of the six pass-3b terms); retained as a separate "
                            "stratum, not scored"),
            "narrow_terms": list(NARROW_PASS3_TERMS),
            "broad_terms": sorted(BROAD_TERMS),
            "pagination_rule": ("this endpoint reports at most 100 totalResults per query, "
                                "so any window returning 100 rows is halved on a day "
                                "boundary and re-queried until every sub-window is under "
                                "the cap"),
            "signal_rule": ("a matched term must appear as a case-insensitive substring of "
                            "the title or abstract; terms matched only through other "
                            "indexed metadata (comments, journal reference) are dropped"),
        },
        "code_detection": {
            "method": ("arXiv HTML (v1/v2) full-text scan for repository URLs, then GitHub "
                       "API existence/license/HEAD checks, then sentence-context "
                       "classification of each link as an authored release or a cited "
                       "dependency"),
            "authored_release_rule": ("release language in the sentence around the link, or "
                                      "a repository whose name shares a content word with "
                                      "the paper title, with no dependency language"),
            "known_limitation": ("papers with no arXiv HTML rendering, or code linked only "
                                 "from a PDF figure or an external project page, can be "
                                 "undercounted"),
            "chrome_exclusion": ("arXiv's HTML renderer links github.com/arXiv/html_feedback "
                                 "and github.com/brucemiller/LaTeXML on every page; both are "
                                 "page furniture and are excluded before classification"),
        },
        "counts": {
            "arxiv_hits_unique": raw["unique_hits"],
            "after_category_filter": raw["after_category_filter"],
            "with_interpretability_signal": raw["with_interpretability_signal"],
            "tier_a": len(papers),
            "tier_b": len(tier_b),
            "tier_b_matched_pass3b_terms_only": pass3b_only,
            "by_code_status": by_status,
            "public_repo_with_spdx_license": len(licensed),
            "public_repo_without_license": len(released) - len(licensed),
        },
        "coverage": {
            "arxiv_query_truncations": raw.get("truncation_events", []),
            "github_rate_limit_events": len(raw.get("rate_limit_events", [])),
            "tier_a_papers_audited": len(papers),
            "tier_a_papers_unaudited": 0,
            "tier_b_papers_audited": 0,
        },
        "papers": papers,
        "tier_b_papers": tier_b,
    }
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="window_from", required=True,
                        help="arXiv submittedDate window start, YYYYMMDDHHMM")
    parser.add_argument("--to", dest="window_to", required=True,
                        help="arXiv submittedDate window end, YYYYMMDDHHMM")
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--generated", default=None,
                        help="pin the frame's generated timestamp "
                             "(for re-rendering an existing frame)")
    parser.add_argument("--out", type=Path, default=None,
                        help="raw census dump; required unless --render-from is used")
    parser.add_argument("--terms", default="pass3+pass3b",
                        choices=["pass3", "pass3b", "pass3+pass3b"])
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--request-delay", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frame-out", type=Path, default=None,
                        help="also render the frame JSON here")
    parser.add_argument("--render-from", type=Path, default=None,
                        help="skip the network and render --frame-out from an existing raw dump")
    args = parser.parse_args(argv)
    if args.render_from is not None:
        if args.frame_out is None:
            parser.error("--render-from requires --frame-out")
        raw = json.loads(args.render_from.read_text())
        rendered = render_frame(raw, args.frame_id, args.window_from,
                                args.window_to, args.generated)
        args.frame_out.write_text(json.dumps(rendered, indent=1) + "\n")
        print(json.dumps(rendered["counts"], indent=2))
        return 0
    if args.out is None:
        parser.error("--out is required when the census actually runs")
    terms = {"pass3": list(PASS3_TERMS), "pass3b": list(PASS3B_TERMS),
             "pass3+pass3b": list(PASS3_TERMS) + list(PASS3B_TERMS)}[args.terms]
    frame = build_frame(args.window_from, args.window_to, args.frame_id,
                        terms=terms, request_delay=args.request_delay,
                        workers=args.workers, audit=not args.no_audit)
    args.out.write_text(json.dumps(frame, indent=1, sort_keys=True))
    if args.frame_out is not None:
        rendered = render_frame(frame, args.frame_id, args.window_from,
                                args.window_to, args.generated)
        args.frame_out.write_text(json.dumps(rendered, indent=1) + "\n")
    print(json.dumps({k: v for k, v in frame.items()
                      if k in ("unique_hits", "after_category_filter",
                               "with_interpretability_signal")}
                     | {"tier_a": len(frame["tier_a"]),
                        "tier_b": len(frame["tier_b"]),
                        "tier_c": len(frame["tier_c"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
