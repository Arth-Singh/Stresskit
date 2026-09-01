"""Verified claim-level evidence matrix and static public-site publisher."""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .audit_models import AUDIT_SCHEMA_VERSION, AuditBundle, AuditDecision, AuditSpec, ClaimRecord
from .audit_verify import verify_audit_release
from .integrity import ContentAddressedStore, digest_json, require_sha256_digest


class PublicationBlocked(RuntimeError):
    """Raised when a public release gate has not been satisfied."""


def _parse_time(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ISO 8601 timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("publication timestamps must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _abstain_row(claim_id: str, reason: str) -> Dict[str, Any]:
    return {
        "claim_id": claim_id,
        "status": "abstain",
        "publication_state": "abstain",
        "reproduction": {"state": "not_evaluated"},
        "stability_specificity": {"state": "not_evaluated"},
        "utility": {"state": "not_evaluated"},
        "generalization": {"state": "not_evaluated"},
        "evidence_confidence": {"state": "insufficient", "reason": reason},
        "reasons": [reason],
        "external_validation": "not obtained",
    }


def _excluded_row(claim_id: str, reason: str) -> Dict[str, Any]:
    row = _abstain_row(claim_id, reason)
    row["status"] = "excluded"
    row["publication_state"] = "final"
    return row


def _author_response_ready(
    registry_row: Mapping[str, Any], as_of: dt.datetime
) -> bool:
    response = registry_row.get("author_response")
    if not isinstance(response, Mapping):
        return False
    if response.get("response_received") is True:
        response_text = response.get("response_text")
        return isinstance(response_text, str) and bool(response_text.strip())
    notified_at = response.get("notified_at")
    if not isinstance(notified_at, str):
        return False
    return as_of >= _parse_time(notified_at) + dt.timedelta(days=14)


def _validate_registry(registry: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if registry.get("artifact") != "stresskit_release_registry" or \
            registry.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("release registry must be stresskit_release_registry schema 1.0")
    if registry.get("status") != "frozen" or registry.get("outcome_blind") is not True:
        raise ValueError("release registry must be frozen before outcomes")
    claims = registry.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("release registry needs non-empty claims list")
    identifiers = [row.get("claim_id") for row in claims if isinstance(row, Mapping)]
    if len(identifiers) != len(claims) or not all(
            isinstance(value, str) and value.strip() for value in identifiers):
        raise ValueError("every registry row needs claim_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("release registry claim IDs must be unique")
    for row in claims:
        if row.get("disposition") not in ("eligible", "excluded"):
            raise ValueError("registry disposition must be eligible or excluded")
        if row.get("disposition") == "excluded" and not row.get("exclusion_reason"):
            raise ValueError("excluded registry row needs exclusion_reason")
        if row.get("disposition") == "eligible":
            require_sha256_digest(
                row.get("claim_record_digest"), "registry claim_record_digest"
            )
            require_sha256_digest(
                row.get("audit_spec_digest"), "registry audit_spec_digest"
            )
    return claims


def build_evidence_board(
    registry: Mapping[str, Any],
    bundles: Sequence[AuditBundle],
    store: ContentAddressedStore,
    *,
    trusted_plan_keys: Mapping[str, bytes],
    trusted_executor_keys: Mapping[str, bytes],
    as_of: Optional[str] = None,
    agent_only_review: bool = False,
) -> Dict[str, Any]:
    """Verify every supplied bundle and produce complete registry matrix.

    ``agent_only_review`` is an explicit user choice, never inferred.  It does
    not pretend external validation occurred; public rows keep
    ``external_validation: not obtained``.
    """
    claims = _validate_registry(registry)
    as_of_time = _parse_time(as_of) if as_of else dt.datetime.now(dt.timezone.utc)
    by_bundle: Dict[str, AuditBundle] = {}
    for bundle in bundles:
        spec = AuditSpec.from_dict(bundle.audit_spec)
        claim = ClaimRecord.from_dict(spec.claim_record)
        if claim.claim_id in by_bundle:
            raise ValueError(f"duplicate bundle for claim {claim.claim_id}")
        by_bundle[claim.claim_id] = bundle
    registry_ids = {str(row["claim_id"]) for row in claims}
    registry_by_id = {str(row["claim_id"]): row for row in claims}
    unknown = sorted(set(by_bundle) - registry_ids)
    if unknown:
        raise ValueError(f"bundles are absent from frozen registry: {unknown}")
    for claim_id, bundle in by_bundle.items():
        registry_row = registry_by_id[claim_id]
        if registry_row["disposition"] != "eligible":
            raise ValueError(f"excluded registry claim has AuditBundle: {claim_id}")
        spec = AuditSpec.from_dict(bundle.audit_spec)
        claim = ClaimRecord.from_dict(spec.claim_record)
        if spec.digest != registry_row["audit_spec_digest"] or \
                claim.digest != registry_row["claim_record_digest"]:
            raise ValueError(
                f"AuditBundle does not match frozen registry digests: {claim_id}"
            )

    eligible_bundles = [
        by_bundle[str(row["claim_id"])]
        for row in claims
        if row["disposition"] == "eligible" and str(row["claim_id"]) in by_bundle
    ]
    release_verification = (
        verify_audit_release(
            eligible_bundles,
            store,
            trusted_plan_keys=trusted_plan_keys,
            trusted_executor_keys=trusted_executor_keys,
        )
        if eligible_bundles else {"ok": False, "problems": ["no audit bundles"],
                                  "results": {}, "holm": {}}
    )
    results = release_verification.get("results", {})
    rows: List[Dict[str, Any]] = []
    response_blockers: List[str] = []
    for registry_row in claims:
        claim_id = str(registry_row["claim_id"])
        if registry_row["disposition"] == "excluded":
            decision = _excluded_row(claim_id, str(registry_row["exclusion_reason"]))
            verification = {"verified": True, "status": "excluded"}
        elif claim_id not in by_bundle:
            decision = _abstain_row(claim_id, "eligible claim has no complete AuditBundle")
            verification = {"verified": False, "status": "abstain"}
        else:
            result = results.get(claim_id)
            if not isinstance(result, Mapping):
                decision = _abstain_row(claim_id, "release verifier returned no claim result")
                verification = {"verified": False, "status": "abstain"}
            else:
                raw_decision = result.get("decision")
                if isinstance(raw_decision, AuditDecision):
                    decision = raw_decision.to_dict()
                else:
                    decision = _abstain_row(
                        claim_id, "bundle did not produce a verifiable AuditDecision"
                    )
                verification = {
                    "verified": result.get("verified") is True,
                    "problems": list(result.get("problems", [])),
                    "bundle_digest": by_bundle[claim_id].digest,
                }
        if decision["status"] in ("audit_failure", "reproduction_failure") and \
                registry_row.get("named_upstream", True) is True and \
                not _author_response_ready(registry_row, as_of_time):
            response_blockers.append(claim_id)
        comparison_fields = {
            key: registry_row.get(key)
            for key in ("task", "metric", "evaluation_set", "resource_budget")
        }
        row = {
            **decision,
            "paper_id": registry_row.get("paper_id"),
            "paper_title": registry_row.get("paper_title"),
            "method_family": registry_row.get("method_family"),
            "model_family": registry_row.get("model_family"),
            "stratum": registry_row.get("stratum"),
            "claim_locator": registry_row.get("claim_locator"),
            "comparison_key": digest_json(comparison_fields),
            "comparison_fields": comparison_fields,
            "author_response": registry_row.get("author_response"),
            "verification": verification,
        }
        row["external_validation"] = decision.get(
            "external_validation", "not obtained"
        )
        rows.append(row)
    if response_blockers:
        raise PublicationBlocked(
            "14-day upstream-author response gate incomplete for adverse claims: "
            + ", ".join(response_blockers)
        )
    if not agent_only_review:
        unvalidated = [row["claim_id"] for row in rows
                       if row["status"] not in ("excluded", "abstain") and
                       row.get("external_validation") == "not obtained"]
        if unvalidated:
            raise PublicationBlocked(
                "external validation absent; rerun with explicit agent_only_review choice "
                "or obtain validation: " + ", ".join(unvalidated)
            )
    return {
        "artifact": "stresskit_evidence_board",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "release_id": registry.get("release_id"),
        "as_of": as_of_time.isoformat(),
        "registry_digest": digest_json(registry),
        "publication_scope": "claim-level evidence only",
        "paper_verdicts": "not computed",
        "external_validation_policy": (
            "agent-only methodological review explicitly selected; external_validation remains not obtained"
            if agent_only_review else "external validation required"
        ),
        "multiplicity": {
            "method": "holm-bonferroni",
            "scope": "all primary checks in frozen release",
            "results": release_verification.get("holm", {}),
        },
        "rows": rows,
    }


def _state_cell(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return str(value.get("state", "—")) if isinstance(value, Mapping) else "—"


def evidence_markdown(board: Mapping[str, Any]) -> str:
    """Render deterministic evidence matrix without ranking or paper scores."""
    lines = [
        "# StressKit verified evidence board",
        "",
        "Claim-level conditional results only. No paper-quality scalar and no whole-paper truth verdict.",
        "",
        "| claim | stratum | reproduction | stability / specificity | utility | generalization | final status |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in board.get("rows", []):
        cells = [
            row.get("claim_id", "—"),
            row.get("stratum", "—") or "—",
            _state_cell(row, "reproduction"),
            _state_cell(row, "stability_specificity"),
            _state_cell(row, "utility"),
            _state_cell(row, "generalization"),
            row.get("status", "abstain"),
        ]
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in cells]
        lines.append("| " + " | ".join(escaped) + " |")
    lines.extend([
        "",
        "Rows preserve frozen registry order. Excluded and abstained claims remain visible.",
        "Comparisons require identical task, metric, evaluation set, and resource budget; board performs no cross-key ranking.",
        "",
    ])
    return "\n".join(lines)


def _html_page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:.45rem;text-align:left}"
        "code{overflow-wrap:anywhere}.status{font-weight:700}</style>"
        f"<body>{body}</body></html>"
    )


def evidence_html(board: Mapping[str, Any]) -> str:
    """Render escaped static HTML matrix."""
    header = (
        "<h1>StressKit verified evidence board</h1>"
        "<p>Claim-level conditional results only. No whole-paper truth verdict.</p>"
        "<table><thead><tr><th>claim</th><th>stratum</th><th>reproduction</th>"
        "<th>stability / specificity</th><th>utility</th><th>generalization</th>"
        "<th>status</th></tr></thead><tbody>"
    )
    rows = []
    for row in board.get("rows", []):
        values = (
            row.get("claim_id", "—"), row.get("stratum", "—") or "—",
            _state_cell(row, "reproduction"),
            _state_cell(row, "stability_specificity"),
            _state_cell(row, "utility"), _state_cell(row, "generalization"),
            row.get("status", "abstain"),
        )
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
        rows.append("<tr>" + cells + "</tr>")
    return _html_page(
        "StressKit evidence board",
        header + "".join(rows) + "</tbody></table>"
        "<p>Excluded and abstained rows remain visible. Rows are not grade-sorted.</p>",
    )


def write_evidence_site(board: Mapping[str, Any], output_dir: str) -> None:
    """Write JSON, Markdown, matrix HTML, and non-verdict paper pages."""
    root = Path(output_dir)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise PublicationBlocked(
            "evidence output must be a new or empty directory; refusing stale pages"
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "evidence-board.json").write_text(
        json.dumps(board, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(evidence_markdown(board), encoding="utf-8")
    (root / "index.html").write_text(evidence_html(board), encoding="utf-8")
    papers: Dict[str, List[Mapping[str, Any]]] = {}
    for row in board.get("rows", []):
        paper_id = row.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            papers.setdefault(paper_id, []).append(row)
    paper_dir = root / "papers"
    paper_dir.mkdir(exist_ok=True)
    for paper_id, rows in sorted(papers.items()):
        items = "".join(
            "<li><code>" + html.escape(str(row["claim_id"])) + "</code>: "
            + html.escape(str(row["status"])) + "</li>" for row in rows
        )
        title = str(rows[0].get("paper_title") or paper_id)
        body = (
            f"<h1>{html.escape(title)}</h1>"
            "<p>No paper-level verdict is computed. Page lists audited claims only.</p>"
            f"<ul>{items}</ul>"
        )
        safe_base = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in paper_id
        ) or "paper"
        suffix = digest_json(paper_id).split(":", 1)[1][:12]
        safe_name = f"{safe_base}-{suffix}"
        (paper_dir / f"{safe_name}.html").write_text(
            _html_page(title, body), encoding="utf-8"
        )
