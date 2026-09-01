"""Deterministic local-file intake for content-addressed SourceBundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .audit_models import AUDIT_SCHEMA_VERSION, SourceBundle
from .integrity import (
    ContentAddressedStore,
    ContentRef,
    require_sha256_digest,
    verify_digest_closure,
)


_DOCUMENT_KINDS = {
    "paper",
    "repository",
    "dataset",
    "model",
    "artifact",
    "dependency",
    "other",
}

_FORBIDDEN_OUTCOME_KEYS = {
    "result", "grade", "verdict", "passed", "failed", "effect_size",
    "audit_status",
}


def _keys(value: Any) -> set:
    found = set()
    if isinstance(value, Mapping):
        found.update(value)
        for child in value.values():
            found.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keys(child))
    return found


def _path(base_dir: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _check_expected_digest(
    expected: Any, reference: ContentRef, field: str
) -> None:
    if expected is None:
        return
    require_sha256_digest(expected, field)
    if expected != reference.digest:
        raise ValueError(f"{field} does not match file bytes")


def _license_evidence(
    document_id: str, license_row: Mapping[str, Any]
) -> Dict[str, Any]:
    status = license_row.get("status")
    if status not in ("verified_compatible", "unresolved", "incompatible"):
        raise ValueError(f"document {document_id} has unsupported license status")
    identifier = license_row.get("identifier")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError(f"document {document_id} needs license identifier")
    evidence = license_row.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError(f"document {document_id} needs license evidence")
    for name in ("source", "scope", "reviewed_by", "checked_at"):
        if not isinstance(evidence.get(name), str) or not evidence[name].strip():
            raise ValueError(
                f"document {document_id} license evidence needs {name}"
            )
    return {
        "artifact": "stresskit_license_evidence",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "document_id": document_id,
        "status": status,
        "identifier": identifier,
        "evidence": dict(evidence),
    }


def _deduplicate(references: Sequence[ContentRef]) -> List[ContentRef]:
    output = []
    seen = set()
    for reference in references:
        if reference.digest not in seen:
            output.append(reference)
            seen.add(reference.digest)
    return output


def _closure_input_references(
    closure_inputs: Optional[Mapping[str, Any]],
    store: ContentAddressedStore,
    base_dir: Path,
) -> List[ContentRef]:
    """Store explicitly supplied leaves without changing the manifest digest."""
    if closure_inputs is None:
        return []
    if not isinstance(closure_inputs, Mapping):
        raise ValueError("closure_inputs must map expected digests to local paths")
    inputs = []
    for expected_digest, path_value in closure_inputs.items():
        require_sha256_digest(expected_digest, "closure input digest")
        inputs.append((expected_digest, path_value))

    references = []
    for expected_digest, path_value in sorted(inputs):
        path = _path(base_dir, path_value, "closure input path")
        reference = store.put_bytes(path.read_bytes(), role="closure_input")
        _check_expected_digest(
            expected_digest, reference, f"closure input {expected_digest}"
        )
        references.append(reference)
    return references


def build_source_bundle(
    manifest: Mapping[str, Any],
    store: ContentAddressedStore,
    *,
    base_dir: Path,
    closure_inputs: Optional[Mapping[str, Any]] = None,
) -> Tuple[SourceBundle, List[ContentRef]]:
    """Hash local files and return a validated, exactly reachable closure.

    ``closure_inputs`` supplies offline bytes for digest links that are not
    themselves SourceBundle documents, such as a pinned repository LICENSE.
    The mapping stays outside the intake manifest so adding transport paths
    cannot change the frozen manifest or SourceBundle digest.
    """
    if manifest.get("artifact") != "stresskit_source_intake_manifest" or \
            manifest.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("source intake manifest must use schema 1.0")
    if manifest.get("outcome_blind") is not True:
        raise ValueError("source intake manifest must be outcome-blind")
    forbidden = _keys(manifest) & _FORBIDDEN_OUTCOME_KEYS
    if forbidden:
        raise ValueError(f"source intake manifest contains outcome keys: {sorted(forbidden)}")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("source intake manifest needs documents")
    output_documents = []
    references: List[ContentRef] = []
    identifiers = set()
    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("source intake documents must be objects")
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("source intake document needs document_id")
        if document_id in identifiers:
            raise ValueError(f"duplicate source intake document {document_id!r}")
        identifiers.add(document_id)
        kind = document.get("kind")
        if kind not in _DOCUMENT_KINDS:
            raise ValueError(f"document {document_id} has unsupported kind")
        locator = document.get("locator")
        if not isinstance(locator, str) or not locator.strip():
            raise ValueError(f"document {document_id} needs locator")
        media_type = document.get("media_type")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError(f"document {document_id} needs media_type")
        source_path = _path(base_dir, document.get("path"), "document path")
        source_reference = store.put_bytes(
            source_path.read_bytes(), media_type=media_type, role=f"source:{kind}"
        )
        _check_expected_digest(
            document.get("expected_digest"), source_reference,
            f"document {document_id} expected_digest",
        )
        references.append(source_reference)

        row: Dict[str, Any] = {
            "document_id": document_id,
            "kind": kind,
            "locator": locator,
            "source_digest": source_reference.digest,
            "media_type": media_type,
            "content_size": source_reference.size,
        }
        extracted_path = document.get("extracted_text_path")
        if extracted_path is not None:
            text_path = _path(
                base_dir, extracted_path, "document extracted_text_path"
            )
            text_bytes = text_path.read_bytes()
            try:
                text_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"document {document_id} extracted text must be UTF-8"
                ) from exc
            text_reference = store.put_bytes(
                text_bytes, media_type="text/plain; charset=utf-8",
                role="extracted_text",
            )
            _check_expected_digest(
                document.get("expected_extracted_text_digest"), text_reference,
                f"document {document_id} expected_extracted_text_digest",
            )
            row["extracted_text_digest"] = text_reference.digest
            references.append(text_reference)

        license_payload = _license_evidence(
            document_id, document.get("license", {})
        )
        license_reference = store.put_json(
            license_payload, role="license_evidence"
        )
        references.append(license_reference)
        row["license"] = {
            "status": license_payload["status"],
            "identifier": license_payload["identifier"],
            "evidence_digest": license_reference.digest,
        }
        output_documents.append(row)

    references.extend(
        _closure_input_references(closure_inputs, store, base_dir)
    )
    manifest_reference = store.put_json(
        manifest, role="source_intake_manifest"
    )
    references.append(manifest_reference)
    bundle = SourceBundle(
        bundle_id=str(manifest.get("bundle_id", "")),
        documents=output_documents,
        created_at=str(manifest.get("created_at", "")),
        metadata={
            **dict(manifest.get("metadata", {})),
            "intake_manifest_digest": manifest_reference.digest,
            "outcome_blind": True,
        },
    )
    bundle_reference = store.put_json(bundle.to_dict(), role="source_bundle")
    references.append(bundle_reference)
    complete = _deduplicate(references)
    try:
        verify_digest_closure(store, complete, [bundle_reference.digest])
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise ValueError(f"source intake digest closure is incomplete: {exc}") from exc
    return bundle, complete
