"""SourceBundle intake keeps exact bytes, licenses, and closure verifiable."""

import copy

import pytest

from stresskit.audit_cli import _closure_inputs
from stresskit.integrity import (
    ContentAddressedStore,
    digest_json,
    sha256_bytes,
    verify_digest_closure,
)
from stresskit.source_intake import build_source_bundle


def _manifest(tmp_path):
    (tmp_path / "paper.pdf").write_bytes(b"synthetic-pdf-bytes")
    (tmp_path / "paper.txt").write_text(
        "Claim: component alpha is recovered.", encoding="utf-8"
    )
    return {
        "artifact": "stresskit_source_intake_manifest",
        "schema_version": "1.0",
        "bundle_id": "source-test",
        "created_at": "2026-09-01T00:00:00+00:00",
        "outcome_blind": True,
        "metadata": {"candidate_id": "claim-test"},
        "documents": [{
            "document_id": "paper",
            "kind": "paper",
            "locator": "https://example.invalid/paper",
            "path": "paper.pdf",
            "media_type": "application/pdf",
            "expected_digest": sha256_bytes(b"synthetic-pdf-bytes"),
            "extracted_text_path": "paper.txt",
            "expected_extracted_text_digest": sha256_bytes(
                b"Claim: component alpha is recovered."
            ),
            "license": {
                "status": "verified_compatible",
                "identifier": "CC-BY-4.0",
                "evidence": {
                    "source": "https://example.invalid/license",
                    "scope": "quotation and redistribution",
                    "reviewed_by": "synthetic-reviewer",
                    "checked_at": "2026-09-01T00:00:00+00:00",
                },
            },
        }],
    }


def test_source_intake_hashes_raw_text_license_and_bundle(tmp_path):
    store = ContentAddressedStore(str(tmp_path / "cas"))
    manifest = _manifest(tmp_path)
    bundle, references = build_source_bundle(
        manifest, store, base_dir=tmp_path
    )
    document = bundle.documents[0]
    assert document["source_digest"] == manifest["documents"][0]["expected_digest"]
    assert document["extracted_text_digest"] == \
        manifest["documents"][0]["expected_extracted_text_digest"]
    assert document["license"]["status"] == "verified_compatible"
    assert bundle.metadata["outcome_blind"] is True
    assert len(references) == 5
    assert len({reference.digest for reference in references}) == len(references)
    store.verify_refs(references)
    assert verify_digest_closure(
        store, references, [bundle.digest]
    ) == {reference.digest for reference in references}
    assert store.get_json(bundle.metadata["intake_manifest_digest"]) == manifest
    assert store.get_json(document["license"]["evidence_digest"])["artifact"] == \
        "stresskit_license_evidence"


def test_source_intake_requires_explicit_bytes_for_reachable_digest(tmp_path):
    manifest = _manifest(tmp_path)
    license_bytes = b"synthetic license bytes\n"
    (tmp_path / "LICENSE").write_bytes(license_bytes)
    license_digest = sha256_bytes(license_bytes)
    manifest["metadata"]["license_file_digest"] = license_digest
    manifest["documents"][0]["license"]["evidence"][
        "license_file_digest"
    ] = license_digest

    with pytest.raises(ValueError, match="digest closure omits referenced object"):
        build_source_bundle(
            manifest,
            ContentAddressedStore(str(tmp_path / "missing-cas")),
            base_dir=tmp_path,
        )

    store = ContentAddressedStore(str(tmp_path / "cas"))
    bundle, references = build_source_bundle(
        manifest,
        store,
        base_dir=tmp_path,
        closure_inputs={license_digest: "LICENSE"},
    )
    assert bundle.metadata["intake_manifest_digest"] == digest_json(manifest)
    assert store.get_bytes(license_digest) == license_bytes
    assert verify_digest_closure(
        store, references, [bundle.digest]
    ) == {reference.digest for reference in references}

    duplicate = tmp_path / "duplicate-license"
    duplicate.write_bytes(license_bytes)
    second_bundle, _ = build_source_bundle(
        manifest,
        ContentAddressedStore(str(tmp_path / "second-cas")),
        base_dir=tmp_path,
        closure_inputs={license_digest: str(duplicate)},
    )
    assert second_bundle.digest == bundle.digest


def test_source_cli_parses_explicit_closure_inputs():
    digest = sha256_bytes(b"license")
    assert _closure_inputs([f"{digest}=LICENSE"]) == {digest: "LICENSE"}
    with pytest.raises(ValueError, match="duplicate --closure-input"):
        _closure_inputs([f"{digest}=LICENSE", f"{digest}=COPYING"])


def test_source_intake_rejects_tampered_pins_and_outcomes(tmp_path):
    store = ContentAddressedStore(str(tmp_path / "cas"))
    manifest = _manifest(tmp_path)
    changed = copy.deepcopy(manifest)
    changed["documents"][0]["expected_digest"] = sha256_bytes(b"different")
    with pytest.raises(ValueError, match="does not match file bytes"):
        build_source_bundle(changed, store, base_dir=tmp_path)

    changed = copy.deepcopy(manifest)
    changed["verdict"] = "pass"
    with pytest.raises(ValueError, match="outcome keys"):
        build_source_bundle(changed, store, base_dir=tmp_path)


def test_source_intake_requires_utf8_extracted_text(tmp_path):
    manifest = _manifest(tmp_path)
    (tmp_path / "paper.txt").write_bytes(b"\xff\xfe")
    manifest["documents"][0].pop("expected_extracted_text_digest")
    with pytest.raises(ValueError, match="must be UTF-8"):
        build_source_bundle(
            manifest,
            ContentAddressedStore(str(tmp_path / "cas")),
            base_dir=tmp_path,
        )
