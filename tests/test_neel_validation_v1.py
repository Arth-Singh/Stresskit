"""Integrity checks for the transcript-derived Neel criteria registry."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from stresskit.neel_validation import build_validation_report, render_markdown


REPO_ROOT = Path(__file__).parents[1]
REGISTRY_PATH = REPO_ROOT / "benchmark" / "neel_criteria_v1.json"
EXTERNAL_TRANSCRIPT = (
    Path.home()
    / "Downloads"
    / "NeelNanda_transcription"
    / "NeelNanda_transcript.txt"
)

EXPECTED_IDS = [f"N{index:02d}" for index in range(1, 49)]
ALLOWED_KINDS = {
    "critique",
    "design_requirement",
    "empirical_claim",
    "open_question",
    "prediction",
    "requested_test",
    "success_criterion",
}
ALLOWED_RELEVANCE = {
    "benchmark_context",
    "core",
    "flagship",
    "out_of_scope",
}
ALLOWED_UNCERTAINTY = {"none", "partial_inaudible", "truncated_context"}


def _registry():
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_neel_registry_identity_and_complete_unique_ids():
    registry = _registry()

    assert registry["artifact"] == "stresskit_neel_criteria_registry"
    assert registry["artifact_version"] == "1.0.0"
    assert registry["schema_version"] == "1.0"
    assert registry["source"]["basename"] == "NeelNanda_transcript.txt"
    assert registry["source"]["line_numbering"] == "one_based"

    ids = [criterion["id"] for criterion in registry["criteria"]]
    assert len(ids) == len(set(ids))
    assert ids == EXPECTED_IDS


def test_neel_registry_fields_use_closed_vocabularies():
    for criterion in _registry()["criteria"]:
        assert criterion["timestamp"] == re.fullmatch(
            r"\d{2}:\d{2}:\d{2}", criterion["timestamp"]
        ).group()
        assert isinstance(criterion["source_line"], int)
        assert criterion["source_line"] > 0
        assert criterion["kinds"]
        assert set(criterion["kinds"]) <= ALLOWED_KINDS
        assert criterion["stresskit_relevance"] in ALLOWED_RELEVANCE
        assert criterion["statement"].strip()
        assert criterion["uncertainty"]["status"] in ALLOWED_UNCERTAINTY


def test_neel_registry_matches_external_transcript_when_available():
    if not EXTERNAL_TRANSCRIPT.exists():
        return

    expected = _registry()["source"]["sha256"]
    actual = hashlib.sha256(EXTERNAL_TRANSCRIPT.read_bytes()).hexdigest()
    assert actual == expected


def test_neel_empirical_items_are_not_marked_verified():
    empirical = [
        criterion
        for criterion in _registry()["criteria"]
        if "empirical_claim" in criterion["kinds"]
    ]

    assert empirical
    assert all(item["verification_status"] != "verified" for item in empirical)
    assert all(item["verification_status"] == "not_verified" for item in empirical)


def test_neel_registry_preserves_inaudible_uncertainty():
    criteria = {item["id"]: item for item in _registry()["criteria"]}

    for criterion_id in ("N03", "N28", "N32", "N33", "N48"):
        uncertainty = criteria[criterion_id]["uncertainty"]
        assert uncertainty["status"] == "partial_inaudible"
        assert uncertainty["note"].strip()


def _report_paths():
    return {
        "criteria_path": REGISTRY_PATH,
        "transcript_path": EXTERNAL_TRANSCRIPT,
        "calibration_path": REPO_ROOT / "artifacts" / "calibration" /
        "v1-audit-profiles-2000.json",
        "compiler_path": REPO_ROOT / "artifacts" / "benchmark" /
        "compiler-evaluation-v1-300.json",
        "qualification_path": REPO_ROOT / "artifacts" / "benchmark" /
        "prefreeze-qualification-report-v1.json",
        "live_decision_path": REPO_ROOT / "benchmark" / "intake" /
        "pyvene_interchange_intervention_ioi" / "discovery-decision.json",
        "provider_attestation_path": REPO_ROOT / "benchmark" / "intake" /
        "pyvene_interchange_intervention_ioi" / "provider-panel.attestation.json",
        "flagship_path": REPO_ROOT / "benchmark" / "flagship" /
        "spec.candidate.json",
        "proof_junit_path": REPO_ROOT / "artifacts" / "validation" /
        "neel-adversarial-proof-v1.xml",
        "proof_selector_path": REPO_ROOT / "artifacts" / "validation" /
        "neel-adversarial-proof-selectors-v1.txt",
        "invalid_panel_execution_path": REPO_ROOT / "benchmark" / "intake" /
        "mechtomo_finite_effect_map_recovery" / "panel-execution.json",
        "invalid_panel_decision_path": REPO_ROOT / "benchmark" / "intake" /
        "mechtomo_finite_effect_map_recovery" / "discovery-decision.json",
        "rejected_attempt_path": REPO_ROOT / "benchmark" / "intake" /
        "mechtomo_finite_effect_map_recovery" / "opinions" /
        "extractor-b-attempt-1-rejected.json",
        "double_rejection_panel_execution_path": REPO_ROOT / "benchmark" /
        "intake" / "acdc_tracr_reverse" / "panel-execution.json",
        "double_rejection_panel_decision_path": REPO_ROOT / "benchmark" /
        "intake" / "acdc_tracr_reverse" / "discovery-decision.json",
        "double_rejection_invalid_attempt_path": REPO_ROOT / "benchmark" /
        "intake" / "acdc_tracr_reverse" / "opinions" /
        "extractor-a-attempt-1-rejected.json",
        "double_rejection_incomplete_attempt_path": REPO_ROOT / "benchmark" /
        "intake" / "acdc_tracr_reverse" / "opinions" /
        "extractor-b-attempt-1-rejected.json",
        "utility_preregistration_path": REPO_ROOT / "benchmark" / "utility" /
        "thought_anchors_solution_outcome" / "preregistration.candidate.json",
        "utility_preflight_path": REPO_ROOT / "benchmark" / "utility" /
        "thought_anchors_solution_outcome" / "blind-metadata-preflight.json",
        "flagship_license_audit_path": REPO_ROOT / "benchmark" / "flagship" /
        "license-audit.v1.json",
        "cot_panel_preflight_path": REPO_ROOT / "benchmark" / "intake" /
        "thought_anchors_counterfactual_importance_r1_qwen14b" /
        "authenticated-preflight-blocker.json",
    }


def test_neel_validation_report_is_conservative_when_local_evidence_exists():
    paths = _report_paths()
    if not all(path.is_file() for path in paths.values()):
        return
    report = build_validation_report(**paths)
    conclusion = report["conclusion"]

    assert conclusion["verifier_mechanics_demonstrated"] is True
    assert conclusion["neel_empirical_claims_verified"] is False
    assert conclusion["registered_external_utility_results_available"] is False
    assert report["benchmark"]["registered_final_audit_bundles"] == 0
    assert report["invalid_evidence_panel"]["invalid_quote_count"] == 4
    assert report["invalid_evidence_panel"]["retry_performed"] is False
    assert report["double_rejection_panel"]["accepted_opinions"] == 0
    assert report["double_rejection_panel"]["invalid_quote_count"] == 3
    assert report["double_rejection_panel"]["checked_quote_count"] == 10
    assert report["double_rejection_panel"][
        "rejected_incomplete_completion"
    ] == 1
    assert report["scientific_readiness"]["external_utility"][
        "minimums_established"
    ] is False
    assert report["scientific_readiness"]["external_utility"][
        "claim_outcome_computed"
    ] is False
    assert report["scientific_readiness"]["gradient_persona_flagship"][
        "blocker_count"
    ] == 7
    assert report["scientific_readiness"]["cot_panel"][
        "routes_passed"
    ] == 3
    assert report["scientific_readiness"]["cot_panel"][
        "chat_completion_calls"
    ] == 0
    assert "not** verified" in render_markdown(report)


def test_neel_validation_rejects_failed_calibration(tmp_path):
    paths = _report_paths()
    if not all(path.is_file() for path in paths.values()):
        return
    calibration = json.loads(paths["calibration_path"].read_text(encoding="utf-8"))
    calibration["acceptance"]["passed"] = False
    tampered = tmp_path / "calibration.json"
    tampered.write_text(json.dumps(calibration), encoding="utf-8")
    paths["calibration_path"] = tampered

    with pytest.raises(ValueError, match="calibration acceptance"):
        build_validation_report(**paths)


@pytest.mark.parametrize("unsupported_count", [0, 2, 3, 4])
def test_neel_validation_rejects_generic_abstention_without_exact_panel(
    tmp_path, unsupported_count
):
    paths = _report_paths()
    if not all(path.is_file() for path in paths.values()):
        return
    decision = json.loads(paths["live_decision_path"].read_text(encoding="utf-8"))
    decision["publication_state"] = "abstain"
    decision["candidates"] = []
    decision["problems"] = ["generic abstention"] + [
        f"opinion {index} marks claim unsupported"
        for index in range(unsupported_count)
    ]
    tampered = tmp_path / "generic-abstention.json"
    tampered.write_text(json.dumps(decision), encoding="utf-8")
    paths["live_decision_path"] = tampered

    with pytest.raises(ValueError, match="exact unsupported panel"):
        build_validation_report(**paths)


def test_neel_validation_rejects_false_invalid_quote_record(tmp_path):
    paths = _report_paths()
    if not all(path.is_file() for path in paths.values()):
        return
    attempt = json.loads(paths["rejected_attempt_path"].read_text(encoding="utf-8"))
    attempt["evidence_quote_checks"][0]["present_in_declared_source_bytes"] = True
    tampered = tmp_path / "attempt.json"
    tampered.write_text(json.dumps(attempt), encoding="utf-8")
    paths["rejected_attempt_path"] = tampered

    with pytest.raises(ValueError, match="not all source-invalid"):
        build_validation_report(**paths)


def test_neel_validation_rejects_forged_clean_incomplete_completion(tmp_path):
    paths = _report_paths()
    if not all(path.is_file() for path in paths.values()):
        return
    key = "double_rejection_incomplete_attempt_path"
    attempt = json.loads(paths[key].read_text(encoding="utf-8"))
    attempt["finish_reason"] = "stop"
    tampered = tmp_path / "incomplete-attempt.json"
    tampered.write_text(json.dumps(attempt), encoding="utf-8")
    paths[key] = tampered

    with pytest.raises(ValueError, match="incomplete extractor attempt"):
        build_validation_report(**paths)
