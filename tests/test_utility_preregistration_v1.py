"""Outcome-blind external-utility preregistration regressions."""

import copy
import json
from pathlib import Path

import pytest

from stresskit.integrity import digest_json
from stresskit.utility_preregistration import (
    ARTIFACT,
    PREFLIGHT_STATUS,
    STATUS,
    build_blind_metadata_preflight,
    validate_blind_metadata_preflight,
    validate_external_utility_preregistration,
)


REPO_ROOT = Path(__file__).parents[1]
PREREGISTRATION = (
    REPO_ROOT
    / "benchmark"
    / "utility"
    / "thought_anchors_solution_outcome"
    / "preregistration.candidate.json"
)
SCHEMA = (
    REPO_ROOT
    / "src"
    / "stresskit"
    / "schemas"
    / "external_utility_preregistration_v1.json"
)
SOURCE_BUNDLE = (
    REPO_ROOT
    / "benchmark"
    / "intake"
    / "thought_anchors_counterfactual_importance_r1_qwen14b"
    / "source-bundle.json"
)
PREFLIGHT = PREREGISTRATION.parent / "blind-metadata-preflight.json"
RAW_ROOT = (
    REPO_ROOT
    / ".stresskit"
    / "intake"
    / "thought_anchors_counterfactual_importance_r1_qwen14b"
    / "raw"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_thought_anchors_utility_candidate_is_valid_but_blocked():
    payload = _load(PREREGISTRATION)
    result = validate_external_utility_preregistration(payload)

    assert result["valid"] is True
    assert result["status"] == STATUS
    assert result["ready_to_freeze"] is False
    assert result["publication_state"] == "abstain"
    assert result["baseline_count"] == 4
    assert result["minimum_independent_units"] == 200
    assert result["digest"] == digest_json(payload)


def test_utility_candidate_binds_registered_claim_and_upstream_revision():
    payload = _load(PREREGISTRATION)
    registry = _load(REPO_ROOT / "benchmark" / "registry.candidates.json")
    claim_id = payload["registered_parent_claim_ids"][0]
    candidate = next(
        row for row in registry["entries"] if row["claim_id"] == claim_id
    )
    upstream = registry["upstreams"][candidate["upstream"]]

    assert candidate["upstream"] == "thought_anchors"
    assert payload["source"]["code"]["revision"] == upstream["commit"]
    assert payload["source"]["code"]["license"] == upstream["source_license"]


def test_utility_candidate_schema_declares_fail_closed_publication_state():
    schema = _load(SCHEMA)

    assert schema["properties"]["artifact"]["const"] == ARTIFACT
    assert schema["properties"]["status"]["const"] == STATUS
    assert schema["properties"]["ready_to_freeze"]["const"] is False
    assert schema["properties"]["outcome_blind"]["const"] is True
    assert schema["properties"]["publication_state"]["const"] == "abstain"
    assert schema["additionalProperties"] is False


def test_utility_candidate_cannot_masquerade_as_frozen_spec():
    payload = _load(PREREGISTRATION)
    payload["ready_to_freeze"] = True
    payload["status"] = "frozen"
    payload["publication_state"] = "final"
    payload["proposed_audit_spec"]["frozen"] = True
    payload["proposed_audit_spec"]["audit_spec_digest"] = digest_json(
        {"forged": "audit spec"}
    )

    with pytest.raises(ValueError, match="blocked and abstained"):
        validate_external_utility_preregistration(payload)


def test_utility_candidate_rejects_target_leakage():
    payload = _load(PREREGISTRATION)
    payload["method"]["feature_allowlist"].append("is_correct")
    payload["method"]["feature_allowlist"].sort()

    with pytest.raises(ValueError, match="overlap"):
        validate_external_utility_preregistration(payload)


def test_utility_candidate_rejects_forged_baseline_implementation():
    payload = _load(PREREGISTRATION)
    payload["baselines"][0]["implementation"]["hash_bins"] = 8

    with pytest.raises(ValueError, match="implementation_digest"):
        validate_external_utility_preregistration(payload)


def test_utility_candidate_rejects_dependent_or_overlapping_units():
    payload = _load(PREREGISTRATION)
    payload["splits"]["partitions"]["generalization"][
        "hash_bucket_inclusive"
    ] = [180, 255]

    with pytest.raises(ValueError, match="overlap"):
        validate_external_utility_preregistration(payload)


def test_utility_candidate_rejects_outcome_exposure():
    payload = _load(PREREGISTRATION)
    payload["outcome_integrity"]["final_dataset_labels_inspected"] = True

    with pytest.raises(ValueError, match="outcome-integrity"):
        validate_external_utility_preregistration(payload)


@pytest.mark.parametrize(
    "code",
    [
        "current_utility_v1_requires_internal_method",
        "independent_cluster_cardinality_unverified",
        "rollout_independence_unattested",
        "underlying_math_mirror_equivalence_unresolved",
        "whitebox_features_not_released",
    ],
)
def test_utility_candidate_cannot_hide_known_blockers(code):
    payload = _load(PREREGISTRATION)
    payload["blockers"] = [
        row for row in payload["blockers"] if row["code"] != code
    ]

    with pytest.raises(ValueError, match="blocking condition"):
        validate_external_utility_preregistration(payload)


def test_utility_candidate_baselines_never_mount_model_internals():
    payload = _load(PREREGISTRATION)
    changed = copy.deepcopy(payload)
    changed["baselines"][0]["allowed_input_kinds"].append("activation")
    changed["baselines"][0]["allowed_input_kinds"].sort()
    changed["baselines"][0]["input_manifest_template"][
        "allowed_input_kinds"
    ] = changed["baselines"][0]["allowed_input_kinds"]
    changed["baselines"][0]["input_manifest_template_digest"] = digest_json(
        changed["baselines"][0]["input_manifest_template"]
    )

    with pytest.raises(ValueError, match="must not mount model internals"):
        validate_external_utility_preregistration(changed)


def test_current_blind_metadata_preflight_fails_without_opening_outcomes():
    payload = _load(PREFLIGHT)
    result = validate_blind_metadata_preflight(payload)

    assert result["valid"] is True
    assert result["status"] == PREFLIGHT_STATUS
    assert result["publication_state"] == "abstain"
    assert result["minimums_established"] is False
    assert payload["observed"]["partition_cluster_counts"] == {
        "train": None,
        "primary": None,
        "generalization": None,
    }
    assert payload["content_rows_read"] is False
    assert payload["labels_read"] is False
    assert payload["predictions_read"] is False
    assert payload["claim_outcome_computed"] is False
    assert payload["gpu_used"] is False


def test_blind_metadata_preflight_recomputes_from_local_metadata_when_available():
    manifest = RAW_ROOT / "math-rollouts-api.json"
    selected = RAW_ROOT / "qwen14b-chunks-labeled-paths.txt"
    if not manifest.is_file() or not selected.is_file():
        return
    rebuilt = build_blind_metadata_preflight(
        _load(PREREGISTRATION),
        _load(SOURCE_BUNDLE),
        revision_manifest_bytes=manifest.read_bytes(),
        selected_path_list_bytes=selected.read_bytes(),
    )

    assert rebuilt == _load(PREFLIGHT)


def test_blind_metadata_preflight_rejects_fabricated_cluster_counts():
    payload = _load(PREFLIGHT)
    payload["observed"]["partition_cluster_counts"]["primary"] = 200
    payload["minimums_established"] = True

    with pytest.raises(ValueError, match="fabricated cluster counts"):
        validate_blind_metadata_preflight(payload)
