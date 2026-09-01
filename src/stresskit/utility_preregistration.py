"""Fail-closed validation for outcome-blind external-utility preregistrations.

This module validates candidate study designs before a ClaimRecord or AuditSpec
can be frozen.  It never loads labels, predictions, or outcome-bearing dataset
content.  A blocked preregistration is evidence about readiness, not an audit
result.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from .audit_models import AUDIT_SCHEMA_VERSION, SourceBundle
from .audit_profiles import get_profile
from .integrity import (
    digest_json,
    require_sha256_digest,
    sha256_bytes,
)
from .utility import (
    INTERNAL_BASELINE_INPUT_KINDS,
    UtilityMetricSpec,
    canonical_baseline_provenance,
    interpretability_phrasing,
)


ARTIFACT = "stresskit_external_utility_preregistration_candidate"
STATUS = "blocked_not_freezable"
PREFLIGHT_ARTIFACT = "stresskit_utility_blind_metadata_preflight"
PREFLIGHT_STATUS = "failed_insufficient_metadata"

_REQUIRED_BLOCKERS = frozenset({
    "current_utility_v1_requires_internal_method",
    "independent_cluster_cardinality_unverified",
    "rollout_independence_unattested",
    "underlying_math_mirror_equivalence_unresolved",
    "whitebox_features_not_released",
})

_REQUIRED_FORBIDDEN_FEATURES = frozenset({
    "accuracy",
    "answer",
    "counterfactual_importance_accuracy",
    "forced_importance_accuracy",
    "gt_answer",
    "is_correct",
    "path",
    "resampling_importance_accuracy",
    "solution_type_directory",
})

_PARTITIONS = ("train", "primary", "generalization")


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sha40(value: Any, field: str) -> str:
    value = _nonempty(value, field)
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a 40-character lowercase git revision")
    return value


def _validate_source(source: Mapping[str, Any]) -> None:
    if not isinstance(source, Mapping) or set(source) != {
        "code",
        "rollout_dataset",
        "task_dataset",
    }:
        raise ValueError("utility preregistration needs exact code and dataset sources")
    code = source["code"]
    if not isinstance(code, Mapping) or code.get("license") != "MIT" or \
            code.get("license_status") != "verified_compatible":
        raise ValueError("Thought Anchors code license must be verified MIT")
    _sha40(code.get("revision"), "code revision")
    _sha40(code.get("tree"), "code tree")
    require_sha256_digest(code.get("archive_digest"), "code archive_digest")
    require_sha256_digest(code.get("license_digest"), "code license_digest")

    rollout = source["rollout_dataset"]
    if not isinstance(rollout, Mapping) or rollout.get("license") != "MIT" or \
            rollout.get("license_status") != "declared_compatible" or \
            rollout.get("public") is not True or rollout.get("gated") is not False:
        raise ValueError("rollout dataset must be public and declare MIT")
    _sha40(rollout.get("revision"), "rollout dataset revision")
    require_sha256_digest(rollout.get("card_digest"), "rollout card_digest")
    if rollout.get("content_rows_inspected") is not False:
        raise ValueError("rollout preregistration must not inspect content rows")

    task = source["task_dataset"]
    if not isinstance(task, Mapping) or task.get("license") != "MIT" or \
            task.get("canonical_license_status") != "verified_compatible" or \
            task.get("mirror_equivalence_status") != "unresolved":
        raise ValueError("task dataset mirror must remain unresolved until verified")
    _sha40(task.get("canonical_revision"), "task dataset canonical_revision")
    _sha40(task.get("used_mirror_revision"), "task dataset used_mirror_revision")


def _validate_provenance_row(row: Mapping[str, Any], *, method: bool) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("method and baseline provenance rows must be objects")
    implementation = row.get("implementation")
    manifest = row.get("input_manifest_template")
    if not isinstance(implementation, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("method and baselines need implementation and input templates")
    implementation_digest = require_sha256_digest(
        row.get("implementation_digest"), "implementation_digest"
    )
    manifest_digest = require_sha256_digest(
        row.get("input_manifest_template_digest"), "input_manifest_template_digest"
    )
    if digest_json(implementation) != implementation_digest:
        raise ValueError("implementation_digest does not recompute")
    if digest_json(manifest) != manifest_digest:
        raise ValueError("input_manifest_template_digest does not recompute")
    if manifest.get("artifact") != "stresskit_utility_input_manifest_template" or \
            manifest.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("utility input manifest template has invalid header")
    kinds = row.get("allowed_input_kinds")
    if not isinstance(kinds, list) or kinds != sorted(set(kinds)) or not kinds:
        raise ValueError("allowed_input_kinds must be non-empty, sorted, and unique")
    manifest_kinds = manifest.get("allowed_input_kinds")
    if manifest_kinds != kinds:
        raise ValueError("input manifest kinds differ from provenance row")
    if INTERNAL_BASELINE_INPUT_KINDS & set(kinds):
        raise ValueError("CPU preregistration must not mount model internals")
    policy = row.get("access_policy")
    expected_policy = {
        "model_internals": "forbidden",
        "mounted_inputs": "manifest_only",
        "network": "disabled",
    }
    if policy != expected_policy:
        raise ValueError("method/baseline access policy is not fail-closed")
    if row.get("uses_internals") is not False:
        raise ValueError("Thought Anchors rollout path must declare no internals")
    if not method:
        canonical = canonical_baseline_provenance({
            "name": row.get("name"),
            "uses_internals": False,
            "implementation_digest": implementation_digest,
            "input_manifest_digest": manifest_digest,
            "allowed_input_kinds": kinds,
            "access_policy": policy,
        })
        if canonical["name"] != row.get("name"):
            raise AssertionError("baseline canonicalization changed its name")


def _validate_features(method: Mapping[str, Any]) -> None:
    allowed = method.get("feature_allowlist")
    forbidden = method.get("forbidden_features")
    if not isinstance(allowed, list) or allowed != sorted(set(allowed)) or not allowed:
        raise ValueError("method feature_allowlist must be sorted and unique")
    if not isinstance(forbidden, list) or forbidden != sorted(set(forbidden)):
        raise ValueError("method forbidden_features must be sorted and unique")
    if set(allowed) & set(forbidden):
        raise ValueError("method allowed and forbidden features overlap")
    if not _REQUIRED_FORBIDDEN_FEATURES <= set(forbidden):
        raise ValueError("method does not forbid every target-leaking feature")
    if any("accuracy" in feature for feature in allowed):
        raise ValueError("label-derived accuracy features cannot enter method")
    if method.get("target_access") != "sealed_until_final_evaluation":
        raise ValueError("method target labels must remain sealed")


def _validate_splits(splits: Mapping[str, Any], minimum: int) -> None:
    if not isinstance(splits, Mapping) or splits.get("held_out_axes") != [
        "model", "unit"
    ] or splits.get("unit_key") != "math_problem_id" or \
            splits.get("cluster_policy") != "same_problem_is_one_unit_across_all_rows":
        raise ValueError("split plan must hold out model and problem clusters")
    rule = splits.get("assignment_rule")
    if not isinstance(rule, Mapping) or rule.get("algorithm") != "sha256_first_byte" or \
            rule.get("label_blind") is not True:
        raise ValueError("split assignment must be deterministic and label blind")
    salt = require_sha256_digest(rule.get("salt_digest"), "split salt_digest")
    if salt != digest_json(rule.get("salt_payload")):
        raise ValueError("split salt_digest does not recompute")
    partitions = splits.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != set(_PARTITIONS):
        raise ValueError("split plan needs train, primary, and generalization")
    covered = set()
    problem_buckets = {}
    for name in _PARTITIONS:
        row = partitions[name]
        if not isinstance(row, Mapping):
            raise ValueError("split partitions must be objects")
        bucket = row.get("hash_bucket_inclusive")
        if not isinstance(bucket, list) or len(bucket) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) for value in bucket
        ) or not 0 <= bucket[0] <= bucket[1] <= 255:
            raise ValueError("split hash bucket must be an inclusive byte range")
        values = set(range(bucket[0], bucket[1] + 1))
        if covered & values:
            raise ValueError("split hash buckets overlap")
        covered.update(values)
        problem_buckets[name] = values
        if row.get("minimum_unique_problem_clusters") != minimum:
            raise ValueError("every evaluation partition needs registered minimum units")
        models = row.get("model_families")
        if not isinstance(models, list) or not models or models != sorted(set(models)):
            raise ValueError("split model families must be non-empty and sorted")
    if covered != set(range(256)):
        raise ValueError("split hash buckets must cover every byte")
    if set(partitions["primary"]["model_families"]) & set(
        partitions["generalization"]["model_families"]
    ):
        raise ValueError("generalization model family must be held out")
    if problem_buckets["primary"] & problem_buckets["generalization"]:
        raise ValueError("generalization problem buckets must be disjoint")


def _validate_blind_preflight(preflight: Mapping[str, Any]) -> None:
    if not isinstance(preflight, Mapping) or \
            preflight.get("status") != "not_run" or \
            preflight.get("reads_content") is not False or \
            preflight.get("reads_labels") is not False or \
            preflight.get("reads_predictions") is not False or \
            preflight.get("emits") != [
                "partition_cluster_counts", "duplicate_cluster_count"
            ]:
        raise ValueError("blind preflight must remain metadata-only and unrun")
    if preflight.get("allowed_dataset_columns") != [
        "extension", "filename", "path", "size_bytes"
    ] or preflight.get("forbidden_dataset_columns") != ["content"]:
        raise ValueError("blind preflight dataset-column policy is not exact")


def validate_external_utility_preregistration(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate and summarize one blocked, outcome-blind utility candidate."""
    if not isinstance(payload, Mapping) or payload.get("artifact") != ARTIFACT or \
            payload.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("invalid external-utility preregistration header")
    if payload.get("status") != STATUS or payload.get("ready_to_freeze") is not False or \
            payload.get("outcome_blind") is not True or \
            payload.get("publication_state") != "abstain":
        raise ValueError("external-utility candidate must remain blocked and abstained")
    _nonempty(payload.get("candidate_id"), "candidate_id")
    parent_ids = payload.get("registered_parent_claim_ids")
    if parent_ids != ["thought_anchors_counterfactual_importance_r1_qwen14b"]:
        raise ValueError("utility candidate must bind registered Thought Anchors claim")
    _validate_source(payload.get("source", {}))

    task = _nonempty(payload.get("external_task"), "external_task")
    jargon = interpretability_phrasing(task)
    if jargon:
        raise ValueError(
            "external task contains interpretability jargon: " + ", ".join(jargon)
        )
    metric_payload = payload.get("metric_spec")
    if not isinstance(metric_payload, Mapping):
        raise ValueError("utility preregistration needs metric_spec")
    metric = UtilityMetricSpec.from_dict(metric_payload)
    if metric.to_dict() != dict(metric_payload):
        raise ValueError("metric_spec must be canonical")
    profile = get_profile("utility_v1")
    if metric.practical_margin != profile.practical_margin or \
            metric.minimum_independent_units != profile.minimum_independent_units or \
            payload.get("profile_digest") != profile.digest:
        raise ValueError("metric thresholds must come from registered utility_v1")

    method = payload.get("method")
    if not isinstance(method, Mapping):
        raise ValueError("utility preregistration needs method provenance")
    _validate_provenance_row(method, method=True)
    _validate_features(method)
    baselines = payload.get("baselines")
    if not isinstance(baselines, list) or len(baselines) < 3:
        raise ValueError("utility preregistration needs at least three strong baselines")
    names = []
    for row in baselines:
        _validate_provenance_row(row, method=False)
        names.append(row.get("name"))
    if len(names) != len(set(names)):
        raise ValueError("utility baseline names must be unique")

    _validate_splits(payload.get("splits", {}), metric.minimum_independent_units)
    _validate_blind_preflight(payload.get("blind_preflight", {}))
    controls = payload.get("controls")
    if not isinstance(controls, Mapping) or set(controls) != {
        "known_truth_positive", "label_randomization_negative"
    } or controls["known_truth_positive"].get("expected_state") != "pass" or \
            controls["label_randomization_negative"].get("expected_state") != "fail":
        raise ValueError("utility preregistration needs exact positive and negative controls")

    blockers = payload.get("blockers")
    if not isinstance(blockers, list) or any(
        not isinstance(row, Mapping) or not isinstance(row.get("code"), str) or
        not isinstance(row.get("resolution"), str) or not row["resolution"].strip()
        for row in blockers
    ):
        raise ValueError("utility preregistration blockers are incomplete")
    codes = {row["code"] for row in blockers}
    if not _REQUIRED_BLOCKERS <= codes:
        raise ValueError("utility preregistration omits a known blocking condition")
    integrity = payload.get("outcome_integrity")
    if integrity != {
        "final_analysis_run": False,
        "final_dataset_labels_inspected": False,
        "final_dataset_predictions_inspected": False,
        "public_schema_examples_visible": True,
        "thresholds_from_registered_profile": True,
        "unrelated_acdc_outcome_exposure_quarantined": True,
    }:
        raise ValueError("outcome-integrity declaration is not exact")
    proposed = payload.get("proposed_audit_spec")
    if not isinstance(proposed, Mapping) or proposed.get("frozen") is not False or \
            proposed.get("profile_id") != "utility_v1" or \
            proposed.get("evaluation_manifest_status") != \
            "pending_blind_materialization" or \
            proposed.get("audit_spec_digest") is not None:
        raise ValueError("blocked candidate cannot masquerade as frozen AuditSpec")
    return {
        "valid": True,
        "status": STATUS,
        "ready_to_freeze": False,
        "publication_state": "abstain",
        "candidate_id": payload["candidate_id"],
        "blocker_codes": sorted(codes),
        "baseline_count": len(baselines),
        "minimum_independent_units": metric.minimum_independent_units,
        "digest": digest_json(payload),
    }


def build_blind_metadata_preflight(
    preregistration: Mapping[str, Any],
    source_bundle_payload: Mapping[str, Any],
    *,
    revision_manifest_bytes: bytes,
    selected_path_list_bytes: bytes,
) -> Dict[str, Any]:
    """Run label-free readiness checks using file metadata only.

    Problem identifiers live inside rollout JSON rows, outside the frozen
    metadata allowlist. The preflight therefore records that partition
    cardinalities are not computable; it never opens those rows or invents a
    weaker independent unit.
    """
    validate_external_utility_preregistration(preregistration)
    source = SourceBundle.from_dict(source_bundle_payload)
    if source.metadata.get("candidate_id") != \
            "thought_anchors_counterfactual_importance_r1_qwen14b":
        raise ValueError("utility preflight SourceBundle targets another claim")
    documents = {row["document_id"]: row for row in source.documents}
    revision_document = documents.get("math-rollouts-revision-manifest")
    if revision_document is None or revision_document.get("source_digest") != \
            sha256_bytes(revision_manifest_bytes):
        raise ValueError("utility preflight revision manifest bytes differ")
    selected_digest = sha256_bytes(selected_path_list_bytes)
    if selected_digest != source.metadata.get("dataset_subset_path_list_digest"):
        raise ValueError("utility preflight selected path-list bytes differ")

    try:
        revision_manifest = json.loads(revision_manifest_bytes.decode("utf-8"))
        selected_paths = [
            row for row in selected_path_list_bytes.decode("utf-8").splitlines()
            if row
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("utility preflight metadata inputs are invalid") from exc
    siblings = revision_manifest.get("siblings") \
        if isinstance(revision_manifest, Mapping) else None
    if not isinstance(siblings, list) or not siblings or any(
        not isinstance(row, Mapping) or
        not isinstance(row.get("rfilename"), str) or
        not row["rfilename"]
        for row in siblings
    ):
        raise ValueError("utility preflight revision metadata lacks filenames")
    revision_paths = [str(row["rfilename"]) for row in siblings]
    if len(revision_paths) != len(set(revision_paths)) or \
            not selected_paths or len(selected_paths) != len(set(selected_paths)):
        raise ValueError("utility preflight path metadata contains duplicates")

    partitions = preregistration["splits"]["partitions"]
    model_families = sorted({
        model
        for partition in partitions.values()
        for model in partition["model_families"]
    })

    def model_counts(paths: Sequence[str]) -> Dict[str, int]:
        return {
            model: sum(
                path.casefold().startswith(model.casefold() + "/")
                for path in paths
            )
            for model in model_families
        }

    minimum = preregistration["metric_spec"]["minimum_independent_units"]
    artifact = {
        "artifact": PREFLIGHT_ARTIFACT,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "candidate_id": preregistration["candidate_id"],
        "source_bundle_digest": source.digest,
        "preregistration_digest": digest_json(preregistration),
        "rollout_dataset_revision": preregistration["source"][
            "rollout_dataset"
        ]["revision"],
        "status": PREFLIGHT_STATUS,
        "publication_state": "abstain",
        "outcome_blind": True,
        "inputs": {
            "revision_manifest_digest": sha256_bytes(revision_manifest_bytes),
            "selected_path_list_digest": selected_digest,
            "allowed_fields_read": ["rfilename"],
        },
        "observed": {
            "revision_metadata_row_count": len(revision_paths),
            "selected_metadata_row_count": len(selected_paths),
            "revision_model_file_counts": model_counts(revision_paths),
            "selected_model_file_counts": model_counts(selected_paths),
            "partition_cluster_counts": {
                name: None for name in _PARTITIONS
            },
            "duplicate_cluster_count": None,
        },
        "independent_unit": "MATH problem_id cluster",
        "minimum_unique_problem_clusters_per_partition": minimum,
        "cluster_identifier_available_in_allowed_metadata": False,
        "minimums_established": False,
        "path_values_emitted": False,
        "content_rows_read": False,
        "labels_read": False,
        "predictions_read": False,
        "claim_outcome_computed": False,
        "reason_code": (
            "problem_cluster_identifier_absent_from_registered_metadata_contract"
        ),
        "next_action": (
            "obtain a separately licensed label-blind problem-ID manifest for both "
            "model families; otherwise retain abstention"
        ),
        "gpu_used": False,
    }
    validate_blind_metadata_preflight(artifact)
    return artifact


def validate_blind_metadata_preflight(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a fail-closed metadata preflight result."""
    if payload.get("artifact") != PREFLIGHT_ARTIFACT or \
            payload.get("schema_version") != AUDIT_SCHEMA_VERSION or \
            payload.get("status") != PREFLIGHT_STATUS or \
            payload.get("publication_state") != "abstain" or \
            payload.get("outcome_blind") is not True:
        raise ValueError("invalid utility blind-preflight header")
    for field in (
        "source_bundle_digest",
        "preregistration_digest",
    ):
        require_sha256_digest(payload.get(field), field)
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("allowed_fields_read") != [
        "rfilename"
    ]:
        raise ValueError("utility blind preflight read unregistered metadata")
    require_sha256_digest(inputs.get("revision_manifest_digest"), "manifest digest")
    require_sha256_digest(inputs.get("selected_path_list_digest"), "path-list digest")
    observed = payload.get("observed")
    if not isinstance(observed, Mapping) or observed.get(
        "partition_cluster_counts"
    ) != {name: None for name in _PARTITIONS} or \
            observed.get("duplicate_cluster_count") is not None:
        raise ValueError("utility blind preflight fabricated cluster counts")
    for field in ("revision_metadata_row_count", "selected_metadata_row_count"):
        value = observed.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("utility blind preflight metadata count is invalid")
    for field in ("revision_model_file_counts", "selected_model_file_counts"):
        counts = observed.get(field)
        if not isinstance(counts, Mapping) or not counts or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts.values()
        ):
            raise ValueError("utility blind preflight model counts are invalid")
    false_fields = (
        "cluster_identifier_available_in_allowed_metadata",
        "minimums_established",
        "path_values_emitted",
        "content_rows_read",
        "labels_read",
        "predictions_read",
        "claim_outcome_computed",
        "gpu_used",
    )
    if any(payload.get(field) is not False for field in false_fields) or \
            payload.get("reason_code") != \
            "problem_cluster_identifier_absent_from_registered_metadata_contract":
        raise ValueError("utility blind preflight is not fail-closed")
    return {
        "valid": True,
        "status": PREFLIGHT_STATUS,
        "publication_state": "abstain",
        "minimums_established": False,
        "digest": digest_json(payload),
    }


__all__: Sequence[str] = (
    "ARTIFACT",
    "PREFLIGHT_ARTIFACT",
    "PREFLIGHT_STATUS",
    "STATUS",
    "build_blind_metadata_preflight",
    "validate_blind_metadata_preflight",
    "validate_external_utility_preregistration",
)
