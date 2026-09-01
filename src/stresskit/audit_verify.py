"""Offline verification of complete StressKit v1 audit bundles and releases."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .audit_compile import detect_prompt_injection, regenerate_run_manifest, validate_agent_panel
from .audit_models import (
    AgentOpinion,
    AuditBundle,
    AuditDecision,
    AuditSpec,
    ClaimRecord,
    ResourcePlan,
    RunAttestation,
    SourceBundle,
)
from .audit_profiles import (
    claim_support,
    finding_similarity,
    get_profile,
    holm_bonferroni,
    reduce_raw_output,
    reducer_digest,
    threshold_check,
    validate_expected_target,
    validate_reducer_config,
)
from .integrity import (
    canonical_json_bytes,
    ContentAddressedStore,
    digest_json,
    verify_digest_closure,
    verify_mapping_signature,
)


_SAFE_EXECUTION = {
    "network": "disabled",
    "credentials": "absent",
    "inputs": "read_only",
    "scratch": "quota_limited",
    "outputs": "allowlisted",
}


def _axis(state: str, checks: Mapping[str, Any], reason: str = "") -> Dict[str, Any]:
    row: Dict[str, Any] = {"state": state, "checks": dict(checks)}
    if reason:
        row["reason"] = reason
    return row


def _status_decision(
    spec: AuditSpec,
    *,
    status: str,
    reproduction: Mapping[str, Any],
    stability_specificity: Mapping[str, Any],
    utility: Mapping[str, Any],
    generalization: Mapping[str, Any],
    evidence_confidence: Mapping[str, Any],
    primary_checks: Mapping[str, Any],
    reasons: Sequence[str],
) -> AuditDecision:
    claim = ClaimRecord.from_dict(spec.claim_record)
    return AuditDecision(
        claim_id=claim.claim_id,
        audit_id=spec.audit_id,
        status=status,
        publication_state="abstain" if status == "abstain" else "final",
        reproduction=reproduction,
        stability_specificity=stability_specificity,
        utility=utility,
        generalization=generalization,
        evidence_confidence=evidence_confidence,
        primary_checks=primary_checks,
        reasons=list(reasons),
        external_validation=spec.external_validation,
    )


def _abstain_without_spec(reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "verified": False,
        "problems": [reason],
        "status": "abstain",
        "publication_state": "abstain",
        "decision": None,
        "primary_p_values": {},
    }


def _utility_baseline_registry(claim: ClaimRecord) -> List[Mapping[str, Any]]:
    registry = claim.reducer.get("config", {}).get("baseline_registry", [])
    return list(registry) if isinstance(registry, list) else []


def _utility_input_manifest_digests(claim: ClaimRecord) -> List[str]:
    return sorted({
        str(row["input_manifest_digest"])
        for row in _utility_baseline_registry(claim) if isinstance(row, Mapping)
    })


def _safe_isolation(
    attestation: RunAttestation, input_manifest_digests: Sequence[str]
) -> bool:
    return all(attestation.isolation.get(key) == value
               for key, value in _SAFE_EXECUTION.items()) and bool(
                   attestation.isolation.get("execution_environment_id")
               ) and attestation.isolation.get("input_manifest_digests") == list(
                   input_manifest_digests
               )


def _numeric_close(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and \
            abs(float(left) - float(right)) <= tolerance
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _numeric_close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numeric_close(a, b, tolerance) for a, b in zip(left, right)
        )
    return left == right


def _utc_timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        dt.timezone.utc
    )


def _disjoint_similarities(
    rows: Sequence[Tuple[RunAttestation, Mapping[str, Any], Mapping[str, Any]]],
    profile_id: str,
) -> List[float]:
    ordered = sorted(rows, key=lambda row: (row[0].dependency_id, row[0].slot_id))
    values = []
    for index in range(0, len(ordered) - 1, 2):
        left, right = ordered[index][0], ordered[index + 1][0]
        if left.dependency_id == right.dependency_id or \
                left.cluster_id == right.cluster_id:
            raise ValueError(
                "dependent or same-cluster runs cannot form an independent stability pair"
            )
        values.append(finding_similarity(
            profile_id, ordered[index][1], ordered[index + 1][1]
        ))
    return values


def _paired_support_differences(
    left_rows: Sequence[Tuple[RunAttestation, Mapping[str, Any], Mapping[str, Any]]],
    right_rows: Sequence[Tuple[RunAttestation, Mapping[str, Any], Mapping[str, Any]]],
    manifest_by_slot: Mapping[str, Mapping[str, Any]],
    profile_id: str,
    expected: Any,
) -> List[float]:
    """Pair independent runs by frozen specification/index, never by outcomes."""
    def key(
        row: Tuple[RunAttestation, Mapping[str, Any], Mapping[str, Any]]
    ) -> Tuple[str, int, str]:
        manifest = manifest_by_slot[row[0].slot_id]
        return (
            str(manifest["specification_id"]),
            int(manifest["index"]),
            row[0].slot_id,
        )
    left_ordered = sorted(left_rows, key=key)
    right_ordered = sorted(right_rows, key=key)
    differences = []
    used_dependencies = set()
    used_clusters = set()
    for left, right in zip(left_ordered, right_ordered):
        attestations = (left[0], right[0])
        dependencies = {row.dependency_id for row in attestations}
        clusters = {row.cluster_id for row in attestations}
        if len(dependencies) != 2 or len(clusters) != 2 or \
                dependencies & used_dependencies or clusters & used_clusters:
            raise ValueError(
                "specificity units reuse a dependency or cluster identifier"
            )
        used_dependencies.update(dependencies)
        used_clusters.update(clusters)
        differences.append(
            claim_support(profile_id, left[1], expected)
            - claim_support(profile_id, right[1], expected)
        )
    return differences


def _load_provenance(
    spec: AuditSpec,
    store: ContentAddressedStore,
) -> Tuple[SourceBundle, List[AgentOpinion], List[str]]:
    claim = ClaimRecord.from_dict(spec.claim_record)
    problems: List[str] = []
    source_payload = store.get_json(claim.source_bundle_digest)
    source = SourceBundle.from_dict(source_payload)
    if source.digest != claim.source_bundle_digest:
        problems.append("SourceBundle canonical digest mismatch")
    opinions = []
    for digest in claim.agent_opinion_digests:
        opinion = AgentOpinion.from_dict(store.get_json(digest))
        if opinion.digest != digest:
            problems.append(f"AgentOpinion canonical digest mismatch: {digest}")
        descriptor = store.get_json(opinion.model_digest)
        expected_descriptor = {
            "provider": opinion.provider,
            "model": opinion.model,
            "family": opinion.model_family,
        }
        if not isinstance(descriptor, Mapping) or any(
                descriptor.get(key) != value
                for key, value in expected_descriptor.items()):
            problems.append(
                f"AgentOpinion model descriptor mismatch: {opinion.opinion_id}"
            )
        opinions.append(opinion)

    source_texts: Dict[str, str] = {}
    for document in source.documents:
        document_id = str(document["document_id"])
        license_row = document["license"]
        license_evidence = store.get_json(license_row["evidence_digest"])
        if not isinstance(license_evidence, Mapping) or any(
                license_evidence.get(key) != license_row[key]
                for key in ("status", "identifier")):
            problems.append(
                f"document {document_id!r} license evidence does not match metadata"
            )
        text_digest = document.get("extracted_text_digest", document["source_digest"])
        try:
            source_texts[document_id] = store.get_bytes(text_digest).decode("utf-8")
        except UnicodeDecodeError:
            problems.append(
                f"document {document_id!r} has no auditable UTF-8 text extraction"
            )
    if claim.code_map["repository_digest"] not in {
            document["source_digest"] for document in source.documents}:
        problems.append("ClaimRecord code_map repository is absent from SourceBundle")
    if not problems:
        problems.extend(validate_agent_panel(source, opinions, source_texts=source_texts))
    for document_id, text in source_texts.items():
        matches = detect_prompt_injection(text)
        if matches:
            problems.append(
                f"document {document_id!r} contains instruction-like text: "
                + ", ".join(matches)
            )
    return source, opinions, problems


def _verify_locator(
    claim: ClaimRecord,
    source: SourceBundle,
    store: ContentAddressedStore,
) -> List[str]:
    problems: List[str] = []
    documents = {str(row["document_id"]): row for row in source.documents}
    locator = claim.claim_locator
    document = documents.get(str(locator.get("document_id")))
    if document is None:
        return ["claim locator document is absent from SourceBundle"]
    if document["source_digest"] != claim.source_digest:
        problems.append("claim source_digest differs from located source document")
    try:
        quote = store.get_bytes(str(locator["quote_digest"]))
    except (KeyError, FileNotFoundError, ValueError) as exc:
        return problems + [f"claim quote is unavailable: {exc}"]
    start, end = locator.get("start"), locator.get("end")
    if isinstance(start, int) and isinstance(end, int):
        text_digest = document.get("extracted_text_digest", document["source_digest"])
        text = store.get_bytes(text_digest)
        if start < 0 or end < start or end > len(text):
            problems.append("claim locator byte range lies outside source text")
        elif text[start:end] != quote:
            problems.append("claim quote does not match exact source byte range")
    return problems


def _closure_roots(spec: AuditSpec, attestations: Sequence[RunAttestation]) -> List[str]:
    claim = ClaimRecord.from_dict(spec.claim_record)
    roots = [claim.source_bundle_digest, *claim.agent_opinion_digests]
    roots.extend([
        claim.code_map["repository_digest"],
        claim.code_map["dependency_manifest_digest"],
        claim.code_map["build_recipe_digest"],
    ])
    universe_digest = claim.reducer.get("config", {}).get(
        "component_universe_digest"
    )
    if isinstance(universe_digest, str):
        roots.append(universe_digest)
    for baseline in _utility_baseline_registry(claim):
        for name in ("implementation_digest", "input_manifest_digest"):
            digest = baseline.get(name) if isinstance(baseline, Mapping) else None
            if isinstance(digest, str):
                roots.append(digest)
    evaluation_manifests = spec.design.get("evaluation_manifests", {})
    if isinstance(evaluation_manifests, Mapping):
        for manifest in evaluation_manifests.values():
            if not isinstance(manifest, Mapping):
                continue
            axis_ids = manifest.get("axis_ids", {})
            if isinstance(axis_ids, Mapping):
                for identifiers in axis_ids.values():
                    if isinstance(identifiers, list):
                        roots.extend(
                            value for value in identifiers if isinstance(value, str)
                        )
    release_digest = spec.multiplicity_family.get("release_manifest_digest")
    if isinstance(release_digest, str):
        roots.append(release_digest)
    for attestation in attestations:
        for digest in (
            attestation.output_digest,
            attestation.error_digest,
        ):
            if digest is not None:
                roots.append(digest)
    return roots


def _apply_holm_to_checks(
    checks: Mapping[str, Any], holm_rows: Mapping[str, Mapping[str, Any]], claim_id: str
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for name, check in checks.items():
        row = dict(check)
        key = f"{claim_id}:{name}"
        correction = holm_rows.get(key)
        if correction is not None:
            row["holm"] = dict(correction)
            if row.get("state") == "pass" and not correction.get("rejected"):
                row["state"] = "inconclusive"
                row["reason"] = "primary result does not survive global Holm-Bonferroni"
        output[name] = row
    return output


def _summarize_state(checks: Iterable[Mapping[str, Any]]) -> str:
    states = [check.get("state") for check in checks]
    if any(state == "fail" for state in states):
        return "fail"
    if states and all(state == "pass" for state in states):
        return "pass"
    return "inconclusive"


def _verify_bundle_base(
    bundle: AuditBundle,
    store: ContentAddressedStore,
    trusted_plan_keys: Mapping[str, bytes],
    trusted_executor_keys: Mapping[str, bytes],
) -> Dict[str, Any]:
    protocol_problems: List[str] = []
    unsafe_problems: List[str] = []
    if set(trusted_plan_keys) & set(trusted_executor_keys):
        protocol_problems.append("plan and executor trust domains reuse key IDs")
    if set(trusted_plan_keys.values()) & set(trusted_executor_keys.values()):
        protocol_problems.append("plan and executor trust domains reuse key material")
    try:
        spec = AuditSpec.from_dict(bundle.audit_spec)
        plan = ResourcePlan.from_dict(bundle.resource_plan)
        claim = ClaimRecord.from_dict(spec.claim_record)
        profile = get_profile(spec.profile_id)
    except (KeyError, TypeError, ValueError) as exc:
        return _abstain_without_spec(str(exc))

    if bundle.audit_spec_digest != spec.digest:
        protocol_problems.append("AuditBundle audit_spec_digest does not recompute")
    if bundle.resource_plan_digest != plan.digest:
        protocol_problems.append("AuditBundle resource_plan_digest does not recompute")
    if plan.audit_spec_digest != spec.digest:
        protocol_problems.append("ResourcePlan targets a different AuditSpec")
    if profile.digest != spec.profile_digest:
        protocol_problems.append("AuditSpec profile_digest differs from registry")
    if claim.profile_id != profile.profile_id or claim.finding_type != profile.finding_type:
        unsafe_problems.append("claim type is unsupported by its frozen profile")
    if claim.reducer.get("name") != profile.reducer_name or \
            claim.reducer.get("implementation_digest") != reducer_digest(profile.reducer_name):
        unsafe_problems.append("claim reducer is absent or unsupported")
    try:
        validate_reducer_config(
            profile.profile_id, claim.reducer.get("config", {})
        )
    except (TypeError, ValueError) as exc:
        unsafe_problems.append(f"invalid frozen reducer config: {exc}")
    try:
        regenerated = regenerate_run_manifest(spec.audit_id, spec.design)
        if regenerated != list(spec.run_manifest):
            protocol_problems.append("run manifest does not regenerate from joint design")
        if digest_json(regenerated) != spec.manifest_digest:
            protocol_problems.append("regenerated manifest digest mismatch")
    except (TypeError, ValueError) as exc:
        protocol_problems.append(f"invalid frozen run manifest: {exc}")
    input_manifest_digests = _utility_input_manifest_digests(claim)
    expected_execution = {
        **_SAFE_EXECUTION,
        "input_manifest_digests": input_manifest_digests,
    }
    if plan.sandbox.get("execution") != expected_execution:
        unsafe_problems.append("ResourcePlan execution sandbox lacks required isolation")
    expected_build = {
        "disposable": True,
        "network": "enabled",
        "credentials": "absent",
        "dependency_manifest_digest": claim.code_map["dependency_manifest_digest"],
        "build_recipe_digest": claim.code_map["build_recipe_digest"],
    }
    if plan.sandbox.get("build") != expected_build:
        protocol_problems.append("ResourcePlan build sandbox differs from ClaimRecord")
    if not verify_mapping_signature(plan.to_dict(), trusted_plan_keys):
        protocol_problems.append("ResourcePlan signature is missing, untrusted, or tampered")
    if plan.hardware_class != spec.reproducibility.get("hardware_class"):
        protocol_problems.append("ResourcePlan hardware class differs from AuditSpec")
    if plan.resources.get("run_slots") != len(spec.run_manifest):
        protocol_problems.append("ResourcePlan run_slots differs from frozen manifest")

    attestations: List[RunAttestation] = []
    plan_created = _utc_timestamp(plan.created_at)
    bundle_created = _utc_timestamp(bundle.created_at)
    for raw in bundle.attestations:
        try:
            attestation = RunAttestation.from_dict(raw)
        except (TypeError, ValueError) as exc:
            protocol_problems.append(f"invalid RunAttestation: {exc}")
            continue
        attestations.append(attestation)
        started = _utc_timestamp(attestation.started_at)
        finished = _utc_timestamp(attestation.finished_at)
        if started < plan_created:
            protocol_problems.append(
                f"slot {attestation.slot_id}: execution predates ResourcePlan"
            )
        if (finished - started).total_seconds() > plan.resources["wall_time_seconds"]:
            protocol_problems.append(
                f"slot {attestation.slot_id}: execution exceeds signed wall time"
            )
        if finished > bundle_created:
            protocol_problems.append(
                f"slot {attestation.slot_id}: execution finishes after AuditBundle creation"
            )
        if not verify_mapping_signature(attestation.to_dict(), trusted_executor_keys):
            protocol_problems.append(
                f"slot {attestation.slot_id}: signature missing, untrusted, or tampered"
            )
        if not _safe_isolation(attestation, input_manifest_digests):
            unsafe_problems.append(
                f"slot {attestation.slot_id}: executor lacks required isolation"
            )
        if attestation.audit_spec_digest != spec.digest:
            protocol_problems.append(
                f"slot {attestation.slot_id}: audit_spec_digest mismatch"
            )
        if attestation.resource_plan_digest != plan.digest:
            protocol_problems.append(
                f"slot {attestation.slot_id}: resource_plan_digest mismatch"
            )
        if attestation.hardware_class != plan.hardware_class:
            protocol_problems.append(
                f"slot {attestation.slot_id}: hardware_class mismatch"
            )
        if attestation.status != "success" and attestation.error_digest is None:
            protocol_problems.append(
                f"slot {attestation.slot_id}: terminal failure has no error artifact"
            )

    manifest_by_slot = {str(row["slot_id"]): row for row in spec.run_manifest}
    attestation_by_slot: Dict[str, RunAttestation] = {}
    for attestation in attestations:
        if attestation.slot_id in attestation_by_slot:
            protocol_problems.append(f"duplicate attestation for slot {attestation.slot_id}")
        attestation_by_slot[attestation.slot_id] = attestation
    missing_slots = sorted(set(manifest_by_slot) - set(attestation_by_slot))
    extra_slots = sorted(set(attestation_by_slot) - set(manifest_by_slot))
    if missing_slots:
        protocol_problems.append(
            f"run slots disappeared from bundle: {missing_slots[:10]}"
        )
    if extra_slots:
        protocol_problems.append(f"unregistered run slots present: {extra_slots[:10]}")
    for slot_id in set(manifest_by_slot) & set(attestation_by_slot):
        manifest_row = manifest_by_slot[slot_id]
        attestation = attestation_by_slot[slot_id]
        if attestation.dependency_id != manifest_row["dependency_id"] or \
                attestation.cluster_id != manifest_row["cluster_id"]:
            protocol_problems.append(
                f"slot {slot_id}: dependency/cluster identity differs from manifest"
            )

    try:
        verify_digest_closure(
            store, list(bundle.content), _closure_roots(spec, attestations)
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        protocol_problems.append(f"content digest closure does not verify: {exc}")

    if _utility_baseline_registry(claim):
        from .utility import validate_utility_input_manifest

        for baseline in _utility_baseline_registry(claim):
            baseline_name = baseline.get("name") if isinstance(baseline, Mapping) else None
            try:
                store.get_bytes(str(baseline["implementation_digest"]))
                input_manifest = store.get_json(str(baseline["input_manifest_digest"]))
                validate_utility_input_manifest(
                    input_manifest, baseline["allowed_input_kinds"]
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
                protocol_problems.append(
                    f"utility baseline {baseline_name!r} provenance does not verify: {exc}"
                )

    provenance_problems: List[str] = []
    try:
        source, _, panel_problems = _load_provenance(spec, store)
        provenance_problems.extend(panel_problems)
        provenance_problems.extend(_verify_locator(claim, source, store))
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        provenance_problems.append(f"source/agent provenance does not verify: {exc}")
    unsafe_problems.extend(provenance_problems)

    if unsafe_problems:
        decision = _status_decision(
            spec,
            status="abstain",
            reproduction=_axis("abstain", {}, "unsafe or unsupported protocol"),
            stability_specificity=_axis("abstain", {}),
            utility=_axis("abstain", {}),
            generalization=_axis("abstain", {}),
            evidence_confidence=_axis("insufficient", {}, "; ".join(unsafe_problems)),
            primary_checks={},
            reasons=unsafe_problems,
        )
        return {
            "ok": False,
            "verified": False,
            "problems": protocol_problems + unsafe_problems,
            "protocol_problems": protocol_problems,
            "unsafe_problems": unsafe_problems,
            "decision": decision,
            "status": decision.status,
            "publication_state": decision.publication_state,
            "primary_p_values": {},
            "spec": spec,
        }
    if protocol_problems:
        decision = _status_decision(
            spec,
            status="protocol_deviation",
            reproduction=_axis("fail", {}, "bundle integrity or protocol mismatch"),
            stability_specificity=_axis("not_evaluated", {}),
            utility=_axis("not_evaluated", {}),
            generalization=_axis("not_evaluated", {}),
            evidence_confidence=_axis("insufficient", {}, "; ".join(protocol_problems)),
            primary_checks={},
            reasons=protocol_problems,
        )
        return {
            "ok": False,
            "verified": False,
            "problems": protocol_problems,
            "protocol_problems": protocol_problems,
            "unsafe_problems": [],
            "decision": decision,
            "status": decision.status,
            "publication_state": decision.publication_state,
            "primary_p_values": {},
            "spec": spec,
        }

    failed_runs = [attestation for attestation in attestations
                   if attestation.status != "success"]
    if failed_runs:
        counts: Dict[str, int] = {}
        for attestation in failed_runs:
            counts[attestation.status] = counts.get(attestation.status, 0) + 1
        decision = _status_decision(
            spec,
            status="reproduction_failure",
            reproduction=_axis("fail", {"terminal_status_counts": counts},
                               "one or more frozen slots did not succeed"),
            stability_specificity=_axis("not_evaluated", {}),
            utility=_axis("not_evaluated", {}),
            generalization=_axis("not_evaluated", {}),
            evidence_confidence=_axis("sufficient", {
                "complete_slots": len(attestations),
                "declared_slots": len(spec.run_manifest),
            }),
            primary_checks={},
            reasons=["failed, crashed, timed-out, or missing runs remain in denominator"],
        )
        return {
            "ok": True,
            "verified": True,
            "problems": [],
            "decision": decision,
            "status": decision.status,
            "publication_state": decision.publication_state,
            "primary_p_values": {},
            "spec": spec,
        }

    component_universe = None
    if profile.reducer_name == "set_graph":
        config = claim.reducer["config"]
        try:
            universe = store.get_json(config["component_universe_digest"])
            if not isinstance(universe, list):
                raise ValueError("component universe object must be a JSON list")
            tokens = [canonical_json_bytes(value).decode("utf-8") for value in universe]
            if len(tokens) != len(set(tokens)):
                raise ValueError("component universe entries must be unique")
            if len(tokens) != config["component_universe_size"]:
                raise ValueError("component universe size differs from ClaimRecord")
            component_universe = set(tokens)
        except (KeyError, TypeError, ValueError) as exc:
            protocol_problems.append(f"component universe does not verify: {exc}")

    config = claim.reducer.get("config", {})
    for target_name, target in (
        ("claim", claim.task.get("expected")),
        ("positive control", claim.controls.get("positive", {}).get("expected")),
        ("negative control", claim.controls.get("negative", {}).get("expected")),
    ):
        try:
            validate_expected_target(
                profile.profile_id,
                target,
                config,
                component_universe=component_universe,
            )
        except (KeyError, TypeError, ValueError) as exc:
            unsafe_problems.append(f"invalid {target_name} target: {exc}")

    loaded: Dict[str, Tuple[RunAttestation, Mapping[str, Any], Mapping[str, Any]]] = {}
    for attestation in attestations:
        try:
            raw = store.get_json(str(attestation.output_digest))
            if not isinstance(raw, Mapping):
                raise ValueError("raw run output must be a JSON object")
            finding = reduce_raw_output(
                profile.profile_id, raw,
                reducer_config=claim.reducer.get("config", {}),
            )
            if component_universe is not None and not set(
                    finding["components"]) <= component_universe:
                raise ValueError("finding contains component outside frozen universe")
            if digest_json(finding) != attestation.finding_digest:
                raise ValueError("stored finding_digest differs from trusted reducer")
            loaded[attestation.slot_id] = (attestation, finding, raw)
        except (KeyError, TypeError, ValueError) as exc:
            protocol_problems.append(f"slot {attestation.slot_id}: {exc}")
    if protocol_problems:
        decision = _status_decision(
            spec,
            status="protocol_deviation",
            reproduction=_axis("fail", {}, "raw outputs or reducers do not verify"),
            stability_specificity=_axis("not_evaluated", {}),
            utility=_axis("not_evaluated", {}),
            generalization=_axis("not_evaluated", {}),
            evidence_confidence=_axis("insufficient", {}, "; ".join(protocol_problems)),
            primary_checks={},
            reasons=protocol_problems,
        )
        return {
            "ok": False,
            "verified": False,
            "problems": protocol_problems,
            "decision": decision,
            "status": decision.status,
            "publication_state": decision.publication_state,
            "primary_p_values": {},
            "spec": spec,
        }

    by_partition: Dict[Tuple[str, str], List[Tuple[RunAttestation, Mapping[str, Any], Mapping[str, Any]]]] = {}
    for slot_id, loaded_row in loaded.items():
        manifest_row = manifest_by_slot[slot_id]
        key = (str(manifest_row["cohort"]), str(manifest_row["partition"]))
        by_partition.setdefault(key, []).append(loaded_row)

    reproduction_checks: Dict[str, Any] = {}
    reproduction_failures: List[str] = []
    final_environment_ids = {
        row[0].isolation.get("execution_environment_id")
        for key, rows in by_partition.items() if key[0] == "final" for row in rows
    }
    replication_environment_ids = {
        row[0].isolation.get("execution_environment_id")
        for key, rows in by_partition.items() if key[0] == "replication" for row in rows
    }
    if final_environment_ids & replication_environment_ids:
        reproduction_failures.append(
            "replication reused a final-cohort execution environment"
        )
    level = str(spec.reproducibility["level"])
    tolerance = float(spec.reproducibility.get("tolerance", 0.0))
    final_by_slot = loaded
    comparisons = 0
    matches = 0
    for row in spec.run_manifest:
        replicate_of = row.get("replicate_of")
        if not isinstance(replicate_of, str):
            continue
        replication = loaded[str(row["slot_id"])]
        original = final_by_slot[replicate_of]
        comparisons += 1
        if level == "bitwise":
            matched = replication[0].output_digest == original[0].output_digest
        elif level == "numeric_with_tolerance":
            matched = _numeric_close(replication[1], original[1], tolerance)
        else:
            matched = finding_similarity(
                profile.profile_id, replication[1], original[1]
            ) >= profile.generalization_min
        matches += int(matched)
    reproduction_checks["independent_rerun"] = {
        "level": level,
        "comparisons": comparisons,
        "matches": matches,
        "state": "pass" if comparisons and matches == comparisons else "fail",
    }
    if not comparisons or matches != comparisons:
        reproduction_failures.append("independent rerun did not meet declared level")

    expected = claim.task.get("expected")
    if expected is None:
        unsafe_problems.append("ClaimRecord task lacks expected target for falsification")
    positive = claim.controls.get("positive", {})
    negative = claim.controls.get("negative", {})
    if "expected" not in positive or "expected" not in negative:
        unsafe_problems.append("known-truth positive and negative targets are required")
    if unsafe_problems:
        decision = _status_decision(
            spec,
            status="abstain",
            reproduction=_axis("pass" if not reproduction_failures else "fail",
                               reproduction_checks),
            stability_specificity=_axis("abstain", {}),
            utility=_axis("abstain", {}),
            generalization=_axis("abstain", {}),
            evidence_confidence=_axis("insufficient", {}, "; ".join(unsafe_problems)),
            primary_checks={},
            reasons=unsafe_problems,
        )
        return {
            "ok": False,
            "verified": False,
            "problems": unsafe_problems,
            "decision": decision,
            "status": decision.status,
            "publication_state": decision.publication_state,
            "primary_p_values": {},
            "spec": spec,
        }

    primary_rows = by_partition[("final", "primary")]
    positive_rows = by_partition[("final", "positive_control")]
    negative_rows = by_partition[("final", "negative_control")]
    generalization_rows = by_partition[("final", "generalization")]

    try:
        primary_support = [claim_support(profile.profile_id, row[1], expected)
                           for row in primary_rows]
        positive_support = [
            claim_support(profile.profile_id, row[1], positive["expected"])
            for row in positive_rows
        ]
        negative_claim_support = [
            claim_support(profile.profile_id, row[1], expected)
            for row in negative_rows
        ]
        negative_truth_support = [
            claim_support(profile.profile_id, row[1], negative["expected"])
            for row in negative_rows
        ]
        generalization_support = [
            claim_support(profile.profile_id, row[1], expected)
            for row in generalization_rows
        ]
        primary_similarity = _disjoint_similarities(primary_rows, profile.profile_id)
        generalization_similarity = _disjoint_similarities(
            generalization_rows, profile.profile_id
        )
        specificity_values = _paired_support_differences(
            primary_rows,
            negative_rows,
            manifest_by_slot,
            profile.profile_id,
            expected,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        unsafe_problems.append(f"claim/control reducer target does not verify: {exc}")
        decision = _status_decision(
            spec,
            status="abstain",
            reproduction=_axis(
                "fail" if reproduction_failures else "pass",
                reproduction_checks,
                "; ".join(reproduction_failures),
            ),
            stability_specificity=_axis("abstain", {}),
            utility=_axis("abstain", {}),
            generalization=_axis("abstain", {}),
            evidence_confidence=_axis("insufficient", {}, "; ".join(unsafe_problems)),
            primary_checks={},
            reasons=unsafe_problems,
        )
        return {
            "ok": False,
            "verified": False,
            "problems": unsafe_problems,
            "decision": decision,
            "status": decision.status,
            "publication_state": decision.publication_state,
            "primary_p_values": {},
            "spec": spec,
        }
    minimum_pairs = max(1, profile.minimum_independent_units // 2)
    checks = {
        "stability": threshold_check(
            primary_similarity, threshold=profile.stability_min,
            direction="higher", minimum_units=minimum_pairs,
        ),
        "claim_support": threshold_check(
            primary_support, threshold=profile.positive_control_min,
            direction="higher", minimum_units=profile.minimum_independent_units,
        ),
        "positive_control": threshold_check(
            positive_support, threshold=profile.positive_control_min,
            direction="higher", minimum_units=profile.minimum_control_units,
        ),
        "negative_falsification": threshold_check(
            negative_claim_support, threshold=profile.negative_control_max,
            direction="lower", minimum_units=profile.minimum_control_units,
        ),
        "negative_control_truth": threshold_check(
            negative_truth_support, threshold=profile.positive_control_min,
            direction="higher", minimum_units=profile.minimum_control_units,
        ),
        "specificity": threshold_check(
            specificity_values,
            threshold=profile.positive_control_min - profile.negative_control_max,
            direction="higher", bounds=(-1.0, 1.0),
            minimum_units=profile.minimum_control_units,
        ),
        "generalization_support": threshold_check(
            generalization_support, threshold=profile.generalization_min,
            direction="higher", minimum_units=profile.minimum_independent_units,
        ),
        "generalization_stability": threshold_check(
            generalization_similarity, threshold=profile.generalization_min,
            direction="higher", minimum_units=minimum_pairs,
        ),
    }

    utility_evidence: Dict[str, Mapping[str, Any]] = {}
    for _, _, raw in primary_rows:
        evidence = raw.get("utility_evidence")
        if isinstance(evidence, Mapping):
            utility_evidence[str(evidence.get("raw_digest", len(utility_evidence)))] = evidence
    utility_required = claim.finding_type == "utility" or \
        claim.task.get("utility_required", True) is True
    if utility_evidence:
        from .utility import UtilityMetricSpec, verify_utility_evidence

        utility_results = [verify_utility_evidence(row)
                           for row in utility_evidence.values()]
        utility_profile = get_profile("utility_v1")
        for evidence, result in zip(utility_evidence.values(), utility_results):
            try:
                metric_spec = UtilityMetricSpec.from_dict(evidence["metric_spec"])
                if metric_spec.practical_margin != utility_profile.practical_margin:
                    raise ValueError(
                        "utility practical_margin differs from registered utility_v1 profile"
                    )
                if metric_spec.minimum_independent_units < \
                        utility_profile.minimum_independent_units:
                    raise ValueError(
                        "utility independent-unit minimum is below utility_v1 profile"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                result["valid"] = False
                result["state"] = "abstain"
                result.setdefault("problems", []).append(str(exc))
        invalid = [row for row in utility_results if not row["valid"]]
        if invalid:
            utility_axis = _axis("abstain", {
                "evidence": utility_results,
            }, "raw utility evidence does not verify")
            unsafe_problems.append("raw utility evidence is invalid or contradictory")
        else:
            utility_state = _summarize_state(utility_results)
            utility_axis = _axis(utility_state, {"evidence": utility_results})
            worst_p = max(float(row["p_value"]) for row in utility_results)
            checks["utility"] = {
                "state": utility_state,
                "p_value": worst_p,
                "estimate": min(float(row["oriented_delta"])
                                for row in utility_results),
                "threshold": utility_profile.practical_margin,
                "direction": "higher",
                "n_independent": min(int(row["n_independent"])
                                     for row in utility_results),
            }
    elif utility_required:
        utility_axis = _axis(
            "inconclusive", {}, "required external-task utility evidence is absent"
        )
    else:
        utility_axis = _axis(
            "not_evaluated", {},
            "ClaimRecord explicitly limits this audit to reproduction/non-utility claims"
        )

    primary_p_values = {
        f"{claim.claim_id}:{name}": float(check.get("p_value", 1.0))
        for name, check in checks.items()
    }
    stability_names = (
        "stability", "claim_support", "positive_control",
        "negative_falsification", "negative_control_truth", "specificity",
    )
    generalization_names = ("generalization_support", "generalization_stability")
    reproduction_state = "fail" if reproduction_failures else "pass"
    reproduction_axis = _axis(
        reproduction_state, reproduction_checks,
        "; ".join(reproduction_failures),
    )
    stability_axis = _axis(
        _summarize_state(checks[name] for name in stability_names),
        {name: checks[name] for name in stability_names},
    )
    generalization_axis = _axis(
        _summarize_state(checks[name] for name in generalization_names),
        {name: checks[name] for name in generalization_names},
    )
    generalization_axis["held_out_axes"] = list(spec.design["held_out_axes"])
    generalization_axis["evaluation_manifests"] = {
        key: {
            "evaluation_id": value["evaluation_id"],
            "manifest_digest": value["manifest_digest"],
        }
        for key, value in spec.design["evaluation_manifests"].items()
    }
    evidence_axis = _axis("sufficient", {
        "complete_slots": len(attestations),
        "declared_slots": len(spec.run_manifest),
        "content_objects": len(bundle.content),
        "source_digest_bound": True,
        "agent_panel_verified": True,
        "independent_unit": spec.design.get("independent_unit"),
    })
    if unsafe_problems:
        status = "abstain"
    elif reproduction_failures:
        status = "reproduction_failure"
    elif stability_axis["state"] == "fail" or generalization_axis["state"] == "fail" \
            or utility_axis["state"] == "fail":
        status = "audit_failure"
    elif stability_axis["state"] != "pass" or generalization_axis["state"] != "pass" \
            or utility_axis["state"] == "inconclusive":
        status = "inconclusive"
    else:
        status = "pass"
    decision = _status_decision(
        spec,
        status=status,
        reproduction=reproduction_axis,
        stability_specificity=stability_axis,
        utility=utility_axis,
        generalization=generalization_axis,
        evidence_confidence=evidence_axis,
        primary_checks=checks,
        reasons=unsafe_problems + reproduction_failures,
    )
    return {
        "ok": not unsafe_problems,
        "verified": not unsafe_problems,
        "problems": unsafe_problems,
        "decision": decision,
        "status": decision.status,
        "publication_state": decision.publication_state,
        "primary_p_values": primary_p_values,
        "spec": spec,
    }


def _finalize_with_holm(base: Mapping[str, Any], holm: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    decision = base.get("decision")
    if not isinstance(decision, AuditDecision) or not base.get("verified"):
        return dict(base)
    checks = _apply_holm_to_checks(
        decision.primary_checks, holm, decision.claim_id
    )
    stability_names = {
        "stability", "claim_support", "positive_control",
        "negative_falsification", "negative_control_truth", "specificity",
    }
    generalization_names = {"generalization_support", "generalization_stability"}
    stability_checks = {key: value for key, value in checks.items()
                        if key in stability_names}
    generalization_checks = {key: value for key, value in checks.items()
                             if key in generalization_names}
    stability_axis = _axis(_summarize_state(stability_checks.values()), stability_checks)
    generalization_axis = _axis(
        _summarize_state(generalization_checks.values()), generalization_checks
    )
    utility_axis = dict(decision.utility)
    if "utility" in checks:
        utility_axis["state"] = checks["utility"]["state"]
        utility_axis["holm"] = checks["utility"].get("holm")
    if decision.status in ("abstain", "protocol_deviation", "reproduction_failure"):
        status = decision.status
    elif stability_axis["state"] == "fail" or generalization_axis["state"] == "fail" \
            or utility_axis.get("state") == "fail":
        status = "audit_failure"
    elif stability_axis["state"] != "pass" or generalization_axis["state"] != "pass" \
            or utility_axis.get("state") == "inconclusive":
        status = "inconclusive"
    else:
        status = "pass"
    spec = base["spec"]
    fresh = _status_decision(
        spec,
        status=status,
        reproduction=decision.reproduction,
        stability_specificity=stability_axis,
        utility=utility_axis,
        generalization=generalization_axis,
        evidence_confidence=decision.evidence_confidence,
        primary_checks=checks,
        reasons=decision.reasons,
    )
    output = dict(base)
    output.update({
        "decision": fresh,
        "status": fresh.status,
        "publication_state": fresh.publication_state,
        "holm": dict(holm),
    })
    return output


def verify_audit_bundle(
    bundle: AuditBundle,
    store: ContentAddressedStore,
    *,
    trusted_plan_keys: Mapping[str, bytes],
    trusted_executor_keys: Mapping[str, bytes],
) -> Dict[str, Any]:
    """Verify one bundle and apply Holm over its declared one-claim family.

    Multi-claim release families must use :func:`verify_audit_release`; this
    function refuses to finalize a bundle whose frozen family names other
    claims.
    """
    base = _verify_bundle_base(
        bundle, store, trusted_plan_keys, trusted_executor_keys
    )
    spec = base.get("spec")
    if not isinstance(spec, AuditSpec) or not base.get("verified"):
        return base
    claim = ClaimRecord.from_dict(spec.claim_record)
    members = spec.multiplicity_family.get("member_claim_ids", [])
    if members != [claim.claim_id]:
        decision = base["decision"]
        assert isinstance(decision, AuditDecision)
        fresh = _status_decision(
            spec,
            status="inconclusive",
            reproduction=decision.reproduction,
            stability_specificity=decision.stability_specificity,
            utility=decision.utility,
            generalization=decision.generalization,
            evidence_confidence=decision.evidence_confidence,
            primary_checks=decision.primary_checks,
            reasons=list(decision.reasons) + [
                "multi-claim Holm family requires release-level verification"
            ],
        )
        return {**base, "decision": fresh, "status": fresh.status,
                "publication_state": fresh.publication_state}
    expected_p_values = {
        f"{claim.claim_id}:{name}": 1.0
        for name in spec.multiplicity_family["primary_check_names"]
    }
    expected_p_values.update(base["primary_p_values"])
    holm = holm_bonferroni(expected_p_values, alpha=float(
        spec.multiplicity_family["alpha"]
    ))
    result = _finalize_with_holm(base, holm)
    stored = bundle.decision
    if stored is not None and stored != result["decision"].to_dict():
        result["ok"] = False
        result["verified"] = False
        result["problems"] = list(result.get("problems", [])) + [
            "stored AuditDecision does not recompute from raw bundle"
        ]
        result["status"] = "protocol_deviation"
    return result


def verify_audit_release(
    bundles: Sequence[AuditBundle],
    store: ContentAddressedStore,
    *,
    trusted_plan_keys: Mapping[str, bytes],
    trusted_executor_keys: Mapping[str, bytes],
) -> Dict[str, Any]:
    """Verify all frozen claims and apply Holm across every primary check."""
    bases = [
        _verify_bundle_base(
            bundle, store, trusted_plan_keys, trusted_executor_keys
        )
        for bundle in bundles
    ]
    by_claim: Dict[str, Dict[str, Any]] = {}
    bundle_by_claim: Dict[str, AuditBundle] = {}
    family_members: Optional[List[str]] = None
    family_id: Optional[str] = None
    alpha: Optional[float] = None
    family_problems: List[str] = []
    for bundle, base in zip(bundles, bases):
        spec = base.get("spec")
        if not isinstance(spec, AuditSpec):
            continue
        claim = ClaimRecord.from_dict(spec.claim_record)
        if claim.claim_id in by_claim:
            family_problems.append(f"duplicate release bundle for {claim.claim_id}")
        by_claim[claim.claim_id] = base
        bundle_by_claim[claim.claim_id] = bundle
        family = spec.multiplicity_family
        members = list(family.get("member_claim_ids", []))
        if family_members is None:
            family_members = members
            family_id = str(family.get("family_id"))
            alpha = float(family.get("alpha"))
        elif members != family_members or family.get("family_id") != family_id or \
                float(family.get("alpha")) != alpha:
            family_problems.append("AuditSpecs disagree on global multiplicity family")
    if family_members is None:
        return {"ok": False, "problems": ["release has no parseable AuditSpec"],
                "results": {}}
    missing = sorted(set(family_members) - set(by_claim))
    extras = sorted(set(by_claim) - set(family_members))
    if missing:
        family_problems.append(f"release multiplicity family missing claims: {missing}")
    if extras:
        family_problems.append(f"release includes claims outside multiplicity family: {extras}")
    all_p_values: Dict[str, float] = {}
    for claim_id, base in by_claim.items():
        spec = base.get("spec")
        if isinstance(spec, AuditSpec):
            for name in spec.multiplicity_family.get("primary_check_names", []):
                all_p_values[f"{claim_id}:{name}"] = 1.0
        if base.get("verified"):
            all_p_values.update(base.get("primary_p_values", {}))
    holm = holm_bonferroni(all_p_values, alpha=float(alpha or 0.05))
    results: Dict[str, Dict[str, Any]] = {}
    for claim_id, base in by_claim.items():
        final = _finalize_with_holm(base, holm)
        stored = bundle_by_claim[claim_id].decision
        if stored is not None and isinstance(final.get("decision"), AuditDecision) and \
                stored != final["decision"].to_dict():
            decision = final["decision"]
            spec = final["spec"]
            fresh = _status_decision(
                spec,
                status="protocol_deviation",
                reproduction=decision.reproduction,
                stability_specificity=decision.stability_specificity,
                utility=decision.utility,
                generalization=decision.generalization,
                evidence_confidence=_axis(
                    "insufficient", decision.evidence_confidence.get("checks", {}),
                    "stored AuditDecision does not recompute from raw release",
                ),
                primary_checks=decision.primary_checks,
                reasons=list(decision.reasons) + [
                    "stored AuditDecision does not recompute from raw release"
                ],
            )
            final.update({"ok": False, "verified": False, "decision": fresh,
                          "status": fresh.status,
                          "problems": list(final.get("problems", [])) + [
                              "stored AuditDecision does not recompute from raw release"
                          ]})
        if family_problems and final.get("verified"):
            decision = final["decision"]
            assert isinstance(decision, AuditDecision)
            spec = final["spec"]
            fresh = _status_decision(
                spec,
                status="protocol_deviation",
                reproduction=decision.reproduction,
                stability_specificity=decision.stability_specificity,
                utility=decision.utility,
                generalization=decision.generalization,
                evidence_confidence=_axis(
                    "insufficient", decision.evidence_confidence.get("checks", {}),
                    "; ".join(family_problems),
                ),
                primary_checks=decision.primary_checks,
                reasons=list(decision.reasons) + family_problems,
            )
            final.update({"ok": False, "verified": False, "decision": fresh,
                          "status": fresh.status})
        results[claim_id] = final
    return {
        "ok": not family_problems and all(row.get("verified") for row in results.values()),
        "family_id": family_id,
        "family_members": family_members,
        "problems": family_problems,
        "holm": holm,
        "results": results,
    }
