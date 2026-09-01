"""Outcome-blind qualification and freezing for StressKit candidate claims.

This module turns the broad candidate frame into a complete pre-freeze ledger.
It validates typed evidence for every gate, retains exclusions, and refuses to
freeze while any candidate is pending or the launch breadth gate is unmet.
Benchmark outcomes are forbidden inputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from stresskit.audit_compile import regenerate_run_manifest, validate_agent_panel
from stresskit.audit_models import (
    AUDIT_SCHEMA_VERSION,
    AgentOpinion,
    AuditSpec,
    ClaimRecord,
    SourceBundle,
)
from stresskit.audit_profiles import get_profile, reducer_digest
from stresskit.integrity import (
    ContentAddressedStore,
    ContentRef,
    digest_json,
    require_sha256_digest,
    sha256_bytes,
    verify_digest_closure,
)


ARTIFACT = "stresskit_prefreeze_qualification"
REPORT_ARTIFACT = "stresskit_prefreeze_report"
RELEASE_ARTIFACT = "stresskit_release_registry"

PREFREEZE_GATES = (
    "source_bundle",
    "agent_panel",
    "claim_record",
    "license_closure",
    "execution_smoke",
    "audit_spec",
    "protocol_review",
    "resource_estimate",
)

STRATA = (
    "cot_trajectories",
    "probes_monitoring",
    "steering_control",
    "lenses_model_diffing",
    "intervention_prediction",
    "circuits_saes",
)

FAMILY_STRATA = {
    "cot_sentence_resampling": "cot_trajectories",
    "evaluation_awareness_model_organism": "probes_monitoring",
    "introspective_self_report": "probes_monitoring",
    "natural_language_activation_reader": "probes_monitoring",
    "persona_space_axis": "probes_monitoring",
    "probe_direction_geometry": "probes_monitoring",
    "unsupervised_latent_knowledge_probe": "probes_monitoring",
    "contrastive_activation_steering": "steering_control",
    "finetuning_direction_ablation": "steering_control",
    "persona_direction_monitoring_steering": "steering_control",
    "representation_engineering": "steering_control",
    "sae_feature_steering": "steering_control",
    "model_diffing": "lenses_model_diffing",
    "representation_lens": "lenses_model_diffing",
    "weight_space_communication_map": "lenses_model_diffing",
    "activation_direction_causal_intervention": "intervention_prediction",
    "causal_abstraction_intervention": "intervention_prediction",
    "causal_tracing": "intervention_prediction",
    "representation_patching": "intervention_prediction",
    "attention_pattern_analysis": "circuits_saes",
    "automated_circuit_discovery": "circuits_saes",
    "sae_evaluation": "circuits_saes",
    "sparse_autoencoder_training": "circuits_saes",
    "sparse_feature_circuits": "circuits_saes",
    "transcoder_attribution_graph": "circuits_saes",
}

COMPUTE_TIERS = {
    "cpu_posthoc_from_released_rollouts": 0,
    "cpu_or_single_gpu_small": 0,
    "single_gpu_small": 1,
    "single_gpu_medium": 2,
    "single_gpu_large": 3,
    "multi_gpu": 4,
}

_FORBIDDEN_OUTCOME_KEYS = {
    "result",
    "grade",
    "verdict",
    "passed",
    "failed",
    "effect_size",
    "audit_status",
    "score",
    "rank",
}

_ISOLATION = {
    "network": "disabled",
    "credentials": "absent",
    "inputs": "read_only",
    "scratch": "quota_limited",
    "outputs": "allowlisted",
}

_AGENT_PANEL_ABSTENTION_REASONS = frozenset({
    "agent_disagreement",
    "frozen_candidate_wording_not_explicitly_supported_by_pinned_sources",
    "invalid_evidence_anchor",
    "missing_evidence",
    "provider_or_model_family_collapse",
    "unsupported_wording",
})

_OPENROUTER_REQUEST_POLICY = {
    "allow_fallbacks": False,
    "data_collection": "deny",
    "require_parameters": True,
    "zdr": True,
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


def _require_outcome_blind(value: Any, label: str) -> None:
    forbidden = _keys(value) & _FORBIDDEN_OUTCOME_KEYS
    if forbidden:
        raise ValueError(f"{label} contains outcome keys: {sorted(forbidden)}")


def _require_header(payload: Mapping[str, Any], artifact: str) -> None:
    if payload.get("artifact") != artifact or \
            payload.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"expected {artifact} schema {AUDIT_SCHEMA_VERSION}")


def _parse_timestamp(value: str, name: str) -> None:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{name} must be ISO 8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include timezone")


def _candidate_rows(registry: Mapping[str, Any]) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    if registry.get("status") != "candidate_frame_not_frozen" or \
            registry.get("outcome_blind") is not True:
        raise ValueError("candidate registry must be outcome-blind and not frozen")
    if registry.get("schema_version") != "0.1":
        raise ValueError("candidate registry schema_version must be '0.1'")
    _require_outcome_blind(registry, "candidate registry")
    rows = registry.get("entries")
    upstreams = registry.get("upstreams")
    if not isinstance(rows, list) or not rows or not isinstance(upstreams, Mapping):
        raise ValueError("candidate registry needs entries and upstreams")
    identifiers = [row.get("claim_id") for row in rows if isinstance(row, Mapping)]
    if len(identifiers) != len(rows) or not all(
            isinstance(value, str) and value for value in identifiers):
        raise ValueError("every candidate needs claim_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate claim IDs must be unique")
    for row in rows:
        upstream = row.get("upstream")
        if upstream not in upstreams:
            raise ValueError(f"candidate {row['claim_id']} names unknown upstream")
        family = row.get("family")
        if family not in FAMILY_STRATA:
            raise ValueError(f"candidate {row['claim_id']} has unmapped family {family!r}")
        tier = row.get("compute_tier")
        if tier not in COMPUTE_TIERS:
            raise ValueError(f"candidate {row['claim_id']} has unknown compute tier")
    return list(rows), upstreams


def _pending_gate(note: str) -> Dict[str, Any]:
    return {
        "status": "pending",
        "evidence": None,
        "evidence_digest": None,
        "note": note,
    }


def _exclusion_gate(
    candidate: Mapping[str, Any], upstream: Mapping[str, Any]
) -> Dict[str, Any]:
    evidence = {
        "artifact": "stresskit_prefreeze_exclusion_evidence",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "claim_id": candidate["claim_id"],
        "category": candidate["eligibility"],
        "reason": candidate["exclusion_reason"],
        "candidate_digest": digest_json(candidate),
        "upstream_record_digest": digest_json(upstream),
        "outcome_blind": True,
    }
    return {
        "status": "fail",
        "evidence": evidence,
        "evidence_digest": digest_json(evidence),
        "note": "pre-freeze exclusion carried from recorded candidate frame",
    }


def scaffold_qualification(registry: Mapping[str, Any]) -> Dict[str, Any]:
    """Create deterministic gate records without claiming uncollected evidence."""
    candidates, upstreams = _candidate_rows(registry)
    records = []
    for candidate in candidates:
        excluded = str(candidate["eligibility"]).startswith("excluded_pre_freeze")
        gates = {
            gate: _pending_gate(
                "evidence not yet registered; legacy state: "
                + str(candidate["eligibility"])
            )
            for gate in PREFREEZE_GATES
        }
        if excluded:
            gates["license_closure"] = _exclusion_gate(
                candidate, upstreams[candidate["upstream"]]
            )
        records.append({
            "claim_id": candidate["claim_id"],
            "candidate_digest": digest_json(candidate),
            "disposition": "excluded" if excluded else "pending",
            "stratum": FAMILY_STRATA[candidate["family"]],
            "compute_tier": COMPUTE_TIERS[candidate["compute_tier"]],
            "claim_record_digest": None,
            "audit_spec_digest": None,
            "comparison": {
                "task": candidate["task"],
                "metric": None,
                "evaluation_set": None,
                "resource_budget": None,
            },
            "gates": gates,
            "exclusion_reason": candidate.get("exclusion_reason") if excluded else None,
        })
    return {
        "artifact": ARTIFACT,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "qualification_in_progress",
        "outcome_blind": True,
        "candidate_registry_digest": digest_json(registry),
        "required_gates": list(PREFREEZE_GATES),
        "records": records,
    }


def _validate_gate(
    claim_id: str, gate_name: str, gate: Mapping[str, Any]
) -> str:
    if not isinstance(gate, Mapping):
        raise ValueError(f"{claim_id} gate {gate_name} must be an object")
    status = gate.get("status")
    if status not in ("pending", "pass", "fail"):
        raise ValueError(f"{claim_id} gate {gate_name} has invalid status")
    evidence = gate.get("evidence")
    evidence_digest = gate.get("evidence_digest")
    if status == "pending":
        if evidence is not None or evidence_digest is not None:
            raise ValueError(f"{claim_id} pending gate {gate_name} carries evidence")
        return str(status)
    if not isinstance(evidence, Mapping):
        raise ValueError(f"{claim_id} gate {gate_name} needs evidence object")
    require_sha256_digest(evidence_digest, f"{claim_id} {gate_name} evidence_digest")
    if digest_json(evidence) != evidence_digest:
        raise ValueError(f"{claim_id} gate {gate_name} evidence digest mismatch")
    _require_outcome_blind(evidence, f"{claim_id} gate {gate_name} evidence")
    if gate_name != "audit_spec" and \
            evidence.get("claim_id") != claim_id:
        raise ValueError(f"{claim_id} gate {gate_name} evidence targets another claim")
    return str(status)


def _embedded_artifact(
    evidence: Mapping[str, Any],
    payload_field: str,
    digest_field: str,
) -> Mapping[str, Any]:
    payload = evidence.get(payload_field)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"agent panel abstention needs embedded {payload_field}"
        )
    expected = require_sha256_digest(
        evidence.get(digest_field), f"agent panel abstention {digest_field}"
    )
    if digest_json(payload) != expected:
        raise ValueError(
            f"agent panel abstention {payload_field} digest mismatch"
        )
    return payload


def _validate_panel_plan(
    panel: Mapping[str, Any],
    candidate: Mapping[str, Any],
    opinions: Sequence[AgentOpinion],
) -> Mapping[str, Mapping[str, Any]]:
    _require_header(panel, "stresskit_agent_panel_plan")
    claim_id = candidate["claim_id"]
    if panel.get("candidate_id") != claim_id or \
            panel.get("outcome_blind") is not True:
        raise ValueError("agent panel plan targets another candidate")
    if panel.get("status") not in (
        "prefrozen_awaiting_authenticated_preflight",
        "frozen",
    ):
        raise ValueError("agent panel plan is not prefrozen or frozen")
    requests = panel.get("requests")
    if not isinstance(requests, list) or len(requests) != 3 or any(
        not isinstance(row, Mapping) for row in requests
    ):
        raise ValueError("agent panel plan needs exactly three request rows")
    identifiers = [row.get("opinion_id") for row in requests]
    if len(set(identifiers)) != 3 or set(identifiers) != {
        opinion.opinion_id for opinion in opinions
    }:
        raise ValueError("agent panel plan opinion IDs do not match opinions")
    by_id = {str(row["opinion_id"]): row for row in requests}
    for opinion in opinions:
        row = by_id[opinion.opinion_id]
        catalog = row.get("catalog")
        if not isinstance(catalog, Mapping):
            raise ValueError("agent panel plan request needs catalog metadata")
        if row.get("role") != opinion.role or \
                row.get("provider_name") != opinion.provider or \
                row.get("model_family") != opinion.model_family or \
                catalog.get("canonical_slug") != opinion.model:
            raise ValueError("agent panel plan request differs from opinion")
        for name in ("model_request_id", "provider_endpoint"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"agent panel plan request needs {name}")
    return by_id


def _validate_provider_attestation(
    attestation: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source: SourceBundle,
    panel_digest: str,
    panel_requests: Mapping[str, Mapping[str, Any]],
    opinions: Sequence[AgentOpinion],
    store: Optional[ContentAddressedStore],
) -> None:
    if store is None:
        raise ValueError("provider attestation verification needs content-addressed store")
    _require_header(attestation, "stresskit_openrouter_panel_attestation")
    if attestation.get("candidate_id") != candidate["claim_id"] or \
            attestation.get("source_bundle_digest") != source.digest or \
            attestation.get("panel_plan_digest") != panel_digest:
        raise ValueError("provider attestation targets another panel")
    if attestation.get("status") != "verified_from_accepted_responses" or \
            attestation.get("authenticated") is not True or \
            attestation.get("credential_serialized") is not False:
        raise ValueError("provider attestation is not authenticated and accepted")
    credential_source = attestation.get("credential_source")
    if not isinstance(credential_source, str) or not credential_source.startswith(
        "environment:"
    ):
        raise ValueError("provider attestation needs environment credential source")
    _parse_timestamp(
        attestation.get("catalog_observed_at"),
        "provider attestation catalog_observed_at",
    )
    _parse_timestamp(
        attestation.get("attestation_created_at"),
        "provider attestation attestation_created_at",
    )
    if attestation.get("request_policy") != _OPENROUTER_REQUEST_POLICY:
        raise ValueError("provider attestation request policy is not exact")
    routes = attestation.get("routes")
    if not isinstance(routes, list) or len(routes) != 3 or any(
        not isinstance(row, Mapping) for row in routes
    ):
        raise ValueError("provider attestation needs exactly three routes")
    route_ids = [row.get("opinion_id") for row in routes]
    if len(set(route_ids)) != 3 or set(route_ids) != {
        opinion.opinion_id for opinion in opinions
    }:
        raise ValueError("provider attestation routes do not match opinions")
    by_id = {opinion.opinion_id: opinion for opinion in opinions}
    for route in routes:
        opinion = by_id[str(route["opinion_id"])]
        request = panel_requests[opinion.opinion_id]
        for field in (
            "request_receipt_digest",
            "model_descriptor_digest",
            "raw_response_digest",
        ):
            require_sha256_digest(
                route.get(field), f"provider attestation route {field}"
            )
        if route.get("request_receipt_digest") != opinion.request_digest or \
                route.get("model_descriptor_digest") != opinion.model_digest:
            raise ValueError("provider attestation provenance differs from opinion")
        if route.get("role") != opinion.role or \
                route.get("requested_model") != request["model_request_id"] or \
                route.get("requested_provider_endpoint") != \
                request["provider_endpoint"] or \
                route.get("selected_provider") != opinion.provider or \
                route.get("selected_canonical_model") != opinion.model:
            raise ValueError("provider attestation route differs from frozen panel")
        if route.get("strategy") != "direct" or \
                route.get("router_attempt") != 1 or \
                route.get("available_endpoint_count") != 1 or \
                route.get("pipeline") not in (None, []) or \
                route.get("accepted") is not True:
            raise ValueError("provider attestation route was not directly accepted")
        inventory = route.get("endpoint_inventory_count")
        if not isinstance(inventory, int) or isinstance(inventory, bool) or \
                inventory < 1:
            raise ValueError("provider attestation endpoint inventory is invalid")
        response_id = route.get("response_id")
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("provider attestation route needs response_id")
        _parse_timestamp(
            route.get("created_at"), "provider attestation route created_at"
        )
        try:
            descriptor = store.get_json(opinion.model_digest)
            receipt = store.get_json(opinion.request_digest)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("provider route provenance is unavailable") from exc
        if not isinstance(descriptor, Mapping) or not isinstance(receipt, Mapping):
            raise ValueError("provider route provenance must be JSON objects")
        _require_header(descriptor, "stresskit_agent_model_descriptor")
        _require_header(receipt, "stresskit_agent_request_receipt")
        route_binding_digest = require_sha256_digest(
            descriptor.get("route_binding_digest"),
            "model descriptor route_binding_digest",
        )
        if receipt.get("route_binding_digest") != route_binding_digest or \
                descriptor.get("panel_plan_digest") != panel_digest or \
                receipt.get("panel_plan_digest") != panel_digest:
            raise ValueError("provider provenance panel or route binding mismatch")
        if descriptor.get("provider") != opinion.provider or \
                descriptor.get("model") != opinion.model or \
                descriptor.get("family") != opinion.model_family or \
                descriptor.get("requested_model") != request["model_request_id"] or \
                descriptor.get("requested_provider_endpoint") != \
                request["provider_endpoint"] or \
                descriptor.get("response_id") != route["response_id"] or \
                descriptor.get("raw_response_digest") != \
                route["raw_response_digest"]:
            raise ValueError("provider model descriptor differs from attested route")
        response = receipt.get("response")
        body = receipt.get("body")
        if not isinstance(response, Mapping) or not isinstance(body, Mapping):
            raise ValueError("provider request receipt is incomplete")
        expected_policy = {
            **_OPENROUTER_REQUEST_POLICY,
            "only": [request["provider_endpoint"]],
        }
        if receipt.get("transport") != "openrouter" or \
                receipt.get("method") != "POST" or \
                receipt.get("prompt_digest") != opinion.prompt_digest or \
                body.get("model") != request["model_request_id"] or \
                body.get("provider") != expected_policy or \
                response.get("response_id") != route["response_id"] or \
                response.get("raw_digest") != route["raw_response_digest"]:
            raise ValueError("provider request receipt differs from attested route")


def _validate_abstention_closure_assertions(
    evidence: Mapping[str, Any],
    opinion_digests: Sequence[str],
    *,
    source_digest: str,
    decision_digest: str,
    attempt_digest: str,
    attestation_digest: str,
    store: Optional[ContentAddressedStore],
) -> None:
    if store is None:
        raise ValueError(
            "agent panel abstention verification needs content-addressed store"
        )
    source_closure_digest = require_sha256_digest(
        evidence.get("source_closure_digest"),
        "agent panel abstention source_closure_digest",
    )
    try:
        source_closure_payload = store.get_json(source_closure_digest)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("agent panel source closure is unavailable") from exc
    if not isinstance(source_closure_payload, list):
        raise ValueError("agent panel source closure manifest must be a list")
    source_references = [
        ContentRef.from_dict(row) for row in source_closure_payload
    ]
    verify_digest_closure(store, source_references, [source_digest])

    rows = evidence.get("opinion_closure_digests")
    if not isinstance(rows, list) or len(rows) != 3 or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise ValueError("agent panel abstention needs three opinion closure digests")
    mapped: Dict[str, str] = {}
    for row in rows:
        opinion_digest = require_sha256_digest(
            row.get("opinion_digest"), "opinion closure opinion_digest"
        )
        closure_digest = require_sha256_digest(
            row.get("closure_digest"), "opinion closure closure_digest"
        )
        if opinion_digest in mapped:
            raise ValueError("duplicate opinion closure assertion")
        mapped[opinion_digest] = closure_digest
    if set(mapped) != set(opinion_digests):
        raise ValueError("opinion closure assertions do not match opinions")
    for opinion_digest, closure_digest in mapped.items():
        try:
            closure_payload = store.get_json(closure_digest)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("agent opinion closure is unavailable") from exc
        if not isinstance(closure_payload, list):
            raise ValueError("agent opinion closure manifest must be a list")
        references = [ContentRef.from_dict(row) for row in closure_payload]
        verify_digest_closure(store, references, [opinion_digest])

    panel_closure = evidence.get("panel_closure")
    if not isinstance(panel_closure, list):
        raise ValueError("agent panel abstention needs embedded panel closure")
    panel_closure_digest = require_sha256_digest(
        evidence.get("panel_closure_digest"),
        "agent panel abstention panel_closure_digest",
    )
    if digest_json(panel_closure) != panel_closure_digest:
        raise ValueError("agent panel closure digest mismatch")
    panel_references = [ContentRef.from_dict(row) for row in panel_closure]
    expected_roots = [
        source_closure_digest,
        *(mapped[digest] for digest in opinion_digests),
        decision_digest,
        attempt_digest,
        attestation_digest,
    ]
    roots = evidence.get("panel_closure_roots")
    if roots != expected_roots:
        raise ValueError("agent panel closure roots are not exact")
    try:
        verify_digest_closure(store, panel_references, roots)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise ValueError("agent panel digest closure is incomplete") from exc


def _validate_unanimous_agent_panel_abstention(
    evidence: Mapping[str, Any],
    candidate: Mapping[str, Any],
    store: Optional[ContentAddressedStore],
) -> None:
    """Validate a self-contained, conservative live-panel exclusion."""
    _require_header(evidence, "stresskit_agent_panel_abstention")
    claim_id = candidate["claim_id"]
    if evidence.get("claim_id") != claim_id or \
            evidence.get("outcome_blind") is not True or \
            evidence.get("publication_state") != "abstain":
        raise ValueError("agent panel abstention targets another candidate")
    reason = evidence.get("reason")
    if reason not in _AGENT_PANEL_ABSTENTION_REASONS:
        raise ValueError("agent panel abstention reason is not recognized")
    if not isinstance(evidence.get("prompt_injection_detected"), bool):
        raise ValueError("agent panel abstention prompt-injection flag must be boolean")
    if evidence["prompt_injection_detected"]:
        raise ValueError("unsupported-wording abstention cannot hide prompt injection")

    source_payload = _embedded_artifact(
        evidence, "source_bundle", "source_bundle_digest"
    )
    source = SourceBundle.from_dict(source_payload)
    if source.metadata.get("candidate_id") != claim_id:
        raise ValueError("embedded SourceBundle targets another candidate")

    opinion_payloads = evidence.get("opinions")
    if not isinstance(opinion_payloads, list) or len(opinion_payloads) != 3 or any(
        not isinstance(row, Mapping) for row in opinion_payloads
    ):
        raise ValueError("agent panel abstention needs three embedded opinions")
    opinions = [AgentOpinion.from_dict(row) for row in opinion_payloads]
    opinion_digests = [opinion.digest for opinion in opinions]
    recorded_digests = evidence.get("agent_opinion_digests")
    if not isinstance(recorded_digests, list) or len(recorded_digests) != 3:
        raise ValueError("agent panel abstention needs three opinion digests")
    for digest in recorded_digests:
        require_sha256_digest(digest, "agent panel abstention opinion digest")
    if list(recorded_digests) != opinion_digests or \
            len(set(opinion_digests)) != 3 or \
            len({opinion.opinion_id for opinion in opinions}) != 3:
        raise ValueError("embedded agent opinions do not match recorded digests")
    if sorted(opinion.role for opinion in opinions) != [
        "critic", "extractor", "extractor"
    ]:
        raise ValueError("agent panel abstention needs two extractors and one critic")
    if len({opinion.provider for opinion in opinions}) != 3 or \
            len({opinion.model_family for opinion in opinions}) != 3:
        raise ValueError("agent panel providers and model families must be distinct")
    if any(opinion.source_bundle_digest != source.digest for opinion in opinions):
        raise ValueError("agent panel opinion targets another SourceBundle")
    if any(opinion.statement != candidate["statement_to_extract"] for opinion in opinions):
        raise ValueError("agent panel opinion changed frozen candidate wording")
    if any(opinion.supported for opinion in opinions) or any(
        opinion.prompt_injection_detected for opinion in opinions
    ):
        raise ValueError("agent panel abstention is not unanimously unsupported")
    if any(not opinion.issues for opinion in opinions):
        raise ValueError("unsupported agent opinion needs explicit issues")

    panel = _embedded_artifact(
        evidence, "panel_plan", "panel_plan_digest"
    )
    panel_requests = _validate_panel_plan(panel, candidate, opinions)

    decision = _embedded_artifact(
        evidence, "discovery_decision", "discovery_decision_digest"
    )
    _require_header(decision, "stresskit_claim_candidates")
    if decision.get("source_bundle_digest") != source.digest or \
            decision.get("publication_state") != "abstain" or \
            decision.get("candidates") != []:
        raise ValueError("embedded discovery decision is not an abstention")
    problems = decision.get("problems")
    if not isinstance(problems, list) or not problems or any(
        not isinstance(problem, str) or not problem for problem in problems
    ):
        raise ValueError("embedded discovery abstention needs explicit problems")
    problem_text = "\n".join(problems)
    if any(opinion.opinion_id not in problem_text for opinion in opinions):
        raise ValueError("discovery decision does not account for every opinion")

    attempt = _embedded_artifact(
        evidence,
        "rejected_attempt_record",
        "rejected_attempt_record_digest",
    )
    _require_header(attempt, "stresskit_agent_attempt_record")
    attempt_opinion = next(
        (opinion for opinion in opinions
         if opinion.opinion_id == attempt.get("opinion_id")),
        None,
    )
    if attempt.get("candidate_id") != claim_id or \
            attempt.get("source_bundle_digest") != source.digest or \
            attempt.get("panel_plan_digest") != evidence["panel_plan_digest"] or \
            attempt.get("role") != "extractor" or \
            attempt_opinion is None or attempt_opinion.role != "extractor":
        raise ValueError("rejected attempt targets another panel")
    if attempt.get("status") != "rejected_before_opinion" or \
            attempt.get("attempt") != 1 or \
            attempt.get("outcome_blind") is not True or \
            attempt.get("completion_content_inspected") is not False or \
            attempt.get("retry_performed_automatically") is not False:
        raise ValueError("rejected attempt record is not conservative")
    for field in (
        "route_binding_digest", "prompt_digest", "raw_response_digest"
    ):
        require_sha256_digest(attempt.get(field), f"rejected attempt {field}")
    if attempt.get("prompt_digest") != attempt_opinion.prompt_digest:
        raise ValueError("rejected attempt prompt differs from accepted retry")
    _parse_timestamp(attempt.get("observed_at"), "rejected attempt observed_at")

    attestation = _embedded_artifact(
        evidence, "provider_attestation", "provider_attestation_digest"
    )
    if attestation.get("catalog_observed_at") != panel.get("catalog_observed_at"):
        raise ValueError("provider attestation catalog snapshot differs from panel")
    _validate_provider_attestation(
        attestation,
        candidate,
        source,
        evidence["panel_plan_digest"],
        panel_requests,
        opinions,
        store,
    )
    _validate_abstention_closure_assertions(
        evidence,
        opinion_digests,
        source_digest=source.digest,
        decision_digest=evidence["discovery_decision_digest"],
        attempt_digest=evidence["rejected_attempt_record_digest"],
        attestation_digest=evidence["provider_attestation_digest"],
        store=store,
    )


def _execution_panel_requests(
    panel: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Mapping[str, Mapping[str, Any]]:
    """Validate a frozen two-extractor/one-critic execution plan."""
    _require_header(panel, "stresskit_agent_panel_plan")
    if panel.get("candidate_id") != candidate["claim_id"] or \
            panel.get("outcome_blind") is not True:
        raise ValueError("agent panel execution plan targets another candidate")
    if panel.get("status") not in (
        "prefrozen_awaiting_authenticated_preflight",
        "frozen",
    ):
        raise ValueError("agent panel execution plan is not frozen")
    _parse_timestamp(panel.get("catalog_observed_at"), "panel catalog_observed_at")
    require_sha256_digest(
        panel.get("claim_query_digest"), "panel claim_query_digest"
    )
    constraints = panel.get("constraints")
    if not isinstance(constraints, Mapping) or any((
        constraints.get("allow_fallbacks") is not False,
        constraints.get("data_collection") != "deny",
        constraints.get("no_plugins_or_tools") is not True,
        constraints.get("require_parameters") is not True,
        constraints.get("router_pipeline") != "must_be_empty",
        constraints.get("selected_attempt") != 1,
        constraints.get("zdr") is not True,
    )):
        raise ValueError("agent panel execution constraints are not fail-closed")

    requests = panel.get("requests")
    if not isinstance(requests, list) or len(requests) != 3 or any(
        not isinstance(row, Mapping) for row in requests
    ):
        raise ValueError("agent panel execution needs exactly three request rows")
    identifiers = [row.get("opinion_id") for row in requests]
    if len(set(identifiers)) != 3 or any(
        not isinstance(value, str) or not value for value in identifiers
    ):
        raise ValueError("agent panel execution opinion IDs must be distinct")
    if sorted(str(row.get("role")) for row in requests) != [
        "critic", "extractor", "extractor"
    ]:
        raise ValueError("agent panel execution needs two extractors and one critic")
    for field in ("provider_name", "model_family"):
        values = [row.get(field) for row in requests]
        if len(set(values)) != 3 or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"agent panel execution needs three distinct {field}s")
    canonical_models = []
    for row in requests:
        for field in ("model_request_id", "provider_endpoint"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"agent panel execution request needs {field}")
        catalog = row.get("catalog")
        if not isinstance(catalog, Mapping) or not isinstance(
            catalog.get("canonical_slug"), str
        ) or not catalog["canonical_slug"].strip():
            raise ValueError("agent panel execution request needs canonical model")
        canonical_models.append(catalog["canonical_slug"])
        parameters = row.get("request_parameters")
        if not isinstance(parameters, Mapping) or \
                parameters.get("temperature") != 0 or \
                not isinstance(parameters.get("seed"), int) or \
                isinstance(parameters.get("seed"), bool) or \
                not isinstance(parameters.get("max_tokens"), int) or \
                isinstance(parameters.get("max_tokens"), bool) or \
                parameters["max_tokens"] < 1:
            raise ValueError("agent panel execution parameters are not deterministic")
    if len(set(canonical_models)) != 3:
        raise ValueError("agent panel execution canonical models must be distinct")
    if panel.get("transport") != "openrouter":
        raise ValueError("agent panel execution transport must be openrouter")
    return {str(row["opinion_id"]): row for row in requests}


def _stored_json_object(
    store: Optional[ContentAddressedStore], digest: str, label: str
) -> Mapping[str, Any]:
    if store is None:
        raise ValueError(
            "agent panel execution abstention needs content-addressed store"
        )
    try:
        payload = store.get_json(digest)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{label} provenance is unavailable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} provenance must be a JSON object")
    return payload


def _validate_route_binding(
    route: Mapping[str, Any],
    request: Mapping[str, Any],
    candidate: Mapping[str, Any],
    panel: Mapping[str, Any],
    panel_digest: str,
) -> None:
    _require_header(route, "stresskit_openrouter_route_binding")
    catalog = request["catalog"]
    if route.get("candidate_id") != candidate["claim_id"] or \
            route.get("panel_plan_digest") != panel_digest or \
            route.get("panel_status") != panel.get("status") or \
            route.get("claim_query_digest") != panel["claim_query_digest"] or \
            route.get("opinion_id") != request["opinion_id"] or \
            route.get("role") != request["role"] or \
            route.get("provider_name") != request["provider_name"] or \
            route.get("provider_endpoint") != request["provider_endpoint"] or \
            route.get("model_family") != request["model_family"] or \
            route.get("model_request_id") != request["model_request_id"] or \
            route.get("canonical_slug") != catalog["canonical_slug"] or \
            route.get("request_parameters") != request["request_parameters"] or \
            route.get("routing_constraints") != panel["constraints"]:
        raise ValueError("agent route binding differs from frozen panel request")


def _validate_source_closure(
    evidence: Mapping[str, Any],
    candidate: Mapping[str, Any],
    store: Optional[ContentAddressedStore],
) -> SourceBundle:
    if store is None:
        raise ValueError(
            "agent panel execution abstention needs content-addressed store"
        )
    source_payload = _embedded_artifact(
        evidence, "source_bundle", "source_bundle_digest"
    )
    source = SourceBundle.from_dict(source_payload)
    if source.metadata.get("candidate_id") != candidate["claim_id"] or \
            source.metadata.get("outcome_blind") is not True:
        raise ValueError("agent panel execution SourceBundle targets another candidate")
    source_closure_digest = require_sha256_digest(
        evidence.get("source_closure_digest"),
        "agent panel execution source_closure_digest",
    )
    try:
        closure_payload = store.get_json(source_closure_digest)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("agent panel execution source closure is unavailable") from exc
    if not isinstance(closure_payload, list):
        raise ValueError("agent panel execution source closure must be a list")
    try:
        references = [ContentRef.from_dict(row) for row in closure_payload]
        verify_digest_closure(store, references, [source.digest])
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise ValueError("agent panel execution source closure is incomplete") from exc
    return source


def _validate_accepted_extractor(
    evidence: Mapping[str, Any],
    source: SourceBundle,
    candidate: Mapping[str, Any],
    panel: Mapping[str, Any],
    request: Mapping[str, Any],
    store: Optional[ContentAddressedStore],
) -> AgentOpinion:
    if store is None:
        raise ValueError(
            "agent panel execution abstention needs content-addressed store"
        )
    payload = _embedded_artifact(
        evidence, "accepted_opinion", "accepted_opinion_digest"
    )
    opinion = AgentOpinion.from_dict(payload)
    catalog = request["catalog"]
    if opinion.opinion_id != request["opinion_id"] or \
            opinion.role != "extractor" or \
            opinion.provider != request["provider_name"] or \
            opinion.model_family != request["model_family"] or \
            opinion.model != catalog["canonical_slug"] or \
            opinion.source_bundle_digest != source.digest or \
            opinion.statement != candidate["statement_to_extract"]:
        raise ValueError("accepted extractor differs from frozen panel request")
    if opinion.prompt_injection_detected:
        raise ValueError("accepted extractor detected prompt injection")
    if _stored_json_object(store, opinion.digest, "accepted extractor") != \
            opinion.to_dict():
        raise ValueError("accepted extractor CAS object differs from embedded opinion")

    closure_digest = require_sha256_digest(
        evidence.get("accepted_opinion_closure_digest"),
        "accepted extractor closure_digest",
    )
    try:
        closure_payload = store.get_json(closure_digest)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("accepted extractor closure is unavailable") from exc
    if not isinstance(closure_payload, list):
        raise ValueError("accepted extractor closure must be a list")
    try:
        references = [ContentRef.from_dict(row) for row in closure_payload]
        verify_digest_closure(store, references, [opinion.digest])
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise ValueError("accepted extractor closure is incomplete") from exc

    documents = {row["document_id"]: row for row in source.documents}
    for anchor in opinion.evidence_anchors:
        document = documents.get(anchor["document_id"])
        if document is None or anchor["source_digest"] != document["source_digest"]:
            raise ValueError("accepted extractor anchor names another source")
        text_digest = document.get(
            "extracted_text_digest", document["source_digest"]
        )
        if anchor["text_digest"] != text_digest:
            raise ValueError("accepted extractor anchor text digest mismatch")
        try:
            text = store.get_bytes(text_digest)
            quote = store.get_bytes(anchor["quote_digest"])
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("accepted extractor anchor bytes are unavailable") from exc
        start, end = int(anchor["start"]), int(anchor["end"])
        if end > len(text) or text[start:end] != quote or \
                sha256_bytes(quote) != anchor["quote_digest"]:
            raise ValueError("accepted extractor anchor does not match source bytes")

    descriptor = _stored_json_object(store, opinion.model_digest, "model descriptor")
    prompt = _stored_json_object(store, opinion.prompt_digest, "agent prompt")
    receipt = _stored_json_object(store, opinion.request_digest, "request receipt")
    _require_header(descriptor, "stresskit_agent_model_descriptor")
    _require_header(prompt, "stresskit_agent_prompt")
    _require_header(receipt, "stresskit_agent_request_receipt")
    route_digest = require_sha256_digest(
        descriptor.get("route_binding_digest"), "model route_binding_digest"
    )
    route = _stored_json_object(store, route_digest, "route binding")
    _validate_route_binding(
        route, request, candidate, panel, evidence["panel_plan_digest"]
    )
    raw_digest = require_sha256_digest(
        descriptor.get("raw_response_digest"), "model raw_response_digest"
    )
    response = receipt.get("response")
    body = receipt.get("body")
    authorization = receipt.get("authorization")
    expected_policy = {
        **_OPENROUTER_REQUEST_POLICY,
        "only": [request["provider_endpoint"]],
    }
    if descriptor.get("panel_plan_digest") != evidence["panel_plan_digest"] or \
            descriptor.get("provider") != opinion.provider or \
            descriptor.get("model") != opinion.model or \
            descriptor.get("family") != opinion.model_family or \
            descriptor.get("requested_model") != request["model_request_id"] or \
            descriptor.get("requested_provider_endpoint") != \
            request["provider_endpoint"]:
        raise ValueError("accepted extractor model descriptor differs from panel")
    if prompt.get("role") != "extractor" or \
            prompt.get("source_bundle_digest") != source.digest or \
            prompt.get("route_binding_digest") != route_digest or \
            prompt.get("claim_query_digest") != panel["claim_query_digest"]:
        raise ValueError("accepted extractor prompt differs from panel")
    if not isinstance(response, Mapping) or not isinstance(body, Mapping) or \
            not isinstance(authorization, Mapping) or \
            receipt.get("transport") != "openrouter" or \
            receipt.get("method") != "POST" or \
            receipt.get("panel_plan_digest") != evidence["panel_plan_digest"] or \
            receipt.get("route_binding_digest") != route_digest or \
            receipt.get("prompt_digest") != opinion.prompt_digest or \
            body.get("model") != request["model_request_id"] or \
            body.get("provider") != expected_policy or \
            response.get("raw_digest") != raw_digest or \
            response.get("response_id") != descriptor.get("response_id") or \
            authorization.get("serialized") is not False or \
            not str(authorization.get("source", "")).startswith("environment:"):
        raise ValueError("accepted extractor request receipt differs from panel")
    return opinion


def _validate_rejected_attempt_provenance(
    attempt: Mapping[str, Any],
    source: SourceBundle,
    candidate: Mapping[str, Any],
    panel: Mapping[str, Any],
    request: Mapping[str, Any],
    panel_digest: str,
    store: Optional[ContentAddressedStore],
    *,
    completion_content_inspected: bool,
) -> Mapping[str, Any]:
    _require_header(attempt, "stresskit_agent_attempt_record")
    if attempt.get("candidate_id") != candidate["claim_id"] or \
            attempt.get("source_bundle_digest") != source.digest or \
            attempt.get("panel_plan_digest") != panel_digest or \
            attempt.get("opinion_id") != request["opinion_id"] or \
            attempt.get("role") != "extractor":
        raise ValueError("rejected extractor attempt targets another panel slot")
    if attempt.get("status") != "rejected_before_opinion" or \
            attempt.get("attempt") != 1 or \
            attempt.get("completion_content_inspected") is not \
            completion_content_inspected or \
            attempt.get("retry_performed") is not False or \
            attempt.get("critic_called") is not False:
        raise ValueError("rejected extractor attempt is not fail-closed")
    if attempt.get("completion_content_human_inspected") not in (None, False):
        raise ValueError("rejected extractor attempt was human-inspected")
    if not isinstance(attempt.get("reason"), str) or not attempt["reason"].strip():
        raise ValueError("rejected extractor attempt needs explicit reason")
    _parse_timestamp(attempt.get("observed_at"), "rejected attempt observed_at")
    route_digest = require_sha256_digest(
        attempt.get("route_binding_digest"), "rejected route_binding_digest"
    )
    prompt_digest = require_sha256_digest(
        attempt.get("prompt_digest"), "rejected prompt_digest"
    )
    raw_digest = require_sha256_digest(
        attempt.get("raw_response_digest"), "rejected raw_response_digest"
    )
    route = _stored_json_object(store, route_digest, "rejected route binding")
    prompt = _stored_json_object(store, prompt_digest, "rejected agent prompt")
    raw = _stored_json_object(store, raw_digest, "rejected raw response")
    _validate_route_binding(route, request, candidate, panel, panel_digest)
    _require_header(prompt, "stresskit_agent_prompt")
    if prompt.get("role") != "extractor" or \
            prompt.get("source_bundle_digest") != source.digest or \
            prompt.get("route_binding_digest") != route_digest or \
            prompt.get("claim_query_digest") != panel["claim_query_digest"]:
        raise ValueError("rejected extractor prompt differs from panel")
    if raw.get("id") != attempt.get("response_id") or \
            raw.get("provider") != request["provider_name"] or \
            raw.get("model") not in (
                request["model_request_id"], request["catalog"]["canonical_slug"]
            ):
        raise ValueError("rejected raw response differs from frozen route")
    route_metadata = attempt.get("safe_route_metadata")
    available = route_metadata.get("available_endpoints") \
        if isinstance(route_metadata, Mapping) else None
    if not isinstance(route_metadata, Mapping) or \
            route_metadata.get("strategy") != "direct" or \
            route_metadata.get("selected_attempt") != 1 or \
            route_metadata.get("pipeline") is not None or \
            route_metadata.get("requested_model") != request["model_request_id"] or \
            route_metadata.get("response_model") not in (
                request["model_request_id"], request["catalog"]["canonical_slug"]
            ) or \
            not isinstance(available, list) or len(available) != 1 or \
            not isinstance(available[0], Mapping) or \
            available[0].get("provider") != request["provider_name"] or \
            available[0].get("model") != request["catalog"]["canonical_slug"] or \
            available[0].get("selected") is not True:
        raise ValueError("rejected attempt route metadata is not fail-closed")
    return raw


def _validate_rejected_extractor_attempt(
    attempt: Mapping[str, Any],
    source: SourceBundle,
    candidate: Mapping[str, Any],
    panel: Mapping[str, Any],
    request: Mapping[str, Any],
    panel_digest: str,
    store: Optional[ContentAddressedStore],
) -> None:
    raw = _validate_rejected_attempt_provenance(
        attempt,
        source,
        candidate,
        panel,
        request,
        panel_digest,
        store,
        completion_content_inspected=True,
    )
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or \
            not isinstance(choices[0], Mapping) or \
            not isinstance(choices[0].get("message"), Mapping) or \
            not isinstance(choices[0]["message"].get("content"), str):
        raise ValueError("rejected raw response has no inspectable completion")
    try:
        completion = json.loads(choices[0]["message"]["content"])
    except json.JSONDecodeError as exc:
        raise ValueError("rejected completion is not structured JSON") from exc
    quotes = completion.get("evidence_quotes") \
        if isinstance(completion, Mapping) else None
    checks = attempt.get("evidence_quote_checks")
    if not isinstance(quotes, list) or not quotes or \
            not isinstance(checks, list) or len(checks) != len(quotes):
        raise ValueError("rejected attempt quote accounting is incomplete")
    documents = {row["document_id"]: row for row in source.documents}
    absent_quotes = 0
    for quote_row, check in zip(quotes, checks):
        if not isinstance(quote_row, Mapping) or \
                not isinstance(check, Mapping) or \
                not isinstance(quote_row.get("quote"), str) or \
                not quote_row["quote"]:
            raise ValueError("rejected attempt quote accounting is invalid")
        quote_bytes = quote_row["quote"].encode("utf-8")
        quote_digest = sha256_bytes(quote_bytes)
        document_id = quote_row.get("document_id")
        document = documents.get(document_id)
        recorded_presence = check.get("present_in_declared_source_bytes")
        if document is None or check.get("document_id") != document_id or \
                check.get("quote_digest") != quote_digest or \
                not isinstance(recorded_presence, bool):
            raise ValueError("rejected attempt quote check differs from raw response")
        text_digest = document.get(
            "extracted_text_digest", document["source_digest"]
        )
        try:
            source_text = store.get_bytes(text_digest) if store else b""
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("rejected attempt source bytes are unavailable") from exc
        present = quote_bytes in source_text
        if recorded_presence is not present:
            raise ValueError("rejected attempt quote check differs from source bytes")
        absent_quotes += int(not present)
    if absent_quotes < 1:
        raise ValueError("rejected invalid-evidence attempt has no absent quote")


def _validate_incomplete_extractor_attempt(
    attempt: Mapping[str, Any],
    source: SourceBundle,
    candidate: Mapping[str, Any],
    panel: Mapping[str, Any],
    request: Mapping[str, Any],
    panel_digest: str,
    store: Optional[ContentAddressedStore],
) -> None:
    raw = _validate_rejected_attempt_provenance(
        attempt,
        source,
        candidate,
        panel,
        request,
        panel_digest,
        store,
        completion_content_inspected=False,
    )
    if attempt.get("evidence_quote_checks") != []:
        raise ValueError("incomplete extractor attempt cannot carry quote checks")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or \
            not isinstance(choices[0], Mapping) or \
            not isinstance(choices[0].get("message"), Mapping):
        raise ValueError("incomplete raw response has no completion slot")
    choice = choices[0]
    if choice.get("finish_reason") != "length" or \
            attempt.get("finish_reason") != "length":
        raise ValueError("incomplete extractor finish reason is not length")
    native_finish_reason = choice.get("native_finish_reason")
    if native_finish_reason is not None and \
            attempt.get("native_finish_reason") != native_finish_reason:
        raise ValueError("incomplete extractor native finish reason mismatch")


def _validate_plural_agent_panel_execution_abstention(
    evidence: Mapping[str, Any],
    candidate: Mapping[str, Any],
    store: Optional[ContentAddressedStore],
) -> None:
    """Validate a panel where neither extractor produced an AgentOpinion."""
    _require_header(evidence, "stresskit_agent_panel_execution_abstention")
    if evidence.get("claim_id") != candidate["claim_id"] or \
            evidence.get("outcome_blind") is not True or \
            evidence.get("publication_state") != "abstain" or \
            evidence.get("reason") != "invalid_agent_outputs" or \
            evidence.get("prompt_injection_detected") is not False:
        raise ValueError("agent panel execution abstention targets another candidate")
    if evidence.get("accepted_opinions") != []:
        raise ValueError("zero-opinion panel abstention cannot accept an opinion")

    source = _validate_source_closure(evidence, candidate, store)
    panel = _embedded_artifact(evidence, "panel_plan", "panel_plan_digest")
    requests = _execution_panel_requests(panel, candidate)
    request_rows = list(panel["requests"])

    execution = _embedded_artifact(
        evidence, "panel_execution", "panel_execution_digest"
    )
    _require_header(execution, "stresskit_agent_panel_execution")
    if execution.get("candidate_id") != candidate["claim_id"] or \
            execution.get("source_bundle_digest") != source.digest or \
            execution.get("panel_plan_digest") != evidence["panel_plan_digest"] or \
            execution.get("outcome_blind") is not True or \
            execution.get("status") != "abstain" or \
            execution.get("publication_state") != "abstain" or \
            execution.get("retry_policy") != "no_retry" or \
            execution.get("complete_slots") is not True:
        raise ValueError("agent panel execution is not a complete abstention")
    slots = execution.get("slots")
    if not isinstance(slots, list) or len(slots) != 3 or any(
        not isinstance(row, Mapping) for row in slots
    ) or [row.get("opinion_id") for row in slots] != [
        row["opinion_id"] for row in request_rows
    ] or [row.get("role") for row in slots] != [
        row["role"] for row in request_rows
    ]:
        raise ValueError("agent panel execution slots are not exact")
    extractor_slots = [row for row in slots if row["role"] == "extractor"]
    critic_slots = [row for row in slots if row["role"] == "critic"]
    allowed_rejections = {
        "rejected_invalid_evidence",
        "rejected_incomplete_completion",
    }
    if len(extractor_slots) != 2 or len(critic_slots) != 1 or any(
        row.get("status") not in allowed_rejections for row in extractor_slots
    ) or critic_slots[0].get("status") != "not_run_dependency_failure":
        raise ValueError("agent panel execution slot states are not fail-closed")

    rejected_rows = evidence.get("rejected_attempts")
    if not isinstance(rejected_rows, list) or len(rejected_rows) != 2 or any(
        not isinstance(row, Mapping) for row in rejected_rows
    ):
        raise ValueError("zero-opinion panel needs two rejected attempts")
    attempts: Dict[str, Tuple[Mapping[str, Any], str]] = {}
    for row in rejected_rows:
        attempt = _embedded_artifact(
            row, "attempt_record", "attempt_record_digest"
        )
        opinion_id = attempt.get("opinion_id")
        if not isinstance(opinion_id, str) or opinion_id in attempts or \
                opinion_id not in requests or \
                requests[opinion_id]["role"] != "extractor" or \
                row.get("opinion_id", opinion_id) != opinion_id:
            raise ValueError("rejected attempt list does not match extractor slots")
        attempts[opinion_id] = (attempt, row["attempt_record_digest"])

    expected_extractor_ids = [
        str(row["opinion_id"])
        for row in request_rows
        if row["role"] == "extractor"
    ]
    if list(attempts) != expected_extractor_ids:
        raise ValueError("rejected attempt list order is not exact")
    for slot in extractor_slots:
        opinion_id = str(slot["opinion_id"])
        attempt, attempt_digest = attempts[opinion_id]
        if slot.get("attempt_digest") != attempt_digest or \
                slot.get("raw_response_digest") != \
                attempt.get("raw_response_digest"):
            raise ValueError("rejected execution slot differs from attempt evidence")
        request = requests[opinion_id]
        if slot["status"] == "rejected_invalid_evidence":
            _validate_rejected_extractor_attempt(
                attempt,
                source,
                candidate,
                panel,
                request,
                evidence["panel_plan_digest"],
                store,
            )
        else:
            _validate_incomplete_extractor_attempt(
                attempt,
                source,
                candidate,
                panel,
                request,
                evidence["panel_plan_digest"],
                store,
            )

    critic_slot = critic_slots[0]
    critic_request = requests[str(critic_slot["opinion_id"])]
    if critic_slot.get("opinion_id") != critic_request["opinion_id"] or \
            critic_slot.get("depends_on") != expected_extractor_ids or \
            not isinstance(critic_slot.get("reason"), str) or \
            not critic_slot["reason"].strip():
        raise ValueError("critic dependency-failure slot is incomplete")

    decision = _embedded_artifact(
        evidence, "discovery_decision", "discovery_decision_digest"
    )
    _require_header(decision, "stresskit_claim_candidates")
    if decision.get("source_bundle_digest") != source.digest or \
            decision.get("panel_execution_digest") != \
            evidence["panel_execution_digest"] or \
            decision.get("publication_state") != "abstain" or \
            decision.get("candidates") != []:
        raise ValueError("agent panel execution discovery decision is not abstain")
    problems = decision.get("problems")
    problem_text = "\n".join(problems) if isinstance(problems, list) and all(
        isinstance(row, str) and row for row in problems
    ) else ""
    failed_ids = [*expected_extractor_ids, str(critic_slot["opinion_id"])]
    if not problem_text or any(value not in problem_text for value in failed_ids):
        raise ValueError("discovery decision does not account for failed slots")

    panel_closure = evidence.get("panel_closure")
    if not isinstance(panel_closure, list):
        raise ValueError("agent panel execution needs embedded panel closure")
    closure_digest = require_sha256_digest(
        evidence.get("panel_closure_digest"),
        "agent panel execution panel_closure_digest",
    )
    if digest_json(panel_closure) != closure_digest:
        raise ValueError("agent panel execution closure digest mismatch")
    expected_roots = [
        evidence["source_closure_digest"],
        evidence["discovery_decision_digest"],
    ]
    if evidence.get("panel_closure_roots") != expected_roots:
        raise ValueError("agent panel execution closure roots are not exact")
    if store is None:
        raise ValueError(
            "agent panel execution abstention needs content-addressed store"
        )
    try:
        references = [ContentRef.from_dict(row) for row in panel_closure]
        verify_digest_closure(store, references, expected_roots)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise ValueError("agent panel execution digest closure is incomplete") from exc


def _validate_agent_panel_execution_abstention(
    evidence: Mapping[str, Any],
    candidate: Mapping[str, Any],
    store: Optional[ContentAddressedStore],
) -> None:
    """Validate complete slot accounting when one extractor output is invalid."""
    if "accepted_opinions" in evidence or "rejected_attempts" in evidence:
        _validate_plural_agent_panel_execution_abstention(
            evidence, candidate, store
        )
        return
    _require_header(evidence, "stresskit_agent_panel_execution_abstention")
    if evidence.get("claim_id") != candidate["claim_id"] or \
            evidence.get("outcome_blind") is not True or \
            evidence.get("publication_state") != "abstain" or \
            evidence.get("reason") != "invalid_evidence_anchor" or \
            evidence.get("prompt_injection_detected") is not False:
        raise ValueError("agent panel execution abstention targets another candidate")
    source = _validate_source_closure(evidence, candidate, store)
    panel = _embedded_artifact(evidence, "panel_plan", "panel_plan_digest")
    requests = _execution_panel_requests(panel, candidate)
    request_rows = list(panel["requests"])

    execution = _embedded_artifact(
        evidence, "panel_execution", "panel_execution_digest"
    )
    _require_header(execution, "stresskit_agent_panel_execution")
    if execution.get("candidate_id") != candidate["claim_id"] or \
            execution.get("source_bundle_digest") != source.digest or \
            execution.get("panel_plan_digest") != evidence["panel_plan_digest"] or \
            execution.get("outcome_blind") is not True or \
            execution.get("status") != "abstain" or \
            execution.get("publication_state") != "abstain" or \
            execution.get("retry_policy") != "no_retry" or \
            execution.get("complete_slots") is not True:
        raise ValueError("agent panel execution is not a complete abstention")
    slots = execution.get("slots")
    if not isinstance(slots, list) or len(slots) != 3 or any(
        not isinstance(row, Mapping) for row in slots
    ) or [row.get("opinion_id") for row in slots] != [
        row["opinion_id"] for row in request_rows
    ]:
        raise ValueError("agent panel execution slots are not exact")
    by_status = {str(row.get("status")): row for row in slots}
    if set(by_status) != {
        "accepted", "rejected_invalid_evidence", "not_run_dependency_failure"
    }:
        raise ValueError("agent panel execution slot states are not exact")
    accepted_slot = by_status["accepted"]
    rejected_slot = by_status["rejected_invalid_evidence"]
    critic_slot = by_status["not_run_dependency_failure"]
    if accepted_slot.get("role") != "extractor" or \
            rejected_slot.get("role") != "extractor" or \
            critic_slot.get("role") != "critic":
        raise ValueError("agent panel execution slot roles are invalid")

    accepted_request = requests[str(accepted_slot["opinion_id"])]
    rejected_request = requests[str(rejected_slot["opinion_id"])]
    critic_request = requests[str(critic_slot["opinion_id"])]
    accepted = _validate_accepted_extractor(
        evidence, source, candidate, panel, accepted_request, store
    )
    accepted_closure = require_sha256_digest(
        evidence.get("accepted_opinion_closure_digest"),
        "accepted extractor closure_digest",
    )
    if accepted_slot.get("opinion_digest") != accepted.digest or \
            accepted_slot.get("closure_digest") != accepted_closure:
        raise ValueError("accepted execution slot differs from extractor evidence")

    attempt = _embedded_artifact(
        evidence, "rejected_attempt_record", "rejected_attempt_record_digest"
    )
    _validate_rejected_extractor_attempt(
        attempt,
        source,
        candidate,
        panel,
        rejected_request,
        evidence["panel_plan_digest"],
        store,
    )
    if rejected_slot.get("attempt_digest") != \
            evidence["rejected_attempt_record_digest"] or \
            rejected_slot.get("raw_response_digest") != \
            attempt["raw_response_digest"]:
        raise ValueError("rejected execution slot differs from attempt evidence")
    extractor_ids = [
        str(row["opinion_id"])
        for row in request_rows
        if row["role"] == "extractor"
    ]
    if critic_slot.get("opinion_id") != critic_request["opinion_id"] or \
            critic_slot.get("depends_on") != extractor_ids or \
            not isinstance(critic_slot.get("reason"), str) or \
            not critic_slot["reason"].strip():
        raise ValueError("critic dependency-failure slot is incomplete")

    decision = _embedded_artifact(
        evidence, "discovery_decision", "discovery_decision_digest"
    )
    _require_header(decision, "stresskit_claim_candidates")
    if decision.get("source_bundle_digest") != source.digest or \
            decision.get("panel_execution_digest") != \
            evidence["panel_execution_digest"] or \
            decision.get("publication_state") != "abstain" or \
            decision.get("candidates") != []:
        raise ValueError("agent panel execution discovery decision is not abstain")
    problems = decision.get("problems")
    problem_text = "\n".join(problems) if isinstance(problems, list) and all(
        isinstance(row, str) and row for row in problems
    ) else ""
    if not problem_text or rejected_slot["opinion_id"] not in problem_text or \
            critic_slot["opinion_id"] not in problem_text:
        raise ValueError("discovery decision does not account for failed slots")

    panel_closure = evidence.get("panel_closure")
    if not isinstance(panel_closure, list):
        raise ValueError("agent panel execution needs embedded panel closure")
    closure_digest = require_sha256_digest(
        evidence.get("panel_closure_digest"),
        "agent panel execution panel_closure_digest",
    )
    if digest_json(panel_closure) != closure_digest:
        raise ValueError("agent panel execution closure digest mismatch")
    roots = evidence.get("panel_closure_roots")
    expected_roots = [
        evidence["source_closure_digest"],
        evidence["discovery_decision_digest"],
    ]
    if roots != expected_roots:
        raise ValueError("agent panel execution closure roots are not exact")
    if store is None:
        raise ValueError(
            "agent panel execution abstention needs content-addressed store"
        )
    try:
        references = [ContentRef.from_dict(row) for row in panel_closure]
        verify_digest_closure(store, references, expected_roots)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise ValueError("agent panel execution digest closure is incomplete") from exc


def _validate_agent_panel_abstention(
    evidence: Mapping[str, Any],
    candidate: Mapping[str, Any],
    store: Optional[ContentAddressedStore],
) -> None:
    artifact = evidence.get("artifact")
    if artifact == "stresskit_agent_panel_abstention":
        _validate_unanimous_agent_panel_abstention(evidence, candidate, store)
        return
    if artifact == "stresskit_agent_panel_execution_abstention":
        _validate_agent_panel_execution_abstention(evidence, candidate, store)
        return
    raise ValueError("agent panel fail evidence has unsupported artifact type")


def _validate_license_closure(
    evidence: Mapping[str, Any], source: SourceBundle, claim: ClaimRecord
) -> None:
    _require_header(evidence, "stresskit_license_closure")
    if evidence.get("complete") is not True or \
            evidence.get("determined_without_outcomes") is not True:
        raise ValueError("license closure must be complete and outcome-blind")
    if evidence.get("claim_record_digest") != claim.digest:
        raise ValueError("license closure targets another ClaimRecord")
    if evidence.get("dependency_manifest_digest") != \
            claim.code_map["dependency_manifest_digest"]:
        raise ValueError("license closure dependency manifest differs from ClaimRecord")
    items = evidence.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("license closure needs at least one item")
    identities = []
    document_ids = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("license closure items must be objects")
        if item.get("status") != "verified_compatible":
            raise ValueError("every license item must be verified compatible")
        for name in ("kind", "identifier", "revision", "license"):
            if not isinstance(item.get(name), str) or not item[name].strip():
                raise ValueError(f"license item needs {name}")
        require_sha256_digest(item.get("evidence_digest"), "license item evidence_digest")
        identities.append((item["kind"], item["identifier"], item["revision"]))
        if isinstance(item.get("document_id"), str):
            document_ids.add(item["document_id"])
    if len(identities) != len(set(identities)):
        raise ValueError("license closure contains duplicate items")
    expected_documents = {row["document_id"] for row in source.documents}
    if document_ids != expected_documents:
        raise ValueError("license closure must cover every SourceBundle document")


def _validate_smoke(
    evidence: Mapping[str, Any], candidate: Mapping[str, Any], upstream: Mapping[str, Any],
    claim: ClaimRecord,
) -> None:
    _require_header(evidence, "stresskit_execution_smoke_evidence")
    if evidence.get("status") != "pass" or \
            evidence.get("not_claim_reproduction") is not True or \
            evidence.get("not_benchmark_outcome") is not True or \
            evidence.get("claim_map_exercised") is not True:
        raise ValueError("execution smoke must pass and exercise the frozen claim map")
    if evidence.get("upstream") != candidate["upstream"] or \
            evidence.get("upstream_commit") != upstream["commit"]:
        raise ValueError("execution smoke upstream revision mismatch")
    if evidence.get("claim_record_digest") != claim.digest:
        raise ValueError("execution smoke targets another ClaimRecord")
    require_sha256_digest(evidence.get("raw_artifact_digest"), "smoke raw_artifact_digest")
    if evidence.get("execution_isolation") != _ISOLATION:
        raise ValueError("execution smoke lacks required network/credential isolation")


def _validate_review(evidence: Mapping[str, Any], spec: AuditSpec) -> None:
    _require_header(evidence, "stresskit_protocol_review")
    if evidence.get("status") != "approved" or evidence.get("issues") != []:
        raise ValueError("protocol review must be approved with no open issues")
    if evidence.get("audit_spec_digest") != spec.digest:
        raise ValueError("protocol review targets another AuditSpec")
    mode = evidence.get("review_mode")
    if mode not in ("external", "agent_only"):
        raise ValueError("protocol review mode must be external or agent_only")
    if mode == "agent_only" and evidence.get("user_authorized") is not True:
        raise ValueError("agent-only review needs explicit user authorization")
    if not isinstance(evidence.get("reviewer_id"), str) or not evidence["reviewer_id"]:
        raise ValueError("protocol review needs reviewer_id")
    _parse_timestamp(evidence.get("reviewed_at"), "protocol review reviewed_at")


def _validate_resource_estimate(
    evidence: Mapping[str, Any], candidate: Mapping[str, Any], spec: AuditSpec
) -> Mapping[str, Any]:
    _require_header(evidence, "stresskit_resource_estimate")
    if evidence.get("estimated_without_outcomes") is not True or \
            evidence.get("audit_spec_digest") != spec.digest:
        raise ValueError("resource estimate must bind AuditSpec without outcomes")
    label = candidate["compute_tier"]
    if evidence.get("compute_tier_label") != label or \
            evidence.get("compute_tier") != COMPUTE_TIERS[label]:
        raise ValueError("resource estimate compute tier differs from candidate frame")
    if evidence.get("hardware_class") != spec.design.get("hardware_class"):
        raise ValueError("resource estimate hardware class differs from AuditSpec")
    resources = evidence.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("resource estimate needs resources object")
    for name in ("gpu_count", "cpu_count", "wall_time_seconds", "storage_bytes"):
        value = resources.get(name)
        minimum = 0 if name == "gpu_count" else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"resource estimate {name} is invalid")
    if COMPUTE_TIERS[label] > 0 and resources["gpu_count"] < 1:
        raise ValueError("GPU compute tier needs at least one GPU")
    return resources


def _validate_pass_bundle(
    record: Mapping[str, Any], candidate: Mapping[str, Any], upstream: Mapping[str, Any]
) -> AuditSpec:
    gates = record["gates"]
    source_evidence = gates["source_bundle"]["evidence"]
    _require_header(source_evidence, "stresskit_source_bundle_evidence")
    source = SourceBundle.from_dict(source_evidence.get("source_bundle", {}))
    source_texts = source_evidence.get("source_texts")
    if not isinstance(source_texts, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in source_texts.items()):
        raise ValueError("SourceBundle evidence needs exact UTF-8 source texts")

    panel = gates["agent_panel"]["evidence"]
    _require_header(panel, "stresskit_agent_panel")
    if panel.get("source_bundle_digest") != source.digest:
        raise ValueError("agent panel targets another SourceBundle")
    opinions_payload = panel.get("opinions")
    if not isinstance(opinions_payload, list):
        raise ValueError("agent panel needs opinions")
    opinions = [AgentOpinion.from_dict(row) for row in opinions_payload]
    panel_problems = validate_agent_panel(
        source, opinions, source_texts=source_texts
    )
    if panel_problems:
        raise ValueError("invalid agent panel: " + "; ".join(panel_problems))

    claim = ClaimRecord.from_dict(gates["claim_record"]["evidence"])
    if claim.claim_id != candidate["claim_id"] or \
            claim.source_bundle_digest != source.digest:
        raise ValueError("ClaimRecord candidate or SourceBundle mismatch")
    if list(claim.agent_opinion_digests) != [opinion.digest for opinion in opinions]:
        raise ValueError("ClaimRecord agent opinion closure mismatch")
    verified_texts = claim.metadata.get("source_text_verification", {}).get(
        "document_digests", {}
    )
    expected_texts = {
        row["document_id"]: row.get("extracted_text_digest", row["source_digest"])
        for row in source.documents
        if "extracted_text_digest" in row or row["document_id"] in verified_texts
    }
    if verified_texts != expected_texts:
        raise ValueError("ClaimRecord source-text closure differs from SourceBundle")
    if claim.code_map["revision"] != upstream["commit"]:
        raise ValueError("ClaimRecord revision differs from candidate upstream")
    entrypoints = [part.strip() for part in str(candidate["entrypoint"]).split(";")]
    if not set(entrypoints) <= set(claim.code_map["entrypoints"]):
        raise ValueError("ClaimRecord omits candidate entrypoint")
    profile = get_profile(claim.profile_id)
    if claim.finding_type != profile.finding_type or \
            claim.reducer.get("name") != profile.reducer_name or \
            claim.reducer.get("implementation_digest") != reducer_digest(
                profile.reducer_name
            ):
        raise ValueError("ClaimRecord reducer/profile mismatch")

    _validate_license_closure(
        gates["license_closure"]["evidence"], source, claim
    )
    _validate_smoke(
        gates["execution_smoke"]["evidence"], candidate, upstream, claim
    )

    spec = AuditSpec.from_dict(gates["audit_spec"]["evidence"])
    if spec.claim_record_digest != claim.digest or spec.claim_record != claim.to_dict():
        raise ValueError("AuditSpec embeds another ClaimRecord")
    if spec.profile_digest != profile.digest:
        raise ValueError("AuditSpec profile digest is not registered")
    regenerated = regenerate_run_manifest(spec.audit_id, spec.design)
    if regenerated != list(spec.run_manifest):
        raise ValueError("AuditSpec run manifest does not regenerate")

    _validate_review(gates["protocol_review"]["evidence"], spec)
    resources = _validate_resource_estimate(
        gates["resource_estimate"]["evidence"], candidate, spec
    )

    if record.get("claim_record_digest") != claim.digest or \
            record.get("audit_spec_digest") != spec.digest:
        raise ValueError("qualification record artifact digest mismatch")
    if record.get("stratum") != FAMILY_STRATA[candidate["family"]] or \
            record.get("compute_tier") != COMPUTE_TIERS[candidate["compute_tier"]]:
        raise ValueError("qualification stratum or compute tier mismatch")
    comparison = record.get("comparison")
    if not isinstance(comparison, Mapping) or set(comparison) != {
            "task", "metric", "evaluation_set", "resource_budget"}:
        raise ValueError("eligible record needs exact comparison fields")
    if comparison["task"] != candidate["task"]:
        raise ValueError("comparison task differs from candidate frame")
    for name in ("metric", "evaluation_set"):
        if not isinstance(comparison[name], str) or not comparison[name].strip():
            raise ValueError(f"comparison {name} must not be empty")
    if comparison["resource_budget"] != resources:
        raise ValueError("comparison resource budget differs from estimate")
    return spec


def _record_state(
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    upstream: Mapping[str, Any],
    store: Optional[ContentAddressedStore],
) -> Tuple[str, Optional[AuditSpec]]:
    if record.get("claim_id") != candidate["claim_id"] or \
            record.get("candidate_digest") != digest_json(candidate):
        raise ValueError(f"qualification record does not bind {candidate['claim_id']}")
    gates = record.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(PREFREEZE_GATES):
        raise ValueError(f"{candidate['claim_id']} must contain every pre-freeze gate")
    statuses = {
        name: _validate_gate(candidate["claim_id"], name, gates[name])
        for name in PREFREEZE_GATES
    }
    if statuses["agent_panel"] == "fail":
        _validate_agent_panel_abstention(
            gates["agent_panel"]["evidence"], candidate, store
        )
    preexcluded = str(candidate["eligibility"]).startswith("excluded_pre_freeze")
    if preexcluded or "fail" in statuses.values():
        expected = "excluded"
        reason = record.get("exclusion_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"excluded claim {candidate['claim_id']} needs reason")
        spec = None
    elif all(status == "pass" for status in statuses.values()):
        expected = "eligible"
        spec = _validate_pass_bundle(record, candidate, upstream)
    else:
        expected = "pending"
        spec = None
    if record.get("disposition") != expected:
        raise ValueError(
            f"{candidate['claim_id']} disposition must be derived as {expected}"
        )
    return expected, spec


def evaluate_qualification(
    registry: Mapping[str, Any],
    qualification: Mapping[str, Any],
    *,
    store: Optional[ContentAddressedStore] = None,
) -> Dict[str, Any]:
    """Validate every row and report blockers without reading outcomes."""
    candidates, upstreams = _candidate_rows(registry)
    _require_header(qualification, ARTIFACT)
    if qualification.get("outcome_blind") is not True or \
            qualification.get("candidate_registry_digest") != digest_json(registry):
        raise ValueError("qualification does not bind outcome-blind candidate registry")
    if qualification.get("required_gates") != list(PREFREEZE_GATES):
        raise ValueError("qualification required_gates differ from v1 protocol")
    _require_outcome_blind(qualification, "qualification")
    records = qualification.get("records")
    if not isinstance(records, list) or len(records) != len(candidates):
        raise ValueError("qualification needs exactly one row per candidate")
    if [record.get("claim_id") for record in records] != [
            candidate["claim_id"] for candidate in candidates]:
        raise ValueError("qualification must preserve candidate registry order")

    states = []
    specs: Dict[str, AuditSpec] = {}
    gate_counts = {gate: Counter() for gate in PREFREEZE_GATES}
    for candidate, record in zip(candidates, records):
        state, spec = _record_state(
            record, candidate, upstreams[candidate["upstream"]], store
        )
        states.append(state)
        for gate in PREFREEZE_GATES:
            gate_counts[gate][record["gates"][gate]["status"]] += 1
        if spec is not None:
            specs[candidate["claim_id"]] = spec

    all_resolved = "pending" not in states
    eligible_ids = [
        candidate["claim_id"]
        for candidate, state in zip(candidates, states)
        if state == "eligible"
    ]
    if all_resolved and eligible_ids:
        expected_ids = sorted(eligible_ids)
        family_ids = set()
        for claim_id, spec in specs.items():
            members = spec.multiplicity_family.get("member_claim_ids")
            if sorted(members or []) != expected_ids:
                raise ValueError(
                    f"{claim_id} multiplicity family does not cover all eligible claims"
                )
            if spec.multiplicity_family.get("release_manifest_digest") != \
                    digest_json(expected_ids):
                raise ValueError(f"{claim_id} release multiplicity digest mismatch")
            family_ids.add(spec.multiplicity_family.get("family_id"))
        if len(family_ids) != 1:
            raise ValueError("eligible AuditSpecs must share one multiplicity family")

    eligible_candidates = [
        candidate for candidate, state in zip(candidates, states)
        if state == "eligible"
    ]
    breadth = {
        "eligible_claims": len(eligible_candidates),
        "method_families": len({row["family"] for row in eligible_candidates}),
        "model_families": len({row["model_family"] for row in eligible_candidates}),
        "strata": len({FAMILY_STRATA[row["family"]] for row in eligible_candidates}),
        "minimums": {
            "eligible_claims": 20,
            "method_families": 6,
            "model_families": 3,
        },
    }
    breadth_met = all(
        breadth[name] >= breadth["minimums"][name]
        for name in ("eligible_claims", "method_families", "model_families")
    )
    disposition_counts = Counter(states)
    return {
        "artifact": REPORT_ARTIFACT,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "candidate_registry_digest": digest_json(registry),
        "qualification_digest": digest_json(qualification),
        "outcome_blind": True,
        "status": "freeze_ready" if all_resolved and breadth_met else (
            "backward_extension_required" if all_resolved else "qualification_in_progress"
        ),
        "freeze_ready": all_resolved and breadth_met,
        "backward_extension_required": all_resolved and not breadth_met,
        "dispositions": {
            name: disposition_counts.get(name, 0)
            for name in ("eligible", "excluded", "pending")
        },
        "gate_counts": {
            gate: {
                name: gate_counts[gate].get(name, 0)
                for name in ("pass", "fail", "pending")
            }
            for gate in PREFREEZE_GATES
        },
        "breadth": breadth,
    }


def freeze_registry(
    registry: Mapping[str, Any],
    qualification: Mapping[str, Any],
    *,
    release_id: str,
    frozen_at: str,
    store: Optional[ContentAddressedStore] = None,
) -> Dict[str, Any]:
    """Create final release registry only after every pre-freeze gate passes."""
    if not isinstance(release_id, str) or not release_id.strip():
        raise ValueError("release_id must not be empty")
    _parse_timestamp(frozen_at, "frozen_at")
    report = evaluate_qualification(registry, qualification, store=store)
    if not report["freeze_ready"]:
        raise ValueError(
            "release registry cannot freeze: " + report["status"]
        )
    candidates, upstreams = _candidate_rows(registry)
    claims = []
    for candidate, record in zip(candidates, qualification["records"]):
        upstream = upstreams[candidate["upstream"]]
        base = {
            "claim_id": candidate["claim_id"],
            "candidate_digest": record["candidate_digest"],
            "paper_id": upstream.get("paper"),
            "paper_title": None,
            "named_upstream": candidate["target_type"] == "upstream_claim",
            "method_family": candidate["family"],
            "model_family": candidate["model_family"],
            "stratum": record["stratum"],
            "task": record["comparison"]["task"],
            "metric": record["comparison"]["metric"],
            "evaluation_set": record["comparison"]["evaluation_set"],
            "resource_budget": record["comparison"]["resource_budget"],
            "compute_tier": record["compute_tier"],
            "claim_locator": candidate["claim_locator"],
            "external_validation": "not obtained",
        }
        if record["disposition"] == "eligible":
            claim = ClaimRecord.from_dict(
                record["gates"]["claim_record"]["evidence"]
            )
            base.update({
                "disposition": "eligible",
                "claim_locator": dict(claim.claim_locator),
                "claim_record_digest": record["claim_record_digest"],
                "audit_spec_digest": record["audit_spec_digest"],
            })
        else:
            base.update({
                "disposition": "excluded",
                "exclusion_reason": record["exclusion_reason"],
                "claim_record_digest": None,
                "audit_spec_digest": None,
            })
        claims.append(base)
    frozen = {
        "artifact": RELEASE_ARTIFACT,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "frozen",
        "release_id": release_id,
        "frozen_at": frozen_at,
        "outcome_blind": True,
        "publication_scope": "claim-level evidence only",
        "paper_verdicts": "not computed",
        "external_validation": "not obtained",
        "candidate_registry_digest": digest_json(registry),
        "qualification_digest": digest_json(qualification),
        "breadth": report["breadth"],
        "claims": claims,
    }
    _require_outcome_blind(frozen, "frozen release registry")
    return frozen


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(payload: Mapping[str, Any], path: Optional[Path]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(text, end="")
    else:
        path.write_text(text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run scaffold, report, or freeze operation."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    scaffold = commands.add_parser("scaffold")
    scaffold.add_argument("registry", type=Path)
    scaffold.add_argument("--out", type=Path)

    report = commands.add_parser("report")
    report.add_argument("registry", type=Path)
    report.add_argument("qualification", type=Path)
    report.add_argument("--cas", required=True, type=Path)
    report.add_argument("--out", type=Path)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("registry", type=Path)
    freeze.add_argument("qualification", type=Path)
    freeze.add_argument("--cas", required=True, type=Path)
    freeze.add_argument("--release-id", required=True)
    freeze.add_argument("--frozen-at", required=True)
    freeze.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    registry = _load(args.registry)
    if args.command == "scaffold":
        payload = scaffold_qualification(registry)
    elif args.command == "report":
        payload = evaluate_qualification(
            registry,
            _load(args.qualification),
            store=ContentAddressedStore(str(args.cas)),
        )
    else:
        payload = freeze_registry(
            registry,
            _load(args.qualification),
            release_id=args.release_id,
            frozen_at=args.frozen_at,
            store=ContentAddressedStore(str(args.cas)),
        )
    _write(payload, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
