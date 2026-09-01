"""Outcome-blind candidate qualification and release-freeze regressions."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from stresskit.audit_compile import freeze_audit_spec
from stresskit.audit_models import AgentOpinion, ClaimRecord, SourceBundle
from stresskit.audit_profiles import reducer_digest
from stresskit.agent_runner import _complete_digest_closure
from stresskit.integrity import ContentAddressedStore, digest_json, sha256_bytes


REPO_ROOT = Path(__file__).parents[1]
PATH = REPO_ROOT / "benchmark" / "qualify_candidates.py"
SPEC = importlib.util.spec_from_file_location("qualify_candidates", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


FAMILIES = (
    "cot_sentence_resampling",
    "probe_direction_geometry",
    "contrastive_activation_steering",
    "representation_lens",
    "causal_tracing",
    "automated_circuit_discovery",
)


def _evaluation_manifest(evaluation_id, axis_ids):
    payload = {"evaluation_id": evaluation_id, "axis_ids": axis_ids}
    return {**payload, "manifest_digest": digest_json(payload)}


def _evaluation_axis_id(axis, value):
    return digest_json({"axis": axis, "value": value})


def _candidate_registry(count=20):
    commit = "a" * 40
    entries = []
    for index in range(count):
        entries.append({
            "claim_id": f"claim-{index:02d}",
            "target_type": "upstream_claim",
            "family": FAMILIES[index % len(FAMILIES)],
            "upstream": "synthetic",
            "model_family": f"model-{index % 3}",
            "model": f"model-{index % 3}",
            "task": f"task-{index}",
            "statement_to_extract": "Registered synthetic statement.",
            "claim_locator": "paper section 1",
            "entrypoint": "run.py",
            "finding_type": "categorical",
            "perturbation_axes": ["seed"],
            "null": "label permutation",
            "compute_tier": "single_gpu_small",
            "eligibility": "candidate_needs_execution_smoke",
        })
    return {
        "schema_version": "0.1",
        "status": "candidate_frame_not_frozen",
        "outcome_blind": True,
        "upstreams": {
            "synthetic": {
                "repository": "https://example.invalid/synthetic",
                "paper": "https://example.invalid/paper",
                "commit": commit,
                "source_license": "MIT",
            }
        },
        "entries": entries,
    }


def _pass_gate(evidence):
    return {
        "status": "pass",
        "evidence": evidence,
        "evidence_digest": digest_json(evidence),
        "note": "synthetic qualification fixture",
    }


def _claim_artifacts(candidate, all_claim_ids):
    claim_id = candidate["claim_id"]
    paper_text = "Registered synthetic statement."
    paper_digest = sha256_bytes(paper_text.encode("utf-8"))
    repository_digest = digest_json({"repository": claim_id})
    paper_license_digest = digest_json({"license": "CC-BY-4.0", "claim": claim_id})
    code_license_digest = digest_json({"license": "MIT", "claim": claim_id})
    source = SourceBundle(
        bundle_id=f"source-{claim_id}",
        documents=[
            {
                "document_id": "paper",
                "locator": "paper.txt",
                "source_digest": paper_digest,
                "extracted_text_digest": paper_digest,
                "license": {
                    "status": "verified_compatible",
                    "identifier": "CC-BY-4.0",
                    "evidence_digest": paper_license_digest,
                },
            },
            {
                "document_id": "repository",
                "locator": "repository",
                "source_digest": repository_digest,
                "license": {
                    "status": "verified_compatible",
                    "identifier": "MIT",
                    "evidence_digest": code_license_digest,
                },
            },
        ],
        created_at="2026-09-01T00:00:00+00:00",
    )
    anchor = {
        "document_id": "paper",
        "locator": "bytes:0-10",
        "start": 0,
        "end": len(paper_text.encode("utf-8")),
        "quote_digest": paper_digest,
        "source_digest": paper_digest,
        "text_digest": paper_digest,
    }
    opinions = []
    for index, (role, provider, family) in enumerate((
        ("extractor", "provider-a", "family-a"),
        ("extractor", "provider-b", "family-b"),
        ("critic", "provider-c", "family-c"),
    )):
        opinions.append(AgentOpinion(
            opinion_id=f"{claim_id}-opinion-{index}",
            role=role,
            provider=provider,
            model=f"model-{index}",
            model_family=family,
            source_bundle_digest=source.digest,
            model_digest=digest_json({"model": index, "claim": claim_id}),
            prompt_digest=digest_json({"prompt": index, "claim": claim_id}),
            request_digest=digest_json({"request": index, "claim": claim_id}),
            statement="Registered synthetic statement.",
            evidence_anchors=[anchor],
            supported=True,
        ))
    dependency_digest = digest_json({"dependencies": claim_id})
    build_digest = digest_json({"build": claim_id})
    claim = ClaimRecord(
        claim_id=claim_id,
        statement="Registered synthetic statement.",
        source_bundle_digest=source.digest,
        source_digest=paper_digest,
        claim_locator=anchor,
        finding_type="categorical",
        profile_id="categorical_v1",
        reducer={
            "name": "categorical",
            "version": "1",
            "implementation_digest": reducer_digest("categorical"),
            "config": {"classes": ["yes", "no"]},
        },
        code_map={
            "repository_digest": repository_digest,
            "revision": "a" * 40,
            "entrypoints": ["run.py"],
            "dependency_manifest_digest": dependency_digest,
            "build_recipe_digest": build_digest,
        },
        controls={
            "positive": {"control_id": "known-truth", "expected": "yes"},
            "negative": {"control_id": "permuted", "expected": "no"},
        },
        task={"expected": "yes", "utility_required": False},
        agent_opinion_digests=[opinion.digest for opinion in opinions],
        metadata={
            "source_text_verification": {
                "status": "verified",
                "document_digests": {"paper": paper_digest},
            }
        },
    )
    design = {
        "held_out_axes": ["dataset", "unit"],
        "evaluation_manifests": {
            "primary": _evaluation_manifest(
                f"primary-{claim_id}",
                {
                    "dataset": [_evaluation_axis_id(
                        "dataset", f"primary-dataset-{claim_id}"
                    )],
                    "model": [_evaluation_axis_id("model", "synthetic-model")],
                    "unit": [_evaluation_axis_id(
                        "unit", f"primary-units-{claim_id}"
                    )],
                },
            ),
            "generalization": _evaluation_manifest(
                f"generalization-{claim_id}",
                {
                    "dataset": [_evaluation_axis_id(
                        "dataset", f"held-out-dataset-{claim_id}"
                    )],
                    "model": [_evaluation_axis_id("model", "synthetic-model")],
                    "unit": [_evaluation_axis_id(
                        "unit", f"held-out-units-{claim_id}"
                    )],
                },
            ),
        },
        "joint_distribution": [{
            "specification_id": "spec-1",
            "values": {"seed_policy": "registered"},
            "weight": 1.0,
        }],
        "runs_per_partition": {
            "primary": 1,
            "positive_control": 1,
            "negative_control": 1,
            "generalization": 1,
        },
        "cohorts": ["final", "replication"],
        "seed": 11,
        "independent_unit": "fresh execution",
        "hardware_class": "gpu-test",
    }
    family = {
        "family_id": "release-test",
        "method": "holm-bonferroni",
        "alpha": 0.05,
        "member_claim_ids": list(all_claim_ids),
        "release_manifest_digest": digest_json(sorted(all_claim_ids)),
    }
    spec = freeze_audit_spec(
        claim,
        design,
        audit_id=f"audit-{claim_id}",
        frozen_at="2026-09-01T00:00:00+00:00",
        multiplicity_family=family,
        reproducibility={"level": "bitwise", "hardware_class": "gpu-test"},
    )
    license_closure = {
        "artifact": "stresskit_license_closure",
        "schema_version": "1.0",
        "claim_id": claim_id,
        "claim_record_digest": claim.digest,
        "dependency_manifest_digest": dependency_digest,
        "complete": True,
        "determined_without_outcomes": True,
        "items": [
            {
                "kind": "paper",
                "identifier": "synthetic-paper",
                "revision": paper_digest,
                "license": "CC-BY-4.0",
                "status": "verified_compatible",
                "evidence_digest": paper_license_digest,
                "document_id": "paper",
            },
            {
                "kind": "source_code",
                "identifier": "synthetic-repository",
                "revision": "a" * 40,
                "license": "MIT",
                "status": "verified_compatible",
                "evidence_digest": code_license_digest,
                "document_id": "repository",
            },
        ],
    }
    smoke = {
        "artifact": "stresskit_execution_smoke_evidence",
        "schema_version": "1.0",
        "claim_id": claim_id,
        "claim_record_digest": claim.digest,
        "upstream": "synthetic",
        "upstream_commit": "a" * 40,
        "status": "pass",
        "not_claim_reproduction": True,
        "not_benchmark_outcome": True,
        "claim_map_exercised": True,
        "raw_artifact_digest": digest_json({"smoke": claim_id}),
        "execution_isolation": dict(MODULE._ISOLATION),
    }
    review = {
        "artifact": "stresskit_protocol_review",
        "schema_version": "1.0",
        "claim_id": claim_id,
        "audit_spec_digest": spec.digest,
        "status": "approved",
        "issues": [],
        "review_mode": "external",
        "reviewer_id": "synthetic-reviewer",
        "reviewed_at": "2026-09-01T01:00:00+00:00",
    }
    resources = {
        "gpu_count": 1,
        "cpu_count": 4,
        "wall_time_seconds": 600,
        "storage_bytes": 1_000_000,
    }
    estimate = {
        "artifact": "stresskit_resource_estimate",
        "schema_version": "1.0",
        "claim_id": claim_id,
        "audit_spec_digest": spec.digest,
        "estimated_without_outcomes": True,
        "compute_tier_label": "single_gpu_small",
        "compute_tier": 1,
        "hardware_class": "gpu-test",
        "resources": resources,
    }
    panel = {
        "artifact": "stresskit_agent_panel",
        "schema_version": "1.0",
        "claim_id": claim_id,
        "source_bundle_digest": source.digest,
        "opinions": [opinion.to_dict() for opinion in opinions],
    }
    source_evidence = {
        "artifact": "stresskit_source_bundle_evidence",
        "schema_version": "1.0",
        "claim_id": claim_id,
        "source_bundle": source.to_dict(),
        "source_texts": {"paper": paper_text},
    }
    return (
        source_evidence, panel, claim, license_closure, smoke, spec, review,
        estimate,
    )


def _agent_panel_abstention_evidence(candidate, store):
    def unique_references(references):
        return [
            {reference.digest: reference for reference in references}[digest]
            for digest in sorted({reference.digest for reference in references})
        ]

    paper_ref = store.put_bytes(
        candidate["statement_to_extract"].encode("utf-8"), role="source:paper"
    )
    license_ref = store.put_json(
        {"artifact": "license", "identifier": "synthetic"},
        role="license_evidence",
    )
    source = SourceBundle(
        bundle_id=f"source-{candidate['claim_id']}",
        documents=[{
            "document_id": "paper",
            "locator": "paper.txt",
            "source_digest": paper_ref.digest,
            "extracted_text_digest": paper_ref.digest,
            "license": {
                "status": "verified_compatible",
                "identifier": "synthetic",
                "evidence_digest": license_ref.digest,
            },
        }],
        created_at="2026-09-01T00:00:00+00:00",
        metadata={
            "candidate_id": candidate["claim_id"],
            "outcome_blind": True,
        },
    )
    source_ref = store.put_json(source.to_dict(), role="source_bundle")
    source_references = unique_references([paper_ref, license_ref, source_ref])
    source_closure_ref = store.put_json(
        [reference.to_dict() for reference in source_references],
        role="source_closure_manifest",
    )

    identities = (
        ("extractor", "provider-a", "model-a", "family-a"),
        ("extractor", "provider-b", "model-b", "family-b"),
        ("critic", "provider-c", "model-c", "family-c"),
    )
    claim_query_ref = store.put_bytes(
        candidate["statement_to_extract"].encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        role="agent_claim_query",
    )
    requests = [
        {
            "opinion_id": f"{candidate['claim_id']}-opinion-{index}",
            "role": role,
            "model_request_id": f"author/request-{index}",
            "model_family": family,
            "provider_endpoint": f"provider-{index}/zdr",
            "provider_name": provider,
            "catalog": {"canonical_slug": model},
            "request_parameters": {
                "max_tokens": 512,
                "seed": 100 + index,
                "temperature": 0,
            },
        }
        for index, (role, provider, model, family) in enumerate(identities)
    ]
    panel_plan = {
        "artifact": "stresskit_agent_panel_plan",
        "schema_version": "1.0",
        "candidate_id": candidate["claim_id"],
        "outcome_blind": True,
        "catalog_observed_at": "2026-09-01T00:00:00+00:00",
        "status": "frozen",
        "transport": "openrouter",
        "claim_query_digest": claim_query_ref.digest,
        "constraints": {
            "account_prompt_logging": "must_not_be_opted_in",
            "allow_fallbacks": False,
            "data_collection": "deny",
            "no_plugins_or_tools": True,
            "require_parameters": True,
            "router_pipeline": "must_be_empty",
            "selected_attempt": 1,
            "zdr": True,
        },
        "requests": requests,
    }
    panel_ref = store.put_json(panel_plan, role="agent_panel_plan")

    opinions = []
    opinion_closure_refs = []
    all_references = [
        *source_references, source_closure_ref, claim_query_ref, panel_ref,
    ]
    accepted_raw_refs = []
    for index, ((role, provider, model, family), request) in enumerate(
        zip(identities, requests)
    ):
        route_ref = store.put_json(
            {
                "artifact": "stresskit_openrouter_route_binding",
                "schema_version": "1.0",
                "candidate_id": candidate["claim_id"],
                "opinion_id": request["opinion_id"],
                "role": role,
                "provider_name": provider,
                "provider_endpoint": request["provider_endpoint"],
                "model_family": family,
                "model_request_id": request["model_request_id"],
                "canonical_slug": model,
                "claim_query_digest": claim_query_ref.digest,
                "panel_plan_digest": panel_ref.digest,
                "panel_status": panel_plan["status"],
                "request_parameters": request["request_parameters"],
                "routing_constraints": panel_plan["constraints"],
            },
            role="agent_route_binding",
        )
        prompt_ref = store.put_json(
            {
                "artifact": "stresskit_agent_prompt",
                "schema_version": "1.0",
                "role": role,
                "claim_query_digest": claim_query_ref.digest,
                "source_bundle_digest": source.digest,
                "route_binding_digest": route_ref.digest,
            },
            role="agent_prompt",
        )
        raw_ref = store.put_json(
            {"response_id": f"response-{index}"}, role="agent_raw_response"
        )
        model_ref = store.put_json(
            {
                "artifact": "stresskit_agent_model_descriptor",
                "schema_version": "1.0",
                "provider": provider,
                "model": model,
                "family": family,
                "transport": "openrouter",
                "requested_model": request["model_request_id"],
                "requested_provider_endpoint": request["provider_endpoint"],
                "panel_plan_digest": panel_ref.digest,
                "route_binding_digest": route_ref.digest,
                "response_id": f"response-{index}",
                "raw_response_digest": raw_ref.digest,
            },
            role="agent_model",
        )
        request_ref = store.put_json(
            {
                "artifact": "stresskit_agent_request_receipt",
                "schema_version": "1.0",
                "transport": "openrouter",
                "method": "POST",
                "panel_plan_digest": panel_ref.digest,
                "route_binding_digest": route_ref.digest,
                "prompt_digest": prompt_ref.digest,
                "authorization": {
                    "scheme": "Bearer",
                    "source": "environment:OPENROUTER_API_KEY",
                    "serialized": False,
                },
                "body": {
                    "model": request["model_request_id"],
                    "provider": {
                        **MODULE._OPENROUTER_REQUEST_POLICY,
                        "only": [request["provider_endpoint"]],
                    },
                },
                "response": {
                    "response_id": f"response-{index}",
                    "raw_digest": raw_ref.digest,
                },
            },
            role="agent_request",
        )
        opinion = AgentOpinion(
            opinion_id=request["opinion_id"],
            role=role,
            provider=provider,
            model=model,
            model_family=family,
            source_bundle_digest=source.digest,
            model_digest=model_ref.digest,
            prompt_digest=prompt_ref.digest,
            request_digest=request_ref.digest,
            statement=candidate["statement_to_extract"],
            evidence_anchors=[{
                "document_id": "paper",
                "locator": "paper.txt#bytes=0-end",
                "start": 0,
                "end": len(candidate["statement_to_extract"].encode("utf-8")),
                "quote_digest": paper_ref.digest,
                "source_digest": paper_ref.digest,
                "text_digest": paper_ref.digest,
            }],
            supported=False,
            issues=["frozen wording is not explicit in source"],
        )
        opinion_ref = store.put_json(opinion.to_dict(), role="agent_opinion")
        opinion_references = unique_references([
            *source_references,
            panel_ref,
            claim_query_ref,
            route_ref,
            prompt_ref,
            raw_ref,
            model_ref,
            request_ref,
            opinion_ref,
        ])
        closure_ref = store.put_json(
            [reference.to_dict() for reference in opinion_references],
            role="agent_opinion_closure_manifest",
        )
        opinions.append(opinion)
        opinion_closure_refs.append(closure_ref)
        accepted_raw_refs.append(raw_ref)
        all_references.extend([*opinion_references, closure_ref])

    decision = {
        "artifact": "stresskit_claim_candidates",
        "schema_version": "1.0",
        "source_bundle_digest": source.digest,
        "publication_state": "abstain",
        "candidates": [],
        "problems": [
            f"opinion {opinion.opinion_id!r} marks claim unsupported"
            for opinion in opinions
        ],
    }
    decision_ref = store.put_json(decision, role="claim_discovery_decision")
    rejected_route_ref = store.put_json(
        {"panel_plan_digest": panel_ref.digest, "route": "rejected"},
        role="agent_route_binding",
    )
    rejected_raw_ref = store.put_json(
        {"response_id": "rejected-response"}, role="agent_raw_response"
    )
    attempt = {
        "artifact": "stresskit_agent_attempt_record",
        "schema_version": "1.0",
        "candidate_id": candidate["claim_id"],
        "opinion_id": opinions[0].opinion_id,
        "role": "extractor",
        "attempt": 1,
        "observed_at": "2026-09-01T00:01:00+00:00",
        "outcome_blind": True,
        "status": "rejected_before_opinion",
        "source_bundle_digest": source.digest,
        "panel_plan_digest": panel_ref.digest,
        "route_binding_digest": rejected_route_ref.digest,
        "prompt_digest": opinions[0].prompt_digest,
        "raw_response_digest": rejected_raw_ref.digest,
        "completion_content_inspected": False,
        "retry_performed_automatically": False,
    }
    attempt_ref = store.put_json(attempt, role="agent_attempt_record")
    routes = [
        {
            "opinion_id": opinion.opinion_id,
            "role": opinion.role,
            "created_at": f"2026-09-01T00:0{index + 2}:00+00:00",
            "request_receipt_digest": opinion.request_digest,
            "model_descriptor_digest": opinion.model_digest,
            "raw_response_digest": accepted_raw_refs[index].digest,
            "response_id": f"response-{index}",
            "requested_model": requests[index]["model_request_id"],
            "requested_provider_endpoint": requests[index]["provider_endpoint"],
            "selected_provider": opinion.provider,
            "selected_canonical_model": opinion.model,
            "strategy": "direct",
            "router_attempt": 1,
            "endpoint_inventory_count": index + 1,
            "available_endpoint_count": 1,
            "pipeline": None,
            "accepted": True,
        }
        for index, opinion in enumerate(opinions)
    ]
    attestation = {
        "artifact": "stresskit_openrouter_panel_attestation",
        "schema_version": "1.0",
        "candidate_id": candidate["claim_id"],
        "source_bundle_digest": source.digest,
        "panel_plan_digest": panel_ref.digest,
        "catalog_observed_at": "2026-09-01T00:00:00+00:00",
        "attestation_created_at": "2026-09-01T00:05:00+00:00",
        "status": "verified_from_accepted_responses",
        "authenticated": True,
        "credential_source": "environment:OPENROUTER_API_KEY",
        "credential_serialized": False,
        "request_policy": dict(MODULE._OPENROUTER_REQUEST_POLICY),
        "routes": routes,
    }
    attestation_ref = store.put_json(
        attestation, role="agent_panel_attestation"
    )
    all_references.extend([
        decision_ref,
        rejected_route_ref,
        rejected_raw_ref,
        attempt_ref,
        attestation_ref,
    ])
    panel_references = unique_references(all_references)
    panel_closure = [reference.to_dict() for reference in panel_references]
    return {
        "artifact": "stresskit_agent_panel_abstention",
        "schema_version": "1.0",
        "claim_id": candidate["claim_id"],
        "outcome_blind": True,
        "publication_state": "abstain",
        "reason": (
            "frozen_candidate_wording_not_explicitly_supported_by_pinned_sources"
        ),
        "prompt_injection_detected": False,
        "source_bundle_digest": source.digest,
        "source_bundle": source.to_dict(),
        "source_closure_digest": source_closure_ref.digest,
        "panel_plan_digest": panel_ref.digest,
        "panel_plan": panel_plan,
        "agent_opinion_digests": [opinion.digest for opinion in opinions],
        "opinions": [opinion.to_dict() for opinion in opinions],
        "opinion_closure_digests": [
            {
                "opinion_digest": opinion.digest,
                "closure_digest": closure_ref.digest,
            }
            for opinion, closure_ref in zip(opinions, opinion_closure_refs)
        ],
        "discovery_decision_digest": decision_ref.digest,
        "discovery_decision": decision,
        "rejected_attempt_record_digest": attempt_ref.digest,
        "rejected_attempt_record": attempt,
        "provider_attestation_digest": attestation_ref.digest,
        "provider_attestation": attestation,
        "panel_closure_digest": digest_json(panel_closure),
        "panel_closure_roots": [
            source_closure_ref.digest,
            *(reference.digest for reference in opinion_closure_refs),
            decision_ref.digest,
            attempt_ref.digest,
            attestation_ref.digest,
        ],
        "panel_closure": panel_closure,
    }


def _agent_panel_execution_abstention_evidence(candidate, store):
    base = _agent_panel_abstention_evidence(candidate, store)
    source = SourceBundle.from_dict(base["source_bundle"])
    panel = base["panel_plan"]
    opinions = [AgentOpinion.from_dict(row) for row in base["opinions"]]
    accepted = opinions[0]
    rejected_request = panel["requests"][1]
    critic_request = panel["requests"][2]
    rejected_descriptor = store.get_json(opinions[1].model_digest)
    route_digest = rejected_descriptor["route_binding_digest"]

    invalid_quote = "This quote is absent from every declared source byte."
    invalid_quote_ref = store.put_bytes(
        invalid_quote.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        role="agent_evidence_quote_rejected",
    )
    response_id = "rejected-response-1"
    raw_response = {
        "id": response_id,
        "object": "chat.completion",
        "model": rejected_request["model_request_id"],
        "provider": rejected_request["provider_name"],
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "statement": candidate["statement_to_extract"],
                    "supported": True,
                    "prompt_injection_detected": False,
                    "issues": [],
                    "evidence_quotes": [{
                        "document_id": "paper",
                        "quote": invalid_quote,
                    }],
                }),
            },
        }],
    }
    raw_ref = store.put_json(raw_response, role="agent_raw_response")
    attempt = {
        "artifact": "stresskit_agent_attempt_record",
        "schema_version": "1.0",
        "candidate_id": candidate["claim_id"],
        "opinion_id": rejected_request["opinion_id"],
        "role": "extractor",
        "attempt": 1,
        "observed_at": "2026-09-01T00:06:00+00:00",
        "status": "rejected_before_opinion",
        "source_bundle_digest": source.digest,
        "panel_plan_digest": base["panel_plan_digest"],
        "route_binding_digest": route_digest,
        "prompt_digest": opinions[1].prompt_digest,
        "raw_response_digest": raw_ref.digest,
        "response_id": response_id,
        "completion_content_inspected": True,
        "retry_performed": False,
        "critic_called": False,
        "reason": "completion evidence quote absent from declared source bytes",
        "evidence_quote_checks": [{
            "document_id": "paper",
            "quote_digest": invalid_quote_ref.digest,
            "present_in_declared_source_bytes": False,
        }],
        "safe_route_metadata": {
            "available_endpoints": [{
                "model": rejected_request["catalog"]["canonical_slug"],
                "provider": rejected_request["provider_name"],
                "selected": True,
            }],
            "pipeline": None,
            "requested_model": rejected_request["model_request_id"],
            "response_model": rejected_request["model_request_id"],
            "selected_attempt": 1,
            "strategy": "direct",
        },
    }
    attempt_ref = store.put_json(attempt, role="agent_attempt_record")
    accepted_closure_digest = base["opinion_closure_digests"][0][
        "closure_digest"
    ]
    execution = {
        "artifact": "stresskit_agent_panel_execution",
        "schema_version": "1.0",
        "candidate_id": candidate["claim_id"],
        "source_bundle_digest": source.digest,
        "panel_plan_digest": base["panel_plan_digest"],
        "outcome_blind": True,
        "status": "abstain",
        "publication_state": "abstain",
        "retry_policy": "no_retry",
        "complete_slots": True,
        "slots": [
            {
                "opinion_id": accepted.opinion_id,
                "role": "extractor",
                "status": "accepted",
                "opinion_digest": accepted.digest,
                "closure_digest": accepted_closure_digest,
            },
            {
                "opinion_id": rejected_request["opinion_id"],
                "role": "extractor",
                "status": "rejected_invalid_evidence",
                "attempt_digest": attempt_ref.digest,
                "raw_response_digest": raw_ref.digest,
            },
            {
                "opinion_id": critic_request["opinion_id"],
                "role": "critic",
                "status": "not_run_dependency_failure",
                "depends_on": [
                    accepted.opinion_id,
                    rejected_request["opinion_id"],
                ],
                "reason": "critic requires two accepted extractor opinions",
            },
        ],
    }
    execution_ref = store.put_json(execution, role="agent_panel_execution")
    decision = {
        "artifact": "stresskit_claim_candidates",
        "schema_version": "1.0",
        "source_bundle_digest": source.digest,
        "panel_execution_digest": execution_ref.digest,
        "publication_state": "abstain",
        "candidates": [],
        "problems": [
            f"{rejected_request['opinion_id']} supplied invalid byte anchors",
            f"{critic_request['opinion_id']} not run after extractor rejection",
        ],
    }
    decision_ref = store.put_json(decision, role="claim_discovery_decision")
    roots = [base["source_closure_digest"], decision_ref.digest]
    closure = _complete_digest_closure(store, roots, [])
    panel_closure = [reference.to_dict() for reference in closure]
    return {
        "artifact": "stresskit_agent_panel_execution_abstention",
        "schema_version": "1.0",
        "claim_id": candidate["claim_id"],
        "outcome_blind": True,
        "publication_state": "abstain",
        "reason": "invalid_evidence_anchor",
        "prompt_injection_detected": False,
        "source_bundle_digest": source.digest,
        "source_bundle": source.to_dict(),
        "source_closure_digest": base["source_closure_digest"],
        "panel_plan_digest": base["panel_plan_digest"],
        "panel_plan": panel,
        "accepted_opinion_digest": accepted.digest,
        "accepted_opinion": accepted.to_dict(),
        "accepted_opinion_closure_digest": accepted_closure_digest,
        "rejected_attempt_record_digest": attempt_ref.digest,
        "rejected_attempt_record": attempt,
        "panel_execution_digest": execution_ref.digest,
        "panel_execution": execution,
        "discovery_decision_digest": decision_ref.digest,
        "discovery_decision": decision,
        "panel_closure_digest": digest_json(panel_closure),
        "panel_closure_roots": roots,
        "panel_closure": panel_closure,
    }


def _plural_agent_panel_execution_abstention_evidence(candidate, store):
    base = _agent_panel_abstention_evidence(candidate, store)
    source = SourceBundle.from_dict(base["source_bundle"])
    panel = base["panel_plan"]
    opinions = [AgentOpinion.from_dict(row) for row in base["opinions"]]
    extractor_requests = [
        row for row in panel["requests"] if row["role"] == "extractor"
    ]
    critic_request = next(
        row for row in panel["requests"] if row["role"] == "critic"
    )

    present_quote = candidate["statement_to_extract"]
    absent_quote = "This exact quote is absent from the declared source."
    absent_quote_ref = store.put_bytes(
        absent_quote.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        role="agent_evidence_quote_rejected",
    )
    first_request = extractor_requests[0]
    first_descriptor = store.get_json(opinions[0].model_digest)
    first_response_id = "rejected-mixed-quote-response"
    first_raw = {
        "id": first_response_id,
        "object": "chat.completion",
        "model": first_request["model_request_id"],
        "provider": first_request["provider_name"],
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "native_finish_reason": "completed",
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "statement": candidate["statement_to_extract"],
                    "supported": True,
                    "prompt_injection_detected": False,
                    "issues": [],
                    "evidence_quotes": [
                        {"document_id": "paper", "quote": present_quote},
                        {"document_id": "paper", "quote": absent_quote},
                    ],
                }),
            },
        }],
    }
    first_raw_ref = store.put_json(first_raw, role="agent_raw_response")
    first_attempt = {
        "artifact": "stresskit_agent_attempt_record",
        "schema_version": "1.0",
        "candidate_id": candidate["claim_id"],
        "opinion_id": first_request["opinion_id"],
        "role": "extractor",
        "attempt": 1,
        "observed_at": "2026-09-01T00:06:00+00:00",
        "status": "rejected_before_opinion",
        "source_bundle_digest": source.digest,
        "panel_plan_digest": base["panel_plan_digest"],
        "route_binding_digest": first_descriptor["route_binding_digest"],
        "prompt_digest": opinions[0].prompt_digest,
        "raw_response_digest": first_raw_ref.digest,
        "response_id": first_response_id,
        "completion_content_inspected": True,
        "completion_content_human_inspected": False,
        "retry_performed": False,
        "critic_called": False,
        "reason": "one evidence quote is absent from declared source bytes",
        "evidence_quote_checks": [
            {
                "document_id": "paper",
                "quote_digest": sha256_bytes(present_quote.encode("utf-8")),
                "present_in_declared_source_bytes": True,
            },
            {
                "document_id": "paper",
                "quote_digest": absent_quote_ref.digest,
                "present_in_declared_source_bytes": False,
            },
        ],
        "safe_route_metadata": {
            "available_endpoints": [{
                "model": first_request["catalog"]["canonical_slug"],
                "provider": first_request["provider_name"],
                "selected": True,
            }],
            "pipeline": None,
            "requested_model": first_request["model_request_id"],
            "response_model": first_request["model_request_id"],
            "selected_attempt": 1,
            "strategy": "direct",
        },
    }
    first_attempt_ref = store.put_json(
        first_attempt, role="agent_attempt_record"
    )

    second_request = extractor_requests[1]
    second_descriptor = store.get_json(opinions[1].model_digest)
    second_response_id = "rejected-length-response"
    second_raw = {
        "id": second_response_id,
        "object": "chat.completion",
        "model": second_request["model_request_id"],
        "provider": second_request["provider_name"],
        "choices": [{
            "index": 0,
            "finish_reason": "length",
            "native_finish_reason": "length",
            "message": {"role": "assistant", "content": None},
        }],
    }
    second_raw_ref = store.put_json(second_raw, role="agent_raw_response")
    second_attempt = {
        "artifact": "stresskit_agent_attempt_record",
        "schema_version": "1.0",
        "candidate_id": candidate["claim_id"],
        "opinion_id": second_request["opinion_id"],
        "role": "extractor",
        "attempt": 1,
        "observed_at": "2026-09-01T00:07:00+00:00",
        "status": "rejected_before_opinion",
        "source_bundle_digest": source.digest,
        "panel_plan_digest": base["panel_plan_digest"],
        "route_binding_digest": second_descriptor["route_binding_digest"],
        "prompt_digest": opinions[1].prompt_digest,
        "raw_response_digest": second_raw_ref.digest,
        "response_id": second_response_id,
        "completion_content_inspected": False,
        "completion_content_human_inspected": False,
        "retry_performed": False,
        "critic_called": False,
        "finish_reason": "length",
        "native_finish_reason": "length",
        "reason": "completion reached frozen max-token limit",
        "evidence_quote_checks": [],
        "safe_route_metadata": {
            "available_endpoints": [{
                "model": second_request["catalog"]["canonical_slug"],
                "provider": second_request["provider_name"],
                "selected": True,
            }],
            "pipeline": None,
            "requested_model": second_request["model_request_id"],
            "response_model": second_request["model_request_id"],
            "selected_attempt": 1,
            "strategy": "direct",
        },
    }
    second_attempt_ref = store.put_json(
        second_attempt, role="agent_attempt_record"
    )

    execution = {
        "artifact": "stresskit_agent_panel_execution",
        "schema_version": "1.0",
        "candidate_id": candidate["claim_id"],
        "source_bundle_digest": source.digest,
        "panel_plan_digest": base["panel_plan_digest"],
        "outcome_blind": True,
        "status": "abstain",
        "publication_state": "abstain",
        "retry_policy": "no_retry",
        "complete_slots": True,
        "slots": [
            {
                "opinion_id": first_request["opinion_id"],
                "role": "extractor",
                "status": "rejected_invalid_evidence",
                "attempt_digest": first_attempt_ref.digest,
                "raw_response_digest": first_raw_ref.digest,
            },
            {
                "opinion_id": second_request["opinion_id"],
                "role": "extractor",
                "status": "rejected_incomplete_completion",
                "attempt_digest": second_attempt_ref.digest,
                "raw_response_digest": second_raw_ref.digest,
            },
            {
                "opinion_id": critic_request["opinion_id"],
                "role": "critic",
                "status": "not_run_dependency_failure",
                "depends_on": [
                    first_request["opinion_id"], second_request["opinion_id"]
                ],
                "reason": "critic requires two accepted extractor opinions",
            },
        ],
    }
    execution_ref = store.put_json(execution, role="agent_panel_execution")
    decision = {
        "artifact": "stresskit_claim_candidates",
        "schema_version": "1.0",
        "source_bundle_digest": source.digest,
        "panel_execution_digest": execution_ref.digest,
        "publication_state": "abstain",
        "candidates": [],
        "problems": [
            f"{first_request['opinion_id']} supplied invalid byte anchors",
            f"{second_request['opinion_id']} did not finish cleanly",
            f"{critic_request['opinion_id']} not run after extractor rejection",
        ],
    }
    decision_ref = store.put_json(decision, role="claim_discovery_decision")
    roots = [base["source_closure_digest"], decision_ref.digest]
    closure = _complete_digest_closure(store, roots, [])
    panel_closure = [reference.to_dict() for reference in closure]
    return {
        "artifact": "stresskit_agent_panel_execution_abstention",
        "schema_version": "1.0",
        "claim_id": candidate["claim_id"],
        "outcome_blind": True,
        "publication_state": "abstain",
        "reason": "invalid_agent_outputs",
        "prompt_injection_detected": False,
        "source_bundle_digest": source.digest,
        "source_bundle": source.to_dict(),
        "source_closure_digest": base["source_closure_digest"],
        "panel_plan_digest": base["panel_plan_digest"],
        "panel_plan": panel,
        "accepted_opinions": [],
        "rejected_attempts": [
            {
                "attempt_record_digest": first_attempt_ref.digest,
                "attempt_record": first_attempt,
            },
            {
                "attempt_record_digest": second_attempt_ref.digest,
                "attempt_record": second_attempt,
            },
        ],
        "panel_execution_digest": execution_ref.digest,
        "panel_execution": execution,
        "discovery_decision_digest": decision_ref.digest,
        "discovery_decision": decision,
        "panel_closure_digest": digest_json(panel_closure),
        "panel_closure_roots": roots,
        "panel_closure": panel_closure,
    }


def _refresh_execution_abstention_digests(evidence):
    attempt = evidence["rejected_attempt_record"]
    evidence["rejected_attempt_record_digest"] = digest_json(attempt)
    execution = evidence["panel_execution"]
    rejected = next(
        row for row in execution["slots"]
        if row["status"] == "rejected_invalid_evidence"
    )
    rejected["attempt_digest"] = evidence["rejected_attempt_record_digest"]
    rejected["raw_response_digest"] = attempt["raw_response_digest"]
    evidence["panel_execution_digest"] = digest_json(execution)
    decision = evidence["discovery_decision"]
    decision["panel_execution_digest"] = evidence["panel_execution_digest"]
    evidence["discovery_decision_digest"] = digest_json(decision)


def _complete_qualification(registry):
    qualification = MODULE.scaffold_qualification(registry)
    claim_ids = [row["claim_id"] for row in registry["entries"]]
    for candidate, record in zip(registry["entries"], qualification["records"]):
        source_evidence, panel, claim, licenses, smoke, spec, review, estimate = \
            _claim_artifacts(candidate, claim_ids)
        evidence = {
            "source_bundle": source_evidence,
            "agent_panel": panel,
            "claim_record": claim.to_dict(),
            "license_closure": licenses,
            "execution_smoke": smoke,
            "audit_spec": spec.to_dict(),
            "protocol_review": review,
            "resource_estimate": estimate,
        }
        record["gates"] = {
            name: _pass_gate(evidence[name]) for name in MODULE.PREFREEZE_GATES
        }
        record["disposition"] = "eligible"
        record["claim_record_digest"] = claim.digest
        record["audit_spec_digest"] = spec.digest
        record["comparison"] = {
            "task": candidate["task"],
            "metric": "accuracy",
            "evaluation_set": "held-out synthetic set",
            "resource_budget": estimate["resources"],
        }
    return qualification


def test_current_registry_scaffold_keeps_every_blocker_visible():
    registry = MODULE._load(REPO_ROOT / "benchmark" / "registry.candidates.json")
    scaffold = MODULE.scaffold_qualification(registry)
    qualification = MODULE._load(
        REPO_ROOT / "benchmark" / "qualification.prefreeze.json"
    )
    store = ContentAddressedStore(str(REPO_ROOT / ".stresskit" / "cas"))
    report = MODULE.evaluate_qualification(
        registry, qualification, store=store
    )
    scaffold_pyvene = next(
        row for row in scaffold["records"]
        if row["claim_id"] == "pyvene_interchange_intervention_ioi"
    )
    qualified_pyvene = next(
        row for row in qualification["records"]
        if row["claim_id"] == "pyvene_interchange_intervention_ioi"
    )
    assert scaffold_pyvene["disposition"] == "pending"
    assert qualified_pyvene["disposition"] == "excluded"
    assert qualified_pyvene["gates"]["agent_panel"]["status"] == "fail"
    assert qualified_pyvene["gates"]["agent_panel"]["evidence"][
        "publication_state"
    ] == "abstain"
    attempt = MODULE._load(
        REPO_ROOT / "benchmark" / "intake"
        / "pyvene_interchange_intervention_ioi" / "opinions"
        / "extractor-a-attempt-1-rejected.json"
    )
    assert attempt["status"] == "rejected_before_opinion"
    assert attempt["completion_content_inspected"] is False
    assert digest_json(attempt) == qualified_pyvene["gates"]["agent_panel"][
        "evidence"
    ]["rejected_attempt_record_digest"]
    qualified_acdc = next(
        row for row in qualification["records"]
        if row["claim_id"] == "acdc_tracr_reverse"
    )
    acdc_evidence = qualified_acdc["gates"]["agent_panel"]["evidence"]
    assert qualified_acdc["disposition"] == "excluded"
    assert qualified_acdc["gates"]["agent_panel"]["status"] == "fail"
    assert acdc_evidence["publication_state"] == "abstain"
    assert acdc_evidence["accepted_opinions"] == []
    assert len(acdc_evidence["rejected_attempts"]) == 2
    assert acdc_evidence["panel_execution"]["slots"][2]["status"] == \
        "not_run_dependency_failure"
    assert report == MODULE._load(
        REPO_ROOT / "artifacts" / "benchmark"
        / "prefreeze-qualification-report-v1.json"
    )
    assert report["dispositions"] == {
        "eligible": 0,
        "excluded": 15,
        "pending": 53,
    }
    assert report["freeze_ready"] is False
    assert report["backward_extension_required"] is False
    assert report["gate_counts"]["agent_panel"]["fail"] == 3
    assert report["gate_counts"]["license_closure"]["fail"] == 12
    qualified_mechtomo = next(
        row for row in qualification["records"]
        if row["claim_id"] == "mechtomo_finite_effect_map_recovery"
    )
    assert qualified_mechtomo["disposition"] == "excluded"
    assert qualified_mechtomo["gates"]["agent_panel"]["evidence"][
        "artifact"
    ] == "stresskit_agent_panel_execution_abstention"
    with pytest.raises(ValueError, match="qualification_in_progress"):
        MODULE.freeze_registry(
            registry,
            qualification,
            release_id="v1",
            frozen_at="2026-09-01T00:00:00+00:00",
            store=store,
        )


def test_disposition_and_evidence_digest_cannot_be_forged():
    registry = _candidate_registry(1)
    qualification = MODULE.scaffold_qualification(registry)
    qualification["records"][0]["disposition"] = "eligible"
    with pytest.raises(ValueError, match="derived as pending"):
        MODULE.evaluate_qualification(registry, qualification)

    qualification = MODULE.scaffold_qualification(registry)
    gate = qualification["records"][0]["gates"]["execution_smoke"]
    gate.update({
        "status": "fail",
        "evidence": {"claim_id": "claim-00", "reason": "smoke failed"},
        "evidence_digest": digest_json({"different": True}),
    })
    qualification["records"][0]["disposition"] = "excluded"
    qualification["records"][0]["exclusion_reason"] = "smoke failed"
    with pytest.raises(ValueError, match="evidence digest mismatch"):
        MODULE.evaluate_qualification(registry, qualification)


def test_typed_agent_panel_abstention_accepts_self_contained_evidence(tmp_path):
    registry = _candidate_registry(1)
    qualification = MODULE.scaffold_qualification(registry)
    store = ContentAddressedStore(str(tmp_path / "cas"))
    evidence = _agent_panel_abstention_evidence(
        registry["entries"][0], store
    )
    record = qualification["records"][0]
    record["gates"]["agent_panel"] = _pass_gate(evidence)
    record["gates"]["agent_panel"]["status"] = "fail"
    record["disposition"] = "excluded"
    record["exclusion_reason"] = "agent panel abstained"

    report = MODULE.evaluate_qualification(
        registry, qualification, store=store
    )

    assert report["dispositions"]["excluded"] == 1
    assert report["gate_counts"]["agent_panel"]["fail"] == 1


def test_invalid_extractor_panel_abstains_with_complete_slot_accounting(tmp_path):
    registry = _candidate_registry(1)
    qualification = MODULE.scaffold_qualification(registry)
    store = ContentAddressedStore(str(tmp_path / "cas"))
    evidence = _agent_panel_execution_abstention_evidence(
        registry["entries"][0], store
    )
    record = qualification["records"][0]
    record["gates"]["agent_panel"] = _pass_gate(evidence)
    record["gates"]["agent_panel"]["status"] = "fail"
    record["disposition"] = "excluded"
    record["exclusion_reason"] = "extractor evidence anchors invalid"

    report = MODULE.evaluate_qualification(
        registry, qualification, store=store
    )

    assert report["dispositions"]["excluded"] == 1
    assert report["gate_counts"]["agent_panel"]["fail"] == 1


def test_two_rejected_extractors_abstain_with_zero_accepted_opinions(tmp_path):
    registry = _candidate_registry(1)
    qualification = MODULE.scaffold_qualification(registry)
    store = ContentAddressedStore(str(tmp_path / "cas"))
    evidence = _plural_agent_panel_execution_abstention_evidence(
        registry["entries"][0], store
    )
    record = qualification["records"][0]
    record["gates"]["agent_panel"] = _pass_gate(evidence)
    record["gates"]["agent_panel"]["status"] = "fail"
    record["disposition"] = "excluded"
    record["exclusion_reason"] = "both extractor outputs rejected"

    report = MODULE.evaluate_qualification(
        registry, qualification, store=store
    )

    assert evidence["accepted_opinions"] == []
    assert report["dispositions"]["excluded"] == 1
    assert report["gate_counts"]["agent_panel"]["fail"] == 1


def test_rejected_extractor_mixed_quote_checks_are_recomputed(tmp_path):
    registry = _candidate_registry(1)
    candidate = registry["entries"][0]
    store = ContentAddressedStore(str(tmp_path / "cas"))
    evidence = _plural_agent_panel_execution_abstention_evidence(
        candidate, store
    )
    source = SourceBundle.from_dict(evidence["source_bundle"])
    panel = evidence["panel_plan"]
    attempt = copy.deepcopy(
        evidence["rejected_attempts"][0]["attempt_record"]
    )
    request = panel["requests"][0]

    MODULE._validate_rejected_extractor_attempt(
        attempt,
        source,
        candidate,
        panel,
        request,
        evidence["panel_plan_digest"],
        store,
    )
    attempt["evidence_quote_checks"][0][
        "present_in_declared_source_bytes"
    ] = False
    with pytest.raises(ValueError, match="differs from source bytes"):
        MODULE._validate_rejected_extractor_attempt(
            attempt,
            source,
            candidate,
            panel,
            request,
            evidence["panel_plan_digest"],
            store,
        )


def test_incomplete_extractor_requires_bound_length_finish(tmp_path):
    registry = _candidate_registry(1)
    candidate = registry["entries"][0]
    store = ContentAddressedStore(str(tmp_path / "cas"))
    evidence = _plural_agent_panel_execution_abstention_evidence(
        candidate, store
    )
    source = SourceBundle.from_dict(evidence["source_bundle"])
    panel = evidence["panel_plan"]
    attempt = copy.deepcopy(
        evidence["rejected_attempts"][1]["attempt_record"]
    )
    request = panel["requests"][1]

    MODULE._validate_incomplete_extractor_attempt(
        attempt,
        source,
        candidate,
        panel,
        request,
        evidence["panel_plan_digest"],
        store,
    )
    attempt["finish_reason"] = "stop"
    with pytest.raises(ValueError, match="finish reason is not length"):
        MODULE._validate_incomplete_extractor_attempt(
            attempt,
            source,
            candidate,
            panel,
            request,
            evidence["panel_plan_digest"],
            store,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evidence: evidence["panel_execution"].update(
                {"complete_slots": False}
            ),
            "complete abstention",
        ),
        (
            lambda evidence: evidence["rejected_attempt_record"][
                "evidence_quote_checks"
            ][0].update({"present_in_declared_source_bytes": True}),
            "quote check differs",
        ),
        (
            lambda evidence: next(
                row for row in evidence["panel_execution"]["slots"]
                if row["role"] == "critic"
            ).update({"depends_on": []}),
            "dependency-failure slot",
        ),
    ],
)
def test_invalid_extractor_panel_rejects_incomplete_or_false_accounting(
    mutation, message, tmp_path
):
    registry = _candidate_registry(1)
    qualification = MODULE.scaffold_qualification(registry)
    store = ContentAddressedStore(str(tmp_path / "cas"))
    evidence = _agent_panel_execution_abstention_evidence(
        registry["entries"][0], store
    )
    mutation(evidence)
    _refresh_execution_abstention_digests(evidence)
    record = qualification["records"][0]
    record["gates"]["agent_panel"] = _pass_gate(evidence)
    record["gates"]["agent_panel"]["status"] = "fail"
    record["disposition"] = "excluded"
    record["exclusion_reason"] = "invalid panel accounting"

    with pytest.raises(ValueError, match=message):
        MODULE.evaluate_qualification(
            registry, qualification, store=store
        )


def test_invalid_extractor_panel_requires_exact_cas_closure(tmp_path):
    registry = _candidate_registry(1)
    qualification = MODULE.scaffold_qualification(registry)
    populated = ContentAddressedStore(str(tmp_path / "populated"))
    evidence = _agent_panel_execution_abstention_evidence(
        registry["entries"][0], populated
    )
    record = qualification["records"][0]
    record["gates"]["agent_panel"] = _pass_gate(evidence)
    record["gates"]["agent_panel"]["status"] = "fail"
    record["disposition"] = "excluded"
    record["exclusion_reason"] = "extractor evidence anchors invalid"

    empty = ContentAddressedStore(str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="source closure is unavailable"):
        MODULE.evaluate_qualification(
            registry, qualification, store=empty
        )


def test_agent_panel_fail_rejects_nonexistent_forged_digests(tmp_path):
    registry = _candidate_registry(1)
    qualification = MODULE.scaffold_qualification(registry)
    store = ContentAddressedStore(str(tmp_path / "cas"))
    evidence = _agent_panel_abstention_evidence(
        registry["entries"][0], store
    )
    evidence["agent_opinion_digests"] = [
        "sha256:" + str(index) * 64 for index in range(3)
    ]
    record = qualification["records"][0]
    record["gates"]["agent_panel"] = _pass_gate(evidence)
    record["gates"]["agent_panel"]["status"] = "fail"
    record["disposition"] = "excluded"
    record["exclusion_reason"] = "forged exclusion"

    with pytest.raises(ValueError, match="embedded agent opinions"):
        MODULE.evaluate_qualification(
            registry, qualification, store=store
        )


def test_agent_panel_fail_requires_resolvable_cas_closure(tmp_path):
    registry = _candidate_registry(1)
    qualification = MODULE.scaffold_qualification(registry)
    populated = ContentAddressedStore(str(tmp_path / "populated"))
    evidence = _agent_panel_abstention_evidence(
        registry["entries"][0], populated
    )
    record = qualification["records"][0]
    record["gates"]["agent_panel"] = _pass_gate(evidence)
    record["gates"]["agent_panel"]["status"] = "fail"
    record["disposition"] = "excluded"
    record["exclusion_reason"] = "agent panel abstained"

    with pytest.raises(ValueError, match="content-addressed store"):
        MODULE.evaluate_qualification(registry, qualification)
    empty = ContentAddressedStore(str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="provenance is unavailable"):
        MODULE.evaluate_qualification(
            registry, qualification, store=empty
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda evidence: evidence.pop("provider_attestation"),
            "embedded provider_attestation",
        ),
        (
            lambda evidence: evidence["rejected_attempt_record"].update(
                {"candidate_id": "another-claim"}
            ),
            "rejected_attempt_record digest mismatch",
        ),
        (
            lambda evidence: evidence["provider_attestation"][
                "request_policy"
            ].update({"zdr": False}),
            "provider_attestation digest mismatch",
        ),
        (
            lambda evidence: evidence["panel_closure_roots"].reverse(),
            "closure roots are not exact",
        ),
    ],
)
def test_agent_panel_fail_rejects_incomplete_or_unbound_evidence(
    mutation, message, tmp_path
):
    registry = _candidate_registry(1)
    qualification = MODULE.scaffold_qualification(registry)
    store = ContentAddressedStore(str(tmp_path / "cas"))
    evidence = _agent_panel_abstention_evidence(
        registry["entries"][0], store
    )
    mutation(evidence)
    record = qualification["records"][0]
    record["gates"]["agent_panel"] = _pass_gate(evidence)
    record["gates"]["agent_panel"]["status"] = "fail"
    record["disposition"] = "excluded"
    record["exclusion_reason"] = "invalid exclusion evidence"

    with pytest.raises(ValueError, match=message):
        MODULE.evaluate_qualification(
            registry, qualification, store=store
        )


def test_backward_extension_waits_until_every_row_has_final_disposition():
    registry = _candidate_registry(6)
    qualification = MODULE.scaffold_qualification(registry)
    for record in qualification["records"]:
        evidence = {
            "artifact": "stresskit_prefreeze_exclusion_evidence",
            "schema_version": "1.0",
            "claim_id": record["claim_id"],
            "reason": "incompatible dependency license",
        }
        record["gates"]["license_closure"] = {
            "status": "fail",
            "evidence": evidence,
            "evidence_digest": digest_json(evidence),
            "note": "fixture",
        }
        record["disposition"] = "excluded"
        record["exclusion_reason"] = "incompatible dependency license"
    report = MODULE.evaluate_qualification(registry, qualification)
    assert report["status"] == "backward_extension_required"
    assert report["backward_extension_required"] is True


def test_complete_typed_evidence_freezes_one_global_multiplicity_family():
    registry = _candidate_registry()
    qualification = _complete_qualification(registry)
    report = MODULE.evaluate_qualification(registry, qualification)
    assert report["freeze_ready"] is True
    assert report["breadth"]["eligible_claims"] == 20
    frozen = MODULE.freeze_registry(
        registry,
        qualification,
        release_id="synthetic-v1",
        frozen_at="2026-09-01T02:00:00+00:00",
    )
    assert frozen["artifact"] == "stresskit_release_registry"
    assert frozen["status"] == "frozen"
    assert len(frozen["claims"]) == 20
    assert {row["disposition"] for row in frozen["claims"]} == {"eligible"}


def test_global_multiplicity_family_cannot_drop_eligible_claim():
    registry = _candidate_registry()
    qualification = _complete_qualification(registry)
    changed = copy.deepcopy(qualification)
    for record in changed["records"]:
        spec_gate = record["gates"]["audit_spec"]
        spec_gate["evidence"]["multiplicity_family"]["member_claim_ids"].pop()
        spec_digest = digest_json(spec_gate["evidence"])
        spec_gate["evidence_digest"] = spec_digest
        record["audit_spec_digest"] = spec_digest
        for gate_name in ("protocol_review", "resource_estimate"):
            gate = record["gates"][gate_name]
            gate["evidence"]["audit_spec_digest"] = spec_digest
            gate["evidence_digest"] = digest_json(gate["evidence"])
    with pytest.raises(ValueError, match="all eligible claims"):
        MODULE.evaluate_qualification(registry, changed)
