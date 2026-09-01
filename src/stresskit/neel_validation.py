"""Build a conservative evidence report against Neel Nanda transcript criteria."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from stresskit.integrity import digest_json
from stresskit.utility_preregistration import (
    validate_blind_metadata_preflight,
    validate_external_utility_preregistration,
)


THEMES = (
    {
        "theme_id": "scientific_integrity",
        "criteria": ["N03", "N07", "N11", "N14", "N16"],
        "status": "verifier_mechanics_demonstrated",
        "finding": (
            "Adversarial tests demonstrate fail-closed handling of stable nonsense, "
            "missing runs, forged evidence, unsafe agent input, and multiplicity."
        ),
        "limitation": (
            "Software checks and calibrated inference do not establish construct "
            "validity or convergence to scientific truth."
        ),
    },
    {
        "theme_id": "external_utility",
        "criteria": ["N04", "N05", "N06", "N08", "N09", "N10", "N12", "N13", "N41"],
        "status": "protocol_hardened_benchmark_pending",
        "finding": (
            "Utility verifier recomputes external-task metrics and comparison against "
            "the strongest registered non-internals baseline from raw evidence."
        ),
        "limitation": "No final external-task utility AuditBundle exists.",
    },
    {
        "theme_id": "generalization",
        "criteria": ["N17", "N27", "N28", "N32", "N40"],
        "status": "protocol_hardened_benchmark_pending",
        "finding": (
            "Generalization is a separate primary axis and must use frozen held-out "
            "partition bindings rather than a relabeled primary split."
        ),
        "limitation": "No final held-out claim result exists.",
    },
    {
        "theme_id": "cot_probe_steering",
        "criteria": ["N19", "N20", "N21", "N22", "N35", "N36"],
        "status": "protocol_hardened_benchmark_pending",
        "finding": (
            "Registry covers CoT, probes, and steering; ordered CoT trajectories retain "
            "event order and multiplicity."
        ),
        "limitation": "Coverage is designed but no frozen claim in these strata has run.",
    },
    {
        "theme_id": "verifiable_agent_auditing",
        "criteria": ["N16", "N25", "N26"],
        "status": "mechanism_demonstrated_live_sample_insufficient",
        "finding": (
            "Planted compiler cases and live panel executions demonstrate provenance "
            "binding, unsupported-wording abstention, and rejection of invented quotes."
        ),
        "limitation": (
            "The small live sample cannot estimate extractor recall, precision, provider "
            "drift, or collusion resistance."
        ),
    },
    {
        "theme_id": "gradient_persona_flagship",
        "criteria": ["N31", "N33", "N34"],
        "status": "blocked_abstain",
        "finding": (
            "Study protocol binds residual-stream loss-gradient projections, held-out "
            "behavior, non-internals baselines, and falsification controls."
        ),
        "limitation": "Required licensed persona and misalignment artifacts are unresolved.",
    },
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _proof_summary(path: Path) -> Dict[str, Any]:
    root = ET.fromstring(path.read_bytes())
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("proof JUnit contains no testsuite")
    counts = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    cases = sorted(
        {
            f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
            for suite in suites for case in suite.findall("testcase")
        }
    )
    if counts["tests"] != len(cases):
        raise ValueError("proof JUnit test count differs from unique testcase list")
    if counts["failures"] or counts["errors"] or counts["skipped"]:
        raise ValueError("proof suite must pass without failures, errors, or skips")
    return {**counts, "testcases": cases, "artifact_digest": _sha256(path)}


def _invalid_panel_summary(
    execution_path: Path,
    decision_path: Path,
    attempt_path: Path,
    incomplete_attempt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Validate one no-retry panel stopped before an auditable claim existed."""
    execution = _load(execution_path)
    decision = _load(decision_path)
    attempt = _load(attempt_path)
    incomplete_attempt = (
        _load(incomplete_attempt_path)
        if incomplete_attempt_path is not None else None
    )
    if execution.get("artifact") != "stresskit_agent_panel_execution" or \
            execution.get("status") != "abstain" or \
            execution.get("publication_state") != "abstain" or \
            execution.get("retry_policy") != "no_retry" or \
            execution.get("complete_slots") is not True:
        raise ValueError("invalid-evidence panel execution is not fail-closed")
    slots = execution.get("slots")
    if not isinstance(slots, list) or len(slots) != 3:
        raise ValueError("invalid-evidence panel must account for three slots")
    expected = (
        [
            ("extractor", "rejected_invalid_evidence"),
            ("extractor", "rejected_incomplete_completion"),
            ("critic", "not_run_dependency_failure"),
        ]
        if incomplete_attempt is not None else
        [
            ("extractor", "accepted"),
            ("extractor", "rejected_invalid_evidence"),
            ("critic", "not_run_dependency_failure"),
        ]
    )
    if [(row.get("role"), row.get("status")) for row in slots] != expected:
        raise ValueError("invalid-evidence panel slot states differ from protocol")
    rejected = next(
        row for row in slots
        if row.get("status") == "rejected_invalid_evidence"
    )
    critic = slots[2]
    extractor_ids = [row.get("opinion_id") for row in slots[:2]]
    if critic.get("depends_on") != extractor_ids:
        raise ValueError("invalid-evidence critic dependency is not exact")

    if attempt.get("artifact") != "stresskit_agent_attempt_record" or \
            attempt.get("status") != "rejected_before_opinion" or \
            attempt.get("completion_content_inspected") is not True or \
            attempt.get("retry_performed") is not False or \
            attempt.get("critic_called") is not False:
        raise ValueError("invalid-evidence attempt record is inconsistent")
    quote_checks = attempt.get("evidence_quote_checks")
    if not isinstance(quote_checks, list) or not quote_checks:
        raise ValueError("rejected extractor quote accounting is empty")
    invalid_quote_count = sum(
        row.get("present_in_declared_source_bytes") is False
        for row in quote_checks
    )
    if invalid_quote_count < 1:
        raise ValueError("rejected extractor has no source-invalid quote")
    if incomplete_attempt is None and invalid_quote_count != len(quote_checks):
        raise ValueError("rejected extractor quotes were not all source-invalid")
    if rejected.get("attempt_digest") != digest_json(attempt) or \
            rejected.get("raw_response_digest") != attempt.get("raw_response_digest"):
        raise ValueError("invalid-evidence slot does not bind rejected attempt")

    candidate_id = execution.get("candidate_id")
    attempts = [attempt]
    incomplete_slot = None
    if incomplete_attempt is not None:
        incomplete_slot = next(
            row for row in slots
            if row.get("status") == "rejected_incomplete_completion"
        )
        if incomplete_attempt.get("artifact") != "stresskit_agent_attempt_record" or \
                incomplete_attempt.get("status") != "rejected_before_opinion" or \
                incomplete_attempt.get("completion_content_inspected") is not False or \
                incomplete_attempt.get("retry_performed") is not False or \
                incomplete_attempt.get("critic_called") is not False or \
                incomplete_attempt.get("finish_reason") != "length" or \
                incomplete_attempt.get("evidence_quote_checks") != []:
            raise ValueError("incomplete extractor attempt record is inconsistent")
        if incomplete_slot.get("attempt_digest") != digest_json(incomplete_attempt) or \
                incomplete_slot.get("raw_response_digest") != \
                incomplete_attempt.get("raw_response_digest"):
            raise ValueError("incomplete slot does not bind rejected attempt")
        attempts.append(incomplete_attempt)
    if any(
        row.get("candidate_id") != candidate_id or
        row.get("source_bundle_digest") != execution.get("source_bundle_digest") or
        row.get("panel_plan_digest") != execution.get("panel_plan_digest")
        for row in attempts
    ):
        raise ValueError("rejected attempt targets another panel")
    if decision.get("artifact") != "stresskit_claim_candidates" or \
            decision.get("publication_state") != "abstain" or \
            decision.get("candidates") != [] or \
            decision.get("panel_execution_digest") != digest_json(execution):
        raise ValueError("invalid-evidence decision is not bound abstention")
    problems = "\n".join(str(row) for row in decision.get("problems", []))
    stopped_ids = [
        row.get("opinion_id") for row in slots
        if row.get("status") != "accepted"
    ]
    if any(opinion_id not in problems for opinion_id in stopped_ids):
        raise ValueError("invalid-evidence decision omits stopped panel slots")
    return {
        "candidate_id": candidate_id,
        "publication_state": "abstain",
        "accepted_opinions": sum(
            row.get("status") == "accepted" for row in slots
        ),
        "rejected_invalid_evidence": 1,
        "rejected_incomplete_completion": int(incomplete_slot is not None),
        "dependent_slots_not_run": 1,
        "checked_quote_count": len(quote_checks),
        "invalid_quote_count": invalid_quote_count,
        "claim_outcome_verified": False,
        "retry_performed": False,
    }


def build_validation_report(
    *,
    criteria_path: Path,
    transcript_path: Path,
    calibration_path: Path,
    compiler_path: Path,
    qualification_path: Path,
    live_decision_path: Path,
    provider_attestation_path: Path,
    flagship_path: Path,
    proof_junit_path: Path,
    proof_selector_path: Optional[Path] = None,
    invalid_panel_execution_path: Optional[Path] = None,
    invalid_panel_decision_path: Optional[Path] = None,
    rejected_attempt_path: Optional[Path] = None,
    double_rejection_panel_execution_path: Optional[Path] = None,
    double_rejection_panel_decision_path: Optional[Path] = None,
    double_rejection_invalid_attempt_path: Optional[Path] = None,
    double_rejection_incomplete_attempt_path: Optional[Path] = None,
    utility_preregistration_path: Optional[Path] = None,
    utility_preflight_path: Optional[Path] = None,
    flagship_license_audit_path: Optional[Path] = None,
    cot_panel_preflight_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Recompute a scoped Neel-criteria validation report from frozen artifacts."""
    criteria = _load(criteria_path)
    calibration = _load(calibration_path)
    compiler = _load(compiler_path)
    qualification = _load(qualification_path)
    live_decision = _load(live_decision_path)
    provider_attestation = _load(provider_attestation_path)
    flagship = _load(flagship_path)

    if criteria.get("artifact") != "stresskit_neel_criteria_registry":
        raise ValueError("invalid Neel criteria registry artifact")
    criterion_rows = criteria.get("criteria")
    if not isinstance(criterion_rows, list):
        raise ValueError("Neel criteria registry needs criteria list")
    criterion_ids = [row.get("id") for row in criterion_rows if isinstance(row, Mapping)]
    expected_ids = [f"N{index:02d}" for index in range(1, 49)]
    if criterion_ids != expected_ids:
        raise ValueError("Neel criteria registry must contain ordered N01-N48")
    if _sha256(transcript_path) != "sha256:" + str(criteria["source"]["sha256"]):
        raise ValueError("transcript bytes differ from criteria source digest")
    empirical = [
        row for row in criterion_rows
        if "empirical_claim" in row.get("kinds", [])
    ]
    if any(row.get("verification_status") == "verified" for row in empirical):
        raise ValueError("criteria registry cannot pre-mark empirical claims verified")

    acceptance = calibration.get("acceptance", {})
    if acceptance.get("passed") is not True or \
            float(acceptance.get("observed_minimum_coverage", 0.0)) < \
            float(acceptance.get("minimum_coverage", 1.0)) or \
            float(acceptance.get("observed_maximum_known_invalid_false_pass_rate", 1.0)) > \
            float(acceptance.get("maximum_known_invalid_false_pass_rate", 0.0)):
        raise ValueError("calibration acceptance gate did not pass")

    counts = compiler.get("counts", {})
    compiler_cases = sum(int(row.get("cases", 0)) for row in counts.values())
    if compiler.get("acceptance_passed") is not True or compiler_cases != 300:
        raise ValueError("300-case compiler gate did not pass")

    dispositions = qualification.get("dispositions", {})
    total_candidates = sum(int(value) for value in dispositions.values())
    if total_candidates < 1 or qualification.get("freeze_ready") is not False:
        raise ValueError("qualification report must remain an unfinished prefreeze ledger")

    routes = provider_attestation.get("routes")
    if provider_attestation.get("artifact") != \
            "stresskit_openrouter_panel_attestation" or \
            provider_attestation.get("status") != \
            "verified_from_accepted_responses" or \
            not isinstance(routes, list) or len(routes) != 3 or \
            not all(row.get("accepted") is True for row in routes) or \
            sorted(row.get("role") for row in routes) != [
                "critic", "extractor", "extractor"
            ] or \
            len({row.get("opinion_id") for row in routes}) != 3 or \
            len({row.get("selected_provider") for row in routes}) != 3 or \
            len({row.get("selected_canonical_model") for row in routes}) != 3 or \
            len({row.get("requested_provider_endpoint") for row in routes}) != 3:
        raise ValueError("live panel needs three accepted distinct-provider routes")
    if live_decision.get("publication_state") != "abstain" or \
            live_decision.get("candidates") != [] or \
            live_decision.get("source_bundle_digest") != \
            provider_attestation.get("source_bundle_digest"):
        raise ValueError("live decision must demonstrate unsupported-wording abstention")
    decision_problems = live_decision.get("problems")
    route_opinion_ids = [str(row["opinion_id"]) for row in routes]
    expected_markers = {
        opinion_id: f"opinion '{opinion_id}' marks claim unsupported"
        for opinion_id in route_opinion_ids
    }
    if not isinstance(decision_problems, list) or \
            sum("marks claim unsupported" in str(problem)
                for problem in decision_problems) != 3 or \
            any(
                sum(marker in str(problem) for problem in decision_problems) != 1
                for marker in expected_markers.values()
            ):
        raise ValueError("live decision does not bind exact unsupported panel")
    if flagship.get("status") != "abstain" or \
            flagship.get("substitution_allowed") is not False:
        raise ValueError("flagship must remain fail-closed while licenses are unresolved")

    utility_paths = (utility_preregistration_path, utility_preflight_path)
    if any(path is not None for path in utility_paths) and \
            not all(path is not None for path in utility_paths):
        raise ValueError("utility readiness needs preregistration and preflight")
    utility_readiness = None
    if all(path is not None for path in utility_paths):
        assert utility_preregistration_path is not None
        assert utility_preflight_path is not None
        utility_preregistration = _load(utility_preregistration_path)
        utility_preflight = _load(utility_preflight_path)
        preregistration_summary = validate_external_utility_preregistration(
            utility_preregistration
        )
        preflight_summary = validate_blind_metadata_preflight(utility_preflight)
        if utility_preflight.get("preregistration_digest") != \
                preregistration_summary["digest"]:
            raise ValueError("utility preflight targets another preregistration")
        utility_readiness = {
            "status": preregistration_summary["status"],
            "publication_state": preregistration_summary["publication_state"],
            "baseline_count": preregistration_summary["baseline_count"],
            "minimum_independent_units": preregistration_summary[
                "minimum_independent_units"
            ],
            "preflight_status": preflight_summary["status"],
            "minimums_established": preflight_summary["minimums_established"],
            "content_rows_read": utility_preflight["content_rows_read"],
            "claim_outcome_computed": utility_preflight[
                "claim_outcome_computed"
            ],
            "gpu_used": utility_preflight["gpu_used"],
        }

    flagship_license_readiness = None
    if flagship_license_audit_path is not None:
        license_audit = _load(flagship_license_audit_path)
        blockers = license_audit.get("blockers")
        if license_audit.get("artifact") != \
                "stresskit_flagship_license_audit" or \
                license_audit.get("schema_version") != "1.0" or \
                license_audit.get("study_id") != \
                "gradient-projection-behavior-v1" or \
                license_audit.get("status") != "blocked" or \
                license_audit.get("freeze_eligible") is not False or \
                license_audit.get("experiments_run") is not False or \
                license_audit.get("substitution_allowed") is not False or \
                not isinstance(blockers, list) or not blockers:
            raise ValueError("flagship license audit is not fail-closed")
        flagship_license_readiness = {
            "status": "blocked",
            "freeze_eligible": False,
            "experiments_run": False,
            "substitution_allowed": False,
            "blocker_count": len(blockers),
        }

    cot_panel_readiness = None
    if cot_panel_preflight_path is not None:
        cot_preflight = _load(cot_panel_preflight_path)
        route_checks = cot_preflight.get("route_checks")
        accounting = cot_preflight.get("execution_accounting")
        logging_check = cot_preflight.get("account_prompt_logging_check")
        if cot_preflight.get("artifact") != \
                "stresskit_openrouter_authenticated_preflight_blocker" or \
                cot_preflight.get("schema_version") != "1.0" or \
                cot_preflight.get("status") != \
                "blocked_awaiting_account_logging_attestation" or \
                cot_preflight.get("gate_publication_state") != "abstain" or \
                cot_preflight.get("candidate_disposition") != "pending" or \
                not isinstance(route_checks, list) or len(route_checks) != 3 or \
                any(row.get("result") != "pass" for row in route_checks) or \
                not isinstance(accounting, Mapping) or \
                accounting.get("chat_completion_calls") != 0 or \
                accounting.get("opinion_slots_started") != [] or \
                accounting.get("critic_called") is not False or \
                accounting.get("retry_performed") is not False or \
                accounting.get("gpu_calls") != 0 or \
                not isinstance(logging_check, Mapping) or \
                logging_check.get("observed") != "unverified" or \
                logging_check.get("http_status") != 401:
            raise ValueError("CoT provider preflight is not fail-closed")
        cot_panel_readiness = {
            "status": cot_preflight["status"],
            "publication_state": cot_preflight["gate_publication_state"],
            "candidate_disposition": cot_preflight["candidate_disposition"],
            "routes_passed": len(route_checks),
            "chat_completion_calls": 0,
            "opinions_created": 0,
            "critic_called": False,
            "gpu_used": False,
        }

    invalid_paths = (
        invalid_panel_execution_path,
        invalid_panel_decision_path,
        rejected_attempt_path,
    )
    if any(path is not None for path in invalid_paths) and \
            not all(path is not None for path in invalid_paths):
        raise ValueError("invalid-evidence panel requires execution, decision, and attempt")
    invalid_panel = None
    if all(path is not None for path in invalid_paths):
        assert invalid_panel_execution_path is not None
        assert invalid_panel_decision_path is not None
        assert rejected_attempt_path is not None
        invalid_panel = _invalid_panel_summary(
            invalid_panel_execution_path,
            invalid_panel_decision_path,
            rejected_attempt_path,
        )

    double_rejection_paths = (
        double_rejection_panel_execution_path,
        double_rejection_panel_decision_path,
        double_rejection_invalid_attempt_path,
        double_rejection_incomplete_attempt_path,
    )
    if any(path is not None for path in double_rejection_paths) and \
            not all(path is not None for path in double_rejection_paths):
        raise ValueError(
            "double-rejection panel requires execution, decision, and both attempts"
        )
    double_rejection_panel = None
    if all(path is not None for path in double_rejection_paths):
        assert double_rejection_panel_execution_path is not None
        assert double_rejection_panel_decision_path is not None
        assert double_rejection_invalid_attempt_path is not None
        assert double_rejection_incomplete_attempt_path is not None
        double_rejection_panel = _invalid_panel_summary(
            double_rejection_panel_execution_path,
            double_rejection_panel_decision_path,
            double_rejection_invalid_attempt_path,
            double_rejection_incomplete_attempt_path,
        )

    proof = _proof_summary(proof_junit_path)
    if proof_selector_path is not None:
        selectors = [
            line.strip()
            for line in proof_selector_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not selectors or len(selectors) != len(set(selectors)) or any(
            "::" not in selector for selector in selectors
        ):
            raise ValueError("proof selector manifest is empty, duplicated, or invalid")
        selector_functions = [selector.rsplit("::", 1)[1] for selector in selectors]
        case_functions = [case.rsplit("::", 1)[1] for case in proof["testcases"]]
        if any(
            not any(
                case == function or case.startswith(function + "[")
                for case in case_functions
            )
            for function in selector_functions
        ) or any(
            not any(
                case == function or case.startswith(function + "[")
                for function in selector_functions
            )
            for case in case_functions
        ):
            raise ValueError("proof JUnit differs from selector manifest")
        proof.update({
            "selector_count": len(selectors),
            "selector_manifest_digest": _sha256(proof_selector_path),
        })
    evidence_paths = {
        "criteria": criteria_path,
        "transcript": transcript_path,
        "calibration": calibration_path,
        "compiler_evaluation": compiler_path,
        "qualification": qualification_path,
        "live_decision": live_decision_path,
        "provider_attestation": provider_attestation_path,
        "flagship_candidate": flagship_path,
    }
    if proof_selector_path is not None:
        evidence_paths["proof_selectors"] = proof_selector_path
    if invalid_panel_execution_path is not None:
        evidence_paths.update({
            "invalid_panel_execution": invalid_panel_execution_path,
            "invalid_panel_decision": invalid_panel_decision_path,
            "rejected_agent_attempt": rejected_attempt_path,
        })
    if double_rejection_panel_execution_path is not None:
        evidence_paths.update({
            "double_rejection_panel_execution":
                double_rejection_panel_execution_path,
            "double_rejection_panel_decision":
                double_rejection_panel_decision_path,
            "double_rejection_invalid_attempt":
                double_rejection_invalid_attempt_path,
            "double_rejection_incomplete_attempt":
                double_rejection_incomplete_attempt_path,
        })
    if utility_preregistration_path is not None:
        evidence_paths.update({
            "utility_preregistration": utility_preregistration_path,
            "utility_blind_preflight": utility_preflight_path,
        })
    if flagship_license_audit_path is not None:
        evidence_paths["flagship_license_audit"] = flagship_license_audit_path
    if cot_panel_preflight_path is not None:
        evidence_paths["cot_panel_authenticated_preflight"] = \
            cot_panel_preflight_path
    return {
        "artifact": "stresskit_neel_validation_report",
        "schema_version": "1.0",
        "source_criteria_digest": _sha256(criteria_path),
        "evidence_digests": {
            name: _sha256(path) for name, path in sorted(evidence_paths.items())
        },
        "proof_suite": proof,
        "calibration": {
            "passed": True,
            "observed_minimum_coverage": acceptance["observed_minimum_coverage"],
            "observed_maximum_known_invalid_false_pass_rate": acceptance[
                "observed_maximum_known_invalid_false_pass_rate"
            ],
            "scope": calibration.get("interpretation"),
        },
        "agent_compiler": {
            "passed": True,
            "cases": compiler_cases,
            "metrics": compiler.get("metrics"),
            "scope": compiler.get("scope"),
        },
        "live_panel": {
            "publication_state": "abstain",
            "providers": [row["selected_provider"] for row in routes],
            "supported_candidates": 0,
        },
        "invalid_evidence_panel": invalid_panel,
        "double_rejection_panel": double_rejection_panel,
        "scientific_readiness": {
            "cot_panel": cot_panel_readiness,
            "external_utility": utility_readiness,
            "gradient_persona_flagship": flagship_license_readiness,
        },
        "benchmark": {
            "candidate_count": total_candidates,
            "dispositions": dict(dispositions),
            "freeze_ready": False,
            "registered_final_audit_bundles": 0,
        },
        "themes": [dict(theme) for theme in THEMES],
        "conclusion": {
            "verifier_mechanics_demonstrated": True,
            "neel_empirical_claims_verified": False,
            "registered_external_utility_results_available": False,
            "registered_august_claim_results_available": False,
            "flagship_study_completed": False,
            "safe_claim": (
                "StressKit defensively verifies audit provenance and rejects several "
                "known false-pass paths; external scientific usefulness remains unproven."
            ),
            "forbidden_claim": "StressKit has verified Neel Nanda's empirical claims.",
        },
        "external_validation": "not obtained",
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render conservative human-readable validation summary."""
    benchmark = report["benchmark"]
    calibration = report["calibration"]
    compiler = report["agent_compiler"]
    proof = report["proof_suite"]
    live_lines = [
        "- Live panel 1: three distinct attested provider routes; three unsupported "
        "opinions produced abstention."
    ]
    invalid_panel = report.get("invalid_evidence_panel")
    if invalid_panel is not None:
        live_lines.append(
            "- Live panel 2: one extractor accepted; one rejected after "
            f"{invalid_panel['invalid_quote_count']} source-quote mismatches; dependent "
            "critic not run; no retry; abstention."
        )
    double_rejection_panel = report.get("double_rejection_panel")
    if double_rejection_panel is not None:
        live_lines.append(
            "- Live panel 3 (ACDC): both extractors rejected; "
            f"{double_rejection_panel['invalid_quote_count']} of "
            f"{double_rejection_panel['checked_quote_count']} quotes failed exact-source "
            "checks and one completion hit its frozen token limit; critic not run; "
            "no retry; claim abstained."
        )
    live_panel_count = 1 + int(invalid_panel is not None) + int(
        double_rejection_panel is not None
    )
    readiness_lines = []
    readiness = report.get("scientific_readiness", {})
    cot_panel = readiness.get("cot_panel") \
        if isinstance(readiness, Mapping) else None
    if isinstance(cot_panel, Mapping):
        readiness_lines.append(
            "- Thought Anchors CoT panel: three live routes passed catalog and ZDR "
            "checks, but account prompt-logging state was not readable; protocol "
            "stopped before sending source text; zero opinions, critic calls, or GPU."
        )
    utility = readiness.get("external_utility") \
        if isinstance(readiness, Mapping) else None
    if isinstance(utility, Mapping):
        readiness_lines.append(
            "- Thought Anchors external utility/generalization: four non-internals "
            "baselines and held-out design registered, but metadata-only preflight "
            "could not establish 200 independent problem clusters per partition; "
            "abstain; no labels, outcome, or GPU used."
        )
    flagship_readiness = readiness.get("gradient_persona_flagship") \
        if isinstance(readiness, Mapping) else None
    if isinstance(flagship_readiness, Mapping):
        readiness_lines.append(
            "- Gradient-persona flagship: license audit records "
            f"{flagship_readiness['blocker_count']} unresolved blockers; no "
            "experiment or artifact substitution; abstain."
        )
    lines = [
        "# StressKit validation against Neel Nanda transcript criteria",
        "",
        "## Verdict",
        "",
        "StressKit verifier mechanics are demonstrated. Neel's empirical claims are **not** "
        "verified, and no final external-utility or August claim result is registered yet.",
        "",
        f"Safe claim: {report['conclusion']['safe_claim']}",
        "",
        "## Evidence available now",
        "",
        f"- Adversarial proof slice: {proof['tests']} passed; zero failures/errors/skips.",
        f"- Compiler evaluation: {compiler['cases']} planted cases; gate passed.",
        "- Calibration: minimum observed coverage "
        f"{100 * calibration['observed_minimum_coverage']:.2f}%; maximum known-invalid "
        f"false-pass {100 * calibration['observed_maximum_known_invalid_false_pass_rate']:.2f}%.",
        *live_lines,
        *readiness_lines,
        f"- Prefreeze registry: {benchmark['dispositions'].get('eligible', 0)} eligible, "
        f"{benchmark['dispositions'].get('pending', 0)} pending, "
        f"{benchmark['dispositions'].get('excluded', 0)} excluded.",
        "",
        "## Criteria themes",
        "",
        "| Theme | Status | What evidence says | Limitation |",
        "|---|---|---|---|",
    ]
    for theme in report["themes"]:
        lines.append(
            f"| `{theme['theme_id']}` | `{theme['status']}` | "
            f"{theme['finding']} | {theme['limitation']} |"
        )
    lines.extend([
        "",
        "## Boundaries",
        "",
        "- Calibration covers bounded inference, not task choice, null validity, or scientific truth.",
        "- Planted compiler cases do not estimate live extractor recall or provider drift.",
        f"- {live_panel_count} live abstentions demonstrate fail-closed behavior, "
        "not successful claim auditing rate.",
        "- Persona-gradient flagship remains abstained until every required artifact license resolves.",
        "- `external_validation: not obtained`.",
        "",
        "Full transcript-derived N01-N48 registry: `benchmark/neel_criteria_v1.json`.",
        "Proof selector manifest: "
        "`artifacts/validation/neel-adversarial-proof-selectors-v1.txt`.",
        "Machine report: `artifacts/validation/neel-validation-v1.json`.",
        "",
    ])
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate JSON and Markdown reports from explicit evidence inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--criteria", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--live-decision", required=True, type=Path)
    parser.add_argument("--provider-attestation", required=True, type=Path)
    parser.add_argument("--flagship", required=True, type=Path)
    parser.add_argument("--proof-junit", required=True, type=Path)
    parser.add_argument("--proof-selectors", type=Path)
    parser.add_argument("--invalid-panel-execution", type=Path)
    parser.add_argument("--invalid-panel-decision", type=Path)
    parser.add_argument("--rejected-attempt", type=Path)
    parser.add_argument("--double-rejection-panel-execution", type=Path)
    parser.add_argument("--double-rejection-panel-decision", type=Path)
    parser.add_argument("--double-rejection-invalid-attempt", type=Path)
    parser.add_argument("--double-rejection-incomplete-attempt", type=Path)
    parser.add_argument("--utility-preregistration", type=Path)
    parser.add_argument("--utility-preflight", type=Path)
    parser.add_argument("--flagship-license-audit", type=Path)
    parser.add_argument("--cot-panel-preflight", type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build_validation_report(
        criteria_path=args.criteria,
        transcript_path=args.transcript,
        calibration_path=args.calibration,
        compiler_path=args.compiler,
        qualification_path=args.qualification,
        live_decision_path=args.live_decision,
        provider_attestation_path=args.provider_attestation,
        flagship_path=args.flagship,
        proof_junit_path=args.proof_junit,
        proof_selector_path=args.proof_selectors,
        invalid_panel_execution_path=args.invalid_panel_execution,
        invalid_panel_decision_path=args.invalid_panel_decision,
        rejected_attempt_path=args.rejected_attempt,
        double_rejection_panel_execution_path=
            args.double_rejection_panel_execution,
        double_rejection_panel_decision_path=
            args.double_rejection_panel_decision,
        double_rejection_invalid_attempt_path=
            args.double_rejection_invalid_attempt,
        double_rejection_incomplete_attempt_path=
            args.double_rejection_incomplete_attempt,
        utility_preregistration_path=args.utility_preregistration,
        utility_preflight_path=args.utility_preflight,
        flagship_license_audit_path=args.flagship_license_audit,
        cot_panel_preflight_path=args.cot_panel_preflight,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
