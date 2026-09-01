"""Outcome-blind claim compilation, design freezing, and resource planning."""

from __future__ import annotations

import datetime as dt
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .audit_models import (
    AUDIT_SCHEMA_VERSION,
    AgentOpinion,
    AuditSpec,
    ClaimRecord,
    ResourcePlan,
    SourceBundle,
)
from .audit_profiles import (
    get_profile,
    reducer_digest,
    validate_expected_target,
    validate_reducer_config,
)
from .integrity import digest_json, require_sha256_digest, sha256_bytes


_INJECTION_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b",
    r"\b(system|developer)\s+prompt\b",
    r"\bdo\s+not\s+(audit|extract|report|follow)\b",
    r"\byou\s+are\s+(chatgpt|an?\s+assistant|an?\s+language\s+model)\b",
    r"<\s*/?\s*(system|assistant|developer)\s*>",
    r"\btool\s+call\b.*\b(execute|run|delete|upload)\b",
))


def utc_now() -> str:
    """Return second-resolution UTC timestamp for artifact creation."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def detect_prompt_injection(text: str) -> List[str]:
    """Return conservative document-instruction matches requiring abstention."""
    if not isinstance(text, str):
        raise TypeError("prompt-injection scan expects text")
    matches = []
    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            matches.append(match.group(0))
    return matches


def _normalize_statement(value: str) -> str:
    return " ".join(value.split()).casefold()


def validate_agent_panel(
    source: SourceBundle,
    opinions: Sequence[AgentOpinion],
    *,
    source_texts: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Validate two isolated cross-provider extractors and one unanimous critic."""
    problems: List[str] = []
    extractors = [opinion for opinion in opinions if opinion.role == "extractor"]
    critics = [opinion for opinion in opinions if opinion.role == "critic"]
    if len(extractors) != 2:
        problems.append("agent panel needs exactly two extractors")
    if len(critics) != 1:
        problems.append("agent panel needs exactly one critic")
    if len({opinion.opinion_id for opinion in opinions}) != len(opinions):
        problems.append("agent opinion IDs are not unique")
    if len({opinion.request_digest for opinion in opinions}) != len(opinions):
        problems.append("agent requests are not isolated: duplicate request digest")
    if len(extractors) == 2:
        if len({opinion.provider for opinion in extractors}) != 2:
            problems.append("extractors must use distinct providers")
        if len({opinion.model_family for opinion in extractors}) != 2:
            problems.append("extractors must use distinct model families")
    for opinion in opinions:
        if opinion.source_bundle_digest != source.digest:
            problems.append(
                f"opinion {opinion.opinion_id!r} targets a different SourceBundle"
            )
        if not opinion.supported:
            problems.append(f"opinion {opinion.opinion_id!r} marks claim unsupported")
        if opinion.prompt_injection_detected:
            problems.append(f"opinion {opinion.opinion_id!r} detected prompt injection")
        if opinion.issues:
            problems.append(
                f"opinion {opinion.opinion_id!r} reports issues: "
                + "; ".join(str(issue) for issue in opinion.issues)
            )
    statements = {_normalize_statement(opinion.statement) for opinion in opinions}
    if len(statements) != 1:
        problems.append("extractors and critic do not support identical claim wording")

    documents = {row["document_id"]: row for row in source.documents}
    if source_texts is None:
        problems.append("exact UTF-8 source texts are required for anchor verification")
    else:
        unknown_texts = sorted(set(source_texts) - set(documents))
        if unknown_texts:
            problems.append(
                "source texts name unknown documents: " + ", ".join(unknown_texts)
            )
    for document_id, document in documents.items():
        if document["license"]["status"] != "verified_compatible":
            problems.append(
                f"document {document_id!r} license is not verified compatible"
            )
    for opinion in opinions:
        for anchor in opinion.evidence_anchors:
            document = documents.get(anchor.get("document_id"))
            if document is None:
                problems.append(
                    f"opinion {opinion.opinion_id!r} anchor names unknown document"
                )
            elif anchor.get("source_digest", document["source_digest"]) != \
                    document["source_digest"]:
                problems.append(
                    f"opinion {opinion.opinion_id!r} anchor source digest mismatch"
                )
            elif anchor.get("text_digest") != document.get(
                    "extracted_text_digest", document["source_digest"]):
                problems.append(
                    f"opinion {opinion.opinion_id!r} anchor text digest mismatch"
                )
            elif source_texts is None or anchor["document_id"] not in source_texts:
                problems.append(
                    f"opinion {opinion.opinion_id!r} anchor lacks exact source text"
                )
            else:
                text = source_texts[anchor["document_id"]].encode("utf-8")
                start, end = int(anchor["start"]), int(anchor["end"])
                if end > len(text) or sha256_bytes(text[start:end]) != \
                        anchor["quote_digest"]:
                    problems.append(
                        f"opinion {opinion.opinion_id!r} anchor does not match "
                        "exact source bytes"
                    )

    if source_texts is not None:
        for document_id, document in documents.items():
            text = source_texts.get(document_id)
            if text is None and "extracted_text_digest" in document:
                problems.append(f"source text missing for document {document_id!r}")
                continue
            if text is None:
                continue
            actual = sha256_bytes(text.encode("utf-8"))
            expected_text_digest = document.get(
                "extracted_text_digest", document["source_digest"]
            )
            if actual != expected_text_digest:
                problems.append(f"source text digest mismatch for document {document_id!r}")
            matches = detect_prompt_injection(text)
            if matches:
                problems.append(
                    f"document {document_id!r} contains instruction-like text: "
                    + ", ".join(matches)
                )
    return problems


def discover_claims(
    source: SourceBundle,
    opinions: Sequence[AgentOpinion],
    *,
    source_texts: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Create a deterministic candidate artifact or explicit abstention."""
    problems = validate_agent_panel(source, opinions, source_texts=source_texts)
    candidates = []
    if not problems:
        candidates.append({
            "statement": opinions[0].statement,
            "evidence_anchors": [
                dict(anchor) for opinion in opinions
                for anchor in opinion.evidence_anchors
            ],
            "agent_opinion_digests": [opinion.digest for opinion in opinions],
        })
    return {
        "artifact": "stresskit_claim_candidates",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source_bundle_digest": source.digest,
        "publication_state": "abstain" if problems else "final",
        "candidates": candidates,
        "problems": problems,
    }


def compile_claim_record(
    source: SourceBundle,
    opinions: Sequence[AgentOpinion],
    template: Mapping[str, Any],
    *,
    source_texts: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Compile one claim only when every evidence and support gate succeeds."""
    discovery = discover_claims(source, opinions, source_texts=source_texts)
    problems = list(discovery["problems"])
    try:
        profile = get_profile(str(template.get("profile_id", "")))
    except ValueError as exc:
        problems.append(str(exc))
        profile = None
    finding_type = str(template.get("finding_type", ""))
    if profile is not None and finding_type != profile.finding_type:
        problems.append(
            f"finding_type {finding_type!r} does not match profile "
            f"{profile.finding_type!r}"
        )
    reducer_config: Dict[str, Any] = {}
    if profile is not None:
        try:
            reducer_config = dict(template.get("reducer_config", {}))
            validate_reducer_config(profile.profile_id, reducer_config)
            controls = template.get("controls")
            task = template.get("task")
            if not isinstance(controls, Mapping) or not isinstance(task, Mapping):
                raise ValueError("claim template needs controls and task objects")
            validate_expected_target(
                profile.profile_id, task.get("expected"), reducer_config
            )
            for control_name in ("positive", "negative"):
                control = controls.get(control_name)
                if not isinstance(control, Mapping):
                    raise ValueError(f"claim template needs {control_name} control")
                validate_expected_target(
                    profile.profile_id, control.get("expected"), reducer_config
                )
        except (TypeError, ValueError) as exc:
            problems.append(str(exc))
    if problems:
        return {
            "artifact": "stresskit_compilation_result",
            "schema_version": AUDIT_SCHEMA_VERSION,
            "publication_state": "abstain",
            "status": "abstain",
            "problems": problems,
            "source_bundle_digest": source.digest,
            "agent_opinion_digests": [opinion.digest for opinion in opinions],
        }

    assert profile is not None
    anchors = discovery["candidates"][0]["evidence_anchors"]
    locator = dict(template.get("claim_locator", anchors[0]))
    documents = {row["document_id"]: row for row in source.documents}
    document = documents.get(locator.get("document_id"))
    if document is None:
        return {
            "artifact": "stresskit_compilation_result",
            "schema_version": AUDIT_SCHEMA_VERSION,
            "publication_state": "abstain",
            "status": "abstain",
            "problems": ["claim locator does not identify a SourceBundle document"],
        }
    try:
        record = ClaimRecord(
            claim_id=str(template.get("claim_id", "")),
            statement=discovery["candidates"][0]["statement"],
            source_bundle_digest=source.digest,
            source_digest=str(document["source_digest"]),
            claim_locator=locator,
            finding_type=finding_type,
            profile_id=profile.profile_id,
            reducer={
                "name": profile.reducer_name,
                "version": "1",
                "implementation_digest": reducer_digest(profile.reducer_name),
                "config": reducer_config,
            },
            code_map=dict(template.get("code_map", {})),
            controls=dict(template.get("controls", {})),
            task=dict(template.get("task", {})),
            agent_opinion_digests=[opinion.digest for opinion in opinions],
            metadata={
                **dict(template.get("metadata", {})),
                "source_text_verification": {
                    "status": "verified",
                    "document_digests": {
                        document_id: sha256_bytes(text.encode("utf-8"))
                        for document_id, text in sorted(
                            (source_texts or {}).items()
                        )
                    },
                },
            },
        )
    except (TypeError, ValueError) as exc:
        return {
            "artifact": "stresskit_compilation_result",
            "schema_version": AUDIT_SCHEMA_VERSION,
            "publication_state": "abstain",
            "status": "abstain",
            "problems": [str(exc)],
        }
    if record.code_map["repository_digest"] not in {
            row["source_digest"] for row in source.documents}:
        return {
            "artifact": "stresskit_compilation_result",
            "schema_version": AUDIT_SCHEMA_VERSION,
            "publication_state": "abstain",
            "status": "abstain",
            "problems": ["code_map repository_digest is absent from SourceBundle"],
        }
    return {
        "artifact": "stresskit_compilation_result",
        "schema_version": AUDIT_SCHEMA_VERSION,
        "publication_state": "final",
        "status": "compiled",
        "problems": [],
        "claim_record": record.to_dict(),
        "claim_record_digest": record.digest,
    }


def _weighted_counts(rows: Sequence[Mapping[str, Any]], total: int) -> Dict[str, int]:
    exact = [(str(row["specification_id"]), total * float(row["weight"]))
             for row in rows]
    counts = {name: int(math.floor(value)) for name, value in exact}
    remaining = total - sum(counts.values())
    order = sorted(exact, key=lambda row: (-(row[1] - math.floor(row[1])), row[0]))
    for name, _ in order[:remaining]:
        counts[name] += 1
    return counts


def _validate_joint_distribution(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("design needs a non-empty joint_distribution")
    identifiers = [row.get("specification_id") for row in rows]
    if not all(isinstance(value, str) and value.strip() for value in identifiers):
        raise ValueError("every joint specification needs specification_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("joint specification IDs must be unique")
    weights = [float(row.get("weight", 0.0)) for row in rows]
    if not all(math.isfinite(value) and value > 0 for value in weights):
        raise ValueError("joint specification weights must be finite and positive")
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("joint specification weights must sum to exactly 1 within 1e-12")
    if not all(isinstance(row.get("values"), Mapping) for row in rows):
        raise ValueError("every joint specification needs values object")


_EVALUATION_AXES = ("dataset", "model", "prompt", "unit")
_EVALUATION_PARTITIONS = {
    "primary": "primary",
    "positive_control": "primary",
    "negative_control": "primary",
    "generalization": "generalization",
}


def _validate_evaluation_design(
    design: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Validate explicit, semantically disjoint evaluation manifests."""
    held_out_axes = design.get("held_out_axes")
    if not isinstance(held_out_axes, list) or not held_out_axes or any(
            axis not in _EVALUATION_AXES for axis in held_out_axes):
        raise ValueError(
            "design held_out_axes must name dataset, model, prompt, or unit"
        )
    expected_order = [axis for axis in _EVALUATION_AXES if axis in held_out_axes]
    if held_out_axes != expected_order:
        raise ValueError("design held_out_axes must be unique and canonically ordered")
    raw_manifests = design.get("evaluation_manifests")
    if not isinstance(raw_manifests, Mapping) or set(raw_manifests) != {
            "primary", "generalization"}:
        raise ValueError(
            "design needs exact primary and generalization evaluation_manifests"
        )
    manifests: Dict[str, Dict[str, Any]] = {}
    evaluation_ids = set()
    axis_names: Optional[set] = None
    for partition in ("primary", "generalization"):
        raw = raw_manifests[partition]
        if not isinstance(raw, Mapping) or set(raw) != {
                "evaluation_id", "axis_ids", "manifest_digest"}:
            raise ValueError(
                f"{partition} evaluation manifest needs evaluation_id, axis_ids, "
                "and manifest_digest"
            )
        evaluation_id = raw.get("evaluation_id")
        if not isinstance(evaluation_id, str) or not evaluation_id.strip() or \
                evaluation_id in evaluation_ids:
            raise ValueError("evaluation manifest IDs must be non-empty and distinct")
        evaluation_ids.add(evaluation_id)
        raw_axes = raw.get("axis_ids")
        if not isinstance(raw_axes, Mapping) or not raw_axes or any(
                axis not in _EVALUATION_AXES for axis in raw_axes):
            raise ValueError("evaluation manifest axis_ids use unsupported axes")
        current_axis_names = set(raw_axes)
        if axis_names is None:
            axis_names = current_axis_names
        elif current_axis_names != axis_names:
            raise ValueError("evaluation manifests must bind identical axis names")
        if not set(held_out_axes) <= current_axis_names:
            raise ValueError("every held_out_axis must appear in both evaluation manifests")
        clean_axes: Dict[str, List[str]] = {}
        for axis in _EVALUATION_AXES:
            if axis not in raw_axes:
                continue
            values = raw_axes[axis]
            if not isinstance(values, list) or not values or any(
                    not isinstance(value, str) for value in values):
                raise ValueError(
                    f"evaluation axis {axis!r} needs content-addressed IDs"
                )
            for value in values:
                require_sha256_digest(value, f"evaluation axis {axis!r} ID")
            if values != sorted(set(values)):
                raise ValueError(
                    f"evaluation axis {axis!r} IDs must be unique and sorted"
                )
            clean_axes[axis] = list(values)
        digest_payload = {
            "evaluation_id": evaluation_id,
            "axis_ids": clean_axes,
        }
        expected_digest = digest_json(digest_payload)
        if raw.get("manifest_digest") != expected_digest:
            raise ValueError(f"{partition} evaluation manifest digest does not recompute")
        manifests[partition] = {
            **digest_payload,
            "manifest_digest": expected_digest,
        }
    primary_axes = manifests["primary"]["axis_ids"]
    generalization_axes = manifests["generalization"]["axis_ids"]
    for axis in primary_axes:
        primary_ids = set(primary_axes[axis])
        generalization_ids = set(generalization_axes[axis])
        if axis in held_out_axes:
            if primary_ids & generalization_ids:
                raise ValueError(
                    f"held-out evaluation axis {axis!r} overlaps primary IDs"
                )
        elif primary_ids != generalization_ids:
            raise ValueError(
                f"evaluation axis {axis!r} changes without held_out_axes registration"
            )
    return manifests


def regenerate_run_manifest(audit_id: str, design: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Regenerate every run slot from frozen joint design without outcomes."""
    rows = list(design.get("joint_distribution", []))
    for name in ("dependency_manifest_digest", "build_recipe_digest"):
        require_sha256_digest(design.get(name), f"design {name}")
    _validate_joint_distribution(rows)
    stored_distribution_digest = design.get("joint_distribution_digest")
    distribution_digest = digest_json(rows)
    if stored_distribution_digest != distribution_digest:
        raise ValueError("joint_distribution_digest does not recompute")
    runs = design.get("runs_per_partition")
    if not isinstance(runs, Mapping):
        raise ValueError("design needs runs_per_partition object")
    required_partitions = (
        "primary", "positive_control", "negative_control", "generalization"
    )
    if set(runs) != set(required_partitions):
        raise ValueError(
            "runs_per_partition must contain primary, positive_control, "
            "negative_control, and generalization"
        )
    cohorts = design.get("cohorts")
    if cohorts != ["final", "replication"]:
        raise ValueError("design cohorts must be ['final', 'replication']")
    base_seed = design.get("seed")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("design seed must be a non-negative integer")
    if not isinstance(design.get("independent_unit"), str) or \
            not design["independent_unit"].strip():
        raise ValueError("design needs named independent_unit")
    evaluation_manifests = _validate_evaluation_design(design)
    held_out_axes = list(design["held_out_axes"])
    by_id = {str(row["specification_id"]): row for row in rows}
    manifest: List[Dict[str, Any]] = []
    final_slots: Dict[Tuple[str, int], str] = {}
    for cohort in cohorts:
        for partition in required_partitions:
            evaluation_partition = _EVALUATION_PARTITIONS[partition]
            evaluation = evaluation_manifests[evaluation_partition]
            count = runs[partition]
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(f"runs_per_partition.{partition} must be positive integer")
            allocation = _weighted_counts(rows, count)
            schedule = []
            for specification_id in sorted(allocation):
                schedule.extend([specification_id] * allocation[specification_id])
            schedule.sort(key=lambda specification_id: digest_json({
                "audit_id": audit_id,
                "partition": partition,
                "specification_id": specification_id,
                "evaluation_id": evaluation["evaluation_id"],
                "evaluation_manifest_digest": evaluation["manifest_digest"],
                "seed": base_seed,
            }))
            for index, specification_id in enumerate(schedule):
                identity = {
                    "audit_id": audit_id,
                    "cohort": cohort,
                    "partition": partition,
                    "index": index,
                    "specification_id": specification_id,
                    "evaluation_id": evaluation["evaluation_id"],
                    "evaluation_manifest_digest": evaluation["manifest_digest"],
                }
                slot_id = "slot-" + digest_json(identity).split(":", 1)[1][:24]
                dependency_id = "dep-" + digest_json({
                    **identity, "kind": "independent-execution"
                }).split(":", 1)[1][:24]
                row: Dict[str, Any] = {
                    "slot_id": slot_id,
                    "cohort": cohort,
                    "partition": partition,
                    "index": index,
                    "run_seed": base_seed + index,
                    "specification_id": specification_id,
                    "specification": dict(by_id[specification_id]["values"]),
                    "sampling_weight": float(by_id[specification_id]["weight"]),
                    "evaluation_partition": evaluation_partition,
                    "evaluation_id": evaluation["evaluation_id"],
                    "evaluation_manifest_digest": evaluation["manifest_digest"],
                    "evaluation_axis_ids": dict(evaluation["axis_ids"]),
                    "held_out_axes": held_out_axes,
                    "dependency_id": dependency_id,
                    "cluster_id": dependency_id,
                }
                key = (partition, index)
                if cohort == "final":
                    final_slots[key] = slot_id
                else:
                    row["replicate_of"] = final_slots[key]
                manifest.append(row)
    return manifest


def freeze_audit_spec(
    claim: ClaimRecord,
    design: Mapping[str, Any],
    *,
    audit_id: str,
    frozen_at: Optional[str] = None,
    multiplicity_family: Mapping[str, Any],
    reproducibility: Mapping[str, Any],
    external_validation: str = "not obtained",
) -> AuditSpec:
    """Freeze claim, profile, joint distribution, manifest, and stopping rule."""
    profile = get_profile(claim.profile_id)
    if claim.finding_type != profile.finding_type:
        raise ValueError("ClaimRecord finding_type does not match threshold profile")
    if claim.reducer.get("name") != profile.reducer_name or \
            claim.reducer.get("implementation_digest") != reducer_digest(profile.reducer_name):
        raise ValueError("ClaimRecord reducer does not match frozen profile reducer")
    config = claim.reducer.get("config", {})
    validate_reducer_config(profile.profile_id, config)
    validate_expected_target(
        profile.profile_id, claim.task.get("expected"), config
    )
    for control_name in ("positive", "negative"):
        validate_expected_target(
            profile.profile_id,
            claim.controls.get(control_name, {}).get("expected"),
            config,
        )
    clean_design = dict(design)
    for name in ("dependency_manifest_digest", "build_recipe_digest"):
        frozen_value = claim.code_map[name]
        if name in clean_design and clean_design[name] != frozen_value:
            raise ValueError(f"design {name} differs from ClaimRecord code_map")
        clean_design[name] = frozen_value
    distribution = list(clean_design.get("joint_distribution", []))
    _validate_joint_distribution(distribution)
    clean_design["joint_distribution"] = distribution
    clean_design["joint_distribution_digest"] = digest_json(distribution)
    clean_design["evaluation_manifests"] = _validate_evaluation_design(clean_design)
    members = multiplicity_family.get("member_claim_ids")
    if not isinstance(members, list) or claim.claim_id not in members or \
            len(set(members)) != len(members):
        raise ValueError(
            "multiplicity family needs unique member_claim_ids including this claim"
        )
    clean_family = dict(multiplicity_family)
    if clean_family.get("method") != "holm-bonferroni":
        raise ValueError("v1 multiplicity method must be holm-bonferroni")
    if float(clean_family.get("alpha", -1)) != profile.alpha:
        raise ValueError("multiplicity alpha must match registered profile alpha")
    expected_release_digest = digest_json(sorted(members))
    if clean_family.get("release_manifest_digest") != expected_release_digest:
        raise ValueError("multiplicity release_manifest_digest does not recompute")
    primary_check_names = [
        "stability", "claim_support", "positive_control",
        "negative_falsification", "negative_control_truth", "specificity",
        "generalization_support", "generalization_stability",
    ]
    if claim.finding_type == "utility" or claim.task.get("utility_required", True) is True:
        primary_check_names.append("utility")
    supplied_names = clean_family.get("primary_check_names")
    if supplied_names is not None and supplied_names != primary_check_names:
        raise ValueError("multiplicity primary_check_names do not match ClaimRecord")
    clean_family["primary_check_names"] = primary_check_names
    if reproducibility.get("hardware_class") != clean_design.get("hardware_class"):
        raise ValueError("reproducibility hardware_class must match design")
    if reproducibility.get("level") == "numeric_with_tolerance":
        tolerance = reproducibility.get("tolerance")
        if not isinstance(tolerance, (int, float)) or not 0 <= tolerance < 1:
            raise ValueError("numeric reproducibility needs tolerance in [0, 1)")
    manifest = regenerate_run_manifest(audit_id, clean_design)
    stopping_rule = {
        "type": "fixed",
        "total_run_slots": len(manifest),
        "runs_per_partition": dict(clean_design["runs_per_partition"]),
        "outcome_blind": True,
    }
    return AuditSpec(
        audit_id=audit_id,
        claim_record=claim.to_dict(),
        claim_record_digest=claim.digest,
        profile_id=profile.profile_id,
        profile_digest=profile.digest,
        design=clean_design,
        run_manifest=manifest,
        manifest_digest=digest_json(manifest),
        stopping_rule=stopping_rule,
        multiplicity_family=clean_family,
        reproducibility=dict(reproducibility),
        frozen_at=frozen_at or utc_now(),
        external_validation=external_validation,
    )


def make_resource_plan(
    spec: AuditSpec,
    resources: Mapping[str, Any],
    *,
    key: bytes,
    key_id: str,
    signing_algorithm: str = "hmac-sha256",
    created_at: Optional[str] = None,
    allowed_outputs: Sequence[str] = ("raw_output.json", "stderr.txt", "attestation.json"),
) -> ResourcePlan:
    """Create and sign a plan that separates networked build from isolated run."""
    clean_resources = dict(resources)
    clean_resources["run_slots"] = len(spec.run_manifest)
    claim = ClaimRecord.from_dict(spec.claim_record)
    baseline_registry = claim.reducer.get("config", {}).get(
        "baseline_registry", []
    )
    input_manifest_digests = sorted({
        str(row["input_manifest_digest"])
        for row in baseline_registry if isinstance(row, Mapping)
    })
    plan_id = "plan-" + digest_json({
        "audit_spec_digest": spec.digest,
        "resources": clean_resources,
    }).split(":", 1)[1][:24]
    sandbox = {
        "build": {
            "disposable": True,
            "network": "enabled",
            "credentials": "absent",
            "dependency_manifest_digest": spec.design["dependency_manifest_digest"],
            "build_recipe_digest": spec.design["build_recipe_digest"],
        },
        "execution": {
            "network": "disabled",
            "credentials": "absent",
            "inputs": "read_only",
            "scratch": "quota_limited",
            "outputs": "allowlisted",
            "input_manifest_digests": input_manifest_digests,
        },
    }
    plan = ResourcePlan(
        plan_id=plan_id,
        audit_spec_digest=spec.digest,
        hardware_class=str(spec.design.get("hardware_class", "")),
        resources=clean_resources,
        sandbox=sandbox,
        allowed_outputs=list(allowed_outputs),
        created_at=created_at or utc_now(),
    )
    return plan.signed(key, key_id, algorithm=signing_algorithm)
