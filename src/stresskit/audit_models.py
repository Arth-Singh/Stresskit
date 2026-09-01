"""Versioned public data model for autonomous claim audits.

Artifacts are plain dataclasses with deterministic JSON representations.  Old
Stability Cards remain separate and keep their original schema and grades.
"""

from __future__ import annotations

import json
import datetime as dt
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence

from .integrity import (
    ContentRef,
    digest_json,
    require_sha256_digest,
    sign_mapping,
    unsigned_payload,
)


AUDIT_SCHEMA_VERSION = "1.0"
PUBLICATION_STATES = ("final", "abstain")
AUDIT_STATUSES = (
    "pass",
    "audit_failure",
    "reproduction_failure",
    "inconclusive",
    "protocol_deviation",
    "excluded",
    "abstain",
)
RUN_STATUSES = ("success", "failed", "crashed", "timed_out", "missing")
REPRODUCIBILITY_LEVELS = ("bitwise", "numeric_with_tolerance", "statistical")


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _header(payload: Mapping[str, Any], artifact: str) -> None:
    if payload.get("artifact") != artifact:
        raise ValueError(f"expected artifact {artifact!r}, got {payload.get('artifact')!r}")
    if payload.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(
            f"{artifact} schema_version must be {AUDIT_SCHEMA_VERSION!r}"
        )


def _nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _timestamp(value: str, field_name: str) -> dt.datetime:
    _nonempty(value, field_name)
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO 8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include timezone")
    return parsed.astimezone(dt.timezone.utc)


@dataclass(frozen=True)
class SourceBundle:
    """Content-addressed papers, repositories, and supplementary sources."""

    bundle_id: str
    documents: Sequence[Mapping[str, Any]]
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    ARTIFACT = "stresskit_source_bundle"

    def __post_init__(self) -> None:
        _nonempty(self.bundle_id, "bundle_id")
        _timestamp(self.created_at, "created_at")
        if not self.documents:
            raise ValueError("SourceBundle needs at least one document")
        identifiers = set()
        for row in self.documents:
            if not isinstance(row, Mapping):
                raise ValueError("SourceBundle documents must be objects")
            document_id = row.get("document_id")
            _nonempty(document_id, "document_id")
            if document_id in identifiers:
                raise ValueError(f"duplicate SourceBundle document_id {document_id!r}")
            identifiers.add(document_id)
            _nonempty(row.get("locator"), "document locator")
            require_sha256_digest(row.get("source_digest"), "document source_digest")
            license_row = row.get("license")
            if not isinstance(license_row, Mapping):
                raise ValueError("SourceBundle document needs license object")
            if license_row.get("status") not in (
                "verified_compatible", "unresolved", "incompatible"
            ):
                raise ValueError("document license status is unsupported")
            _nonempty(license_row.get("identifier"), "document license identifier")
            require_sha256_digest(
                license_row.get("evidence_digest"), "document license evidence_digest"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return canonical artifact JSON."""
        return {
            "artifact": self.ARTIFACT,
            "schema_version": AUDIT_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "documents": _copy_json(list(self.documents)),
            "metadata": _copy_json(dict(self.metadata)),
        }

    @property
    def digest(self) -> str:
        """Return canonical artifact digest."""
        return digest_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceBundle":
        """Load and validate artifact JSON."""
        _header(payload, cls.ARTIFACT)
        return cls(
            bundle_id=str(payload.get("bundle_id", "")),
            documents=list(payload.get("documents", [])),
            created_at=str(payload.get("created_at", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class AgentOpinion:
    """One isolated extractor or critic output with exact provenance."""

    opinion_id: str
    role: str
    provider: str
    model: str
    model_family: str
    source_bundle_digest: str
    model_digest: str
    prompt_digest: str
    request_digest: str
    statement: str
    evidence_anchors: Sequence[Mapping[str, Any]]
    supported: bool
    prompt_injection_detected: bool = False
    issues: Sequence[str] = field(default_factory=tuple)

    ARTIFACT = "stresskit_agent_opinion"

    def __post_init__(self) -> None:
        _nonempty(self.opinion_id, "opinion_id")
        if self.role not in ("extractor", "critic"):
            raise ValueError("AgentOpinion role must be 'extractor' or 'critic'")
        for name, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("model_family", self.model_family),
            ("statement", self.statement),
        ):
            _nonempty(value, name)
        for name, value in (
            ("source_bundle_digest", self.source_bundle_digest),
            ("model_digest", self.model_digest),
            ("prompt_digest", self.prompt_digest),
            ("request_digest", self.request_digest),
        ):
            require_sha256_digest(value, name)
        if not self.evidence_anchors:
            raise ValueError("AgentOpinion needs at least one exact evidence anchor")
        for anchor in self.evidence_anchors:
            if not isinstance(anchor, Mapping):
                raise ValueError("evidence anchors must be objects")
            _nonempty(anchor.get("document_id"), "anchor document_id")
            _nonempty(anchor.get("locator"), "anchor locator")
            require_sha256_digest(anchor.get("quote_digest"), "anchor quote_digest")
            require_sha256_digest(anchor.get("source_digest"), "anchor source_digest")
            require_sha256_digest(anchor.get("text_digest"), "anchor text_digest")
            start, end = anchor.get("start"), anchor.get("end")
            if not isinstance(start, int) or isinstance(start, bool) or \
                    not isinstance(end, int) or isinstance(end, bool) or \
                    start < 0 or end <= start:
                raise ValueError("evidence anchor needs a non-empty byte range")

    def to_dict(self) -> Dict[str, Any]:
        """Return canonical artifact JSON."""
        return {
            "artifact": self.ARTIFACT,
            "schema_version": AUDIT_SCHEMA_VERSION,
            "opinion_id": self.opinion_id,
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "model_family": self.model_family,
            "source_bundle_digest": self.source_bundle_digest,
            "model_digest": self.model_digest,
            "prompt_digest": self.prompt_digest,
            "request_digest": self.request_digest,
            "statement": self.statement,
            "evidence_anchors": _copy_json(list(self.evidence_anchors)),
            "supported": self.supported,
            "prompt_injection_detected": self.prompt_injection_detected,
            "issues": list(self.issues),
        }

    @property
    def digest(self) -> str:
        """Return canonical opinion digest."""
        return digest_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentOpinion":
        """Load and validate artifact JSON."""
        _header(payload, cls.ARTIFACT)
        return cls(
            opinion_id=str(payload.get("opinion_id", "")),
            role=str(payload.get("role", "")),
            provider=str(payload.get("provider", "")),
            model=str(payload.get("model", "")),
            model_family=str(payload.get("model_family", "")),
            source_bundle_digest=str(payload.get("source_bundle_digest", "")),
            model_digest=str(payload.get("model_digest", "")),
            prompt_digest=str(payload.get("prompt_digest", "")),
            request_digest=str(payload.get("request_digest", "")),
            statement=str(payload.get("statement", "")),
            evidence_anchors=list(payload.get("evidence_anchors", [])),
            supported=payload.get("supported") is True,
            prompt_injection_detected=payload.get("prompt_injection_detected") is True,
            issues=list(payload.get("issues", [])),
        )


@dataclass(frozen=True)
class ClaimRecord:
    """Falsifiable claim bound to source, reducer, controls, and agent panel."""

    claim_id: str
    statement: str
    source_bundle_digest: str
    source_digest: str
    claim_locator: Mapping[str, Any]
    finding_type: str
    profile_id: str
    reducer: Mapping[str, Any]
    code_map: Mapping[str, Any]
    controls: Mapping[str, Any]
    task: Mapping[str, Any]
    agent_opinion_digests: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    ARTIFACT = "stresskit_claim_record"

    def __post_init__(self) -> None:
        _nonempty(self.claim_id, "claim_id")
        _nonempty(self.statement, "statement")
        _nonempty(self.finding_type, "finding_type")
        _nonempty(self.profile_id, "profile_id")
        require_sha256_digest(self.source_bundle_digest, "source_bundle_digest")
        require_sha256_digest(self.source_digest, "source_digest")
        if not isinstance(self.claim_locator, Mapping):
            raise ValueError("claim_locator must be an object")
        _nonempty(self.claim_locator.get("document_id"), "claim locator document_id")
        _nonempty(self.claim_locator.get("locator"), "claim locator locator")
        require_sha256_digest(
            self.claim_locator.get("quote_digest"), "claim locator quote_digest"
        )
        start, end = self.claim_locator.get("start"), self.claim_locator.get("end")
        if not isinstance(start, int) or isinstance(start, bool) or \
                not isinstance(end, int) or isinstance(end, bool) or \
                start < 0 or end <= start:
            raise ValueError("claim locator needs a non-empty byte range")
        if not isinstance(self.reducer, Mapping):
            raise ValueError("reducer must be an object")
        _nonempty(self.reducer.get("name"), "reducer name")
        require_sha256_digest(
            self.reducer.get("implementation_digest"), "reducer implementation_digest"
        )
        if not isinstance(self.code_map, Mapping):
            raise ValueError("code_map must be an object")
        for name in (
            "repository_digest", "dependency_manifest_digest", "build_recipe_digest"
        ):
            require_sha256_digest(self.code_map.get(name), f"code_map {name}")
        _nonempty(self.code_map.get("revision"), "code_map revision")
        entrypoints = self.code_map.get("entrypoints")
        if not isinstance(entrypoints, list) or not entrypoints or \
                len(entrypoints) != len(set(entrypoints)):
            raise ValueError("code_map needs unique non-empty entrypoints")
        for value in entrypoints:
            if not isinstance(value, str) or not value or "\\" in value:
                raise ValueError("code_map entrypoints must be POSIX relative paths")
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("code_map entrypoints must stay inside repository")
        for control in ("positive", "negative"):
            row = self.controls.get(control) if isinstance(self.controls, Mapping) else None
            if not isinstance(row, Mapping):
                raise ValueError(f"ClaimRecord needs claim-specific {control} control")
            _nonempty(row.get("control_id"), f"{control} control_id")
            if "expected" not in row:
                raise ValueError(f"{positive_or_negative(control)} control needs expected target")
        if len(self.agent_opinion_digests) < 3:
            raise ValueError("ClaimRecord must bind two extractors and one critic")
        for index, value in enumerate(self.agent_opinion_digests):
            require_sha256_digest(value, f"agent_opinion_digests[{index}]")
        verification = self.metadata.get("source_text_verification") \
            if isinstance(self.metadata, Mapping) else None
        if not isinstance(verification, Mapping) or \
                verification.get("status") != "verified":
            raise ValueError("ClaimRecord needs verified source-text closure")
        document_digests = verification.get("document_digests")
        if not isinstance(document_digests, Mapping) or not document_digests:
            raise ValueError("ClaimRecord source-text closure must not be empty")
        for document_id, value in document_digests.items():
            _nonempty(document_id, "source-text document_id")
            require_sha256_digest(value, "source-text document digest")
        if self.claim_locator.get("document_id") not in document_digests:
            raise ValueError("ClaimRecord locator is absent from source-text closure")

    def to_dict(self) -> Dict[str, Any]:
        """Return canonical artifact JSON."""
        return {
            "artifact": self.ARTIFACT,
            "schema_version": AUDIT_SCHEMA_VERSION,
            "claim_id": self.claim_id,
            "statement": self.statement,
            "source_bundle_digest": self.source_bundle_digest,
            "source_digest": self.source_digest,
            "claim_locator": _copy_json(dict(self.claim_locator)),
            "finding_type": self.finding_type,
            "profile_id": self.profile_id,
            "reducer": _copy_json(dict(self.reducer)),
            "code_map": _copy_json(dict(self.code_map)),
            "controls": _copy_json(dict(self.controls)),
            "task": _copy_json(dict(self.task)),
            "agent_opinion_digests": list(self.agent_opinion_digests),
            "metadata": _copy_json(dict(self.metadata)),
        }

    @property
    def digest(self) -> str:
        """Return canonical claim-record digest."""
        return digest_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClaimRecord":
        """Load and validate artifact JSON."""
        _header(payload, cls.ARTIFACT)
        return cls(
            claim_id=str(payload.get("claim_id", "")),
            statement=str(payload.get("statement", "")),
            source_bundle_digest=str(payload.get("source_bundle_digest", "")),
            source_digest=str(payload.get("source_digest", "")),
            claim_locator=dict(payload.get("claim_locator", {})),
            finding_type=str(payload.get("finding_type", "")),
            profile_id=str(payload.get("profile_id", "")),
            reducer=dict(payload.get("reducer", {})),
            code_map=dict(payload.get("code_map", {})),
            controls=dict(payload.get("controls", {})),
            task=dict(payload.get("task", {})),
            agent_opinion_digests=list(payload.get("agent_opinion_digests", [])),
            metadata=dict(payload.get("metadata", {})),
        )


def positive_or_negative(value: str) -> str:
    """Keep control validation messages readable without accepting other names."""
    return value


@dataclass(frozen=True)
class AuditSpec:
    """Frozen joint audit design and complete deterministic run manifest."""

    audit_id: str
    claim_record: Mapping[str, Any]
    claim_record_digest: str
    profile_id: str
    profile_digest: str
    design: Mapping[str, Any]
    run_manifest: Sequence[Mapping[str, Any]]
    manifest_digest: str
    stopping_rule: Mapping[str, Any]
    multiplicity_family: Mapping[str, Any]
    reproducibility: Mapping[str, Any]
    frozen_at: str
    external_validation: str = "not obtained"

    ARTIFACT = "stresskit_audit_spec"

    def __post_init__(self) -> None:
        _nonempty(self.audit_id, "audit_id")
        _nonempty(self.profile_id, "profile_id")
        _timestamp(self.frozen_at, "frozen_at")
        require_sha256_digest(self.claim_record_digest, "claim_record_digest")
        require_sha256_digest(self.profile_digest, "profile_digest")
        require_sha256_digest(self.manifest_digest, "manifest_digest")
        if digest_json(self.claim_record) != self.claim_record_digest:
            raise ValueError("embedded ClaimRecord does not match claim_record_digest")
        if digest_json(list(self.run_manifest)) != self.manifest_digest:
            raise ValueError("run_manifest does not match manifest_digest")
        if self.stopping_rule.get("type") != "fixed":
            raise ValueError("v1 stopping_rule must be fixed before outcomes")
        _nonempty(self.multiplicity_family.get("family_id"), "multiplicity family_id")
        primary_names = self.multiplicity_family.get("primary_check_names")
        if not isinstance(primary_names, list) or not primary_names or \
                len(set(primary_names)) != len(primary_names) or \
                not all(isinstance(name, str) and name for name in primary_names):
            raise ValueError("multiplicity family needs unique primary_check_names")
        if self.reproducibility.get("level") not in REPRODUCIBILITY_LEVELS:
            raise ValueError(
                "reproducibility.level must be bitwise, numeric_with_tolerance, or statistical"
            )
        if self.external_validation != "not obtained":
            raise ValueError(
                "v1 AuditSpec external_validation must remain 'not obtained'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return canonical artifact JSON."""
        return {
            "artifact": self.ARTIFACT,
            "schema_version": AUDIT_SCHEMA_VERSION,
            "audit_id": self.audit_id,
            "claim_record": _copy_json(dict(self.claim_record)),
            "claim_record_digest": self.claim_record_digest,
            "profile_id": self.profile_id,
            "profile_digest": self.profile_digest,
            "design": _copy_json(dict(self.design)),
            "run_manifest": _copy_json(list(self.run_manifest)),
            "manifest_digest": self.manifest_digest,
            "stopping_rule": _copy_json(dict(self.stopping_rule)),
            "multiplicity_family": _copy_json(dict(self.multiplicity_family)),
            "reproducibility": _copy_json(dict(self.reproducibility)),
            "frozen_at": self.frozen_at,
            "external_validation": self.external_validation,
        }

    @cached_property
    def digest(self) -> str:
        """Return canonical frozen-spec digest."""
        return digest_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuditSpec":
        """Load and validate artifact JSON."""
        _header(payload, cls.ARTIFACT)
        return cls(
            audit_id=str(payload.get("audit_id", "")),
            claim_record=dict(payload.get("claim_record", {})),
            claim_record_digest=str(payload.get("claim_record_digest", "")),
            profile_id=str(payload.get("profile_id", "")),
            profile_digest=str(payload.get("profile_digest", "")),
            design=dict(payload.get("design", {})),
            run_manifest=list(payload.get("run_manifest", [])),
            manifest_digest=str(payload.get("manifest_digest", "")),
            stopping_rule=dict(payload.get("stopping_rule", {})),
            multiplicity_family=dict(payload.get("multiplicity_family", {})),
            reproducibility=dict(payload.get("reproducibility", {})),
            frozen_at=str(payload.get("frozen_at", "")),
            external_validation=str(payload.get("external_validation", "not obtained")),
        )


@dataclass(frozen=True)
class ResourcePlan:
    """Signed compute, time, storage, hardware, and isolation request."""

    plan_id: str
    audit_spec_digest: str
    hardware_class: str
    resources: Mapping[str, Any]
    sandbox: Mapping[str, Any]
    allowed_outputs: Sequence[str]
    created_at: str
    signature: Optional[Mapping[str, Any]] = None

    ARTIFACT = "stresskit_resource_plan"

    def __post_init__(self) -> None:
        _nonempty(self.plan_id, "plan_id")
        _nonempty(self.hardware_class, "hardware_class")
        _timestamp(self.created_at, "created_at")
        require_sha256_digest(self.audit_spec_digest, "audit_spec_digest")
        for key in ("gpu_count", "cpu_count", "wall_time_seconds", "storage_bytes"):
            if not isinstance(self.resources.get(key), int) or \
                    isinstance(self.resources[key], bool) or self.resources[key] < 0:
                raise ValueError(f"ResourcePlan resources.{key} must be a non-negative integer")
        if self.resources["cpu_count"] < 1 or \
                self.resources["wall_time_seconds"] < 1 or \
                self.resources["storage_bytes"] < 1:
            raise ValueError("ResourcePlan CPU, wall time, and storage must be positive")
        run_slots = self.resources.get("run_slots")
        if not isinstance(run_slots, int) or isinstance(run_slots, bool) or run_slots < 1:
            raise ValueError("ResourcePlan resources.run_slots must be a positive integer")
        if not self.allowed_outputs:
            raise ValueError("ResourcePlan needs an output allowlist")
        if len(set(self.allowed_outputs)) != len(self.allowed_outputs):
            raise ValueError("ResourcePlan output allowlist contains duplicates")
        for value in self.allowed_outputs:
            if not isinstance(value, str) or not value or "\\" in value:
                raise ValueError("ResourcePlan output paths must be non-empty POSIX paths")
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("ResourcePlan output paths must stay inside scratch output")

    def to_dict(self) -> Dict[str, Any]:
        """Return canonical artifact JSON."""
        payload = {
            "artifact": self.ARTIFACT,
            "schema_version": AUDIT_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "audit_spec_digest": self.audit_spec_digest,
            "hardware_class": self.hardware_class,
            "resources": _copy_json(dict(self.resources)),
            "sandbox": _copy_json(dict(self.sandbox)),
            "allowed_outputs": list(self.allowed_outputs),
            "created_at": self.created_at,
        }
        if self.signature is not None:
            payload["signature"] = _copy_json(dict(self.signature))
        return payload

    @property
    def digest(self) -> str:
        """Return digest of signed artifact when signed, otherwise unsigned artifact."""
        return digest_json(self.to_dict())

    @property
    def unsigned_digest(self) -> str:
        """Return digest of fields covered by signature."""
        return digest_json(unsigned_payload(self.to_dict()))

    def signed(
        self, key: bytes, key_id: str, *, algorithm: str = "hmac-sha256"
    ) -> "ResourcePlan":
        """Return a copy carrying configured canonical signature."""
        payload = self.to_dict()
        signature = sign_mapping(payload, key, key_id, algorithm=algorithm)
        return ResourcePlan(
            self.plan_id,
            self.audit_spec_digest,
            self.hardware_class,
            self.resources,
            self.sandbox,
            self.allowed_outputs,
            self.created_at,
            signature,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourcePlan":
        """Load and validate artifact JSON."""
        _header(payload, cls.ARTIFACT)
        return cls(
            plan_id=str(payload.get("plan_id", "")),
            audit_spec_digest=str(payload.get("audit_spec_digest", "")),
            hardware_class=str(payload.get("hardware_class", "")),
            resources=dict(payload.get("resources", {})),
            sandbox=dict(payload.get("sandbox", {})),
            allowed_outputs=list(payload.get("allowed_outputs", [])),
            created_at=str(payload.get("created_at", "")),
            signature=(dict(payload["signature"]) if "signature" in payload else None),
        )


@dataclass(frozen=True)
class RunAttestation:
    """Signed terminal state for one frozen run slot, including failures."""

    slot_id: str
    status: str
    audit_spec_digest: str
    resource_plan_digest: str
    executor_id: str
    hardware_class: str
    dependency_id: str
    cluster_id: str
    isolation: Mapping[str, Any]
    started_at: str
    finished_at: str
    output_digest: Optional[str] = None
    finding_digest: Optional[str] = None
    error_digest: Optional[str] = None
    signature: Optional[Mapping[str, Any]] = None

    ARTIFACT = "stresskit_run_attestation"

    def __post_init__(self) -> None:
        for name, value in (
            ("slot_id", self.slot_id),
            ("executor_id", self.executor_id),
            ("hardware_class", self.hardware_class),
            ("dependency_id", self.dependency_id),
            ("cluster_id", self.cluster_id),
            ("started_at", self.started_at),
            ("finished_at", self.finished_at),
        ):
            _nonempty(value, name)
        if self.status not in RUN_STATUSES:
            raise ValueError(f"RunAttestation status must be one of {RUN_STATUSES}")
        started = _timestamp(self.started_at, "started_at")
        finished = _timestamp(self.finished_at, "finished_at")
        if finished < started:
            raise ValueError("RunAttestation finished_at precedes started_at")
        require_sha256_digest(self.audit_spec_digest, "audit_spec_digest")
        require_sha256_digest(self.resource_plan_digest, "resource_plan_digest")
        if self.status == "success":
            require_sha256_digest(self.output_digest, "successful output_digest")
            require_sha256_digest(self.finding_digest, "successful finding_digest")
        elif self.output_digest is not None:
            require_sha256_digest(self.output_digest, "output_digest")
        if self.error_digest is not None:
            require_sha256_digest(self.error_digest, "error_digest")

    def to_dict(self) -> Dict[str, Any]:
        """Return canonical artifact JSON."""
        payload: Dict[str, Any] = {
            "artifact": self.ARTIFACT,
            "schema_version": AUDIT_SCHEMA_VERSION,
            "slot_id": self.slot_id,
            "status": self.status,
            "audit_spec_digest": self.audit_spec_digest,
            "resource_plan_digest": self.resource_plan_digest,
            "executor_id": self.executor_id,
            "hardware_class": self.hardware_class,
            "dependency_id": self.dependency_id,
            "cluster_id": self.cluster_id,
            "isolation": _copy_json(dict(self.isolation)),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        for key, value in (
            ("output_digest", self.output_digest),
            ("finding_digest", self.finding_digest),
            ("error_digest", self.error_digest),
        ):
            if value is not None:
                payload[key] = value
        if self.signature is not None:
            payload["signature"] = _copy_json(dict(self.signature))
        return payload

    @property
    def digest(self) -> str:
        """Return canonical signed-attestation digest."""
        return digest_json(self.to_dict())

    def signed(
        self, key: bytes, key_id: str, *, algorithm: str = "hmac-sha256"
    ) -> "RunAttestation":
        """Return a copy carrying configured canonical signature."""
        signature = sign_mapping(
            self.to_dict(), key, key_id, algorithm=algorithm
        )
        return RunAttestation(
            slot_id=self.slot_id,
            status=self.status,
            audit_spec_digest=self.audit_spec_digest,
            resource_plan_digest=self.resource_plan_digest,
            executor_id=self.executor_id,
            hardware_class=self.hardware_class,
            dependency_id=self.dependency_id,
            cluster_id=self.cluster_id,
            isolation=self.isolation,
            started_at=self.started_at,
            finished_at=self.finished_at,
            output_digest=self.output_digest,
            finding_digest=self.finding_digest,
            error_digest=self.error_digest,
            signature=signature,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunAttestation":
        """Load and validate artifact JSON."""
        _header(payload, cls.ARTIFACT)
        return cls(
            slot_id=str(payload.get("slot_id", "")),
            status=str(payload.get("status", "")),
            audit_spec_digest=str(payload.get("audit_spec_digest", "")),
            resource_plan_digest=str(payload.get("resource_plan_digest", "")),
            executor_id=str(payload.get("executor_id", "")),
            hardware_class=str(payload.get("hardware_class", "")),
            dependency_id=str(payload.get("dependency_id", "")),
            cluster_id=str(payload.get("cluster_id", "")),
            isolation=dict(payload.get("isolation", {})),
            started_at=str(payload.get("started_at", "")),
            finished_at=str(payload.get("finished_at", "")),
            output_digest=payload.get("output_digest"),
            finding_digest=payload.get("finding_digest"),
            error_digest=payload.get("error_digest"),
            signature=(dict(payload["signature"]) if "signature" in payload else None),
        )


@dataclass(frozen=True)
class AuditDecision:
    """Claim-level final status with distinct evidence axes and no paper score."""

    claim_id: str
    audit_id: str
    status: str
    publication_state: str
    reproduction: Mapping[str, Any]
    stability_specificity: Mapping[str, Any]
    utility: Mapping[str, Any]
    generalization: Mapping[str, Any]
    evidence_confidence: Mapping[str, Any]
    primary_checks: Mapping[str, Any]
    reasons: Sequence[str] = field(default_factory=tuple)
    external_validation: str = "not obtained"

    ARTIFACT = "stresskit_audit_decision"

    def __post_init__(self) -> None:
        _nonempty(self.claim_id, "claim_id")
        _nonempty(self.audit_id, "audit_id")
        if self.status not in AUDIT_STATUSES:
            raise ValueError(f"AuditDecision status must be one of {AUDIT_STATUSES}")
        if self.publication_state not in PUBLICATION_STATES:
            raise ValueError(
                f"publication_state must be one of {PUBLICATION_STATES}"
            )
        if self.status == "abstain" and self.publication_state != "abstain":
            raise ValueError("abstain status requires publication_state='abstain'")
        if self.status != "abstain" and self.publication_state != "final":
            raise ValueError("non-abstain status requires publication_state='final'")
        if self.external_validation != "not obtained":
            raise ValueError(
                "v1 AuditDecision external_validation must remain 'not obtained'"
            )
        for name, value in (
            ("reproduction", self.reproduction),
            ("stability_specificity", self.stability_specificity),
            ("utility", self.utility),
            ("generalization", self.generalization),
            ("evidence_confidence", self.evidence_confidence),
            ("primary_checks", self.primary_checks),
        ):
            if not isinstance(value, Mapping):
                raise ValueError(f"AuditDecision {name} must be an object")

    def to_dict(self) -> Dict[str, Any]:
        """Return canonical artifact JSON."""
        return {
            "artifact": self.ARTIFACT,
            "schema_version": AUDIT_SCHEMA_VERSION,
            "claim_id": self.claim_id,
            "audit_id": self.audit_id,
            "status": self.status,
            "publication_state": self.publication_state,
            "reproduction": _copy_json(dict(self.reproduction)),
            "stability_specificity": _copy_json(dict(self.stability_specificity)),
            "utility": _copy_json(dict(self.utility)),
            "generalization": _copy_json(dict(self.generalization)),
            "evidence_confidence": _copy_json(dict(self.evidence_confidence)),
            "primary_checks": _copy_json(dict(self.primary_checks)),
            "reasons": list(self.reasons),
            "external_validation": self.external_validation,
        }

    @property
    def digest(self) -> str:
        """Return canonical decision digest."""
        return digest_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuditDecision":
        """Load and validate artifact JSON."""
        _header(payload, cls.ARTIFACT)
        return cls(
            claim_id=str(payload.get("claim_id", "")),
            audit_id=str(payload.get("audit_id", "")),
            status=str(payload.get("status", "")),
            publication_state=str(payload.get("publication_state", "")),
            reproduction=dict(payload.get("reproduction", {})),
            stability_specificity=dict(payload.get("stability_specificity", {})),
            utility=dict(payload.get("utility", {})),
            generalization=dict(payload.get("generalization", {})),
            evidence_confidence=dict(payload.get("evidence_confidence", {})),
            primary_checks=dict(payload.get("primary_checks", {})),
            reasons=list(payload.get("reasons", [])),
            external_validation=str(payload.get("external_validation", "not obtained")),
        )


@dataclass(frozen=True)
class AuditBundle:
    """Complete digest closure for a frozen spec, plan, runs, and decision."""

    bundle_id: str
    audit_spec: Mapping[str, Any]
    audit_spec_digest: str
    resource_plan: Mapping[str, Any]
    resource_plan_digest: str
    attestations: Sequence[Mapping[str, Any]]
    content: Sequence[ContentRef]
    decision: Optional[Mapping[str, Any]]
    created_at: str

    ARTIFACT = "stresskit_audit_bundle"

    def __post_init__(self) -> None:
        _nonempty(self.bundle_id, "bundle_id")
        _timestamp(self.created_at, "created_at")
        require_sha256_digest(self.audit_spec_digest, "audit_spec_digest")
        require_sha256_digest(self.resource_plan_digest, "resource_plan_digest")
        if digest_json(self.audit_spec) != self.audit_spec_digest:
            raise ValueError("embedded AuditSpec does not match audit_spec_digest")
        if digest_json(self.resource_plan) != self.resource_plan_digest:
            raise ValueError("embedded ResourcePlan does not match resource_plan_digest")

    def to_dict(self) -> Dict[str, Any]:
        """Return canonical artifact JSON."""
        payload: Dict[str, Any] = {
            "artifact": self.ARTIFACT,
            "schema_version": AUDIT_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "audit_spec": _copy_json(dict(self.audit_spec)),
            "audit_spec_digest": self.audit_spec_digest,
            "resource_plan": _copy_json(dict(self.resource_plan)),
            "resource_plan_digest": self.resource_plan_digest,
            "attestations": _copy_json(list(self.attestations)),
            "content": [reference.to_dict() for reference in self.content],
            "created_at": self.created_at,
        }
        if self.decision is not None:
            payload["decision"] = _copy_json(dict(self.decision))
        return payload

    @property
    def digest(self) -> str:
        """Return canonical bundle digest."""
        return digest_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuditBundle":
        """Load and validate artifact JSON."""
        _header(payload, cls.ARTIFACT)
        return cls(
            bundle_id=str(payload.get("bundle_id", "")),
            audit_spec=dict(payload.get("audit_spec", {})),
            audit_spec_digest=str(payload.get("audit_spec_digest", "")),
            resource_plan=dict(payload.get("resource_plan", {})),
            resource_plan_digest=str(payload.get("resource_plan_digest", "")),
            attestations=list(payload.get("attestations", [])),
            content=[ContentRef.from_dict(row) for row in payload.get("content", [])],
            decision=(dict(payload["decision"]) if "decision" in payload else None),
            created_at=str(payload.get("created_at", "")),
        )
