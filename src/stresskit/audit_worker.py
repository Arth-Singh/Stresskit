"""Outbound-only executor protocol and deterministic run attestation helpers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .audit_compile import utc_now
from .audit_models import (
    AuditBundle,
    AuditDecision,
    AuditSpec,
    ClaimRecord,
    ResourcePlan,
    RunAttestation,
)
from .audit_profiles import reduce_raw_output
from .integrity import ContentAddressedStore, ContentRef, digest_json


@dataclass(frozen=True)
class ExecutorCapabilities:
    """Worker hardware and isolation capabilities advertised while polling."""

    executor_id: str
    hardware_classes: Sequence[str]
    gpu_count: int
    cpu_count: int
    storage_bytes: int
    network_namespace: bool
    credential_isolation: bool
    read_only_mounts: bool
    scratch_quota: bool
    output_allowlist: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return public capability JSON."""
        return {
            "executor_id": self.executor_id,
            "hardware_classes": list(self.hardware_classes),
            "gpu_count": self.gpu_count,
            "cpu_count": self.cpu_count,
            "storage_bytes": self.storage_bytes,
            "isolation": {
                "network_namespace": self.network_namespace,
                "credential_isolation": self.credential_isolation,
                "read_only_mounts": self.read_only_mounts,
                "scratch_quota": self.scratch_quota,
                "output_allowlist": self.output_allowlist,
            },
        }


def compatible_executor(
    plan: ResourcePlan, capabilities: ExecutorCapabilities
) -> Tuple[bool, Sequence[str]]:
    """Check signed plan resources and required isolation before assignment."""
    problems = []
    if plan.hardware_class not in capabilities.hardware_classes:
        problems.append("hardware class unavailable")
    if capabilities.gpu_count < int(plan.resources["gpu_count"]):
        problems.append("GPU count below plan")
    if capabilities.cpu_count < int(plan.resources["cpu_count"]):
        problems.append("CPU count below plan")
    if capabilities.storage_bytes < int(plan.resources["storage_bytes"]):
        problems.append("storage below plan")
    for name, present in (
        ("network namespace", capabilities.network_namespace),
        ("credential isolation", capabilities.credential_isolation),
        ("read-only mounts", capabilities.read_only_mounts),
        ("scratch quota", capabilities.scratch_quota),
        ("output allowlist", capabilities.output_allowlist),
    ):
        if not present:
            problems.append(f"missing {name}")
    return not problems, problems


def isolation_attestation(
    execution_environment_id: str,
    input_manifest_digests: Sequence[str] = (),
) -> Dict[str, Any]:
    """Return required execution-isolation statement for one disposable sandbox."""
    if not execution_environment_id.strip():
        raise ValueError("execution_environment_id must not be empty")
    return {
        "network": "disabled",
        "credentials": "absent",
        "inputs": "read_only",
        "scratch": "quota_limited",
        "outputs": "allowlisted",
        "execution_environment_id": execution_environment_id,
        "input_manifest_digests": sorted(set(input_manifest_digests)),
    }


def _utility_input_manifest_digests(spec: AuditSpec) -> List[str]:
    claim = ClaimRecord.from_dict(spec.claim_record)
    registry = claim.reducer.get("config", {}).get("baseline_registry", [])
    return sorted({
        str(row["input_manifest_digest"])
        for row in registry if isinstance(row, Mapping)
    })


def attest_success(
    spec: AuditSpec,
    plan: ResourcePlan,
    slot: Mapping[str, Any],
    raw_output: Mapping[str, Any],
    store: ContentAddressedStore,
    *,
    executor_id: str,
    execution_environment_id: str,
    key: bytes,
    key_id: str,
    signing_algorithm: str = "hmac-sha256",
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
) -> Tuple[RunAttestation, ContentRef]:
    """Reduce, store, and sign one successful frozen slot."""
    claim = ClaimRecord.from_dict(spec.claim_record)
    finding = reduce_raw_output(
        spec.profile_id, raw_output,
        reducer_config=claim.reducer.get("config", {}),
    )
    output_ref = store.put_json(raw_output, role="raw_output")
    attestation = RunAttestation(
        slot_id=str(slot["slot_id"]),
        status="success",
        audit_spec_digest=spec.digest,
        resource_plan_digest=plan.digest,
        executor_id=executor_id,
        hardware_class=plan.hardware_class,
        dependency_id=str(slot["dependency_id"]),
        cluster_id=str(slot["cluster_id"]),
        isolation=isolation_attestation(
            execution_environment_id, _utility_input_manifest_digests(spec)
        ),
        started_at=started_at or utc_now(),
        finished_at=finished_at or utc_now(),
        output_digest=output_ref.digest,
        finding_digest=digest_json(finding),
    ).signed(key, key_id, algorithm=signing_algorithm)
    return attestation, output_ref


def attest_failure(
    spec: AuditSpec,
    plan: ResourcePlan,
    slot: Mapping[str, Any],
    status: str,
    error: Mapping[str, Any],
    store: ContentAddressedStore,
    *,
    executor_id: str,
    execution_environment_id: str,
    key: bytes,
    key_id: str,
    signing_algorithm: str = "hmac-sha256",
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
) -> Tuple[RunAttestation, ContentRef]:
    """Store and sign a failed, crashed, timed-out, or missing slot."""
    if status not in ("failed", "crashed", "timed_out", "missing"):
        raise ValueError("failure status must be failed, crashed, timed_out, or missing")
    error_ref = store.put_json(error, role="run_error")
    attestation = RunAttestation(
        slot_id=str(slot["slot_id"]),
        status=status,
        audit_spec_digest=spec.digest,
        resource_plan_digest=plan.digest,
        executor_id=executor_id,
        hardware_class=plan.hardware_class,
        dependency_id=str(slot["dependency_id"]),
        cluster_id=str(slot["cluster_id"]),
        isolation=isolation_attestation(
            execution_environment_id, _utility_input_manifest_digests(spec)
        ),
        started_at=started_at or utc_now(),
        finished_at=finished_at or utc_now(),
        error_digest=error_ref.digest,
    ).signed(key, key_id, algorithm=signing_algorithm)
    return attestation, error_ref


def assemble_audit_bundle(
    spec: AuditSpec,
    plan: ResourcePlan,
    attestations: Sequence[RunAttestation],
    content: Sequence[ContentRef],
    *,
    created_at: Optional[str] = None,
    decision: Optional[AuditDecision] = None,
) -> AuditBundle:
    """Assemble deterministic complete-closure bundle without hiding run states."""
    by_digest: Dict[str, ContentRef] = {}
    for reference in content:
        previous = by_digest.get(reference.digest)
        if previous is not None and previous != reference:
            raise ValueError(f"conflicting content references for {reference.digest}")
        by_digest[reference.digest] = reference
    bundle_id = "bundle-" + digest_json({
        "audit_spec_digest": spec.digest,
        "resource_plan_digest": plan.digest,
        "attestation_digests": sorted(row.digest for row in attestations),
    }).split(":", 1)[1][:24]
    return AuditBundle(
        bundle_id=bundle_id,
        audit_spec=spec.to_dict(),
        audit_spec_digest=spec.digest,
        resource_plan=plan.to_dict(),
        resource_plan_digest=plan.digest,
        attestations=[row.to_dict() for row in attestations],
        content=[by_digest[key] for key in sorted(by_digest)],
        decision=decision.to_dict() if decision is not None else None,
        created_at=created_at or utc_now(),
    )


def poll_control_plane(
    base_url: str,
    capabilities: ExecutorCapabilities,
    *,
    worker_token: str,
    timeout_seconds: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Poll outward for one compatible plan; worker exposes no inbound socket."""
    body = json.dumps(capabilities.to_dict(), separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/workers/plans/poll",
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer " + worker_token,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status == 204:
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 204:
            return None
        raise RuntimeError(f"control-plane poll failed with HTTP {exc.code}") from exc
    if not isinstance(payload, dict):
        raise ValueError("control-plane poll response must be a JSON object")
    return payload
