"""Hosted state machine remains queue-free and workers poll outward."""

import json
from pathlib import Path

import pytest

import stresskit as sk
from stresskit.audit_models import ResourcePlan
from stresskit.audit_worker import ExecutorCapabilities
from stresskit.cli import build_parser
from stresskit.control_plane import (
    InMemoryAuditRepository,
    create_app,
    require_transition,
)
from stresskit.integrity import S3ContentAddressedStore, digest_json


class _Body:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload


class _S3Client:
    def __init__(self):
        self.objects = {}

    def put_object(self, *, Bucket, Key, Body, **_):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket, Key):
        return {"Body": _Body(self.objects[(Bucket, Key)])}


def _plan():
    return ResourcePlan(
        plan_id="plan-1",
        audit_spec_digest=digest_json({"spec": 1}),
        hardware_class="h100-80gb",
        resources={
            "gpu_count": 1,
            "cpu_count": 8,
            "wall_time_seconds": 3600,
            "storage_bytes": 10_000,
            "run_slots": 10,
        },
        sandbox={
            "execution": {
                "network": "disabled",
                "credentials": "absent",
                "inputs": "read_only",
                "scratch": "quota_limited",
                "outputs": "allowlisted",
                "input_manifest_digests": [],
            }
        },
        allowed_outputs=["raw_output.json"],
        created_at="2026-09-01T00:00:00+00:00",
    ).signed(b"key", "control")


def _capabilities(*, isolated=True):
    return ExecutorCapabilities(
        executor_id="worker-1",
        hardware_classes=["h100-80gb"],
        gpu_count=1,
        cpu_count=8,
        storage_bytes=10_000,
        network_namespace=isolated,
        credential_isolation=isolated,
        read_only_mounts=isolated,
        scratch_quota=isolated,
        output_allowlist=isolated,
    )


def test_postgres_style_state_machine_and_outbound_polling():
    repository = InMemoryAuditRepository()
    row = repository.create_intake("claim-1", {"curated": True})
    repository.transition(row.audit_id, "intake", "compiled")
    repository.transition(row.audit_id, "compiled", "frozen")
    repository.attach_resource_plan(row.audit_id, _plan().to_dict())
    assert repository.poll_plan(_capabilities(isolated=False)) is None
    leased = repository.poll_plan(_capabilities())
    assert leased is not None and leased.state == "leased"
    repository.submit_attestations(
        row.audit_id, "worker-1", [{"slot_id": "slot-1"}]
    )
    submitted = repository.finish_submission(row.audit_id, "worker-1")
    assert submitted.state == "submitted"
    assert [event["to"] for event in repository.events] == [
        "intake", "compiled", "frozen", "awaiting_executor",
        "leased", "running", "submitted",
    ]


def test_state_skips_and_terminal_mutations_are_rejected():
    with pytest.raises(ValueError, match="invalid audit transition"):
        require_transition("intake", "published")
    with pytest.raises(ValueError, match="terminal"):
        require_transition("published", "verified")


def test_public_v1_types_are_exported():
    for name in (
        "SourceBundle", "ClaimRecord", "AgentOpinion", "AuditSpec",
        "ResourcePlan", "RunAttestation", "AuditBundle", "AuditDecision",
    ):
        assert getattr(sk, name).__name__ == name


def test_all_versioned_schemas_are_valid_json():
    schema_root = Path(sk.__file__).parent / "schemas"
    expected = {
        "source_bundle_v1.json", "agent_opinion_v1.json", "claim_record_v1.json",
        "audit_spec_v1.json", "resource_plan_v1.json", "run_attestation_v1.json",
        "audit_bundle_v1.json", "audit_decision_v1.json", "audit_v1.json",
    }
    assert expected <= {path.name for path in schema_root.glob("*_v1.json")}
    for name in expected:
        assert json.loads((schema_root / name).read_text())["$schema"].startswith("https://")


def test_nested_audit_cli_has_complete_lifecycle():
    parser = build_parser()
    for command in (
        "source", "opinion", "discover", "compile", "freeze", "plan", "run", "verify",
        "publish",
    ):
        with pytest.raises(SystemExit) as stopped:
            parser.parse_args(["audit", command, "--help"])
        assert stopped.value.code == 0


def test_s3_store_supports_same_verified_closure_operations_as_local_cas():
    store = S3ContentAddressedStore("audit-bucket", client=_S3Client())
    reference = store.put_json({"raw": [1, 2, 3]}, role="raw_output")
    assert store.get_json(reference.digest) == {"raw": [1, 2, 3]}
    store.verify_refs([reference])


def test_fastapi_intake_is_admin_only_when_optional_stack_is_installed():
    pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")
    app = create_app(
        repository=InMemoryAuditRepository(),
        admin_token="admin-secret",
        worker_token="worker-secret",
    )
    client = testclient.TestClient(app)
    assert client.post("/v1/admin/intake", json={"claim_id": "claim-1"}).status_code == 401
    response = client.post(
        "/v1/admin/intake",
        json={"claim_id": "claim-1", "curated": True},
        headers={"Authorization": "Bearer admin-secret"},
    )
    assert response.status_code == 200
    assert response.json()["state"] == "intake"
