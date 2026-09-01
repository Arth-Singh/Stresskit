"""Optional FastAPI/PostgreSQL control plane with outbound worker polling.

Core StressKit never imports FastAPI, psycopg, or boto3 at module import time.
No Redis or Celery queue is used; PostgreSQL row locking owns lease state.
"""

from __future__ import annotations

import json
import hmac
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .audit_models import ResourcePlan
from .audit_worker import ExecutorCapabilities, compatible_executor


TERMINAL_STATES = {
    "published", "excluded", "abstain", "protocol_deviation",
    "reproduction_failure",
}

STATE_TRANSITIONS = {
    "intake": {"compiled", "excluded", "abstain"},
    "compiled": {"frozen", "abstain"},
    "frozen": {"awaiting_executor", "abstain"},
    "awaiting_executor": {"leased", "abstain"},
    "leased": {"running", "awaiting_executor", "abstain"},
    "running": {"submitted", "reproduction_failure", "abstain"},
    "submitted": {"verified", "protocol_deviation", "abstain"},
    "verified": {"published", "protocol_deviation"},
}


def require_transition(current: str, target: str) -> None:
    """Reject state skips and mutations out of terminal states."""
    if current in TERMINAL_STATES:
        raise ValueError(f"terminal audit state {current!r} cannot transition")
    if target not in STATE_TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid audit transition {current!r} -> {target!r}")


@dataclass
class AuditRow:
    """Control-plane state row used by in-memory tests and local deployments."""

    audit_id: str
    claim_id: str
    state: str
    payload: Dict[str, Any]
    resource_plan: Optional[Dict[str, Any]] = None
    lease_owner: Optional[str] = None
    attestations: List[Dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.attestations is None:
            self.attestations = []

    def to_dict(self) -> Dict[str, Any]:
        """Return API representation."""
        return {
            "audit_id": self.audit_id,
            "claim_id": self.claim_id,
            "state": self.state,
            "payload": self.payload,
            "resource_plan": self.resource_plan,
            "lease_owner": self.lease_owner,
            "attestations": self.attestations,
        }


class InMemoryAuditRepository:
    """Thread-safe reference repository matching PostgreSQL state semantics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: Dict[str, AuditRow] = {}
        self._events: List[Dict[str, Any]] = []

    def create_intake(self, claim_id: str, payload: Mapping[str, Any]) -> AuditRow:
        """Create admin-curated intake; public callers receive no endpoint."""
        if not claim_id.strip():
            raise ValueError("claim_id must not be empty")
        with self._lock:
            audit_id = "audit-" + uuid.uuid4().hex
            row = AuditRow(audit_id, claim_id, "intake", dict(payload))
            self._rows[audit_id] = row
            self._events.append({"audit_id": audit_id, "from": None, "to": "intake"})
            return row

    def get(self, audit_id: str) -> AuditRow:
        """Return one row or raise KeyError."""
        with self._lock:
            return self._rows[audit_id]

    def transition(
        self, audit_id: str, expected_state: str, target_state: str,
        payload_patch: Optional[Mapping[str, Any]] = None,
    ) -> AuditRow:
        """Compare-and-swap one audited state transition."""
        with self._lock:
            row = self._rows[audit_id]
            if row.state != expected_state:
                raise ValueError(
                    f"state conflict: expected {expected_state!r}, found {row.state!r}"
                )
            require_transition(row.state, target_state)
            previous = row.state
            row.state = target_state
            if payload_patch:
                row.payload.update(dict(payload_patch))
            self._events.append({"audit_id": audit_id, "from": previous,
                                 "to": target_state})
            return row

    def attach_resource_plan(
        self, audit_id: str, plan: Mapping[str, Any]
    ) -> AuditRow:
        """Attach signed plan and pause audit for compatible executor."""
        parsed = ResourcePlan.from_dict(plan)
        with self._lock:
            row = self._rows[audit_id]
            if row.state != "frozen":
                raise ValueError("resource plan can attach only to frozen audit")
            require_transition(row.state, "awaiting_executor")
            row.resource_plan = parsed.to_dict()
            row.state = "awaiting_executor"
            self._events.append({"audit_id": audit_id, "from": "frozen",
                                 "to": "awaiting_executor"})
            return row

    def poll_plan(self, capabilities: ExecutorCapabilities) -> Optional[AuditRow]:
        """Lease first compatible waiting plan to outbound-polling worker."""
        with self._lock:
            for audit_id in sorted(self._rows):
                row = self._rows[audit_id]
                if row.state != "awaiting_executor" or row.resource_plan is None:
                    continue
                plan = ResourcePlan.from_dict(row.resource_plan)
                compatible, _ = compatible_executor(plan, capabilities)
                if not compatible:
                    continue
                require_transition(row.state, "leased")
                row.state = "leased"
                row.lease_owner = capabilities.executor_id
                self._events.append({"audit_id": audit_id,
                                     "from": "awaiting_executor", "to": "leased"})
                return row
        return None

    def submit_attestations(
        self, audit_id: str, executor_id: str,
        attestations: Sequence[Mapping[str, Any]],
    ) -> AuditRow:
        """Accept terminal slot attestations only from current lease owner."""
        with self._lock:
            row = self._rows[audit_id]
            if row.lease_owner != executor_id:
                raise ValueError("executor does not own audit lease")
            if row.state not in ("leased", "running"):
                raise ValueError("audit is not accepting worker attestations")
            if row.state == "leased":
                require_transition("leased", "running")
                row.state = "running"
                self._events.append({"audit_id": audit_id, "from": "leased",
                                     "to": "running"})
            row.attestations.extend(dict(value) for value in attestations)
            return row

    def finish_submission(self, audit_id: str, executor_id: str) -> AuditRow:
        """Close worker submission without deciding scientific outcome."""
        with self._lock:
            row = self._rows[audit_id]
            if row.lease_owner != executor_id or row.state != "running":
                raise ValueError("worker cannot finish this audit")
            require_transition("running", "submitted")
            row.state = "submitted"
            self._events.append({"audit_id": audit_id, "from": "running",
                                 "to": "submitted"})
            return row

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Return append-only transition log copy."""
        with self._lock:
            return [dict(event) for event in self._events]


class PostgresAuditRepository:
    """PostgreSQL state repository using transactions and SKIP LOCKED leases."""

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("PostgreSQL DSN must not be empty")
        try:
            import psycopg  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PostgreSQL control plane needs: pip install stress-kit[control]"
            ) from exc
        self.psycopg = psycopg
        self.dsn = dsn

    def create_intake(self, claim_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Insert one admin-curated intake and append its first event."""
        audit_id = "audit-" + uuid.uuid4().hex
        with self.psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO audits (audit_id, claim_id, state, payload) "
                    "VALUES (%s, %s, 'intake', %s::jsonb)",
                    (audit_id, claim_id, json.dumps(payload)),
                )
                cursor.execute(
                    "INSERT INTO audit_events (audit_id, from_state, to_state) "
                    "VALUES (%s, NULL, 'intake')", (audit_id,),
                )
        return {"audit_id": audit_id, "claim_id": claim_id, "state": "intake"}

    def transition(
        self, audit_id: str, expected_state: str, target_state: str,
        payload_patch: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Lock, validate, and update one state with append-only event."""
        require_transition(expected_state, target_state)
        with self.psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT state FROM audits WHERE audit_id=%s FOR UPDATE", (audit_id,)
                )
                found = cursor.fetchone()
                if found is None or found[0] != expected_state:
                    raise ValueError("PostgreSQL audit state conflict")
                cursor.execute(
                    "UPDATE audits SET state=%s, payload=payload || %s::jsonb, "
                    "updated_at=now() WHERE audit_id=%s",
                    (target_state, json.dumps(payload_patch or {}), audit_id),
                )
                cursor.execute(
                    "INSERT INTO audit_events (audit_id, from_state, to_state) "
                    "VALUES (%s, %s, %s)",
                    (audit_id, expected_state, target_state),
                )
        return {"audit_id": audit_id, "state": target_state}

    def attach_resource_plan(
        self, audit_id: str, plan: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Persist signed plan and atomically expose audit for worker polling."""
        parsed = ResourcePlan.from_dict(plan)
        with self.psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT state FROM audits WHERE audit_id=%s FOR UPDATE", (audit_id,)
                )
                found = cursor.fetchone()
                if found is None or found[0] != "frozen":
                    raise ValueError("resource plan requires frozen audit")
                cursor.execute(
                    "UPDATE audits SET state='awaiting_executor', "
                    "resource_plan=%s::jsonb, updated_at=now() WHERE audit_id=%s",
                    (json.dumps(parsed.to_dict()), audit_id),
                )
                cursor.execute(
                    "INSERT INTO audit_events (audit_id, from_state, to_state) "
                    "VALUES (%s, 'frozen', 'awaiting_executor')", (audit_id,),
                )
        return {"audit_id": audit_id, "state": "awaiting_executor",
                "resource_plan": parsed.to_dict()}

    def poll_plan(self, capabilities: ExecutorCapabilities) -> Optional[Dict[str, Any]]:
        """Lease one compatible plan with row locking; no task broker needed."""
        with self.psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT audit_id, resource_plan FROM audits "
                    "WHERE state='awaiting_executor' ORDER BY created_at "
                    "FOR UPDATE SKIP LOCKED"
                )
                for audit_id, raw_plan in cursor.fetchall():
                    plan = ResourcePlan.from_dict(raw_plan)
                    compatible, _ = compatible_executor(plan, capabilities)
                    if not compatible:
                        continue
                    lease_seconds = max(
                        900, int(plan.resources["wall_time_seconds"]) + 300
                    )
                    cursor.execute(
                        "UPDATE audits SET state='leased', lease_owner=%s, "
                        "lease_expires_at=now()+(%s * interval '1 second') "
                        "WHERE audit_id=%s",
                        (capabilities.executor_id, lease_seconds, audit_id),
                    )
                    cursor.execute(
                        "INSERT INTO audit_events (audit_id, from_state, to_state, actor) "
                        "VALUES (%s, 'awaiting_executor', 'leased', %s)",
                        (audit_id, capabilities.executor_id),
                    )
                    return {"audit_id": audit_id, "resource_plan": raw_plan}
        return None

    def submit_attestations(
        self, audit_id: str, executor_id: str,
        attestations: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Insert terminal slot rows idempotently under current worker lease."""
        with self.psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT state, lease_owner, lease_expires_at > now() "
                    "FROM audits WHERE audit_id=%s FOR UPDATE",
                    (audit_id,),
                )
                found = cursor.fetchone()
                if found is None or found[1] != executor_id or found[0] not in (
                    "leased", "running"
                ) or found[2] is not True:
                    raise ValueError("executor does not own an active lease")
                if found[0] == "leased":
                    cursor.execute(
                        "UPDATE audits SET state='running', updated_at=now() "
                        "WHERE audit_id=%s", (audit_id,),
                    )
                    cursor.execute(
                        "INSERT INTO audit_events (audit_id, from_state, to_state, actor) "
                        "VALUES (%s, 'leased', 'running', %s)",
                        (audit_id, executor_id),
                    )
                for attestation in attestations:
                    slot_id = attestation.get("slot_id")
                    if not isinstance(slot_id, str) or not slot_id:
                        raise ValueError("attestation needs slot_id")
                    cursor.execute(
                        "INSERT INTO run_attestations (audit_id, slot_id, attestation) "
                        "VALUES (%s, %s, %s::jsonb) "
                        "ON CONFLICT (audit_id, slot_id) DO UPDATE SET "
                        "attestation=EXCLUDED.attestation",
                        (audit_id, slot_id, json.dumps(attestation)),
                    )
        return {"audit_id": audit_id, "state": "running",
                "accepted": len(attestations)}

    def finish_submission(self, audit_id: str, executor_id: str) -> Dict[str, Any]:
        """Close a worker lease and expose complete rows to offline verifier."""
        with self.psycopg.connect(self.dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE audits SET state='submitted', updated_at=now() "
                    "WHERE audit_id=%s AND state='running' AND lease_owner=%s "
                    "AND lease_expires_at > now()",
                    (audit_id, executor_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("worker cannot finish this audit")
                cursor.execute(
                    "INSERT INTO audit_events (audit_id, from_state, to_state, actor) "
                    "VALUES (%s, 'running', 'submitted', %s)",
                    (audit_id, executor_id),
                )
        return {"audit_id": audit_id, "state": "submitted"}


def _bearer(value: Optional[str], expected: str) -> None:
    wanted = "Bearer " + expected
    if not isinstance(value, str) or not hmac.compare_digest(value, wanted):
        raise PermissionError("invalid bearer token")


def create_app(
    *,
    repository: Any,
    admin_token: str,
    worker_token: str,
    evidence_root: Optional[str] = None,
    object_store: Any = None,
) -> Any:
    """Create optional FastAPI app; core install remains NumPy-only."""
    try:
        from fastapi import FastAPI, Header, HTTPException, Request, Response
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "hosted control plane needs: pip install stress-kit[control]"
        ) from exc
    if not admin_token or not worker_token:
        raise ValueError("admin and worker tokens must not be empty")
    app = FastAPI(title="StressKit control plane", version="1.0")

    @app.post("/v1/admin/intake")
    def intake(payload: Dict[str, Any], authorization: Optional[str] = Header(None)) -> Any:
        try:
            _bearer(authorization, admin_token)
            return repository.create_intake(str(payload["claim_id"]), payload)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/admin/audits/{audit_id}/transition")
    def transition(audit_id: str, payload: Dict[str, Any],
                   authorization: Optional[str] = Header(None)) -> Any:
        try:
            _bearer(authorization, admin_token)
            return repository.transition(
                audit_id, str(payload["expected_state"]), str(payload["target_state"]),
                payload.get("payload_patch"),
            )
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/admin/audits/{audit_id}/resource-plan")
    def resource_plan(audit_id: str, payload: Dict[str, Any],
                      authorization: Optional[str] = Header(None)) -> Any:
        try:
            _bearer(authorization, admin_token)
            row = repository.attach_resource_plan(audit_id, payload)
            return row.to_dict() if hasattr(row, "to_dict") else row
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/workers/plans/poll")
    def poll(payload: Dict[str, Any], response: Response,
             authorization: Optional[str] = Header(None)) -> Any:
        try:
            _bearer(authorization, worker_token)
            isolation = payload.get("isolation", {})
            capabilities = ExecutorCapabilities(
                executor_id=str(payload["executor_id"]),
                hardware_classes=list(payload["hardware_classes"]),
                gpu_count=int(payload["gpu_count"]),
                cpu_count=int(payload["cpu_count"]),
                storage_bytes=int(payload["storage_bytes"]),
                network_namespace=isolation.get("network_namespace") is True,
                credential_isolation=isolation.get("credential_isolation") is True,
                read_only_mounts=isolation.get("read_only_mounts") is True,
                scratch_quota=isolation.get("scratch_quota") is True,
                output_allowlist=isolation.get("output_allowlist") is True,
            )
            row = repository.poll_plan(capabilities)
            if row is None:
                response.status_code = 204
                return None
            return row.to_dict() if hasattr(row, "to_dict") else row
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/workers/audits/{audit_id}/attestations")
    def attest(audit_id: str, payload: Dict[str, Any],
               authorization: Optional[str] = Header(None)) -> Any:
        try:
            _bearer(authorization, worker_token)
            row = repository.submit_attestations(
                audit_id, str(payload["executor_id"]), list(payload["attestations"])
            )
            return row.to_dict() if hasattr(row, "to_dict") else row
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/workers/audits/{audit_id}/finish")
    def finish(audit_id: str, payload: Dict[str, Any],
               authorization: Optional[str] = Header(None)) -> Any:
        try:
            _bearer(authorization, worker_token)
            return repository.finish_submission(audit_id, str(payload["executor_id"]))
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    if object_store is not None:
        @app.post("/v1/workers/objects")
        async def put_object(request: Request,
                             authorization: Optional[str] = Header(None),
                             x_stresskit_digest: Optional[str] = Header(None),
                             content_type: Optional[str] = Header(None)) -> Any:
            try:
                _bearer(authorization, worker_token)
                payload = await request.body()
                reference = object_store.put_bytes(
                    payload, media_type=content_type or "application/octet-stream"
                )
                if x_stresskit_digest and reference.digest != x_stresskit_digest:
                    raise ValueError("uploaded object does not match declared digest")
                return reference.to_dict()
            except PermissionError as exc:
                raise HTTPException(status_code=401, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    if evidence_root:
        app.mount("/evidence", StaticFiles(directory=evidence_root, html=True),
                  name="evidence")
    return app
