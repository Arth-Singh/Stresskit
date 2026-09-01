# StressKit v1 hosted control plane

Hosted deployment is optional. Offline core remains NumPy-only.

## Components

- FastAPI accepts admin-curated intake, resource plans, worker polling,
  attestations, and terminal submission events.
- PostgreSQL owns the state machine, leases, append-only events, and run-slot
  records. `FOR UPDATE SKIP LOCKED` replaces a task broker.
- S3-compatible storage holds immutable SHA-256 objects.
- Static evidence output is mounted read-only after offline verification.
- Workers initiate outbound HTTPS polling. Workers expose no inbound service.

No Redis or Celery dependency exists.

## State machine

```text
intake -> compiled -> frozen -> awaiting_executor -> leased -> running
running -> submitted -> verified -> published
```

Explicit terminal branches are `excluded`, `abstain`, `protocol_deviation`, and
`reproduction_failure`. Compare-and-swap transitions prevent state skips.

## Deployment

Install optional dependencies and initialize PostgreSQL:

```bash
python -m pip install 'stress-kit[control]'
psql "$STRESSKIT_POSTGRES_DSN" -f deploy/postgres.sql
stresskit-control-plane
```

Required environment variables:

- `STRESSKIT_POSTGRES_DSN`;
- `STRESSKIT_ADMIN_TOKEN`;
- `STRESSKIT_WORKER_TOKEN`.

Optional S3 variables are `STRESSKIT_S3_BUCKET`, `STRESSKIT_S3_PREFIX`, and
`STRESSKIT_S3_ENDPOINT`. `STRESSKIT_EVIDENCE_ROOT` mounts an already verified
static site. Default bind address is `127.0.0.1`; public ingress, TLS, secret
rotation, PostgreSQL backups, and object retention belong to deployment policy.

Admin token gates all intake and transitions. Public users cannot enqueue
papers. Worker capability matching checks declared hardware, GPU/CPU/storage,
network namespaces, credential isolation, read-only mounts, scratch quotas, and
output allowlists before a lease is issued.

PostgreSQL lease duration equals signed plan wall time plus five-minute
submission grace, with fifteen-minute minimum. Expired leases cannot submit
attestations or finish. Slot writes remain idempotent by `(audit_id, slot_id)`;
transition events remain append-only.
