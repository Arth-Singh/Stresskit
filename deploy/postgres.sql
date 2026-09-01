CREATE TABLE IF NOT EXISTS audits (
    audit_id text PRIMARY KEY,
    claim_id text NOT NULL,
    state text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    resource_plan jsonb,
    lease_owner text,
    lease_expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS audits_poll_idx
    ON audits (state, created_at)
    WHERE state = 'awaiting_executor';

CREATE TABLE IF NOT EXISTS audit_events (
    event_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    audit_id text NOT NULL REFERENCES audits(audit_id),
    from_state text,
    to_state text NOT NULL,
    actor text,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS run_attestations (
    audit_id text NOT NULL REFERENCES audits(audit_id),
    slot_id text NOT NULL,
    attestation jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (audit_id, slot_id)
);

CREATE TABLE IF NOT EXISTS evidence_publications (
    claim_id text PRIMARY KEY,
    audit_id text NOT NULL REFERENCES audits(audit_id),
    decision jsonb NOT NULL,
    bundle_digest text NOT NULL,
    published_at timestamptz NOT NULL DEFAULT now()
);
