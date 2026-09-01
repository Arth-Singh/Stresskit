"""Launch optional StressKit hosted control plane from environment settings."""

from __future__ import annotations

import os


def main() -> int:
    """Start FastAPI with PostgreSQL state and optional S3-compatible CAS."""
    try:
        import uvicorn  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "control plane needs: pip install stress-kit[control]"
        ) from exc
    from .control_plane import PostgresAuditRepository, create_app
    from .integrity import S3ContentAddressedStore

    required = ("STRESSKIT_POSTGRES_DSN", "STRESSKIT_ADMIN_TOKEN",
                "STRESSKIT_WORKER_TOKEN")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("missing control-plane environment variables: " + ", ".join(missing))
    object_store = None
    bucket = os.environ.get("STRESSKIT_S3_BUCKET")
    if bucket:
        object_store = S3ContentAddressedStore(
            bucket,
            prefix=os.environ.get("STRESSKIT_S3_PREFIX", "stresskit"),
            endpoint_url=os.environ.get("STRESSKIT_S3_ENDPOINT"),
        )
    app = create_app(
        repository=PostgresAuditRepository(os.environ["STRESSKIT_POSTGRES_DSN"]),
        admin_token=os.environ["STRESSKIT_ADMIN_TOKEN"],
        worker_token=os.environ["STRESSKIT_WORKER_TOKEN"],
        evidence_root=os.environ.get("STRESSKIT_EVIDENCE_ROOT"),
        object_store=object_store,
    )
    uvicorn.run(
        app,
        host=os.environ.get("STRESSKIT_HOST", "127.0.0.1"),
        port=int(os.environ.get("STRESSKIT_PORT", "8000")),
        log_level=os.environ.get("STRESSKIT_LOG_LEVEL", "info"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
