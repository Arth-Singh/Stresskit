"""Canonical hashing, signatures, and content-addressed storage for audits.

This module uses only Python standard library until Ed25519 is explicitly used;
that path lazily imports optional ``cryptography``. Offline verifier keeps
separate control-plan and executor trust domains. Algorithm and key identifier
are explicit and covered by each signature.
"""

from __future__ import annotations

import hashlib
import hmac
import base64
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically and reject non-finite numbers."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return a tagged SHA-256 digest for bytes."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def digest_json(value: Any) -> str:
    """Return the canonical SHA-256 digest of a JSON value."""
    return sha256_bytes(canonical_json_bytes(value))


def is_sha256_digest(value: Any) -> bool:
    """Return whether ``value`` is a canonical tagged SHA-256 digest."""
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def require_sha256_digest(value: Any, field: str) -> str:
    """Validate and return a tagged SHA-256 digest."""
    if not is_sha256_digest(value):
        raise ValueError(f"{field} must be 'sha256:' followed by 64 lowercase hex digits")
    return str(value)


def unsigned_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy an artifact while removing its top-level signature."""
    return {key: value for key, value in payload.items() if key != "signature"}


def _signature_message(
    payload: Mapping[str, Any], algorithm: str, key_id: str
) -> bytes:
    return canonical_json_bytes({
        "algorithm": algorithm,
        "key_id": key_id,
        "payload": unsigned_payload(payload),
    })


def sign_mapping(
    payload: Mapping[str, Any],
    key: bytes,
    key_id: str,
    *,
    algorithm: str = "hmac-sha256",
) -> Dict[str, str]:
    """Sign canonical payload with HMAC-SHA256 or optional Ed25519."""
    if not key:
        raise ValueError("signing key must not be empty")
    if not key_id.strip():
        raise ValueError("key_id must not be empty")
    message = _signature_message(payload, algorithm, key_id)
    if algorithm == "hmac-sha256":
        value = hmac.new(key, message, hashlib.sha256).hexdigest()
    elif algorithm == "ed25519":
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Ed25519 signing needs: pip install stress-kit[audit]"
            ) from exc
        if key.lstrip().startswith(b"-----BEGIN"):
            private_key = serialization.load_pem_private_key(key, password=None)
            if not isinstance(private_key, Ed25519PrivateKey):
                raise ValueError("signing key PEM is not Ed25519")
        else:
            private_key = Ed25519PrivateKey.from_private_bytes(key)
        value = base64.b64encode(private_key.sign(message)).decode("ascii")
    else:
        raise ValueError("signature algorithm must be hmac-sha256 or ed25519")
    return {"algorithm": algorithm, "key_id": key_id, "value": value}


def verify_mapping_signature(
    payload: Mapping[str, Any], trusted_keys: Mapping[str, bytes]
) -> bool:
    """Verify explicit HMAC or Ed25519 signature against named trusted key."""
    signature = payload.get("signature")
    if not isinstance(signature, Mapping):
        return False
    algorithm = signature.get("algorithm")
    if algorithm not in ("hmac-sha256", "ed25519"):
        return False
    key_id = signature.get("key_id")
    value = signature.get("value")
    if not isinstance(key_id, str) or not isinstance(value, str):
        return False
    key = trusted_keys.get(key_id)
    if not key:
        return False
    message = _signature_message(payload, str(algorithm), key_id)
    if algorithm == "hmac-sha256":
        expected = hmac.new(key, message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(value, expected)
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        if key.lstrip().startswith(b"-----BEGIN"):
            public_key = serialization.load_pem_public_key(key)
            if not isinstance(public_key, Ed25519PublicKey):
                return False
        else:
            public_key = Ed25519PublicKey.from_public_bytes(key)
        public_key.verify(base64.b64decode(value, validate=True), message)
        return True
    except (ImportError, TypeError, ValueError):
        return False
    except Exception as exc:  # cryptography exposes backend-specific InvalidSignature
        if exc.__class__.__name__ == "InvalidSignature":
            return False
        return False


def read_secret(path: str) -> bytes:
    """Read a signing secret from a file without normalizing its bytes."""
    value = Path(path).read_bytes()
    if not value:
        raise ValueError(f"signing key file is empty: {path}")
    return value


@dataclass(frozen=True)
class ContentRef:
    """One immutable object referenced by an audit bundle."""

    digest: str
    size: int
    media_type: str = "application/octet-stream"
    role: str = "raw_output"

    def __post_init__(self) -> None:
        require_sha256_digest(self.digest, "content digest")
        if self.size < 0:
            raise ValueError("content size must be non-negative")
        if not self.media_type:
            raise ValueError("content media_type must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON representation."""
        return {
            "digest": self.digest,
            "size": self.size,
            "media_type": self.media_type,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContentRef":
        """Build a content reference from JSON."""
        return cls(
            digest=str(payload.get("digest", "")),
            size=int(payload.get("size", -1)),
            media_type=str(payload.get("media_type", "application/octet-stream")),
            role=str(payload.get("role", "raw_output")),
        )


class ContentAddressedStore:
    """Filesystem SHA-256 object store with atomic, idempotent writes."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def _path(self, digest: str) -> Path:
        require_sha256_digest(digest, "object digest")
        hexdigest = digest.split(":", 1)[1]
        return self.root / "objects" / "sha256" / hexdigest[:2] / hexdigest[2:]

    def put_bytes(
        self,
        payload: bytes,
        *,
        media_type: str = "application/octet-stream",
        role: str = "raw_output",
    ) -> ContentRef:
        """Store bytes once and return their immutable reference."""
        digest = sha256_bytes(payload)
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"content collision at {digest}")
        else:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".stresskit-", dir=str(path.parent)
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return ContentRef(digest, len(payload), media_type, role)

    def put_json(self, value: Any, *, role: str = "raw_output") -> ContentRef:
        """Store canonical JSON and return its reference."""
        return self.put_bytes(
            canonical_json_bytes(value), media_type="application/json", role=role
        )

    def has(self, digest: str) -> bool:
        """Return whether an object exists at its digest path."""
        return self._path(digest).is_file()

    def get_bytes(self, digest: str) -> bytes:
        """Load bytes and reject tampering at rest."""
        path = self._path(digest)
        try:
            payload = path.read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"content object missing: {digest}") from exc
        actual = sha256_bytes(payload)
        if actual != digest:
            raise ValueError(f"content object {digest} hashes as {actual}")
        return payload

    def get_json(self, digest: str) -> Any:
        """Load and decode one JSON object after digest verification."""
        try:
            return json.loads(self.get_bytes(digest).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"content object is not valid UTF-8 JSON: {digest}") from exc

    def verify_ref(self, reference: ContentRef) -> None:
        """Verify digest and byte count for one reference."""
        payload = self.get_bytes(reference.digest)
        if len(payload) != reference.size:
            raise ValueError(
                f"content object {reference.digest} has {len(payload)} bytes, "
                f"reference records {reference.size}"
            )

    def verify_refs(self, references: Sequence[ContentRef]) -> None:
        """Verify a complete list of content references."""
        seen: Set[str] = set()
        for reference in references:
            if reference.digest in seen:
                raise ValueError(f"duplicate content reference: {reference.digest}")
            seen.add(reference.digest)
            self.verify_ref(reference)


def referenced_digests(value: Any) -> Set[str]:
    """Collect CAS-link digests recursively from a JSON value.

    ``raw_digest`` is a recomputed integrity assertion over embedded utility
    fields, not a reference to a separately stored object. All other tagged
    digests remain strict closure links.
    """
    found: Set[str] = set()
    if is_sha256_digest(value):
        found.add(str(value))
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if key == "raw_digest":
                continue
            found.update(referenced_digests(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(referenced_digests(child))
    return found


def verify_digest_closure(
    store: ContentAddressedStore,
    references: Sequence[ContentRef],
    roots: Iterable[str],
) -> Set[str]:
    """Verify every JSON-reachable object appears in a bundle's closure.

    Non-JSON objects are leaves. Tagged digests found inside JSON objects are
    traversed. Parsing is content-based: an untrusted media-type label cannot
    hide links. Referenced objects omitted from ``references`` are rejected;
    unrelated extra references are rejected too.
    """
    by_digest = {reference.digest: reference for reference in references}
    if len(by_digest) != len(references):
        raise ValueError("content closure contains duplicate digests")
    pending: List[str] = list(roots)
    reached: Set[str] = set()
    while pending:
        digest = pending.pop()
        require_sha256_digest(digest, "closure root")
        if digest in reached:
            continue
        reference = by_digest.get(digest)
        if reference is None:
            raise ValueError(f"digest closure omits referenced object {digest}")
        store.verify_ref(reference)
        reached.add(digest)
        try:
            parsed = json.loads(store.get_bytes(digest).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        if parsed is not None:
            for child in referenced_digests(parsed):
                if child not in reached:
                    pending.append(child)
    extras = set(by_digest) - reached
    if extras:
        raise ValueError(f"digest closure contains unreachable objects: {sorted(extras)}")
    return reached


class S3ContentAddressedStore:
    """S3-compatible immutable object store; imports boto3 only when used."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "stresskit",
        endpoint_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket must not be empty")
        if client is None:
            try:
                import boto3  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "S3 storage needs the 'control' extra: pip install stress-kit[control]"
                ) from exc
            client = boto3.client("s3", endpoint_url=endpoint_url)
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, digest: str) -> str:
        require_sha256_digest(digest, "object digest")
        hexdigest = digest.split(":", 1)[1]
        base = f"objects/sha256/{hexdigest[:2]}/{hexdigest[2:]}"
        return f"{self.prefix}/{base}" if self.prefix else base

    def put_bytes(
        self,
        payload: bytes,
        *,
        media_type: str = "application/octet-stream",
        role: str = "raw_output",
    ) -> ContentRef:
        """Upload one object under its digest key."""
        digest = sha256_bytes(payload)
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(digest),
            Body=payload,
            ContentType=media_type,
            Metadata={"sha256": digest.split(":", 1)[1]},
        )
        return ContentRef(digest, len(payload), media_type, role)

    def put_json(self, value: Any, *, role: str = "raw_output") -> ContentRef:
        """Upload canonical JSON under its content digest."""
        return self.put_bytes(
            canonical_json_bytes(value), media_type="application/json", role=role
        )

    def get_bytes(self, digest: str) -> bytes:
        """Download one object and verify its content digest."""
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(digest))
        payload = response["Body"].read()
        if sha256_bytes(payload) != digest:
            raise ValueError(f"S3 object does not match {digest}")
        return payload

    def get_json(self, digest: str) -> Any:
        """Download and decode one verified JSON object."""
        try:
            return json.loads(self.get_bytes(digest).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"S3 content object is not valid UTF-8 JSON: {digest}") from exc

    def verify_ref(self, reference: ContentRef) -> None:
        """Verify an S3 object's digest and declared byte count."""
        payload = self.get_bytes(reference.digest)
        if len(payload) != reference.size:
            raise ValueError(
                f"S3 object {reference.digest} has {len(payload)} bytes, "
                f"reference records {reference.size}"
            )

    def verify_refs(self, references: Sequence[ContentRef]) -> None:
        """Verify a duplicate-free S3 content-reference list."""
        seen: Set[str] = set()
        for reference in references:
            if reference.digest in seen:
                raise ValueError(f"duplicate content reference: {reference.digest}")
            seen.add(reference.digest)
            self.verify_ref(reference)
