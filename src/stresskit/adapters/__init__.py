"""Adapters: bridges from the libraries the field already uses into
stresskit.Finding objects. Adapters never hard-depend on the host library —
imports happen lazily so the core stays numpy-only."""

from . import sae  # numpy-only, safe to import eagerly

__all__ = ["sae"]
