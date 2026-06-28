from __future__ import annotations

from uuid import UUID


def normalize_uuid(value: UUID | str) -> UUID:
    """Return a UUID object from an existing UUID or UUID-compatible string."""
    return value if isinstance(value, UUID) else UUID(str(value))
