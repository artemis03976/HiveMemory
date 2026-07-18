"""Read models exposed by MemoryLibrary stores."""

from __future__ import annotations

from typing import Optional

from dataclasses import dataclass


@dataclass
class ArtifactIntegrityResult:
    artifact_id: str
    ok: bool
    stored_hash: Optional[str] = None
    actual_hash: Optional[str] = None


@dataclass(frozen=True)
class StorageHealthComponent:
    name: str
    healthy: bool
    required: bool = True
    detail: Optional[str] = None


@dataclass(frozen=True)
class StorageHealthReport:
    components: tuple[StorageHealthComponent, ...]

    @property
    def healthy(self) -> bool:
        return all(
            component.healthy
            for component in self.components
            if component.required
        )


__all__ = [
    "ArtifactIntegrityResult",
    "StorageHealthComponent",
    "StorageHealthReport",
]
