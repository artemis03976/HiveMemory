"""MemoryLibrary store 暴露的读模型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArtifactIntegrityResult:
    artifact_id: str
    ok: bool
    stored_hash: str | None = None
    actual_hash: str | None = None


@dataclass(frozen=True)
class StorageHealthComponent:
    name: str
    healthy: bool
    required: bool = True
    detail: str | None = None


@dataclass(frozen=True)
class StorageHealthReport:
    components: tuple[StorageHealthComponent, ...]

    @property
    def healthy(self) -> bool:
        return all(component.healthy for component in self.components if component.required)


__all__ = [
    "ArtifactIntegrityResult",
    "StorageHealthComponent",
    "StorageHealthReport",
]
