from __future__ import annotations

from dataclasses import dataclass

from hivememory.core.models.pending import PendingAtomMaterializeTask


@dataclass(frozen=True)
class FrameProducts:
    """Artifacts projected from one successfully finalized frame."""

    artifact_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeProducts:
    """Products handed to Patchouli after a root run finishes."""

    materialize_tasks: tuple[PendingAtomMaterializeTask, ...] = ()


__all__ = ["FrameProducts", "RuntimeProducts"]
