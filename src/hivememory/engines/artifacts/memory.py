"""MemoryArtifactBuilder - 处理 MemoryAtom 创建与更新时的 artifact 写入。"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models.artifact import (
    ArtifactRef,
    MemoryCreationArtifact,
    MemoryInputRef,
    MemoryVersionArtifact,
    MemoryVersionSnapshot,
)
from hivememory.core.models.memory import MemoryAtom
from hivememory.engines.generation.models import GenerationContext
from hivememory.patchouli.memory_library import ArtifactStore
from hivememory.system.config.patchouli import ArtifactComponentConfig


class MemoryCreationBundle(BaseModel):
    """build_for_create 的原子返回值 - 两个强关联 artifact 作为整体返回。"""
    creation_ref: Optional[ArtifactRef] = None
    initial_version_ref: Optional[ArtifactRef] = None  # MemoryVersionArtifact v1

    @property
    def refs(self) -> list[ArtifactRef]:
        return [
            ref
            for ref in (self.initial_version_ref, self.creation_ref)
            if ref is not None
        ]


class MemoryArtifactBuilder:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    async def build_for_create(
        self,
        *,
        memory: MemoryAtom,
        context: GenerationContext,
        source_intent: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"],
        source_artifact_refs: List[ArtifactRef],
        source_memory_refs: Optional[List[MemoryInputRef]] = None,
    ) -> MemoryCreationBundle:
        """原子写入 MemoryVersionArtifact(v1) 与 MemoryCreationArtifact。v1 先写。"""
        memory_id = str(memory.id)
        _require_source_refs_in_workspace(memory, source_artifact_refs)

        # 1. v1 快照 — 初始可变字段全量状态
        v1 = MemoryVersionArtifact(
            memory_id=memory_id,
            workspace_identity=memory.workspace_identity,
            owner_agent_id=memory.meta.source_agent_id,
            version_number=1,
            update_source="CREATE",
            snapshot_before=None,
            snapshot_after=MemoryVersionSnapshot.from_memory_atom(memory),
            changed_at=datetime.now(),
            source_artifacts=source_artifact_refs,
            source_memory_refs=source_memory_refs or [],
        )
        v1_ref = await self._store.put(v1)

        # 2. creation artifact — initial_version_ref 指向 v1
        creation = MemoryCreationArtifact(
            memory_id=memory_id,
            workspace_identity=memory.workspace_identity,
            owner_agent_id=memory.meta.source_agent_id,
            source_intent=source_intent,
            generation_view=context.model_dump(),
            source_artifacts=source_artifact_refs,
            source_memory_refs=source_memory_refs or [],
            initial_version_ref=v1_ref,
        )
        creation_ref = await self._store.put(creation)

        return MemoryCreationBundle(creation_ref=creation_ref, initial_version_ref=v1_ref)

    async def build_for_update(
        self,
        *,
        memory_after: MemoryAtom,
        snapshot_before: Optional[MemoryVersionSnapshot] = None,
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"],
        changelog: Optional[str] = None,
        source_artifact_refs: Optional[List[ArtifactRef]] = None,
        source_memory_refs: Optional[List[MemoryInputRef]] = None,
    ) -> ArtifactRef | None:
        """写入 MemoryVersionArtifact(v2+)，返回 version ref。"""
        _require_source_refs_in_workspace(memory_after, source_artifact_refs or [])
        version = MemoryVersionArtifact(
            memory_id=str(memory_after.id),
            workspace_identity=memory_after.workspace_identity,
            owner_agent_id=memory_after.meta.source_agent_id,
            version_number=memory_after.meta.version,
            update_source=update_source,
            snapshot_before=snapshot_before,
            snapshot_after=MemoryVersionSnapshot.from_memory_atom(memory_after),
            changelog=changelog,
            changed_at=datetime.now(),
            source_artifacts=source_artifact_refs or [],
            source_memory_refs=source_memory_refs or [],
        )
        return await self._store.put(version)


class NoOpMemoryArtifactBuilder:
    async def build_for_create(
        self,
        *,
        memory: MemoryAtom,
        context: GenerationContext,
        source_intent: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"],
        source_artifact_refs: List[ArtifactRef],
        source_memory_refs: Optional[List[MemoryInputRef]] = None,
    ) -> MemoryCreationBundle:
        return MemoryCreationBundle()

    async def build_for_update(
        self,
        *,
        memory_after: MemoryAtom,
        snapshot_before: Optional[MemoryVersionSnapshot] = None,
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"],
        changelog: Optional[str] = None,
        source_artifact_refs: Optional[List[ArtifactRef]] = None,
        source_memory_refs: Optional[List[MemoryInputRef]] = None,
    ) -> ArtifactRef | None:
        return None


def create_memory_builder(
    config: ArtifactComponentConfig,
    store: ArtifactStore | None,
) -> MemoryArtifactBuilder | NoOpMemoryArtifactBuilder:
    if store is None or not config.enabled:
        return NoOpMemoryArtifactBuilder()
    return MemoryArtifactBuilder(store)


def _require_source_refs_in_workspace(
    memory: MemoryAtom,
    source_refs: list[ArtifactRef],
) -> None:
    """拒绝把其他 Workspace 的 provenance 引用写入当前 Memory。"""
    for ref in source_refs:
        if ref.workspace_identity != memory.workspace_identity:
            raise WorkspaceMismatchError(details={"artifact_id": ref.artifact_id})
