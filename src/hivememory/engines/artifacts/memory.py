"""MemoryArtifactBuilder - 处理 MemoryAtom 创建与更新时的 artifact 写入。"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel

from hivememory.core.models.artifact import (
    ArtifactRef,
    MemoryCreationArtifact,
    MemoryInputRef,
    MemoryVersionArtifact,
    MemoryVersionSnapshot,
)
from hivememory.core.models.memory import MemoryAtom
from hivememory.engines.generation.models import GenerationContext
from hivememory.infrastructure.storage.artifact_store import ArtifactStore


class MemoryCreationBundle(BaseModel):
    """build_for_create 的原子返回值 - 两个强关联 artifact 作为整体返回。"""
    creation_ref: ArtifactRef
    initial_version_ref: ArtifactRef  # MemoryVersionArtifact v1


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

        # 1. v1 快照 — 初始可变字段全量状态
        v1 = MemoryVersionArtifact(
            memory_id=memory_id,
            version_number=1,
            update_source="CREATE",
            snapshot_before=None,
            snapshot_after=_snapshot(memory),
            changed_at=datetime.now(),
            source_artifacts=source_artifact_refs,
            source_memory_refs=source_memory_refs or [],
        )
        v1_ref = await self._store.put_json(v1)

        # 2. creation artifact — initial_version_ref 指向 v1
        creation = MemoryCreationArtifact(
            memory_id=memory_id,
            source_intent=source_intent,
            generation_view=context.model_dump(),
            source_artifacts=source_artifact_refs,
            source_memory_refs=source_memory_refs or [],
            initial_version_ref=v1_ref,
        )
        creation_ref = await self._store.put_json(creation)

        return MemoryCreationBundle(creation_ref=creation_ref, initial_version_ref=v1_ref)

    async def build_for_update(
        self,
        *,
        memory_before: MemoryAtom,
        memory_after: MemoryAtom,
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"],
        changelog: Optional[str] = None,
        source_artifact_refs: Optional[List[ArtifactRef]] = None,
        source_memory_refs: Optional[List[MemoryInputRef]] = None,
    ) -> ArtifactRef:
        """写入 MemoryVersionArtifact(v2+)，返回 version ref。"""
        version = MemoryVersionArtifact(
            memory_id=str(memory_after.id),
            version_number=memory_after.meta.version,
            update_source=update_source,
            snapshot_before=_snapshot(memory_before),
            snapshot_after=_snapshot(memory_after),
            changelog=changelog,
            changed_at=datetime.now(),
            source_artifacts=source_artifact_refs or [],
            source_memory_refs=source_memory_refs or [],
        )
        return await self._store.put_json(version)


def _snapshot(memory: MemoryAtom) -> MemoryVersionSnapshot:
    return MemoryVersionSnapshot(
        content=memory.payload.content,
        alias=memory.index.alias,
        title=memory.index.title,
        summary=memory.index.summary,
        tags=list(memory.index.tags),
        memory_type=memory.index.memory_type.value if memory.index.memory_type else None,
    )