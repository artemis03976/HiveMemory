"""MemoryArtifactBuilder - 处理 MemoryAtom 创建与更新时的 artifact 写入。

schema "2" 起，版本记录的 snapshot_before/after 直接嵌入捕获时点完整
MemoryAtom 的 canonical JSON（见 :func:`snapshot_memory_atom`）；捕获时点
与内容提交时点的统一由 Familiar 完整写入路径负责（MVL-2/MVL-3 收敛）。
"""

from typing import Literal

from pydantic import BaseModel

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models.artifact import (
    ArtifactRef,
    MemoryCreationArtifact,
    MemoryInputRef,
    MemoryVersionArtifact,
    snapshot_memory_atom,
)
from hivememory.core.models.memory import MemoryAtom
from hivememory.engines.generation.models import GenerationContext
from hivememory.patchouli.memory_library import ArtifactStore
from hivememory.system.config.patchouli import ArtifactComponentConfig
from hivememory.utils.time import utc_now


class MemoryCreationBundle(BaseModel):
    """build_for_create 的原子返回值 - 两个强关联 artifact 作为整体返回。"""

    creation_ref: ArtifactRef | None = None
    initial_version_ref: ArtifactRef | None = None  # MemoryVersionArtifact v1

    @property
    def refs(self) -> list[ArtifactRef]:
        return [ref for ref in (self.initial_version_ref, self.creation_ref) if ref is not None]


class MemoryArtifactBuilder:
    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    async def build_for_create(
        self,
        *,
        memory: MemoryAtom,
        context: GenerationContext,
        source_intent: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"],
        source_artifact_refs: list[ArtifactRef],
        source_memory_refs: list[MemoryInputRef] | None = None,
    ) -> MemoryCreationBundle:
        """原子写入 MemoryVersionArtifact(v1) 与 MemoryCreationArtifact。v1 先写。"""
        memory_id = str(memory.id)
        _require_source_refs_in_workspace(memory, source_artifact_refs)

        # 1. v1 版本记录 — 捕获时点完整原子；v1 无 snapshot_before
        v1 = MemoryVersionArtifact(
            memory_id=memory_id,
            workspace_identity=memory.workspace_identity,
            provenance=memory.meta.provenance,
            version_number=1,
            update_source="CREATE",
            snapshot_before=None,
            snapshot_after=snapshot_memory_atom(memory),
            changed_at=utc_now(),
            source_artifacts=source_artifact_refs,
            source_memory_refs=source_memory_refs or [],
        )
        v1_ref = await self._store.put(v1)

        # 2. creation artifact — initial_version_ref 指向 v1
        creation = MemoryCreationArtifact(
            memory_id=memory_id,
            workspace_identity=memory.workspace_identity,
            provenance=memory.meta.provenance,
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
        snapshot_before: dict | None = None,
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"],
        changelog: str | None = None,
        source_artifact_refs: list[ArtifactRef] | None = None,
        source_memory_refs: list[MemoryInputRef] | None = None,
    ) -> ArtifactRef | None:
        """写入 MemoryVersionArtifact(v2+)，返回 version ref。

        ``snapshot_before`` 是提交边界捕获的修改前完整原子 canonical JSON。
        """
        _require_source_refs_in_workspace(memory_after, source_artifact_refs or [])
        version = MemoryVersionArtifact(
            memory_id=str(memory_after.id),
            workspace_identity=memory_after.workspace_identity,
            provenance=memory_after.meta.provenance,
            version_number=memory_after.meta.version,
            update_source=update_source,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_memory_atom(memory_after),
            changelog=changelog,
            changed_at=utc_now(),
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
        source_artifact_refs: list[ArtifactRef],
        source_memory_refs: list[MemoryInputRef] | None = None,
    ) -> MemoryCreationBundle:
        return MemoryCreationBundle()

    async def build_for_update(
        self,
        *,
        memory_after: MemoryAtom,
        snapshot_before: dict | None = None,
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"],
        changelog: str | None = None,
        source_artifact_refs: list[ArtifactRef] | None = None,
        source_memory_refs: list[MemoryInputRef] | None = None,
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
