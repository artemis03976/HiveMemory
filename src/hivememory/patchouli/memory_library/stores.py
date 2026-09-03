"""Storage facades for Patchouli memory layers.

Short-term storage deliberately exposes only persistence facts.  Topic state
transitions, compaction, settlement, eviction and routing policies belong to
the Perception layer; this module does not provide workflow-specific methods.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from uuid import UUID, uuid4

from hivememory.core.models import (
    IdentityScope,
    MemoryAtom,
    TopicData,
    WorkspaceMemoryKey,
    require_identity_scope,
)
from hivememory.core.models.artifact import ArtifactRef
from hivememory.engines.lifecycle.models import ArchiveRecord
from hivememory.patchouli.memory_library.adapters.short_term import InMemoryShortTermStorage
from hivememory.patchouli.memory_library.models import (
    ArtifactIntegrityResult,
    StorageHealthComponent,
)
from hivememory.patchouli.memory_library.ports import (
    ArtifactStoragePort,
    LongTermStoragePort,
    MidTermStoragePort,
    ShortTermStoragePort,
)

logger = logging.getLogger(__name__)


class ShortTermMemoryStore:
    """Short-term Topic persistence facade.

    The store owns only the port and the snapshot boundary.  Public methods use
    ``IdentityScope + topic_id``; storage keys and mutable buffers never cross
    this boundary.
    """

    def __init__(self, port: ShortTermStoragePort | None = None) -> None:
        self._port = port or InMemoryShortTermStorage()
        # The lock serializes snapshot touch + write-back.  Workflow state and
        # lifecycle decisions are intentionally not protected here.
        self._lock = threading.RLock()

    def get(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        *,
        touch: bool = True,
    ) -> TopicData | None:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            topic = self._port.get(identity_scope.workspace_identity, topic_id)
            if topic is None or not touch:
                return topic
            touched = topic.model_copy(
                update={"last_accessed_at": datetime.now().timestamp()}
            )
            self._port.put(touched)
            return touched

    def put(self, topic: TopicData) -> None:
        if not isinstance(topic, TopicData):
            raise TypeError("short-term store accepts TopicData snapshots")
        with self._lock:
            self._port.put(topic)

    def create(
        self,
        identity_scope: IdentityScope,
        topic_title: str = "新建话题",
        topic_summary: str = "",
        *,
        topic_id: str | None = None,
    ) -> TopicData:
        identity_scope = require_identity_scope(identity_scope)
        now = datetime.now().timestamp()
        topic = TopicData(
            topic_id=topic_id or str(uuid4()),
            workspace_identity=identity_scope.workspace_identity,
            topic_title=topic_title,
            topic_summary=topic_summary,
            last_update=now,
            last_accessed_at=now,
        )
        self.put(topic)
        return topic

    def delete(self, identity_scope: IdentityScope, topic_id: str) -> bool:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            return self._port.delete(identity_scope.workspace_identity, topic_id)

    def list_by_workspace(
        self,
        identity_scope: IdentityScope,
        *,
        include_empty: bool = True,
    ) -> list[TopicData]:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            topics = self._port.list_by_workspace(identity_scope.workspace_identity)
            if not include_empty:
                topics = [topic for topic in topics if topic.has_content]
            return topics

    def list_all(self) -> list[TopicData]:
        with self._lock:
            return self._port.list_all()

    def count(self, identity_scope: IdentityScope) -> int:
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            return self._port.count(identity_scope.workspace_identity)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


class MidTermMemoryStore:
    """中期记忆存储（向量库）。"""

    def __init__(
        self,
        primary: MidTermStoragePort,
        secondary: list[MidTermStoragePort] | None = None,
    ) -> None:
        self._primary = primary
        self._secondary: list[MidTermStoragePort] = secondary or []

    async def upsert(self, memory: MemoryAtom) -> None:
        await self._primary.upsert(memory)
        for secondary in self._secondary:
            await secondary.upsert(memory)

    async def get(self, scope: IdentityScope, memory_id: UUID) -> MemoryAtom | None:
        return await self._primary.get(require_identity_scope(scope), memory_id)

    async def get_by_alias(self, scope: IdentityScope, alias: str) -> MemoryAtom | None:
        return await self._primary.get_by_alias(require_identity_scope(scope), alias)

    async def get_for_mutation(
        self,
        identity_scope: IdentityScope,
        memory_id: UUID,
    ) -> MemoryAtom | None:
        return await self._primary.get_for_mutation(
            require_identity_scope(identity_scope),
            memory_id,
        )

    async def get_by_key(self, key: WorkspaceMemoryKey) -> MemoryAtom | None:
        return await self._primary.get_by_key(key)

    async def update_access_info(self, identity_scope: IdentityScope, memory_id: UUID) -> None:
        await self._primary.update_access_info(require_identity_scope(identity_scope), memory_id)

    async def delete(self, identity_scope: IdentityScope, memory_id: UUID) -> bool:
        identity_scope = require_identity_scope(identity_scope)
        result = await self._primary.delete(identity_scope, memory_id)
        for secondary in self._secondary:
            await secondary.delete(identity_scope, memory_id)
        return result

    async def delete_by_key(self, key: WorkspaceMemoryKey) -> bool:
        result = await self._primary.delete_by_key(key)
        for secondary in self._secondary:
            await secondary.delete_by_key(key)
        return result

    async def batch_delete(self, identity_scope: IdentityScope, ids: list[UUID]) -> int:
        identity_scope = require_identity_scope(identity_scope)
        count = await self._primary.batch_delete(identity_scope, ids)
        for secondary in self._secondary:
            await secondary.batch_delete(identity_scope, ids)
        return count

    async def search(
        self,
        scope: IdentityScope,
        query: str,
        top_k: int,
        filters=None,
        mode: str = "dense",
        score_threshold: float = 0.0,
    ):
        return await self._primary.search(
            require_identity_scope(scope),
            query,
            top_k,
            filters,
            mode,
            score_threshold,
        )

    async def scroll(
        self,
        scope: IdentityScope,
        filters=None,
        limit: int = 100,
    ) -> list[MemoryAtom]:
        return await self._primary.scroll(require_identity_scope(scope), filters, limit)

    async def count(self, scope: IdentityScope, filters=None) -> int:
        return await self._primary.count(require_identity_scope(scope), filters)

    async def list_all_for_maintenance(self, limit: int = 10000) -> list[MemoryAtom]:
        return await self._primary.list_all_for_maintenance(limit)

    async def check_health(self) -> StorageHealthComponent:
        primary_health = await self._primary.check_health()
        if not primary_health.healthy:
            return primary_health
        for index, secondary in enumerate(self._secondary):
            health = await secondary.check_health()
            if not health.healthy and health.required:
                return StorageHealthComponent(
                    name=f"mid_term.secondary.{index}",
                    healthy=False,
                    required=True,
                    detail=health.detail,
                )
        return primary_health


class LongTermMemoryStore:
    """长期记忆存储（冷存储）。"""

    def __init__(self, port: LongTermStoragePort) -> None:
        self._port = port

    async def persist(self, memory: MemoryAtom) -> None:
        await self._port.persist(memory)

    async def load(self, key: WorkspaceMemoryKey) -> MemoryAtom:
        return await self._port.load(key)

    async def remove(self, key: WorkspaceMemoryKey) -> None:
        await self._port.remove(key)

    async def is_archived(self, key: WorkspaceMemoryKey) -> bool:
        return await self._port.is_archived(key)

    async def query(
        self,
        limit: int = 100,
        vitality_threshold: float | None = None,
    ) -> list[ArchiveRecord]:
        return await self._port.query(limit=limit, vitality_threshold=vitality_threshold)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


class ArtifactStore:
    """Artifact 附属资产仓库。"""

    def __init__(self, port: ArtifactStoragePort) -> None:
        self._port = port

    async def put(self, artifact) -> ArtifactRef:
        return await self._port.put(artifact)

    async def get(self, identity_scope: IdentityScope, ref_or_id) -> dict:
        return await self._port.get(require_identity_scope(identity_scope), ref_or_id)

    async def exists(self, identity_scope: IdentityScope, artifact_id: str) -> bool:
        return await self._port.exists(require_identity_scope(identity_scope), artifact_id)

    async def list_by_memory(
        self,
        identity_scope: IdentityScope,
        memory_id: str,
        artifact_type=None,
    ) -> list:
        return await self._port.list_by_memory(
            require_identity_scope(identity_scope),
            memory_id,
            artifact_type,
        )

    async def verify(self, identity_scope: IdentityScope, ref) -> ArtifactIntegrityResult:
        return await self._port.verify(require_identity_scope(identity_scope), ref)

    async def check_health(self) -> StorageHealthComponent:
        return await self._port.check_health()


__all__ = ["ShortTermMemoryStore", "MidTermMemoryStore", "LongTermMemoryStore", "ArtifactStore"]
