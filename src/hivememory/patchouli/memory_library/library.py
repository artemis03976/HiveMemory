"""
MemoryLibrary — 三级存储协调层（书库）

持有 ShortTermMemoryStore / MidTermMemoryStore / LongTermMemoryStore，
负责各层之间的状态转移（archive / revive）。

状态转移协议：
    短期 → 中期：由 MemoryIngestionPipeline 驱动（Phase 4），Library 提供两端读写能力
    中期 → 长期：MemoryLibrary.archive()  （纯数据搬运）
    长期 → 中期：MemoryLibrary.revive()   （纯数据搬运）

实现阶段: Phase 1 骨架
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from hivememory.core.models import MemoryEventLog, MemoryEventType
from hivememory.patchouli.memory_library.models import (
    StorageHealthComponent,
    StorageHealthReport,
)
from hivememory.patchouli.memory_library.stores import (
    LongTermMemoryStore,
    MidTermMemoryStore,
    ShortTermMemoryStore,
    ArtifactStore,
)

logger = logging.getLogger(__name__)


class MemoryLibrary:
    """
    三级记忆书库。

    图书馆隐喻:
        short_term  — 工作台（内存 buffer，活跃话题）
        mid_term    — 主书库（向量数据库，已入库记忆）
        long_term   — 冷藏库（文件系统，低活跃度记忆）
    """

    def __init__(
        self,
        short_term: ShortTermMemoryStore,
        mid_term: MidTermMemoryStore,
        long_term: LongTermMemoryStore,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> None:
        self.short_term = short_term
        self.mid_term = mid_term
        self.long_term = long_term
        self.artifact_store = artifact_store
        logger.info("MemoryLibrary 初始化完成")

    # ========== 中期 → 长期（归档） ==========

    async def archive(self, memory_id: UUID) -> None:
        """
        将中期记忆迁移到冷存储。

        流程: MidTermStore.get() → LongTermStore.persist() → MidTermStore.delete()
        """
        memory = await self.mid_term.get(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id} not found in mid-term storage")
        memory.payload.artifacts.events.append(
            MemoryEventLog(event_type=MemoryEventType.ARCHIVED)
        )
        await self.long_term.persist(memory)
        await self.mid_term.delete(memory_id)
        logger.info(f"记忆已归档至冷存储: {memory_id}")

    # ========== 长期 → 中期（复活） ==========

    async def revive(self, memory_id: UUID) -> None:
        """
        从冷存储复活记忆到中期存储。

        流程: LongTermStore.load() → MidTermStore.upsert() → LongTermStore.remove()
        """
        memory = await self.long_term.load(memory_id)
        memory.payload.artifacts.events.append(
            MemoryEventLog(event_type=MemoryEventType.REVIVED)
        )
        await self.mid_term.upsert(memory)
        await self.long_term.remove(memory_id)
        logger.info(f"记忆已从冷存储复活至向量库: {memory_id}")


    async def check_storage_health(self) -> StorageHealthReport:
        """Return a health report for the complete memory storage system."""
        components = [
            await self.short_term.check_health(),
            await self.mid_term.check_health(),
            await self.long_term.check_health(),
        ]
        if self.artifact_store is None:
            components.append(
                StorageHealthComponent(
                    name="artifact",
                    healthy=True,
                    required=False,
                    detail="disabled",
                )
            )
        else:
            components.append(await self.artifact_store.check_health())
        return StorageHealthReport(components=tuple(components))


__all__ = ["MemoryLibrary"]
