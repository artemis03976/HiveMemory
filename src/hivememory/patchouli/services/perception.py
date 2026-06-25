"""Patchouli 感知使魔。

承接感知层代理职责，让 PerceptionLayer 保持为不感知总线和生成链路的底层 engine。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from hivememory.core.models import Identity
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.models import FlushReason
from hivememory.patchouli.runtime.memory_tasks import MemoryGenerationTask
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes

if TYPE_CHECKING:
    from hivememory.engines.perception.interfaces import BasePerceptionLayer
    from hivememory.patchouli.memory_library.library import MemoryLibrary
    from hivememory.patchouli.runtime.bus import PatchouliBus
    from hivememory.system.config.patchouli import MemoryPerceptionConfig

logger = logging.getLogger(__name__)


# ========== Application-level response models ==========

class TopicEvictResult(BaseModel):
    success: bool
    message: str


class ShutdownFlushResult(BaseModel):
    success: bool
    trigger_reason: str
    flushed_topics: list[str]
    skipped_topics: list[str]
    archived_blocks: int


# ========== PerceptionFamiliar ==========

class PerceptionFamiliar:
    """感知业务门面，负责摄入与短期话题管理。"""

    def __init__(
        self,
        *,
        perception_layer: "BasePerceptionLayer",
        bus: "PatchouliBus",
        config: "MemoryPerceptionConfig",
        memory_library: "MemoryLibrary",
    ) -> None:
        self.perception_layer = perception_layer
        self._bus = bus
        self._idle_timeout_seconds = config.idle_timeout_seconds
        self._short_term = memory_library.short_term

        logger.info("PerceptionFamiliar 初始化完成")

    async def submit_interaction(
        self,
        payload: InteractionPayload,
        target_topic_id: str = "NEW_TOPIC",
    ) -> str:
        """摄入完整交互载荷，并交给感知层完成话题路由。"""
        logger.info(
            "PerceptionFamiliar 摄入交互载荷: "
            "user='%s...', target_topic_id=%s, traces=%s, tasks=%s",
            payload.user_message[:30],
            target_topic_id,
            len(payload.mtp_traces),
            len(payload.materialize_tasks),
        )

        # 检查是否需要先驱逐 LRU 话题，独立发出 task
        await self._maybe_evict_lru()

        topic_id, settle_payload = await self.perception_layer.route_and_ingest(target_topic_id, payload)

        if settle_payload is not None:
            await self._bus.request(PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, settle_payload)
        return topic_id

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: Optional[str],
        new_topic_summary: Optional[str],
        identity: Identity,
    ) -> str:
        """确保目标短期话题存在，并返回真实 topic_id。"""
        # 检查是否需要先驱逐 LRU 话题，独立发出 task
        await self._maybe_evict_lru()

        return await self.perception_layer.prepare_topic(
            target_topic_id,
            new_topic_title,
            new_topic_summary,
            identity,
        )

    async def _maybe_evict_lru(self) -> None:
        """池满时驱逐 LRU 话题并提交结算任务。"""
        if not self._short_term.needs_eviction():
            return
            
        lru = self._short_term.get_lru_buffer()
        if lru is None:
            return

        settle_payload = await self.perception_layer.settle_topic(lru.topic_id, FlushReason.LRU_EVICTION)

        if settle_payload is not None:
            await self._bus.request(PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, settle_payload)

    async def manual_settle_topic(self, topic_id: Optional[str] = None) -> MemoryGenerationTask | None:
        """手动结算指定话题，返回生成任务句柄（None 表示话题为空无需生成）。"""
        target_id = topic_id or self._short_term.get_last_active_topic()
        if not target_id:
            raise ValueError("未指定 topic_id 且无活跃话题")

        topic = self._short_term.get_topic_data(target_id)
        if topic is None:
            raise KeyError(f"话题 {target_id} 不存在")
 
        if topic.is_empty:
            return None

        settle_payload = await self.perception_layer.settle_topic(
            target_id, FlushReason.MANUAL, wait_for_completion=True
        )
        if settle_payload is None:
            return None

        task: MemoryGenerationTask | None = await self._bus.request(
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
            settle_payload,
        )
        logger.info(
            "manual_settle_topic 完成: topic_id=%s, task_id=%s",
            target_id, task.task_id if task else None,
        )
        return task

    async def evict_topic(self, topic_id: str) -> TopicEvictResult:
        """从活跃话题池中驱逐话题，不触发结算。"""
        removed = self.perception_layer.swap_out_topic(topic_id)
        if not removed:
            return TopicEvictResult(success=False, message="话题不存在或已被驱逐")
        return TopicEvictResult(success=True, message=f"话题 {topic_id} 已删除")

    def discard_if_empty(self, topic_id: str) -> bool:
        """话题为空时清理该话题。"""
        return self.perception_layer.discard_if_empty(topic_id)

    async def scan_idle_buffers_once(self) -> list[str]:
        """扫描并 settle 空闲超时话题，策略由 Familiar 持有。"""
        flushed = []
        for topic in self._short_term.list_topic_data():
            if topic.is_idle(self._idle_timeout_seconds):
                logger.info(
                    "检测到空闲话题: topic_id=%s, idle_time=%.1fs",
                    topic.topic_id,
                    datetime.now().timestamp() - topic.last_update,
                )
                settle_payload = await self.perception_layer.settle_topic(
                    topic.topic_id, FlushReason.IDLE_TIMEOUT
                )
                if settle_payload is not None:
                    await self._bus.request(
                        PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, 
                        settle_payload
                    )
                flushed.append(topic.topic_id)
        return flushed

    async def flush_all_for_shutdown(self) -> ShutdownFlushResult:
        """服务关闭前强制结算所有活跃话题。"""
        flushed, skipped, archived_blocks = [], [], 0
        for topic in self._short_term.list_topic_data():
            if topic.is_empty:
                skipped.append(topic.topic_id)
                continue
            archived_blocks += topic.block_count
            _, settle_payload = await self.perception_layer.settle_topic(
                topic.topic_id, FlushReason.SHUTDOWN, wait_for_completion=True
            )
            if settle_payload is not None:
                await self._bus.request(
                    PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, 
                    settle_payload
                )
            flushed.append(topic.topic_id)

        logger.info(
            "shutdown flush 完成: flushed=%d, skipped=%d, archived_blocks=%d",
            len(flushed), len(skipped), archived_blocks,
        )
        
        return ShutdownFlushResult(
            success=True,
            trigger_reason=FlushReason.SHUTDOWN.value,
            flushed_topics=flushed,
            skipped_topics=skipped,
            archived_blocks=archived_blocks,
        )


__all__ = [
    "PerceptionFamiliar", 
    "TopicEvictResult", 
    "ShutdownFlushResult"
]
