"""Patchouli 感知使魔。

承接感知层代理职责，让 PerceptionLayer 保持为不感知总线和生成链路的底层 engine。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from hivememory.core.models import (
    IdentityScope,
    WorkspaceAccessContext,
    WorkspaceTopicKey,
    require_workspace_access_context,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.models import FlushReason
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.topic_management import (
    TopicEvictionResult,
    TopicSettleResult,
)
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
)
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
from hivememory.patchouli.errors import TopicSettleAdmissionError
from hivememory.patchouli.runtime.models import TopicShutdownFlushReport

if TYPE_CHECKING:
    from hivememory.engines.perception.interfaces import BasePerceptionLayer
    from hivememory.patchouli.memory_library.library import MemoryLibrary
    from hivememory.patchouli.runtime.bus import PatchouliBus
    from hivememory.system.config.patchouli import MemoryPerceptionConfig

logger = logging.getLogger(__name__)


@dataclass
class _InteractionGate:
    """仅在同一 interaction 的并发调用期间存在。"""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


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
        interaction_journal: InMemoryInteractionApplyJournal,
    ) -> None:
        self.perception_layer = perception_layer
        self._bus = bus
        self._idle_timeout_seconds = config.idle_timeout_seconds
        self._short_term = memory_library.short_term
        self._interaction_journal = interaction_journal
        self._interaction_gates: dict[str, _InteractionGate] = {}
        self._interaction_gates_lock = asyncio.Lock()

        logger.info("PerceptionFamiliar 初始化完成")

    async def submit_interaction(
        self,
        payload: InteractionPayload,
        *,
        identity_scope: IdentityScope,
        target_topic_id: str = "NEW_TOPIC",
        interaction_id: str | None = None,
    ) -> str:
        """摄入完整交互载荷，并交给感知层完成话题路由。"""
        if interaction_id:
            gate = await self._acquire_interaction_gate(interaction_id)
            try:
                async with gate.lock:
                    return await self._submit_interaction_once(
                        payload,
                        identity_scope,
                        target_topic_id,
                        interaction_id,
                    )
            finally:
                await self._release_interaction_gate(interaction_id, gate)

        return await self._submit_interaction_once(
            payload,
            identity_scope,
            target_topic_id,
            None,
        )

    async def _acquire_interaction_gate(
        self,
        interaction_id: str,
    ) -> _InteractionGate:
        async with self._interaction_gates_lock:
            gate = self._interaction_gates.get(interaction_id)
            if gate is None:
                gate = _InteractionGate()
                self._interaction_gates[interaction_id] = gate
            gate.users += 1
            return gate

    async def _release_interaction_gate(
        self,
        interaction_id: str,
        gate: _InteractionGate,
    ) -> None:
        async with self._interaction_gates_lock:
            gate.users -= 1
            if (
                gate.users == 0
                and self._interaction_gates.get(interaction_id) is gate
            ):
                self._interaction_gates.pop(interaction_id, None)

    async def _submit_interaction_once(
        self,
        payload: InteractionPayload,
        identity_scope: IdentityScope,
        target_topic_id: str,
        interaction_id: str | None,
    ) -> str:
        """执行一次实际摄入；分阶段幂等真相由 raw perception journal 保存。"""
        apply_record = (
            self._interaction_journal.get(interaction_id)
            if interaction_id
            else None
        )

        logger.info(
            "PerceptionFamiliar 摄入交互载荷: "
            "user='%s...', target_topic_id=%s, traces=%s, tasks=%s",
            payload.user_message[:30],
            target_topic_id,
            len(payload.mtp_traces),
            len(payload.materialize_tasks),
        )

        # retry 已有 apply journal 时不能驱逐刚刚写入的目标话题。
        if apply_record is None:
            await self._maybe_evict_lru(identity_scope, target_topic_id)

        if interaction_id is None:
            topic_id, settle_payload = await self.perception_layer.route_and_ingest(
                target_topic_id,
                payload,
                identity_scope=identity_scope,
            )
        else:
            topic_id, settle_payload = await self.perception_layer.route_and_ingest(
                target_topic_id,
                payload,
                identity_scope=identity_scope,
                interaction_id=interaction_id,
            )

        if settle_payload is not None:
            await self._bus.request(
                PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
                settle_payload,
            )

        if interaction_id:
            apply_record = self._interaction_journal.get(interaction_id)
            if apply_record is not None:
                self._interaction_journal.complete(interaction_id, topic_id)
        return topic_id

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: Optional[str],
        new_topic_summary: Optional[str],
        access_context: WorkspaceAccessContext,
    ) -> str:
        """确保目标短期话题存在，并返回真实 topic_id。"""
        access_context = require_workspace_access_context(access_context)
        # 检查是否需要先驱逐 LRU 话题，独立发出 task
        await self._maybe_evict_lru(access_context, target_topic_id)

        return await self.perception_layer.prepare_topic(
            target_topic_id,
            new_topic_title,
            new_topic_summary,
            access_context,
        )

    async def _maybe_evict_lru(
        self,
        access_context: WorkspaceAccessContext,
        target_topic_id: str,
    ) -> None:
        """需要创建新话题且池满时，驱逐 LRU 话题并提交结算任务。"""
        # 命中已有话题时无需驱逐
        if (
            target_topic_id != "NEW_TOPIC"
            and self._short_term.topic_exists(access_context, target_topic_id)
        ):
            return
        if not self._short_term.needs_eviction(access_context):
            return

        lru_topic_id = self._short_term.get_lru_topic(access_context)
        if lru_topic_id is None:
            return

        settle_payload = await self.perception_layer.settle_topic(
            WorkspaceTopicKey.from_access_context(access_context, lru_topic_id),
            FlushReason.LRU_EVICTION,
        )

        if settle_payload is not None:
            await self._bus.request(PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, settle_payload)

    async def manual_settle_topic(
        self,
        access_context: WorkspaceAccessContext,
        topic_id: Optional[str] = None,
    ) -> TopicSettleResult:
        """手动结算指定话题：prepare -> admission -> evict。

        冻结 settlement 材料，存在可写材料时先可靠接纳 memory generation task，
        接纳成功（或没有可提交材料）后再结束 Topic 生命周期。admission 失败时
        抛出受控错误，Topic、blocks 与 state_summary 保持完整可重试。
        """
        access_context = require_workspace_access_context(access_context)
        target_id = topic_id or self._short_term.get_last_active_topic(access_context)
        if not target_id:
            raise ValueError("未指定 topic_id 且无活跃话题")

        topic = self._short_term.get_topic_data(access_context, target_id)
        if topic is None:
            raise KeyError(f"话题 {target_id} 不存在")

        # 1. prepare：只冻结材料，不清 blocks、不驱逐
        settle_payload = await self.perception_layer.prepare_settlement(
            topic.topic_key
        )

        # 2. admission：存在可提交材料时必须先可靠接纳
        task: MemoryGenerationTask | None = None
        if settle_payload is not None:
            try:
                task = await self._bus.request(
                    PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
                    settle_payload,
                )
            except Exception as exc:
                logger.warning(
                    "manual settle admission 失败: topic_id=%s", target_id,
                    exc_info=True,
                )
                raise TopicSettleAdmissionError(
                    f"结算材料接纳失败，话题内容已保留，可重试: {target_id}"
                ) from exc

        # 3. evict：接纳成功或无可提交材料后结束 Topic 生命周期
        self.perception_layer.swap_out_topic(topic.topic_key)
        logger.info(
            "manual_settle_topic 完成: topic_id=%s, task_id=%s",
            target_id, task.task_id if task else None,
        )
        return TopicSettleResult(
            topic_id=target_id,
            generation_task_id=task.task_id if task else None,
        )

    async def evict_topic(
        self,
        access_context: WorkspaceAccessContext,
        topic_id: str,
    ) -> TopicEvictionResult:
        """从活跃话题池中驱逐话题，不触发结算。"""
        key = WorkspaceTopicKey.from_access_context(
            require_workspace_access_context(access_context),
            topic_id,
        )
        removed = self.perception_layer.swap_out_topic(key)
        return TopicEvictionResult(topic_id=topic_id, removed=removed)

    def discard_if_empty(
        self,
        access_context: WorkspaceAccessContext,
        topic_id: str,
    ) -> bool:
        """话题为空时清理该话题。"""
        return self.perception_layer.discard_if_empty(access_context, topic_id)

    async def scan_idle_buffers_once(self) -> list[str]:
        """扫描并 settle 空闲超时话题，策略由 Familiar 持有。"""
        flushed = []
        for topic in self._short_term.list_all_topic_data_for_maintenance():
            if topic.is_idle(self._idle_timeout_seconds):
                logger.info(
                    "检测到空闲话题: topic_id=%s, idle_time=%.1fs",
                    topic.topic_id,
                    datetime.now(timezone.utc).timestamp() - topic.last_update,
                )
                settle_payload = await self.perception_layer.settle_topic(
                    topic.topic_key, FlushReason.IDLE_TIMEOUT
                )
                if settle_payload is not None:
                    await self._bus.request(
                        PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
                        settle_payload
                    )
                flushed.append(topic.topic_id)
        return flushed

    async def flush_all_for_shutdown(self) -> TopicShutdownFlushReport:
        """
        服务关闭前强制结算并驱逐所有活跃话题。

        真正空 Topic 没有可提交的结算材料，但仍按 SHUTDOWN 矩阵执行 evict，
        不留在活跃池中。
        """
        settled_topic_ids: list[str] = []
        resident_block_count = 0
        for topic in self._short_term.list_all_topic_data_for_maintenance():
            resident_block_count += topic.block_count
            settle_payload = await self.perception_layer.settle_topic(
                topic.topic_key, FlushReason.SHUTDOWN
            )
            if settle_payload is not None:
                await self._bus.request(
                    PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
                    settle_payload
                )
            settled_topic_ids.append(topic.topic_id)

        logger.info(
            "shutdown flush 完成: settled=%d, resident_blocks=%d",
            len(settled_topic_ids), resident_block_count,
        )

        return TopicShutdownFlushReport(
            settled_topic_ids=tuple(settled_topic_ids),
            resident_block_count=resident_block_count,
        )


__all__ = [
    "PerceptionFamiliar",
]
