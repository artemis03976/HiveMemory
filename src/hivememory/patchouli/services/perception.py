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
    WorkspaceAssetRef,
    WorkspaceTopicKey,
    require_identity_scope,
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
from hivememory.patchouli.errors import TopicBusyError, TopicSettleAdmissionError
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
        asset_id_and_refs: tuple[tuple[str, WorkspaceAssetRef], ...] = (),
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
                        asset_id_and_refs,
                    )
            finally:
                await self._release_interaction_gate(interaction_id, gate)

        return await self._submit_interaction_once(
            payload,
            identity_scope,
            target_topic_id,
            None,
            asset_id_and_refs,
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
        asset_id_and_refs: tuple[tuple[str, WorkspaceAssetRef], ...] = (),
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
                asset_id_and_refs=asset_id_and_refs,
            )
        else:
            topic_id, settle_payload = await self.perception_layer.route_and_ingest(
                target_topic_id,
                payload,
                identity_scope=identity_scope,
                interaction_id=interaction_id,
                asset_id_and_refs=asset_id_and_refs,
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
        identity_scope: IdentityScope,
    ) -> str:
        """确保目标短期话题存在，并返回真实 topic_id。"""
        identity_scope = require_identity_scope(identity_scope)
        # 检查是否需要先驱逐 LRU 话题，独立发出 task
        await self._maybe_evict_lru(identity_scope, target_topic_id)

        return await self.perception_layer.prepare_topic(
            target_topic_id,
            new_topic_title,
            new_topic_summary,
            identity_scope,
        )

    async def _maybe_evict_lru(
        self,
        identity_scope: IdentityScope,
        target_topic_id: str,
    ) -> None:
        """需要创建新话题且池满时，驱逐 LRU 话题并提交结算任务。"""
        # 已有话题无需驱逐；未知目标必须直接拒绝，不能被误当成 NEW_TOPIC。
        # 该检查位于 LRU 操作之前，保证跨 Workspace ID 不会产生本域副作用。
        if target_topic_id != "NEW_TOPIC":
            if not self._short_term.topic_exists(
                identity_scope,
                target_topic_id,
                touch=False,
            ):
                raise KeyError(
                    f"topic '{target_topic_id}' does not exist in requested Workspace"
                )
            return
        if not self._short_term.needs_eviction(identity_scope):
            return

        attempted_topic_ids: set[str] = set()
        while self._short_term.needs_eviction(identity_scope):
            lru_topic_id = self._short_term.get_lru_topic(identity_scope)
            if lru_topic_id is None or lru_topic_id in attempted_topic_ids:
                # 池满但无未尝试的 IDLE 候选，无法安全驱逐。
                raise TopicBusyError("LRU 驱逐无 IDLE 候选，稍后重试")

            try:
                settle_result = await self.perception_layer.settle_topic(
                    WorkspaceTopicKey.from_identity_scope(identity_scope, lru_topic_id),
                    FlushReason.LRU_EVICTION,
                )
            except TopicBusyError:
                # 候选在选择后进入 busy，重新读取 Store 以改选其他 IDLE Topic。
                attempted_topic_ids.add(lru_topic_id)
                continue

            if not settle_result.evicted:
                # 候选已被其他生命周期操作移除；重新检查容量，而不是误报驱逐成功。
                attempted_topic_ids.add(lru_topic_id)
                continue
            if settle_result.settlement is not None:
                await self._bus.request(
                    PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
                    settle_result.settlement,
                )
            return

    async def manual_settle_topic(
        self,
        identity_scope: IdentityScope,
        topic_id: Optional[str] = None,
    ) -> TopicSettleResult:
        """手动结算指定话题：FLUSHING prepare -> admission -> commit/abort。

        先取得 FLUSHING 预约并冻结 settlement 材料，存在可写材料时先可靠接纳
        memory generation task，接纳成功（或没有可提交材料）后再结束 Topic 生命
        周期。admission 失败时 abort 恢复 IDLE，Topic、blocks 与 state_summary
        保持完整可重试。
        """
        identity_scope = require_identity_scope(identity_scope)
        target_id = topic_id or self._short_term.get_last_active_topic(identity_scope)
        if not target_id:
            raise ValueError("未指定 topic_id 且无活跃话题")

        topic = self._short_term.get_topic_data(identity_scope, target_id)
        if topic is None:
            raise KeyError(f"话题 {target_id} 不存在")

        # 1. prepare：取得 FLUSHING 并只冻结材料，不清 blocks、不驱逐
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
                self.perception_layer.abort_settlement(topic.topic_key)
                raise TopicSettleAdmissionError(
                    f"结算材料接纳失败，话题内容已保留，可重试: {target_id}"
                ) from exc

        # 3. commit：接纳成功或正常 skip 后结束 Topic 生命周期
        self.perception_layer.commit_settlement(topic.topic_key)
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
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> TopicEvictionResult:
        """从活跃话题池中驱逐话题，不触发结算。"""
        key = WorkspaceTopicKey.from_identity_scope(identity_scope, topic_id)
        removed = self.perception_layer.swap_out_topic(key)
        return TopicEvictionResult(topic_id=topic_id, removed=removed)

    def discard_if_empty(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> bool:
        """话题为空时清理该话题。"""
        return self.perception_layer.discard_if_empty(identity_scope, topic_id)

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
                try:
                    settle_result = await self.perception_layer.settle_topic(
                        topic.topic_key, FlushReason.IDLE_TIMEOUT
                    )
                except TopicBusyError:
                    # snapshot 后进入 busy 的 Topic 留给后续维护轮次处理。
                    continue
                if not settle_result.evicted:
                    continue
                if settle_result.settlement is not None:
                    await self._bus.request(
                        PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
                        settle_result.settlement,
                    )
                flushed.append(topic.topic_id)
        return flushed

    async def flush_all_for_shutdown(self) -> TopicShutdownFlushReport:
        """
        服务关闭前强制结算并驱逐所有活跃话题。

        真正空 Topic 没有可提交的结算材料，但仍按 SHUTDOWN 矩阵执行 evict，
        不留在活跃池中。没有建立 generation task 属于正常 skip；异常仍向上
        传播，不能伪装成 skip。
        """
        settled_topic_ids: list[str] = []
        generation_skipped_topic_ids: list[str] = []
        resident_block_count = 0
        for topic in self._short_term.list_all_topic_data_for_maintenance():
            resident_block_count += topic.block_count
            settle_result = await self.perception_layer.settle_topic(
                topic.topic_key, FlushReason.SHUTDOWN
            )
            if not settle_result.evicted:
                # 目标已由其他生命周期操作移除，不把它伪装成本次 settled。
                continue
            task: MemoryGenerationTask | None = None
            if settle_result.settlement is not None:
                task = await self._bus.request(
                    PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
                    settle_result.settlement,
                )
            settled_topic_ids.append(topic.topic_id)
            if task is None:
                generation_skipped_topic_ids.append(topic.topic_id)

        logger.info(
            "shutdown flush 完成: settled=%d, generation_skipped=%d, "
            "resident_blocks=%d",
            len(settled_topic_ids),
            len(generation_skipped_topic_ids),
            resident_block_count,
        )

        return TopicShutdownFlushReport(
            settled_topic_ids=tuple(settled_topic_ids),
            generation_skipped_topic_ids=tuple(generation_skipped_topic_ids),
            resident_block_count=resident_block_count,
        )


__all__ = [
    "PerceptionFamiliar",
]
