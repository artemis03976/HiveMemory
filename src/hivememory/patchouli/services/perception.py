"""Patchouli 感知使魔。

承接感知层代理职责：接收已接纳的 Interaction、执行统一的 settle 外部时序
（begin -> 锁外 admission -> complete/abort）、把领域结果投影为公开报告。
Topic Buffer 状态、活跃池与 LRU/idle/shutdown 的状态执行由
``TopicBufferService`` 唯一拥有，本使魔不直接读写短期 Store。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from hivememory.core.models import IdentityScope, require_identity_scope
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.models import TopicMaterializeTask, TriggerReason
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
from hivememory.patchouli.services.topic_buffer import SettlementStatus

if TYPE_CHECKING:
    from hivememory.core.models import WorkspaceAssetRef
    from hivememory.engines.perception.interfaces import BasePerceptionLayer
    from hivememory.patchouli.services.topic_buffer import TopicBufferService
    from hivememory.system.config.patchouli import MemoryPerceptionConfig

logger = logging.getLogger(__name__)


# ========== PerceptionFamiliar ==========

class PerceptionFamiliar:
    """感知业务门面，负责摄入与短期话题管理。"""

    def __init__(
        self,
        *,
        perception_layer: "BasePerceptionLayer",
        topic_buffer: "TopicBufferService",
        bus,
        config: "MemoryPerceptionConfig",
        interaction_journal: InMemoryInteractionApplyJournal,
    ) -> None:
        self.perception_layer = perception_layer
        self._topic_buffer = topic_buffer
        self._bus = bus
        self._idle_timeout_seconds = config.idle_timeout_seconds
        config_engine = getattr(config, "engine", config)
        self._max_resident_topics = getattr(config_engine, "max_resident_topics", 5)
        self._last_active_topic_ids: dict[tuple[str, str], str] = {}
        self._interaction_journal = interaction_journal

        logger.info("PerceptionFamiliar 初始化完成")

    async def apply_interaction(
        self,
        payload: InteractionPayload,
        *,
        identity_scope: IdentityScope,
        target_topic_id: str = "NEW_TOPIC",
        interaction_id: str | None = None,
        asset_id_and_refs: tuple[tuple[str, "WorkspaceAssetRef"], ...] = (),
    ) -> str:
        """应用一份已由队列接纳的交互载荷，并完成话题路由。"""
        apply_record = (
            self._interaction_journal.get(interaction_id)
            if interaction_id
            else None
        )

        logger.info(
            "PerceptionFamiliar 应用交互载荷: "
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

        await self._submit_settlement_payload(settle_payload)

        workspace = identity_scope.workspace_identity
        self._topic_buffer.touch_topic(identity_scope, topic_id)
        self._last_active_topic_ids[
            (workspace.owner_user_id, workspace.workspace_id)
        ] = topic_id

        if interaction_id:
            apply_record = self._interaction_journal.get(interaction_id)
            if apply_record is not None:
                self._interaction_journal.complete(interaction_id, topic_id)
        return topic_id

    async def _submit_settlement_payload(
        self,
        settlement: TopicMaterializeTask | None,
    ) -> MemoryGenerationTask | None:
        """提交可选的 Topic settlement payload，并返回已接纳任务。

        ``None`` 表示当前 Topic 没有可提交的 generation 材料，是正常的 skip，
        而不是异常；admission 的异常由调用方按各自来源投影。
        """
        if settlement is None:
            return None
        return await self._bus.request(
            PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT,
            settlement,
        )

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

        real_topic_id = await self.perception_layer.prepare_topic(
            target_topic_id,
            new_topic_title,
            new_topic_summary,
            identity_scope,
        )
        if self._topic_buffer.touch_topic(identity_scope, real_topic_id) is not None:
            workspace = identity_scope.workspace_identity
            self._last_active_topic_ids[
                (workspace.owner_user_id, workspace.workspace_id)
            ] = real_topic_id
        return real_topic_id

    async def _settle_candidate(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        reason: TriggerReason,
        *,
        raise_on_admission_failure: bool,
    ) -> tuple[bool, MemoryGenerationTask | None]:
        """对单个话题执行统一 settle 外部时序：begin -> admission -> complete/abort。

        所有 ``settle=True`` 触发来源共用本时序，差异只体现在 admission 失败的
        投影上——``raise_on_admission_failure=True`` 时恢复 Topic 并向调用方传播
        可重试错误（manual / LRU backpressure），否则记录失败并等待下一次维护
        （idle）。begin 报告目标缺失、admission 失败或 complete 未删除时返回
        ``completed=False``，调用方不得把它当作成功清理。

        Returns:
            (话题生命周期是否已结束, 已接纳的 generation task 或 None)
        """
        # 1. begin：IDLE -> SETTLING，冻结材料（领域锁内）；busy 显式抛出。
        reservation = self._topic_buffer.begin_settlement(identity_scope, topic_id, reason)
        if reservation is None:
            # 目标已被其他生命周期操作移除。
            return False, None

        # 2. admission：领域锁外等待 Generation queue 接纳。
        try:
            task = await self._submit_settlement_payload(reservation.task)
        except Exception:
            self._topic_buffer.abort_settlement(identity_scope, topic_id, reason=reason)
            if raise_on_admission_failure:
                raise
            logger.warning(
                "settle admission 失败，Topic 保留并等待重试: topic_id=%s, reason=%s",
                topic_id,
                reason.value,
                exc_info=True,
            )
            return False, None

        # 3. complete：接纳成功或正常 skip 后结束 Topic 生命周期。
        outcome = self._topic_buffer.complete_settlement(
            identity_scope,
            topic_id,
            generation_task_id=task.task_id if task else None,
            reason=reason,
        )
        if not outcome.removed:
            # 目标在 admission 期间被并发操作移除；已接纳任务属于 Generation
            # 自身的后续终态，不再重开 Topic。
            logger.warning(
                "settle 完成时目标已不存在: topic_id=%s, status=%s",
                topic_id,
                outcome.status.value,
            )
            return False, task
        return True, task

    async def _maybe_evict_lru(
        self,
        identity_scope: IdentityScope,
        target_topic_id: str,
    ) -> None:
        """需要创建新话题且池满时，驱逐 LRU 话题并提交结算任务。"""
        # 已有话题无需驱逐；未知目标必须直接拒绝，不能被误当成 NEW_TOPIC。
        # 该检查位于 LRU 操作之前，保证跨 Workspace ID 不会产生本域副作用。
        if target_topic_id != "NEW_TOPIC":
            if self._topic_buffer.get_topic(identity_scope, target_topic_id, touch=False) is None:
                raise KeyError(
                    f"topic '{target_topic_id}' does not exist in requested Workspace"
                )
            return
        if self._topic_buffer.count_topics(identity_scope) < self._max_resident_topics:
            return

        attempted_topic_ids: set[str] = set()
        while self._topic_buffer.count_topics(identity_scope) >= self._max_resident_topics:
            # 候选选择（只挑 IDLE、排除已尝试）由领域服务决定。
            lru_topic_id = self._topic_buffer.select_lru_candidate(
                identity_scope, exclude_ids=attempted_topic_ids
            )
            if lru_topic_id is None:
                # 池满但无未尝试的 IDLE 候选，无法安全驱逐。
                raise TopicBusyError("LRU 驱逐无 IDLE 候选，稍后重试")

            try:
                completed, _ = await self._settle_candidate(
                    identity_scope,
                    lru_topic_id,
                    TriggerReason.LRU_EVICTION,
                    raise_on_admission_failure=True,
                )
            except TopicBusyError:
                # 候选在选择后进入 busy，改选其他 IDLE Topic。
                attempted_topic_ids.add(lru_topic_id)
                continue
            if not completed:
                # 候选已被其他生命周期操作移除；重新检查容量，而不是误报驱逐成功。
                attempted_topic_ids.add(lru_topic_id)
                continue
            # admission 成功或正常 skip 后 Topic 已被 complete 删除，容量释放。
            return

    async def manual_settle_topic(
        self,
        identity_scope: IdentityScope,
        topic_id: Optional[str] = None,
    ) -> TopicSettleResult:
        """手动结算指定话题：SETTLING begin -> admission -> complete/abort。

        先取得 SETTLING 预约并冻结 settlement 材料，存在可写材料时先可靠接纳
        memory generation task，接纳成功（或没有可提交材料）后再结束 Topic 生
        命周期。admission 失败时 abort 恢复 IDLE，Topic、blocks 与 state_summary
        保持完整可重试。
        """
        identity_scope = require_identity_scope(identity_scope)
        workspace = identity_scope.workspace_identity
        target_id = topic_id or self._last_active_topic_ids.get(
            (workspace.owner_user_id, workspace.workspace_id)
        )
        if not target_id:
            raise ValueError("未指定 topic_id 且无活跃话题")

        topic = self._topic_buffer.get_topic(identity_scope, target_id, touch=True)
        if topic is None:
            raise KeyError(f"话题 {target_id} 不存在")

        try:
            completed, task = await self._settle_candidate(
                identity_scope,
                target_id,
                TriggerReason.MANUAL_SETTLE,
                raise_on_admission_failure=True,
            )
        except TopicBusyError:
            # 目标正忙是瞬态冲突，不是 admission 失败，保持原语义向用户抛出。
            raise
        except Exception as exc:
            raise TopicSettleAdmissionError(
                f"结算材料接纳失败，话题内容已保留，可重试: {target_id}"
            ) from exc
        if not completed:
            if self._topic_buffer.get_topic(identity_scope, target_id, touch=False) is None:
                raise KeyError(f"话题 {target_id} 不存在")
            raise RuntimeError(f"话题 {target_id} 未能完成结算")

        self._last_active_topic_ids.pop(
            (workspace.owner_user_id, workspace.workspace_id),
            None,
        )
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
        removed = self._topic_buffer.delete_if_idle(identity_scope, topic_id)
        if removed:
            workspace = identity_scope.workspace_identity
            scope = (workspace.owner_user_id, workspace.workspace_id)
            if self._last_active_topic_ids.get(scope) == topic_id:
                self._last_active_topic_ids.pop(scope, None)
        return TopicEvictionResult(topic_id=topic_id, removed=removed)

    def discard_if_empty(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> bool:
        """话题为空时清理该话题。"""
        return self._topic_buffer.discard_if_empty(identity_scope, topic_id)

    async def scan_idle_buffers_once(self) -> list[str]:
        """扫描并 settle 空闲超时话题；候选由领域服务提供。"""
        flushed = []
        for candidate in self._topic_buffer.list_idle_candidates(
            self._idle_timeout_seconds
        ):
            logger.info(
                "检测到空闲话题: topic_id=%s, idle_time=%.1fs",
                candidate.topic_id,
                datetime.now(timezone.utc).timestamp() - candidate.last_update,
            )
            try:
                completed, _ = await self._settle_candidate(
                    candidate.identity_scope,
                    candidate.topic_id,
                    TriggerReason.IDLE_TIMEOUT,
                    raise_on_admission_failure=False,
                )
            except TopicBusyError:
                # snapshot 后进入 busy 的 Topic 留给后续维护轮次处理。
                continue
            if completed:
                flushed.append(candidate.topic_id)
        return flushed

    async def flush_all_for_shutdown(self) -> TopicShutdownFlushReport:
        """
        服务关闭前逐 Topic 执行统一 settle 协议并驱逐。

        真正空 Topic 没有可提交的结算材料，但仍按 SHUTDOWN 矩阵执行 evict，
        不留在活跃池中。没有建立 generation task 属于正常 skip；单个 Topic 的
        busy 或 admission 异常被隔离记录到报告的 ``failed_topic_ids``，不阻止
        其余 Topic 清理与已接纳任务的 drain。未完成 admission 的 Topic 通过
        abort 恢复 IDLE，不计入已完成清理。
        """
        candidates = self._topic_buffer.list_shutdown_candidates()
        resident_block_count = sum(candidate.block_count for candidate in candidates)
        settled_topic_ids: list[str] = []
        generation_skipped_topic_ids: list[str] = []
        failed_topic_ids: list[str] = []
        for candidate in candidates:
            try:
                reservation = self._topic_buffer.begin_settlement(
                    candidate.identity_scope,
                    candidate.topic_id,
                    TriggerReason.SHUTDOWN,
                )
            except TopicBusyError:
                # 无法安全结算（仍被 Interaction/compact 占用），隔离记录。
                failed_topic_ids.append(candidate.topic_id)
                continue
            if reservation is None:
                # 目标已被其他生命周期操作移除，不把它伪装成本次 settled。
                continue
            try:
                task = await self._submit_settlement_payload(reservation.task)
            except Exception:
                logger.exception(
                    "shutdown settle admission 失败: topic_id=%s", candidate.topic_id
                )
                self._topic_buffer.abort_settlement(
                    candidate.identity_scope,
                    candidate.topic_id,
                    reason=TriggerReason.SHUTDOWN,
                )
                failed_topic_ids.append(candidate.topic_id)
                continue
            outcome = self._topic_buffer.complete_settlement(
                candidate.identity_scope,
                candidate.topic_id,
                generation_task_id=task.task_id if task else None,
                reason=TriggerReason.SHUTDOWN,
            )
            if not outcome.removed:
                if outcome.status is SettlementStatus.NOT_FOUND:
                    # 已被其他生命周期操作移除，不记为失败也不计入本次 settled。
                    continue
                failed_topic_ids.append(candidate.topic_id)
                continue
            settled_topic_ids.append(candidate.topic_id)
            if task is None:
                generation_skipped_topic_ids.append(candidate.topic_id)

        logger.info(
            "shutdown flush 完成: settled=%d, generation_skipped=%d, failed=%d, "
            "resident_blocks=%d",
            len(settled_topic_ids),
            len(generation_skipped_topic_ids),
            len(failed_topic_ids),
            resident_block_count,
        )

        return TopicShutdownFlushReport(
            settled_topic_ids=tuple(settled_topic_ids),
            generation_skipped_topic_ids=tuple(generation_skipped_topic_ids),
            resident_block_count=resident_block_count,
            failed_topic_ids=tuple(failed_topic_ids),
        )


__all__ = [
    "PerceptionFamiliar",
]
