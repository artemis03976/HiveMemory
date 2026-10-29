"""Patchouli 感知使魔。

感知业务的唯一编排者：持有 Engine（纯算法，含 RelayController）、Store（内容）
与 WorkingSet（驻留与 lease），负责话题路由、Interaction 原子写入、话题折叠、
统一 settle 时序、LRU / idle / shutdown 维护，并把领域结果投影为公开报告。

折叠是话题自身的维护，不属于交互应用：应用在写入 block 后即完成，是否需要
折叠由 ``needs_compact`` 判断，``compact_topic`` 作为独立用例执行。

占用权统一由 WorkingSet 的 lease 表达：所有写路径都是
``acquire → try/finally release``；Relay 摘要生成在 lease 持有期间进行，
其他调用无法取得同一话题的占用权，因此无需旧的记录状态补偿校验。
统一 settle 时序（``_settle_topic``）为所有来源共用；``reason`` 只作为
材料的 provenance 标签，不驱动分支。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from hivememory.components.work_queue import WorkQueueCapacityError, WorkQueueStoppedError
from hivememory.core.errors import ResourceNotFoundError
from hivememory.core.models import (
    ActorIdentity,
    LogicalBlock,
    TopicData,
    WorkspaceAssetRef,
    WorkspaceIdentity,
    merge_interaction_into_topic,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.engines.perception.models import TopicMaterializeTask, TriggerReason
from hivememory.patchouli.contracts.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.topic_management import TopicEvictionResult, TopicSettleResult
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
from hivememory.patchouli.runtime.bus import PatchouliBus
from hivememory.patchouli.runtime.models import TopicShutdownFlushReport
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet

if TYPE_CHECKING:
    from hivememory.config.patchouli import MemoryPerceptionConfig
    from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore

logger = logging.getLogger(__name__)


class PerceptionFamiliar:
    """感知业务门面：Engine + Store + WorkingSet 的编排者。

    ``engine=None`` 表示感知关闭（原 ``NullPerceptionLayer`` 语义）：摄入
    路径无任何存储副作用；维护用例对空存储自然空转。
    """

    def __init__(
        self,
        *,
        engine: MemoryPerceptionEngine | None,
        store: ShortTermMemoryStore,
        working_set: TopicWorkingSet,
        bus: PatchouliBus,
        config: MemoryPerceptionConfig,
    ) -> None:
        self._engine = engine
        self._store = store
        self._working_set = working_set
        self._bus = bus
        self._idle_timeout_seconds = config.idle_timeout_seconds

    # ========== Interaction 摄入 ==========

    async def apply_interaction(
        self,
        payload: InteractionPayload,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        target_topic_id: str = "NEW_TOPIC",
        interaction_id: str | None = None,
        asset_refs: tuple[WorkspaceAssetRef, ...] = (),
        new_topic_title: str | None = None,
        new_topic_summary: str | None = None,
    ) -> str:
        """应用一份已接纳的交互载荷并完成话题路由（核心用例）。

        本轮 block 与首次 asset binding 一次写入话题即完成，不执行折叠。写入是
        这次应用中最后一个可能失败的步骤：应用要么失败且没有写入，要么写入并
        成功。提交队列只允许未成功结束的交互重新提交，因此同一轮不会被写入两次；
        在写入之后加入可能失败的步骤会破坏这一点。
        目标为 ``NEW_TOPIC`` 时以提交携带的标题与摘要建立话题，后续落位
        依赖候选话题的标题与摘要；缺省时使用默认标题。
        """
        if self._engine is None:  # 感知关闭：无任何存储副作用
            return target_topic_id
        if not payload.turn_events:
            raise ValueError(
                "InteractionPayload.turn_events is required; "
                "legacy assistant_message fallback has been removed."
            )

        logger.info("PerceptionFamiliar 应用交互载荷: user='%s...'", payload.user_message[:30])
        await self._maybe_evict_lru(belong_to, target_topic_id)
        topic_id = self._ensure_topic(
            belong_to,
            target_topic_id,
            topic_title=new_topic_title,
            topic_summary=new_topic_summary,
        )
        block = self._engine.build_block(payload, from_actor)

        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法原子摄入 interaction，可稍后重试")
        try:
            topic = self._store.get(belong_to, topic_id)
            if topic is None:
                raise ResourceNotFoundError(details={"resource": "topic", "topic_id": topic_id})
            self._store.put(
                merge_interaction_into_topic(
                    topic, block, asset_refs, interaction_id, payload.model_used
                )
            )
        finally:
            self._working_set.release(lease)

        self._refresh_residency(belong_to, topic_id)
        return topic_id

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        belong_to: WorkspaceIdentity,
    ) -> str:
        """确保目标短期话题存在（必要时先执行 LRU 驱逐），返回真实 topic_id。"""
        if self._engine is None:  # 感知关闭：不创建话题
            self._refresh_residency(belong_to, target_topic_id)
            return target_topic_id
        await self._maybe_evict_lru(belong_to, target_topic_id)
        topic_id = self._ensure_topic(
            belong_to,
            target_topic_id,
            topic_title=new_topic_title,
            topic_summary=new_topic_summary,
        )
        self._refresh_residency(belong_to, topic_id)
        return topic_id

    # ========== 统一 settle 协议 ==========

    async def _settle_topic(
        self,
        belong_to: WorkspaceIdentity,
        topic_id: str,
        *,
        reason: TriggerReason,
        raise_on_admission_failure: bool,
    ) -> tuple[bool, MemoryGenerationTask | None]:
        """统一 settle 时序：获取 lease → 冻结材料 → 锁外 admission → 删除话题。

        所有 settle 来源（manual / idle / LRU / shutdown）共用。admission
        失败时话题原样保留可重试，无需 abort——记录从未被改动。
        返回 (生命周期是否结束, 已接纳 task 或 None)。
        """
        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法开始结算")
        try:
            topic = self._store.get(belong_to, topic_id)
            if topic is None:
                self._working_set.remove(belong_to, topic_id)  # 清理陈旧驻留条目
                return False, None
            # 冻结材料：字段转换与 no-material 判断收口在数据模型内。
            task = TopicMaterializeTask.from_topic_data(topic, belong_to=belong_to, reason=reason)
            if task is None:  # 无可保存材料：正常结束生命周期，不触发生成
                self._store.delete(belong_to, topic_id)
                self._working_set.remove(belong_to, topic_id)
                return True, None

            # admission：lease 持有期间等待 Generation queue 接纳。
            try:
                generation_task = await self._bus.request(
                    PatchouliLocalRoutes.GENERATION_SUBMIT_SETTLEMENT, task
                )
            except Exception:
                if raise_on_admission_failure:
                    raise
                logger.warning(
                    "settle admission 失败，Topic 保留: topic_id=%s, reason=%s",
                    topic_id,
                    reason.value,
                    exc_info=True,
                )
                return False, None

            self._store.delete(belong_to, topic_id)
            self._working_set.remove(belong_to, topic_id)
            return True, generation_task
        finally:
            self._working_set.release(lease)

    async def manual_settle_topic(
        self, belong_to: WorkspaceIdentity, topic_id: str
    ) -> TopicSettleResult:
        """手动结算指定话题；话题不存在时抛出 :class:`ResourceNotFoundError`。

        队列明确拒绝接纳时抛出 :class:`TopicSettleAdmissionError`（话题内容保留
        可重试）；目标正忙是瞬态冲突，保持 :class:`TopicBusyError` 语义。
        """
        if self._store.get(belong_to, topic_id) is None:
            raise ResourceNotFoundError(details={"resource": "topic", "topic_id": topic_id})
        try:
            completed, task = await self._settle_topic(
                belong_to,
                topic_id,
                reason=TriggerReason.MANUAL_SETTLE,
                raise_on_admission_failure=True,
            )
        except (WorkQueueCapacityError, WorkQueueStoppedError) as exc:
            raise TopicSettleAdmissionError(
                f"结算材料接纳失败，话题内容已保留，可重试: {topic_id}"
            ) from exc
        if not completed:
            raise ResourceNotFoundError(details={"resource": "topic", "topic_id": topic_id})
        return TopicSettleResult(
            topic_id=topic_id, generation_task_id=task.task_id if task else None
        )

    # ========== Compact（页面折叠） ==========

    def needs_compact(self, belong_to: WorkspaceIdentity, topic_id: str) -> bool:
        """话题超过折叠阈值且有可折叠的旧前缀时为真；话题不存在或感知关闭时为假。"""
        topic = self._store.get(belong_to, topic_id)
        return topic is not None and bool(self._blocks_to_fold(topic))

    async def compact_topic(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """折叠用例：取得话题的 lease 后，在超过阈值时折叠旧前缀。

        话题已被结算或驱逐时直接结束；正被占用时抛出 :class:`TopicBusyError`
        留待重试。摘要生成期间持有 lease，同一话题的写入与结算不会交错；写回在
        摘要返回后一次完成，中途取消或失败都不改动话题。
        """
        if self._engine is None:  # 感知关闭：不做任何折叠
            return
        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法折叠，可稍后重试")
        try:
            await self._compact_topic_if_needed(belong_to, topic_id)
        finally:
            self._working_set.release(lease)

    def _blocks_to_fold(self, topic: TopicData) -> list[LogicalBlock]:
        """超过折叠阈值时返回保留后缀之前的旧 blocks；未超过或无可折叠前缀时为空。"""
        engine = self._engine
        if engine is None or topic.is_empty or not engine.should_compact(topic.total_tokens):
            return []
        # 保留块数以 Engine 配置为单一事实源。
        return engine.select_blocks_to_fold(topic.blocks, engine.config.fold_retain_recent_blocks)

    async def _compact_topic_if_needed(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """token 溢出时执行 compact：生成折叠摘要并写回保留的近期 blocks。

        调用方必须已持有该话题的 lease：Relay 摘要生成期间其他调用无法取
        得同一话题的占用权，因此写回不再需要旧的 expected_state 补偿校验。
        """
        engine = self._engine
        if engine is None:  # 感知关闭：与 apply 一致，不做任何折叠
            return
        topic = self._store.get(belong_to, topic_id)
        if topic is None:
            return
        blocks_to_fold = self._blocks_to_fold(topic)
        if not blocks_to_fold:
            return
        logger.info("Token 溢出，触发 Page Folding: topic_id=%s", topic_id)
        summary = await engine.generate_fold_summary(blocks_to_fold, topic.state_summary)
        retained = topic.blocks[len(blocks_to_fold) :]
        self._store.put(
            topic.model_copy(
                update={
                    "state_summary": summary,
                    "blocks": retained,
                    "total_tokens": sum(block.total_tokens for block in retained),
                    "last_update": datetime.now().timestamp(),
                }
            )
        )

    # ========== LRU / 驱逐 ==========

    async def _maybe_evict_lru(self, belong_to: WorkspaceIdentity, target_topic_id: str) -> None:
        """需要创建新话题且池满时，驱逐 LRU 话题并提交结算任务。

        已有话题无需驱逐；未知目标必须直接拒绝。容量与候选由 WorkingSet
        决定；候选正被占用时改选其他话题。
        """
        if target_topic_id != "NEW_TOPIC":
            if self._store.get(belong_to, target_topic_id) is None:
                raise ResourceNotFoundError(
                    details={"resource": "topic", "topic_id": target_topic_id}
                )
            return
        if not self._working_set.needs_eviction(belong_to):
            return
        attempted: set[str] = set()
        while self._working_set.needs_eviction(belong_to):
            lru_topic_id = self._working_set.select_lru_candidate(belong_to, exclude=attempted)
            if lru_topic_id is None:  # 池满但全部候选正被占用，无法安全驱逐
                raise TopicBusyError("LRU 驱逐无可占用候选，稍后重试")
            try:
                completed, _ = await self._settle_topic(
                    belong_to,
                    lru_topic_id,
                    reason=TriggerReason.LRU_EVICTION,
                    raise_on_admission_failure=True,
                )
            except TopicBusyError:  # 候选在选择后进入 busy，改选其他话题
                attempted.add(lru_topic_id)
                continue
            if not completed:  # 候选已被其他生命周期操作移除，重查容量后改选
                attempted.add(lru_topic_id)
                continue
            return  # admission 成功或无材料完成后 Topic 已删除，容量释放

    async def evict_topic(self, belong_to: WorkspaceIdentity, topic_id: str) -> TopicEvictionResult:
        """从活跃话题池中驱逐话题，不触发结算；正被占用时报 ``removed=False``。"""
        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:
            return TopicEvictionResult(topic_id=topic_id, removed=False)
        try:
            removed = self._store.delete(belong_to, topic_id)
            if removed:
                self._working_set.remove(belong_to, topic_id)
            return TopicEvictionResult(topic_id=topic_id, removed=removed)
        finally:
            self._working_set.release(lease)

    def discard_if_empty(self, belong_to: WorkspaceIdentity, topic_id: str) -> bool:
        """话题真正为空（无 blocks 且无非空白折叠摘要）时清理并返回 True。"""
        lease = self._working_set.acquire(belong_to, topic_id)
        if lease is None:  # 正被占用（可能正在写入首块），留给后续维护判断
            return False
        try:
            topic = self._store.get(belong_to, topic_id)
            if topic is None or not topic.is_empty:
                return False
            removed = self._store.delete(belong_to, topic_id)
            if removed:
                self._working_set.remove(belong_to, topic_id)
            return removed
        finally:
            self._working_set.release(lease)

    # ========== 维护与 shutdown ==========

    async def scan_idle_buffers_once(self) -> list[str]:
        """扫描并 settle 空闲超时话题；候选由 WorkingSet 提供。"""
        flushed: list[str] = []
        for belong_to, topic_id in self._working_set.list_idle_candidates(
            self._idle_timeout_seconds
        ):
            logger.info("检测到空闲话题: topic_id=%s", topic_id)
            try:
                completed, _ = await self._settle_topic(
                    belong_to,
                    topic_id,
                    reason=TriggerReason.IDLE_TIMEOUT,
                    raise_on_admission_failure=False,
                )
            except TopicBusyError:  # snapshot 后进入 busy，留给后续维护轮次
                continue
            if completed:
                flushed.append(topic_id)
        return flushed

    async def flush_all_for_shutdown(self) -> TopicShutdownFlushReport:
        """服务关闭前逐话题执行统一 settle 协议。

        单个话题的 busy 或 admission 异常被隔离记录到 ``failed_topic_ids``，
        不阻止其余清理；无材料话题正常结束生命周期并计入 generation_skipped。
        """
        candidates = self._working_set.list_shutdown_candidates()
        resident_block_count = 0
        for belong_to, topic_id in candidates:
            topic = self._store.get(belong_to, topic_id)
            if topic is not None:
                resident_block_count += topic.block_count

        settled: list[str] = []
        generation_skipped: list[str] = []
        failed: list[str] = []
        for belong_to, topic_id in candidates:
            try:
                completed, task = await self._settle_topic(
                    belong_to,
                    topic_id,
                    reason=TriggerReason.SHUTDOWN,
                    raise_on_admission_failure=True,
                )
            except Exception:
                logger.exception("shutdown settle 失败: topic_id=%s", topic_id)
                failed.append(topic_id)
                continue
            if not completed:  # 目标已被其他生命周期操作移除，不计入本次 settled
                continue
            settled.append(topic_id)
            if task is None:
                generation_skipped.append(topic_id)

        logger.info(
            "shutdown flush 完成: settled=%d, skipped=%d, failed=%d, blocks=%d",
            len(settled),
            len(generation_skipped),
            len(failed),
            resident_block_count,
        )
        return TopicShutdownFlushReport(
            settled_topic_ids=tuple(settled),
            generation_skipped_topic_ids=tuple(generation_skipped),
            resident_block_count=resident_block_count,
            failed_topic_ids=tuple(failed),
        )

    # ========== 内部实现 ==========

    def _ensure_topic(
        self,
        belong_to: WorkspaceIdentity,
        target_topic_id: str,
        *,
        topic_title: str | None = None,
        topic_summary: str | None = None,
    ) -> str:
        """确保目标话题存在并返回真实 topic_id；未知目标显式拒绝。"""
        if target_topic_id == "NEW_TOPIC":
            topic = self._store.create(
                belong_to,
                topic_title=topic_title or "新建话题",
                topic_summary=topic_summary or "",
            )
            # 创建即入池：避免 Store 有记录而 WorkingSet 不可见的孤儿窗口。
            self._working_set.touch(belong_to, topic.topic_id)
            return topic.topic_id
        if self._store.get(belong_to, target_topic_id) is None:
            raise ResourceNotFoundError(details={"resource": "topic", "topic_id": target_topic_id})
        return target_topic_id

    def _refresh_residency(self, belong_to: WorkspaceIdentity, topic_id: str) -> None:
        """更新驻留记录；话题已不存在时不复活陈旧条目。

        retry 的失败窗口内话题可能已被 settle 删除，此时以 Store 为准跳过
        touch，避免 idle/LRU 扫描空转。
        """
        if self._store.get(belong_to, topic_id) is None:
            return
        self._working_set.touch(belong_to, topic_id)


__all__ = ["PerceptionFamiliar"]
