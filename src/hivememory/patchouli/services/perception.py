"""Patchouli 感知使魔。

感知业务的唯一编排者：持有 Engine（纯算法）、Store（内容）、WorkingSet
（驻留与 lease）、Relay（摘要）与 Interaction journal，负责话题路由、
Interaction 原子写入与 retry、统一 settle 时序、LRU / idle / shutdown 维护，
并把领域结果投影为公开报告。

占用权统一由 WorkingSet 的 lease 表达（见 Plan §2.1/§3.5）：所有写路径都是
``acquire → try/finally release``；Relay 摘要生成在 lease 持有期间进行，
其他调用无法取得同一话题的占用权，因此无需旧的记录状态补偿校验。
统一 settle 时序（``_settle_topic``）为所有来源共用；``reason`` 只作为
材料的 provenance 标签，不驱动分支。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from hivememory.core.models import (
    IdentityScope,
    TopicAssetBinding,
    TopicData,
    WorkspaceAssetRef,
    require_identity_scope,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.engines.perception.memory_perception_engine import MemoryPerceptionEngine
from hivememory.engines.perception.models import TopicMaterializeTask, TriggerReason
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.contracts.topic_management import TopicEvictionResult, TopicSettleResult
from hivememory.patchouli.control.interaction_apply_journal import (
    InMemoryInteractionApplyJournal,
    InteractionApplyStage,
    compute_apply_digest,
)
from hivememory.patchouli.control.memory_generation.models import MemoryGenerationTask
from hivememory.patchouli.errors import TopicBusyError, TopicSettleAdmissionError
from hivememory.patchouli.runtime.models import TopicShutdownFlushReport
from hivememory.patchouli.services.topic_working_set import TopicWorkingSet

if TYPE_CHECKING:
    from hivememory.engines.perception.relay_controller import BaseRelayController
    from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore
    from hivememory.system.config.patchouli import MemoryPerceptionConfig

logger = logging.getLogger(__name__)


class PerceptionFamiliar:
    """感知业务门面：Engine + Store + WorkingSet + Relay + Journal 的编排者。

    ``engine=None`` 表示感知关闭（原 ``NullPerceptionLayer`` 语义）：摄入
    路径无任何存储与 journal 副作用；维护用例对空存储自然空转。
    """

    def __init__(
        self,
        *,
        engine: MemoryPerceptionEngine | None,
        store: ShortTermMemoryStore,
        working_set: TopicWorkingSet,
        relay_controller: BaseRelayController,
        bus,
        config: MemoryPerceptionConfig,
        interaction_journal: InMemoryInteractionApplyJournal,
    ) -> None:
        self._engine = engine
        self._store = store
        self._working_set = working_set
        self._relay_controller = relay_controller
        self._bus = bus
        self._idle_timeout_seconds = config.idle_timeout_seconds
        self._last_active_topic_ids: dict[tuple[str, str], str] = {}
        self._interaction_journal = interaction_journal

    # ========== Interaction 摄入 ==========

    async def apply_interaction(
        self,
        payload: InteractionPayload,
        *,
        identity_scope: IdentityScope,
        target_topic_id: str = "NEW_TOPIC",
        interaction_id: str | None = None,
        asset_id_and_refs: tuple[tuple[str, WorkspaceAssetRef], ...] = (),
    ) -> str:
        """应用一份已接纳的交互载荷并完成话题路由（核心用例）。

        retry 依据 journal 阶段幂等续跑（COMPLETED 直接返回；APPLIED 补跑
        compact），不重复写入 block；等价性校验在取得占用权之前完成。
        """
        identity_scope = require_identity_scope(identity_scope)
        if self._engine is None:  # 感知关闭：无任何存储与 journal 副作用
            return target_topic_id
        if not payload.turn_events:
            raise ValueError(
                "InteractionPayload.turn_events is required; "
                "legacy assistant_message fallback has been removed."
            )

        apply_record = self._interaction_journal.get(interaction_id) if interaction_id else None
        logger.info("PerceptionFamiliar 应用交互载荷: user='%s...'", payload.user_message[:30])

        # retry 已有 apply journal 时不能驱逐刚刚写入的目标话题。
        if apply_record is None:
            await self._maybe_evict_lru(identity_scope, target_topic_id)
        # retry 路由回已记录的话题；首次 apply 确保目标存在。
        topic_id = (
            apply_record.topic_id
            if apply_record is not None
            else self._ensure_topic(identity_scope, target_topic_id)
        )

        # block 构造与 digest 是纯计算；retry 等价性校验在取得占用权之前完成。
        block = self._engine.build_block(payload, identity_scope)
        digest = compute_apply_digest(block, asset_id_and_refs, payload.model_used, identity_scope)
        if apply_record is not None:
            if apply_record.input_digest != digest:
                raise ValueError(
                    f"interaction '{interaction_id}' was already applied with different input"
                )
            if apply_record.stage is InteractionApplyStage.COMPLETED:
                self._refresh_residency(identity_scope, topic_id)
                return topic_id

        lease = self._working_set.acquire(identity_scope, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法原子摄入 interaction，可稍后重试")
        try:
            if apply_record is None:
                topic = self._store.get(identity_scope, topic_id)
                if topic is None:
                    raise KeyError(f"topic '{topic_id}' does not exist in requested Workspace")
                self._store.put(
                    self._merge_interaction(
                        topic, block, asset_id_and_refs, interaction_id, payload.model_used
                    )
                )
                if interaction_id:
                    # journal 必须紧跟实际写入点；后续异常发生时 retry 仍能去重。
                    self._interaction_journal.record_interaction_applied(
                        interaction_id, topic_id, digest
                    )
            if (
                apply_record is None
                or apply_record.stage is InteractionApplyStage.INTERACTION_APPLIED
            ):
                # 后置本地义务：token 溢出 compact；LOCAL_COMPLETED 之后不再重复执行。
                await self._compact_topic_if_needed(identity_scope, topic_id)
        finally:
            self._working_set.release(lease)

        if interaction_id:
            self._interaction_journal.record_local_completed(interaction_id, topic_id, None)
            self._interaction_journal.complete(interaction_id, topic_id)

        self._refresh_residency(identity_scope, topic_id)
        return topic_id

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: str | None,
        new_topic_summary: str | None,
        identity_scope: IdentityScope,
    ) -> str:
        """确保目标短期话题存在（必要时先执行 LRU 驱逐），返回真实 topic_id。"""
        identity_scope = require_identity_scope(identity_scope)
        if self._engine is None:  # 感知关闭：不创建话题
            self._refresh_residency(identity_scope, target_topic_id)
            return target_topic_id
        await self._maybe_evict_lru(identity_scope, target_topic_id)
        topic_id = self._ensure_topic(
            identity_scope,
            target_topic_id,
            topic_title=new_topic_title,
            topic_summary=new_topic_summary,
        )
        self._refresh_residency(identity_scope, topic_id)
        return topic_id

    # ========== 统一 settle 协议 ==========

    async def _settle_topic(
        self,
        identity_scope: IdentityScope,
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
        lease = self._working_set.acquire(identity_scope, topic_id)
        if lease is None:
            raise TopicBusyError(f"topic '{topic_id}' 正忙，无法开始结算")
        try:
            topic = self._store.get(identity_scope, topic_id)
            if topic is None:
                self._working_set.remove(identity_scope, topic_id)  # 清理陈旧驻留条目
                return False, None
            # 冻结材料：字段转换、worth_saving 过滤与 no-material 判断收口在数据模型内。
            task = TopicMaterializeTask.from_topic_data(
                topic, identity_scope=identity_scope, reason=reason
            )
            if task is None:  # 无可保存材料：正常结束生命周期，不触发生成
                self._store.delete(identity_scope, topic_id)
                self._working_set.remove(identity_scope, topic_id)
                self._forget_active_topic(identity_scope, topic_id)
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

            self._store.delete(identity_scope, topic_id)
            self._working_set.remove(identity_scope, topic_id)
            self._forget_active_topic(identity_scope, topic_id)
            return True, generation_task
        finally:
            self._working_set.release(lease)

    async def manual_settle_topic(
        self, identity_scope: IdentityScope, topic_id: str | None = None
    ) -> TopicSettleResult:
        """手动结算指定话题（缺省为最近活跃话题）。

        admission 失败抛出 :class:`TopicSettleAdmissionError`（话题内容保留
        可重试）；目标正忙是瞬态冲突，保持 :class:`TopicBusyError` 语义。
        """
        identity_scope = require_identity_scope(identity_scope)
        workspace = identity_scope.workspace_identity
        scope_key = (workspace.owner_user_id, workspace.workspace_id)
        target_id = topic_id or self._last_active_topic_ids.get(scope_key)
        if not target_id:
            raise ValueError("未指定 topic_id 且无活跃话题")
        if self._store.get(identity_scope, target_id) is None:
            raise KeyError(f"话题 {target_id} 不存在")
        try:
            completed, task = await self._settle_topic(
                identity_scope,
                target_id,
                reason=TriggerReason.MANUAL_SETTLE,
                raise_on_admission_failure=True,
            )
        except TopicBusyError:
            raise
        except Exception as exc:
            raise TopicSettleAdmissionError(
                f"结算材料接纳失败，话题内容已保留，可重试: {target_id}"
            ) from exc
        if not completed:
            raise KeyError(f"话题 {target_id} 不存在")
        return TopicSettleResult(
            topic_id=target_id, generation_task_id=task.task_id if task else None
        )

    # ========== Compact（Page Folding） ==========

    async def _compact_topic_if_needed(self, identity_scope: IdentityScope, topic_id: str) -> None:
        """token 溢出时执行 compact：生成折叠摘要并写回保留的近期 blocks。

        调用方必须已持有该话题的 lease：Relay 摘要生成期间其他调用无法取
        得同一话题的占用权，因此写回不再需要旧的 expected_state 补偿校验。
        """
        topic = self._store.get(identity_scope, topic_id)
        if topic is None or topic.is_empty or not self._engine.should_compact(topic.total_tokens):
            return
        logger.info("Token 溢出，触发 Page Folding: topic_id=%s", topic_id)
        # 保留块数以 Engine 配置为单一事实源。
        blocks_to_fold = self._engine.select_blocks_to_fold(
            topic.blocks, self._engine.config.fold_retain_recent_blocks
        )
        if not blocks_to_fold:
            return
        summary = self._relay_controller.generate_summary(
            blocks_to_fold=blocks_to_fold, previous_summary=topic.state_summary
        )
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

    # ========== LRU / Evict ==========

    async def _maybe_evict_lru(self, identity_scope: IdentityScope, target_topic_id: str) -> None:
        """需要创建新话题且池满时，驱逐 LRU 话题并提交结算任务。

        已有话题无需驱逐；未知目标必须直接拒绝。容量与候选由 WorkingSet
        决定；候选正被占用时改选其他话题。
        """
        if target_topic_id != "NEW_TOPIC":
            if self._store.get(identity_scope, target_topic_id) is None:
                raise KeyError(f"topic '{target_topic_id}' does not exist in requested Workspace")
            return
        if not self._working_set.needs_eviction(identity_scope):
            return
        attempted: set[str] = set()
        while self._working_set.needs_eviction(identity_scope):
            lru_topic_id = self._working_set.select_lru_candidate(identity_scope, exclude=attempted)
            if lru_topic_id is None:  # 池满但全部候选正被占用，无法安全驱逐
                raise TopicBusyError("LRU 驱逐无可占用候选，稍后重试")
            try:
                completed, _ = await self._settle_topic(
                    identity_scope,
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

    async def evict_topic(
        self, identity_scope: IdentityScope, topic_id: str
    ) -> TopicEvictionResult:
        """从活跃话题池中驱逐话题，不触发结算；正被占用时报 ``removed=False``。"""
        identity_scope = require_identity_scope(identity_scope)
        lease = self._working_set.acquire(identity_scope, topic_id)
        if lease is None:
            return TopicEvictionResult(topic_id=topic_id, removed=False)
        try:
            removed = self._store.delete(identity_scope, topic_id)
            if removed:
                self._working_set.remove(identity_scope, topic_id)
                self._forget_active_topic(identity_scope, topic_id)
            return TopicEvictionResult(topic_id=topic_id, removed=removed)
        finally:
            self._working_set.release(lease)

    def discard_if_empty(self, identity_scope: IdentityScope, topic_id: str) -> bool:
        """话题真正为空（无 blocks 且无非空白折叠摘要）时清理并返回 True。"""
        identity_scope = require_identity_scope(identity_scope)
        lease = self._working_set.acquire(identity_scope, topic_id)
        if lease is None:  # 正被占用（可能正在写入首块），留给后续维护判断
            return False
        try:
            topic = self._store.get(identity_scope, topic_id)
            if topic is None or not topic.is_empty:
                return False
            removed = self._store.delete(identity_scope, topic_id)
            if removed:
                self._working_set.remove(identity_scope, topic_id)
            return removed
        finally:
            self._working_set.release(lease)

    # ========== 维护与 shutdown ==========

    async def scan_idle_buffers_once(self) -> list[str]:
        """扫描并 settle 空闲超时话题；候选由 WorkingSet 提供。"""
        flushed: list[str] = []
        for identity_scope, topic_id in self._working_set.list_idle_candidates(
            self._idle_timeout_seconds
        ):
            logger.info("检测到空闲话题: topic_id=%s", topic_id)
            try:
                completed, _ = await self._settle_topic(
                    identity_scope,
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
        for identity_scope, topic_id in candidates:
            topic = self._store.get(identity_scope, topic_id)
            if topic is not None:
                resident_block_count += topic.block_count

        settled: list[str] = []
        generation_skipped: list[str] = []
        failed: list[str] = []
        for identity_scope, topic_id in candidates:
            try:
                completed, task = await self._settle_topic(
                    identity_scope,
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
        identity_scope: IdentityScope,
        target_topic_id: str,
        *,
        topic_title: str | None = None,
        topic_summary: str | None = None,
    ) -> str:
        """确保目标话题存在并返回真实 topic_id；未知目标显式拒绝。"""
        if target_topic_id == "NEW_TOPIC":
            topic = self._store.create(
                identity_scope,
                topic_title=topic_title or "新建话题",
                topic_summary=topic_summary or "",
            )
            # 创建即入池：避免 Store 有记录而 WorkingSet 不可见的孤儿窗口。
            self._working_set.touch(identity_scope, topic.topic_id)
            return topic.topic_id
        if self._store.get(identity_scope, target_topic_id) is None:
            raise KeyError(f"topic '{target_topic_id}' does not exist in requested Workspace")
        return target_topic_id

    @staticmethod
    def _merge_interaction(
        topic: TopicData,
        block,
        asset_id_and_refs: tuple[tuple[str, WorkspaceAssetRef], ...],
        interaction_id: str | None,
        model_used: str | None,
    ) -> TopicData:
        """把一次 Interaction 的 block、首次绑定与元数据并成新快照。

        binding 以 ``asset_id`` 幂等去重；原子性由「frozen 快照 → 新快照 →
        ``Store.put`` 整条替换」承担，互斥由调用方持有的 lease 保证。
        """
        if asset_id_and_refs and not interaction_id:
            raise ValueError("建立 asset binding 必须携带 interaction_id")
        bindings = list(topic.bindings)
        existing_ids = {binding.asset_id for binding in bindings}
        now = datetime.now()
        for asset_id, asset_ref in asset_id_and_refs:
            if not isinstance(asset_id, str) or not asset_id.strip():
                raise ValueError("asset_id 不能为空")
            if not isinstance(asset_ref, WorkspaceAssetRef):
                raise TypeError("asset_ref 必须是 WorkspaceAssetRef")
            if asset_id in existing_ids:
                continue
            bindings.append(
                TopicAssetBinding(
                    asset_id=asset_id.strip(),
                    asset_ref=asset_ref,
                    first_bound_interaction_id=interaction_id,
                    bound_at=now,
                )
            )
            existing_ids.add(asset_id)
        return topic.model_copy(
            update={
                "blocks": (*topic.blocks, block),
                "bindings": tuple(bindings),
                "total_tokens": topic.total_tokens + block.total_tokens,
                "last_update": now.timestamp(),
                "model_used": model_used or topic.model_used,
            }
        )

    def _refresh_residency(self, identity_scope: IdentityScope, topic_id: str) -> None:
        """更新驻留与活跃记录；话题已不存在时不复活陈旧条目。

        retry 的失败窗口内话题可能已被 settle 删除，此时以 Store 为准清理
        活跃记录，避免 idle/LRU 扫描空转或缺省 manual settle 寻址到死话题。
        """
        if self._store.get(identity_scope, topic_id) is None:
            self._forget_active_topic(identity_scope, topic_id)
            return
        self._working_set.touch(identity_scope, topic_id)
        self._remember_active_topic(identity_scope, topic_id)

    def _remember_active_topic(self, identity_scope: IdentityScope, topic_id: str) -> None:
        """记录 Workspace 最近活跃话题，供缺省 manual settle 寻址。"""
        workspace = identity_scope.workspace_identity
        self._last_active_topic_ids[(workspace.owner_user_id, workspace.workspace_id)] = topic_id

    def _forget_active_topic(self, identity_scope: IdentityScope, topic_id: str) -> None:
        """话题生命周期结束时清理最近活跃记录。"""
        workspace = identity_scope.workspace_identity
        scope_key = (workspace.owner_user_id, workspace.workspace_id)
        if self._last_active_topic_ids.get(scope_key) == topic_id:
            self._last_active_topic_ids.pop(scope_key, None)


__all__ = ["PerceptionFamiliar"]
