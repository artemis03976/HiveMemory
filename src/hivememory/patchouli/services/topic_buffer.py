"""Topic Buffer 领域服务。

``TopicBufferService`` 是 Patchouli 内部唯一的 Topic Buffer 状态与活跃池所有者：
持有 ``ShortTermMemoryStore`` 与 Relay 依赖和一把领域锁，负责 ``TopicData``
状态转换、Interaction/绑定写回、Compact、统一 settle 协议、Evict 以及 LRU /
idle / shutdown 池操作。它执行 ``TriggerPlan``，但不解释策略来源，也不提交
generation queue——admission 由上层 Familiar 在领域锁外完成。

锁边界（见 Plan §5.1）：
    领域锁内：读取快照 -> 检查状态 -> 形成预约/新快照 -> Store 写回或删除
    领域锁外：Relay 摘要生成、Generation queue admission、EventBus 调用
    领域锁内：确认预约仍然有效 -> 写回最终快照或放弃本次变换
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from hivememory.core.models import (
    ActorIdentity,
    BufferState,
    IdentityScope,
    LogicalBlock,
    TopicAssetBinding,
    TopicData,
    WorkspaceAssetRef,
    require_identity_scope,
)
from hivememory.engines.perception.models import TriggerReason, TopicMaterializeTask
from hivememory.patchouli.errors import TopicBusyError

if TYPE_CHECKING:
    from hivememory.engines.perception.relay_controller import BaseRelayController
    from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore

logger = logging.getLogger(__name__)


# ========== 触发计划：TriggerReason 的唯一决策矩阵 ==========

@dataclass(frozen=True, slots=True)
class TriggerPlan:
    """触发原因解释后的三动作计划；不承载执行时序或外部副作用。

    ``settle`` 的结果必然是 Topic 生命周期结束，因此必须同时声明 ``evict``；
    ``evict=True`` 可以独立存在（如 ``MANUAL_DELETE``）。
    """

    settle: bool = False
    compact: bool = False
    evict: bool = False

    def __post_init__(self) -> None:
        if self.settle and not self.evict:
            raise ValueError("settle=True requires evict=True")
        if not (self.settle or self.compact or self.evict):
            raise ValueError("TriggerPlan must contain at least one action")


# 七种触发原因 -> 三动作的唯一映射。settle=True 的原因统一进入
# begin -> admission -> complete/abort 协议，不再区分驱逐时机。
TRIGGER_PLANS: dict[TriggerReason, TriggerPlan] = {
    TriggerReason.TOKEN_OVERFLOW: TriggerPlan(compact=True),
    TriggerReason.IDLE_TIMEOUT: TriggerPlan(settle=True, evict=True),
    TriggerReason.LRU_EVICTION: TriggerPlan(settle=True, evict=True),
    TriggerReason.SHUTDOWN: TriggerPlan(settle=True, evict=True),
    TriggerReason.MANUAL_SETTLE: TriggerPlan(settle=True, evict=True),
    TriggerReason.MANUAL_COMPACT: TriggerPlan(compact=True),
    TriggerReason.MANUAL_DELETE: TriggerPlan(evict=True),
}


def resolve_trigger_plan(reason: TriggerReason) -> TriggerPlan:
    """解释触发原因并返回动作计划；未知原因视为矩阵缺口并显式报错。"""
    plan = TRIGGER_PLANS.get(reason)
    if plan is None:
        raise ValueError(f"未知的触发原因: {reason}")
    return plan


# ========== settlement 领域结果 ==========

class SettlementStatus(str, Enum):
    """settle 操作的终态。

    ``ACCEPTED`` 与 ``NO_MATERIAL`` 都表示 Topic 已经结束；``REJECTED`` 表示
    admission 未完成且 Topic 已恢复 ``IDLE``；``NOT_FOUND`` 表示目标已被其他
    生命周期操作移除。该结果表示队列交接与 Topic 生命周期，不表示记忆已经
    生成或写入中期存储。
    """

    ACCEPTED = "accepted"
    NO_MATERIAL = "no_material"
    REJECTED = "rejected"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class SettlementOutcome:
    """统一 settle 协议的终态结果，由 Familiar 投影为不同公共报告。"""

    topic_id: str
    status: SettlementStatus
    removed: bool
    generation_task_id: str | None = None
    reason: TriggerReason | None = None

    def __post_init__(self) -> None:
        if self.status is SettlementStatus.ACCEPTED and self.generation_task_id is None:
            raise ValueError("ACCEPTED 结果必须携带已接纳的 generation task id")
        if self.status is SettlementStatus.NO_MATERIAL and not self.removed:
            raise ValueError("无材料正常完成同样表示 Topic 已结束")
        if self.status in (SettlementStatus.REJECTED, SettlementStatus.NOT_FOUND) and self.removed:
            raise ValueError(f"{self.status.value} 结果不能携带 removed=True")


@dataclass(frozen=True, slots=True)
class SettlementReservation:
    """begin_settlement 的预约结果；调用方在领域锁外完成 admission。

    ``task`` 为 ``None`` 表示没有可生成材料（正常成功，不是失败或 busy），
    调用方仍须调用 ``complete_settlement`` 结束 Topic 生命周期。
    """

    topic_id: str
    reason: TriggerReason
    task: TopicMaterializeTask | None = None


@dataclass(frozen=True, slots=True)
class TriggerExecution:
    """``handle_trigger`` 的执行结果。

    settle 计划返回待 admission 的 :class:`SettlementReservation`；compact /
    evict 计划在返回前已经执行完毕。调用方只依据 ``settlement`` 是否为空决定
    是否继续 admission，不得依据 ``TriggerReason`` 复制分支。
    """

    reason: TriggerReason
    plan: TriggerPlan
    evicted: bool = False
    compacted: bool = False
    settlement: SettlementReservation | None = None


@dataclass(frozen=True, slots=True)
class TopicCandidate:
    """池扫描返回的结算候选。

    ``identity_scope`` 按候选 Topic 内容重建（最后 block 的执行者，缺失时回退
    owner），供锁外 admission 使用；调用方无需自行遍历 ``TopicData``。
    """

    identity_scope: IdentityScope
    topic_id: str
    state: BufferState
    last_update: float
    block_count: int


# ========== TopicBufferService ==========

class TopicBufferService:
    """Topic Buffer 状态与活跃池的唯一领域所有者。

    服务内不解释触发策略来源（矩阵只提供 ``TriggerPlan``），不提交 generation
    queue，也不感知 HTTP、总线或 WorkspaceAssetStore 生命周期。所有 Store 访问
    都收口在此；入口统一为 ``IdentityScope + topic_id``。
    """

    def __init__(
        self,
        store: "ShortTermMemoryStore",
        relay_controller: "BaseRelayController",
        *,
        lock: threading.RLock | None = None,
    ) -> None:
        """
        初始化 TopicBufferService。

        Args:
            store: ShortTermMemoryStore 实例（短期记忆存储）
            relay_controller: Relay 控制器（Compact 时在领域锁外生成摘要）
            lock: 可选的外部领域锁（默认自建 RLock）
        """
        self._store = store
        self._relay_controller = relay_controller
        self._lock = lock or threading.RLock()

    # ========== Topic 创建、查询与池管理 ==========

    def create_topic(
        self,
        identity_scope: IdentityScope,
        *,
        topic_title: str | None = None,
        topic_summary: str | None = None,
    ) -> TopicData:
        """创建新话题并返回初始快照。容量与 LRU 驱逐由调用方先行处理。"""
        identity_scope = require_identity_scope(identity_scope)
        return self._store.create(
            identity_scope,
            topic_title=topic_title or "新建话题",
            topic_summary=topic_summary or "",
        )

    def ensure_topic(
        self,
        identity_scope: IdentityScope,
        target_topic_id: str,
        *,
        topic_title: str | None = None,
        topic_summary: str | None = None,
    ) -> str:
        """确保目标话题存在并返回真实 topic_id。

        ``NEW_TOPIC`` 创建新话题；已有话题返回其全局 ID。未知目标不能被投影成
        当前 Workspace 的新话题，否则跨 Workspace 的真实 ID 会造成隐式副作用。
        """
        if target_topic_id == "NEW_TOPIC":
            data = self.create_topic(
                identity_scope,
                topic_title=topic_title,
                topic_summary=topic_summary,
            )
            return data.topic_id
        if self._store.get(identity_scope, target_topic_id) is None:
            raise KeyError(
                f"topic '{target_topic_id}' does not exist in requested Workspace"
            )
        return target_topic_id

    def get_topic(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> TopicData | None:
        """读取话题快照（纯读；访问追踪由 TopicWorkingSet 负责）。"""
        identity_scope = require_identity_scope(identity_scope)
        return self._store.get(identity_scope, topic_id)

    def touch_topic(self, identity_scope: IdentityScope, topic_id: str) -> TopicData | None:
        """返回话题快照（已废弃的读取兼容入口）。

        访问时间追踪已随 ``last_accessed_at`` 字段移除；驻留刷新统一由
        ``TopicWorkingSet.touch`` 在 Familiar 编排中执行。
        """
        identity_scope = require_identity_scope(identity_scope)
        return self._store.get(identity_scope, topic_id)

    def count_topics(self, identity_scope: IdentityScope) -> int:
        """统计 Workspace 内的话题数量。"""
        identity_scope = require_identity_scope(identity_scope)
        return self._store.count(identity_scope)

    def list_topics(
        self,
        identity_scope: IdentityScope,
        *,
        include_empty: bool = True,
    ) -> list[TopicData]:
        """列出 Workspace 内的话题快照（读模型入口）。"""
        identity_scope = require_identity_scope(identity_scope)
        return self._store.list_by_workspace(identity_scope, include_empty=include_empty)

    def discard_if_empty(self, identity_scope: IdentityScope, topic_id: str) -> bool:
        """话题真正为空（无 blocks 且无非空白折叠摘要）时驱逐并返回 True。"""
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            data = self._store.get(identity_scope, topic_id)
            if data is not None and data.is_empty:
                removed = self._store.delete(identity_scope, topic_id)
                if removed:
                    logger.info("已清理无内容话题: %s", topic_id)
                return removed
            return False

    def select_lru_candidate(
        self,
        identity_scope: IdentityScope,
        *,
        exclude_ids: frozenset[str] | set[str] = frozenset(),
    ) -> str | None:
        """选择最久未访问的 IDLE 话题作为 LRU 驱逐候选。

        非 IDLE（预约中）话题一律跳过；``exclude_ids`` 用于调用方在候选失效后
        改选其他话题。返回 ``None`` 表示当前没有可安全驱逐的候选。
        """
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            candidates = [
                topic
                for topic in self._store.list_by_workspace(identity_scope)
                if topic.state is BufferState.IDLE
                and topic.topic_id not in exclude_ids
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda topic: topic.last_accessed_at).topic_id

    def list_idle_candidates(self, idle_timeout_seconds: int) -> list[TopicCandidate]:
        """扫描全部 Workspace 中空闲超时且处于 IDLE 状态的话题候选。"""
        with self._lock:
            topics = self._store.list_all()
        return [
            self._make_candidate(topic)
            for topic in topics
            if topic.state is BufferState.IDLE and topic.is_idle(idle_timeout_seconds)
        ]

    def list_shutdown_candidates(self) -> list[TopicCandidate]:
        """返回进程关闭前需要逐个执行 settle 协议的全部驻留话题。"""
        with self._lock:
            topics = self._store.list_all()
        return [self._make_candidate(topic) for topic in topics]

    @staticmethod
    def _make_candidate(topic: TopicData) -> TopicCandidate:
        return TopicCandidate(
            identity_scope=TopicBufferService.build_maintenance_scope(topic),
            topic_id=topic.topic_id,
            state=topic.state,
            last_update=topic.last_update,
            block_count=topic.block_count,
        )

    @staticmethod
    def build_maintenance_scope(topic: TopicData) -> IdentityScope:
        """为进程级生命周期扫描重建访问作用域。

        执行者取话题最后一个 block 的身份；话题尚无 block 时回退 Workspace
        owner。``TopicData`` 不保存本次执行者身份，因此维护路径需要在扫描时
        显式重建。
        """
        actor = (
            topic.blocks[-1].identity
            if topic.blocks
            else ActorIdentity(user_id=topic.workspace_identity.owner_user_id)
        )
        return IdentityScope(
            actor_identity=actor,
            workspace_identity=topic.workspace_identity,
        )

    # ========== 状态预约 ==========

    def reserve_processing(self, identity_scope: IdentityScope, topic_id: str) -> bool:
        """取得 ``IDLE -> PROCESSING`` 预约；目标缺失或非 IDLE 时返回 False。"""
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id)
            if snapshot is None or snapshot.state is not BufferState.IDLE:
                return False
            self._store.put(
                snapshot.model_copy(
                    update={
                        "state": BufferState.PROCESSING,
                        "last_update": datetime.now().timestamp(),
                    }
                )
            )
            return True

    def release_processing(self, identity_scope: IdentityScope, topic_id: str) -> None:
        """释放 ``PROCESSING`` 预约并恢复 ``IDLE``；目标缺失时静默忽略。"""
        self._transition_state(
            identity_scope, topic_id, BufferState.PROCESSING, BufferState.IDLE
        )

    # ========== Interaction 写入 ==========

    def apply_interaction(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        block: LogicalBlock,
        *,
        interaction_id: str | None = None,
        asset_id_and_refs: tuple[tuple[str, WorkspaceAssetRef], ...] = (),
        model_used: str | None = None,
    ) -> TopicData:
        """把一次 Interaction 的 block、首次绑定与元数据原子写回话题快照。

        调用方必须先通过 :meth:`reserve_processing` 取得 ``PROCESSING`` 预约；
        binding 以 ``asset_id`` 幂等去重，重复使用只保留首次交互事实。
        """
        if asset_id_and_refs and not interaction_id:
            raise ValueError("建立 asset binding 必须携带 interaction_id")
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id)
            if snapshot is None:
                raise KeyError(f"topic '{topic_id}' does not exist in requested Workspace")
            if snapshot.state is not BufferState.PROCESSING:
                raise TopicBusyError(f"topic '{topic_id}' 未持有 PROCESSING 预约")
            bindings = list(snapshot.bindings)
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
            updated = snapshot.model_copy(
                update={
                    "blocks": (*snapshot.blocks, block),
                    "bindings": tuple(bindings),
                    "total_tokens": snapshot.total_tokens + block.total_tokens,
                    "last_update": now.timestamp(),
                    "model_used": model_used or snapshot.model_used,
                }
            )
            self._store.put(updated)
            return updated

    # ========== 触发计划执行 ==========

    def handle_trigger(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        reason: TriggerReason,
        *,
        retain_recent_blocks: int | None = None,
    ) -> TriggerExecution:
        """唯一计划执行入口：按矩阵执行 compact / settle / evict。

        settle 计划只完成 ``begin_settlement``（进入 ``SETTLING`` 并冻结材料），
        admission 与 complete/abort 由调用方在领域锁外继续；compact 与 evict
        在本方法内执行完毕。
        """
        plan = resolve_trigger_plan(reason)
        identity_scope = require_identity_scope(identity_scope)
        if plan.compact:
            return self._execute_compact(
                identity_scope, topic_id, reason, plan, retain_recent_blocks
            )
        if plan.settle:
            reservation = self.begin_settlement(identity_scope, topic_id, reason)
            return TriggerExecution(
                reason=reason,
                plan=plan,
                settlement=reservation,
            )
        removed = self.delete_if_idle(identity_scope, topic_id)
        return TriggerExecution(reason=reason, plan=plan, evicted=removed)

    # ========== 统一 settle 协议原语 ==========

    def begin_settlement(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        reason: TriggerReason,
    ) -> SettlementReservation | None:
        """开始结算：``IDLE -> SETTLING`` 并冻结生成交接任务。

        所有 ``settle=True`` 的触发原因共用本原语；进入 ``SETTLING`` 后话题
        不再接受新的 Interaction。返回 ``None`` 表示目标已被其他生命周期操作
        移除；非 IDLE 话题通过 ``TopicBusyError`` 显式报 busy。
        """
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id)
            if snapshot is None:
                return None
            if snapshot.state is not BufferState.IDLE:
                raise TopicBusyError(f"topic '{topic_id}' 正忙，无法开始结算")
            settling = snapshot.model_copy(
                update={
                    "state": BufferState.SETTLING,
                    "last_update": datetime.now().timestamp(),
                }
            )
            self._store.put(settling)
        # 字段转换、worth_saving 过滤与 no-material 判断收口在数据模型内。
        task = TopicMaterializeTask.from_topic_data(
            settling,
            identity_scope=identity_scope,
            reason=reason,
        )
        return SettlementReservation(topic_id=topic_id, reason=reason, task=task)

    def complete_settlement(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        *,
        generation_task_id: str | None = None,
        reason: TriggerReason | None = None,
    ) -> SettlementOutcome:
        """完成结算：删除仍处于 ``SETTLING`` 状态的话题。

        ``generation_task_id`` 为空表示没有可生成材料（正常成功）；只删除仍处
        于本次 settling 状态的话题，目标缺失或状态已变时返回 ``NOT_FOUND``。
        """
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id)
            if snapshot is None or snapshot.state is not BufferState.SETTLING:
                return SettlementOutcome(
                    topic_id=topic_id,
                    status=SettlementStatus.NOT_FOUND,
                    removed=False,
                    reason=reason,
                )
            self._store.delete(identity_scope, topic_id)
        status = (
            SettlementStatus.ACCEPTED
            if generation_task_id is not None
            else SettlementStatus.NO_MATERIAL
        )
        logger.debug(
            "settlement 完成: topic_id=%s, status=%s", topic_id, status.value
        )
        return SettlementOutcome(
            topic_id=topic_id,
            status=status,
            removed=True,
            generation_task_id=generation_task_id,
            reason=reason,
        )

    def abort_settlement(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        *,
        reason: TriggerReason | None = None,
    ) -> SettlementOutcome:
        """中止结算：``SETTLING`` 恢复 ``IDLE``，保留 blocks、summary 与 bindings。

        用于 queue 明确拒绝或 admission 抛出异常后的恢复；话题内容完整保留，
        可在后续维护轮次或用户重试中再次结算。
        """
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id)
            if snapshot is None or snapshot.state is not BufferState.SETTLING:
                return SettlementOutcome(
                    topic_id=topic_id,
                    status=SettlementStatus.NOT_FOUND,
                    removed=False,
                    reason=reason,
                )
            self._store.put(
                snapshot.model_copy(
                    update={
                        "state": BufferState.IDLE,
                        "last_update": datetime.now().timestamp(),
                    }
                )
            )
        return SettlementOutcome(
            topic_id=topic_id,
            status=SettlementStatus.REJECTED,
            removed=False,
            reason=reason,
        )

    # ========== 独立 evict ==========

    def delete_if_idle(self, identity_scope: IdentityScope, topic_id: str) -> bool:
        """删除处于 ``IDLE`` 状态的话题（``MANUAL_DELETE`` / 显式换出）。

        不构造 generation task；非 IDLE 话题通过 ``TopicBusyError`` 显式报 busy。
        """
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id)
            if snapshot is None:
                return False
            if snapshot.state is not BufferState.IDLE:
                raise TopicBusyError(f"topic '{topic_id}' 正忙，无法删除")
            return self._store.delete(identity_scope, topic_id)

    # ========== 内部实现 ==========

    def _execute_compact(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        reason: TriggerReason,
        plan: TriggerPlan,
        retain_recent_blocks: int | None,
    ) -> TriggerExecution:
        """执行 compact：锁内取快照与预约，锁外生成摘要，锁内确认后写回。

        manual compact（``MANUAL_COMPACT``）要求话题处于 ``IDLE`` 并自行取得
        ``PROCESSING`` 预约；automatic compact（``TOKEN_OVERFLOW``）复用调用方
        已持有的 ``PROCESSING`` 预约，异常由调用方负责释放预约。
        """
        if retain_recent_blocks is None:
            raise ValueError(f"{reason.value} requires retain_recent_blocks")
        if retain_recent_blocks < 1:
            raise ValueError("retain_recent_blocks must be >= 1")
        manual_reserved = reason is TriggerReason.MANUAL_COMPACT
        reserved_manual = False
        try:
            with self._lock:
                snapshot = self._store.get(identity_scope, topic_id)
                if manual_reserved:
                    if snapshot is None:
                        raise KeyError(
                            f"topic '{topic_id}' does not exist in requested Workspace"
                        )
                    if snapshot.state is not BufferState.IDLE:
                        raise TopicBusyError(
                            f"topic '{topic_id}' 正忙，无法开始 manual compact"
                        )
                    self._store.put(
                        snapshot.model_copy(update={"state": BufferState.PROCESSING})
                    )
                    reserved_manual = True
                    snapshot = self._store.get(identity_scope, topic_id)
                if snapshot is None or snapshot.is_empty:
                    return TriggerExecution(reason=reason, plan=plan)
                expected_state = snapshot.state
                fold_count = max(0, len(snapshot.blocks) - retain_recent_blocks)
                if fold_count == 0:
                    return TriggerExecution(reason=reason, plan=plan)
                fold_prefix = tuple(snapshot.blocks[:fold_count])

            # ---- 领域锁外：Relay 摘要生成，不得阻塞其他话题的状态变换 ----
            summary = self._relay_controller.generate_summary(
                blocks_to_fold=list(fold_prefix),
                previous_summary=snapshot.state_summary,
            )

            # ---- 领域锁内：确认预约仍然有效后写回 ----
            with self._lock:
                current = self._store.get(identity_scope, topic_id)
                if (
                    current is None
                    or current.state is not expected_state
                    or tuple(current.blocks[:fold_count]) != fold_prefix
                ):
                    # 预约或折叠前缀已被并发操作改变；放弃本次写回，避免覆盖
                    # 其他写入者已提交的 block 事实。
                    logger.warning(
                        "compact 写回前预约校验失败，放弃写回: topic_id=%s, reason=%s",
                        topic_id,
                        reason.value,
                    )
                    return TriggerExecution(reason=reason, plan=plan)
                retained = current.blocks[fold_count:]
                updated = current.model_copy(
                    update={
                        "state_summary": summary,
                        "blocks": retained,
                        "total_tokens": sum(block.total_tokens for block in retained),
                        "last_update": datetime.now().timestamp(),
                        # manual compact 完成后归还预约；automatic compact 保持
                        # 调用方的 PROCESSING 预约由其自行释放。
                        "state": (
                            BufferState.IDLE if manual_reserved else current.state
                        ),
                    }
                )
                self._store.put(updated)
            return TriggerExecution(reason=reason, plan=plan, compacted=True)
        finally:
            if reserved_manual:
                # 摘要生成或写回失败也必须结束本次预约，避免话题永久 busy。
                self._recover_manual_compact_reservation(identity_scope, topic_id)

    def _recover_manual_compact_reservation(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
    ) -> None:
        with self._lock:
            current = self._store.get(identity_scope, topic_id)
            if current is not None and current.state is BufferState.PROCESSING:
                self._store.put(
                    current.model_copy(
                        update={
                            "state": BufferState.IDLE,
                            "last_update": datetime.now().timestamp(),
                        }
                    )
                )

    def _transition_state(
        self,
        identity_scope: IdentityScope,
        topic_id: str,
        expected: BufferState,
        target: BufferState,
    ) -> None:
        """在预期状态匹配时推进状态；目标缺失或状态不匹配时静默忽略。"""
        identity_scope = require_identity_scope(identity_scope)
        with self._lock:
            snapshot = self._store.get(identity_scope, topic_id)
            if snapshot is not None and snapshot.state is expected:
                self._store.put(
                    snapshot.model_copy(
                        update={
                            "state": target,
                            "last_update": datetime.now().timestamp(),
                        }
                    )
                )


__all__ = [
    "SettlementOutcome",
    "SettlementReservation",
    "SettlementStatus",
    "TopicBufferService",
    "TopicCandidate",
    "TRIGGER_PLANS",
    "TriggerExecution",
    "TriggerPlan",
    "resolve_trigger_plan",
]
