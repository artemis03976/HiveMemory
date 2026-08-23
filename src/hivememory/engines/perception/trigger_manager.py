"""
HiveMemory Trigger Manager

话题结算调度器，负责根据触发原因执行对应的原子操作组合，并返回结算载荷。

职责:
    - 统一的 FlushEvent 输入协议
    - 根据触发原因执行原子操作组合 (Settle/Compact/Evict)
    - 返回 TopicMaterializeTask 供上层提交给生成链路（不主动触发）

核心概念:
    - Settle (结算): 将 blocks 与 binding 打包为 TopicMaterializeTask 返回给调用方
    - Compact (压缩): 生成 state_summary（同步阻塞）
    - Evict (驱逐): 从活跃池移除 buffer

参考: BufferManagement.md, ShortTermMemory.md §4.2, §5.1

作者: HiveMemory Team
版本: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from hivememory.core.models import (
    IdentityScope,
    TopicData,
    WorkspaceIdentity,
    WorkspaceTopicKey,
)
from hivememory.patchouli.errors import TopicBusyError

from hivememory.engines.perception.models import (
    FlushEvent,
    FlushReason,
    LogicalBlock,
    TopicMaterializeTask,
)

if TYPE_CHECKING:
    from hivememory.engines.perception.relay_controller import BaseRelayController
    from hivememory.patchouli.memory_library.stores import ShortTermMemoryStore

logger = logging.getLogger(__name__)


# ========== 决策矩阵：触发原因 -> 原子操作组合 ==========
DECISION_MATRIX: Dict[FlushReason, Dict[str, bool]] = {
    FlushReason.TOKEN_OVERFLOW: {
        "settle": False,  # 话题未完，暂不结算
        "compact": True,  # 必须留摘要以接力
        "evict": False,   # 保持存活
    },
    FlushReason.IDLE_TIMEOUT: {
        "settle": True,   # 完整事件，落库
        "compact": False, # 无需摘要，人已走
        "evict": True,    # 踢出内存
    },
    FlushReason.LRU_EVICTION: {
        "settle": True,   # 完整事件，落库
        "compact": False,
        "evict": True,    # 踢出内存
    },
    FlushReason.SHUTDOWN: {
        "settle": True,   # 进程关闭前强制结算
        "compact": False,
        "evict": True,    # 清空内存态话题
    },
    FlushReason.MANUAL_SETTLE: {
        "settle": True,   # 冻结材料并结算为记忆资产
        "compact": False, # 手动 settle 不再 compact
        "evict": True,    # 接纳成功后结束 Topic 生命周期
    },
    FlushReason.MANUAL_COMPACT: {
        "settle": False,  # 只压缩工作集
        "compact": True,  # 合并 previous summary 与旧前缀
        "evict": False,   # 保留 Topic
    },
    FlushReason.MANUAL_DELETE: {
        "settle": False,  # 不构造 settlement
        "compact": False, # 不生成摘要
        "evict": True,    # 丢弃 Topic
    },
}


class TriggerManager:
    """
    Flush 触发调度器

    根据 FlushEvent 查表决定执行哪些原子操作组合：
        - Settle: 构建 TopicMaterializeTask 返回给调用方（不主动触发生成）
        - Compact: 生成 state_summary（同步阻塞）
        - Evict: 从活跃池移除 buffer

    决策矩阵 (DECISION_MATRIX):
        | Trigger Reason    | Settle | Compact | Evict |
        |-------------------|--------|---------|-------|
        | TOKEN_OVERFLOW    | ❌     | ✅      | ❌    |
        | IDLE_TIMEOUT      | ✅     | ❌      | ✅    |
        | LRU_EVICTION      | ✅     | ❌      | ✅    |
        | SHUTDOWN          | ✅     | ❌      | ✅    |
        | MANUAL_SETTLE     | ✅     | ❌      | ✅    |
        | MANUAL_COMPACT    | ❌     | ✅      | ❌    |
        | MANUAL_DELETE     | ❌     | ❌      | ✅    |

    依赖:
        - ShortTermMemoryStore: 读取 buffer 状态、执行 evict
        - RelayController: 生成摘要

    Examples:
        >>> trigger_manager = TriggerManager(store, relay_controller)
        >>> payload = await trigger_manager.resolve_topic(
        ...     FlushEvent(topic_id=topic_id, reason=FlushReason.IDLE_TIMEOUT)
        ... )
        >>> if payload:
        ...     await bus.request(GENERATION_SUBMIT_SETTLEMENT, payload)
    """

    def __init__(
        self,
        store: "ShortTermMemoryStore",
        relay_controller: "BaseRelayController",
    ) -> None:
        """
        初始化 TriggerManager

        Args:
            store: ShortTermMemoryStore 实例（短期记忆存储）
            relay_controller: RelayController 实例（用于生成摘要）
        """
        self._store = store
        self._relay_controller = relay_controller
        logger.info("TriggerManager 初始化完成")

    # ========== 统一调度器 ==========

    async def resolve_topic(
        self,
        trigger: FlushEvent,
        *,
        retain_recent_blocks: Optional[int] = None,
    ) -> Optional[TopicMaterializeTask]:
        """
        统一的话题结算调度器。

        Args:
            trigger: 结算触发指令（包含 topic_key 与 reason）
            retain_recent_blocks: compact 路径（TOKEN_OVERFLOW / MANUAL_COMPACT）
                必须提供的最近工作集大小，必须 >= 1；其他触发原因忽略该值

        Returns:
            TopicMaterializeTask 供上层提交给生成链路；如无需结算则返回 None
        """
        actions = DECISION_MATRIX.get(trigger.reason)
        if not actions:
            logger.error(f"未知的触发原因: {trigger.reason}")
            return None

        need_settle = actions["settle"]
        need_compact = actions["compact"]
        need_evict = actions["evict"]

        if trigger.reason is FlushReason.MANUAL_SETTLE:
            # manual settle 使用 FLUSHING prepare -> admission -> evict，不走本调度器。
            raise ValueError("MANUAL_SETTLE 必须通过 prepare_manual_settle 处理")

        if need_settle and need_evict:
            # IDLE/LRU/SHUTDOWN：automatic settle 走原子 freeze-and-evict。
            return self.settle_and_evict(trigger.topic_key, trigger.reason)

        if need_compact:
            # TOKEN_OVERFLOW 复用 Interaction 已持有的 PROCESSING；
            # MANUAL_COMPACT 需要自行预约 PROCESSING。
            return await self._compact_path(trigger, retain_recent_blocks)

        if need_evict:
            # MANUAL_DELETE：只丢弃 Topic，不构造 settlement。
            self._store.pop_buffer_by_key(trigger.topic_key)
            return None

        return None

    def settle_and_evict(
        self,
        topic_key: WorkspaceTopicKey,
        reason: FlushReason,
    ) -> Optional[TopicMaterializeTask]:
        """automatic settle：只接受 IDLE Topic，原子 freeze-and-evict 后冻结载荷。

        ``freeze_and_evict`` 在 Store 临界区内冻结 blocks/state summary/binding
        refs 并移除 buffer；busy（PROCESSING/FLUSHING）或缺失时返回 None，由调用方
        跳过或改选候选，不把已预定离开的 Topic 重新插回池中。
        """
        snapshot = self._store.freeze_and_evict(topic_key)
        if snapshot is None:
            logger.debug("settle_and_evict: topic busy 或不存在，跳过结算")
            return None
        if snapshot.is_empty:
            logger.debug("settle_and_evict: topic 内容为空，已驱逐但无可结算材料")
            return None
        return self._build_settle_payload_from_snapshot(snapshot, reason)

    def prepare_manual_settle(
        self,
        topic_key: WorkspaceTopicKey,
    ) -> Optional[TopicMaterializeTask]:
        """manual settle 的 FLUSHING prepare：冻结材料但不驱逐，不清 blocks。

        只接受 IDLE Topic；成功取得 FLUSHING 后冻结 payload 并返回。admission
        失败由 ``abort_manual_settle`` 恢复 IDLE，成功由 ``commit_manual_settle``
        驱逐。
        """
        snapshot = self._store.freeze_for_manual_settle(topic_key)
        if snapshot is None:
            raise TopicBusyError(
                f"topic '{topic_key.topic_id}' 正忙，无法开始 manual settle"
            )
        if snapshot.is_empty:
            return None
        return self._build_settle_payload_from_snapshot(
            snapshot,
            FlushReason.MANUAL_SETTLE,
        )

    def commit_manual_settle(self, topic_key: WorkspaceTopicKey) -> bool:
        """admission 成功或正常 skip 后驱逐 FLUSHING Topic。"""
        return self._store.commit_flushing(topic_key)

    def abort_manual_settle(self, topic_key: WorkspaceTopicKey) -> None:
        """admission 失败后恢复 FLUSHING Topic 为 IDLE，保留可重试内容。"""
        self._store.abort_flushing(topic_key)

    # ========== compact 路径 ==========

    async def _compact_path(
        self,
        trigger: FlushEvent,
        retain_recent_blocks: Optional[int],
    ) -> Optional[TopicMaterializeTask]:
        """执行 compact；TOKEN_OVERFLOW 复用已持有预约，MANUAL_COMPACT 自行预约。"""
        if retain_recent_blocks is None:
            raise ValueError(f"{trigger.reason.value} requires retain_recent_blocks")

        topic_key = trigger.topic_key
        manual = trigger.reason is FlushReason.MANUAL_COMPACT
        if manual and not self._store.reserve_processing(topic_key):
            raise TopicBusyError(
                f"topic '{topic_key.topic_id}' 正忙，无法开始 manual compact"
            )

        try:
            topic_data = self._store.get_topic_data_by_key(topic_key, touch=False)
            if topic_data is None:
                return None
            if topic_data.is_empty:
                return None
            await self._compact_topic(
                topic_key,
                list(topic_data.blocks),
                topic_data.state_summary,
                retain_recent_blocks=retain_recent_blocks,
            )
            return None
        finally:
            if manual:
                self._store.release_processing(topic_key)

    # ========== 原子操作 ==========

    def _build_settle_payload_from_snapshot(
        self,
        snapshot: TopicData,
        reason: FlushReason,
    ) -> Optional[TopicMaterializeTask]:
        """从冻结 TopicData 快照构建 settle 载荷，并冻结 binding refs。"""
        return self._build_settle_payload(
            topic_id=snapshot.topic_id,
            blocks_snapshot=list(snapshot.blocks),
            state_summary=snapshot.state_summary,
            reason=reason,
            workspace_identity=snapshot.workspace_identity,
            topic_title=snapshot.topic_title,
            topic_summary=snapshot.topic_summary,
            asset_bindings=snapshot.bindings,
        )

    def _build_settle_payload(
        self,
        topic_id: str,
        blocks_snapshot: List[LogicalBlock],
        state_summary: str,
        reason: FlushReason,
        workspace_identity: WorkspaceIdentity,
        topic_title: str = "",
        topic_summary: str = "",
        asset_bindings: tuple = (),
    ) -> Optional[TopicMaterializeTask]:
        """
        Settle 原子操作：过滤 blocks 并构建 TopicMaterializeTask。

        ``asset_bindings`` 在 buffer 清理前冻结进 task，进入 queue 后不再依赖
        SemanticBuffer；后续 codec/retry 必须原样保留 ref。

        Returns:
            TopicMaterializeTask；若所有 blocks 均被过滤则返回 None
        """
        blocks_to_settle = [
            block for block in blocks_snapshot
            if block.worth_saving is not False
        ]
        filtered = len(blocks_snapshot) - len(blocks_to_settle)
        if filtered > 0:
            logger.info(
                f"worth_saving 过滤: 原始 {len(blocks_snapshot)} blocks, "
                f"过滤 {filtered} blocks, 保留 {len(blocks_to_settle)} blocks"
            )

        if not blocks_to_settle:
            logger.debug("所有 blocks 被 worth_saving 过滤，跳过 Settle")
            return None

        return TopicMaterializeTask(
            topic_id=topic_id,
            identity_scope=IdentityScope(
                actor_identity=blocks_to_settle[-1].identity,
                workspace_identity=workspace_identity,
            ),
            topic_title=topic_title,
            topic_summary=topic_summary,
            blocks=blocks_to_settle,
            state_summary=state_summary,
            asset_bindings=tuple(asset_bindings),
            reason=reason,
        )

    async def _compact_topic(
        self,
        topic_key: WorkspaceTopicKey,
        blocks_to_fold: List[LogicalBlock],
        previous_summary: str,
        *,
        retain_recent_blocks: int,
    ) -> int:
        """
        Compact 原子操作：总结待折叠前缀并更新 buffer（同步阻塞）。

        所有 compact 路径都必须至少保留一个最新 block；``retain_recent_blocks``
        必须 >= 1，小于 1 的值在输入边界以具体异常拒绝。只总结保留后缀之前的
        旧 blocks，避免 state_summary 与 recent blocks 重复承载同一轮事实。
        """
        if retain_recent_blocks < 1:
            raise ValueError("retain_recent_blocks must be >= 1")

        fold_count = max(0, len(blocks_to_fold) - retain_recent_blocks)
        if fold_count == 0:
            logger.warning(
                "Compact skipped: no blocks older than retained working set, "
                "topic_id=%s, blocks=%d, retain_recent_blocks=%d",
                topic_key.topic_id,
                len(blocks_to_fold),
                retain_recent_blocks,
            )
            return 0
        blocks_to_fold = blocks_to_fold[:fold_count]

        # 调用 RelayController 生成摘要（已包含 previous_summary 合并逻辑）
        new_summary = self._relay_controller.generate_summary(
            blocks_to_fold=blocks_to_fold,
            previous_summary=previous_summary
        )

        # 计算与写入分离：通过 Store 命名方法写入，不直接操作 buffer 字段
        folded = self._store.apply_compaction(
            topic_key,
            new_summary,
            retain_count=retain_recent_blocks,
        )

        logger.debug(
            "Compact: 生成新摘要，topic_id=%s, folded=%d, retained=%s",
            topic_key.topic_id,
            folded,
            retain_recent_blocks,
        )
        return folded


__all__ = [
    "TriggerManager",
    "DECISION_MATRIX",
]
