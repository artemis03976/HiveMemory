"""
HiveMemory Trigger Manager

话题结算调度器，负责根据触发原因执行对应的原子操作组合，并返回结算载荷。

职责:
    - 统一的 FlushEvent 输入协议
    - 根据触发原因执行原子操作组合 (Settle/Compact/Evict)
    - 返回 TopicMaterializeTask 供上层提交给生成链路（不主动触发）

核心概念:
    - Settle (结算): 将 blocks 打包为 TopicMaterializeTask 返回给调用方
    - Compact (压缩): 生成 state_summary（同步阻塞）
    - Evict (驱逐): 从活跃池移除 buffer

参考: BufferManagement.md, ShortTermMemory.md §4.2, §5.1

作者: HiveMemory Team
版本: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from hivememory.core.models import WorkspaceIdentity, WorkspaceTopicKey

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
    FlushReason.MANUAL: {
        "settle": True,   # 立即结算到 Librarian
        "compact": True,  # 生成摘要保持上下文连续性
        "evict": False,   # 保留话题在活跃池中
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
        | Trigger Reason   | Settle | Compact | Evict |
        |------------------|--------|---------|-------|
        | TOKEN_OVERFLOW   | ❌     | ✅      | ❌    |
        | IDLE_TIMEOUT     | ✅     | ❌      | ✅    |
        | LRU_EVICTION     | ✅     | ❌      | ✅    |
        | SHUTDOWN         | ✅     | ❌      | ✅    |
        | MANUAL           | ✅     | ✅      | ❌    |

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
            trigger: 结算触发指令（包含 topic_id 与 reason）
            retain_recent_blocks: TOKEN_OVERFLOW 时必须提供的最近工作集大小；
                其他触发原因忽略该值

        Returns:
            TopicMaterializeTask 供上层提交给生成链路；如无需结算则返回 None
        """
        topic_data = self._store.get_topic_data_by_key(trigger.topic_key)
        if topic_data is None or topic_data.is_empty:
            logger.debug("resolve_topic: topic_data 为空或不存在，跳过结算")
            return None

        # 1. 查表：决策开关
        actions = DECISION_MATRIX.get(trigger.reason)
        if not actions:
            logger.error(f"未知的触发原因: {trigger.reason}")
            return None

        need_settle = actions["settle"]
        need_compact = actions["compact"]
        need_evict = actions["evict"]

        logger.info(
            f"resolve_topic: topic_id={trigger.topic_id}, "
            f"reason={trigger.reason}, "
            f"actions=[Settle={need_settle}, Compact={need_compact}, Evict={need_evict}], "
            f"blocks={topic_data.block_count}"
        )

        blocks_snapshot = list(topic_data.blocks)
        settle_payload: Optional[TopicMaterializeTask] = None

        # Action 1: Settle — 构建载荷并返回给调用方
        if need_settle:
            settle_payload = self._build_settle_payload(
                topic_id=trigger.topic_id,
                blocks_snapshot=blocks_snapshot,
                state_summary=topic_data.state_summary,
                reason=trigger.reason,
                workspace_identity=topic_data.workspace_identity,
                topic_title=topic_data.topic_title,
                topic_summary=topic_data.topic_summary,
            )

        # Action 2: Compact（同步阻塞）
        if need_compact:
            if (
                trigger.reason == FlushReason.TOKEN_OVERFLOW
                and retain_recent_blocks is None
            ):
                raise ValueError(
                    "TOKEN_OVERFLOW requires retain_recent_blocks"
                )
            await self._compact_topic(
                trigger.topic_key,
                blocks_snapshot,
                topic_data.state_summary,
                retain_recent_blocks=(
                    retain_recent_blocks
                    if trigger.reason == FlushReason.TOKEN_OVERFLOW
                    else None
                ),
            )

        # Settle 之后旧 Blocks 必须清空；纯 Compact 则由 fold 操作保留最近工作集。
        if need_settle:
            self._store.clear_blocks(trigger.topic_key)

        # Action 3: Evict（内存清理）
        if need_evict:
            self._store.pop_buffer_by_key(trigger.topic_key)

        return settle_payload

    # ========== 原子操作 ==========

    def _build_settle_payload(
        self,
        topic_id: str,
        blocks_snapshot: List[LogicalBlock],
        state_summary: str,
        reason: FlushReason,
        workspace_identity: WorkspaceIdentity,
        topic_title: str = "",
        topic_summary: str = "",
    ) -> Optional[TopicMaterializeTask]:
        """
        Settle 原子操作：过滤 blocks 并构建 TopicMaterializeTask。

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
            workspace_identity=workspace_identity,
            topic_title=topic_title,
            topic_summary=topic_summary,
            blocks=blocks_to_settle,
            state_summary=state_summary,
            reason=reason,
        )

    async def _compact_topic(
        self,
        topic_key: WorkspaceTopicKey,
        blocks_to_fold: List[LogicalBlock],
        previous_summary: str,
        *,
        retain_recent_blocks: Optional[int] = None,
    ) -> int:
        """
        Compact 原子操作：总结待折叠前缀并更新 buffer（同步阻塞）。

        ``retain_recent_blocks=None`` 用于 MANUAL 等“总结后结算清空”的路径；
        TOKEN_OVERFLOW 必须传入非负值，只总结保留后缀之前的旧 blocks，避免
        state_summary 与 recent blocks 重复。
        """
        if retain_recent_blocks is not None:
            if retain_recent_blocks < 0:
                raise ValueError(
                    "retain_recent_blocks must be greater than or equal to 0"
                )
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
        if retain_recent_blocks is None:
            self._store.update_summary(topic_key, new_summary)
            folded = 0
        else:
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
