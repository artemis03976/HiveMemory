"""
HiveMemory Trigger Manager

话题结算调度器，负责根据触发原因执行对应的原子操作组合。

职责:
    - 统一的 Flush 触发调度
    - 根据触发原因执行原子操作组合 (Archive/Compact/Evict)
    - 依赖外部服务 (RelayController)

核心概念:
    - Archive (归档): 将 blocks 打包发送给 Librarian（异步非阻塞）
    - Compact (压缩): 生成 state_summary（同步阻塞）
    - Evict (驱逐): 从活跃池移除 buffer

参考: BufferManagement.md, ShortTermMemory.md §4.2, §5.1

作者: HiveMemory Team
版本: 1.1.0
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from hivememory.engines.perception.models import (
    FlushReason,
    LogicalBlock,
    ArchivePayload,
)

if TYPE_CHECKING:
    from hivememory.engines.perception.buffer_manager import SemanticBufferManager
    from hivememory.engines.perception.relay_controller import BaseRelayController

logger = logging.getLogger(__name__)


# ========== 决策矩阵：触发原因 -> 原子操作组合 ==========
DECISION_MATRIX: Dict[FlushReason, Dict[str, bool]] = {
    FlushReason.TOKEN_OVERFLOW: {
        "archive": False,  # 话题未完，暂不归档
        "compact": True,   # 必须留摘要以接力
        "evict": False,    # 保持存活
    },
    FlushReason.IDLE_TIMEOUT: {
        "archive": True,   # 完整事件，落库
        "compact": False,  # 无需摘要，人已走
        "evict": True,     # 踢出内存
    },
    FlushReason.LRU_EVICTION: {
        "archive": True,   # 完整事件，落库
        "compact": False,  # 无需摘要
        "evict": True,     # 踢出内存
    },
    FlushReason.SHUTDOWN: {
        "archive": True,   # 进程关闭前强制归档
        "compact": False,  # 无需摘要
        "evict": True,     # 清空内存态话题
    },
    FlushReason.MANUAL: {
        "archive": True,   # 立即归档到 Librarian
        "compact": True,   # 生成摘要保持上下文连续性
        "evict": False,    # 保留话题在活跃池中
    },
}


class TriggerManager:
    """
    Flush 触发调度器

    根据 FlushReason 查表决定执行哪些原子操作组合：
        - Archive: 发送事件到 Librarian（异步非阻塞）
        - Compact: 生成 state_summary（同步阻塞）
        - Evict: 从活跃池移除 buffer

    决策矩阵 (DECISION_MATRIX):
        | Trigger Reason   | Archive | Compact | Evict |
        |------------------|---------|---------|-------|
        | TOKEN_OVERFLOW   | ❌      | ✅      | ❌    |
        | IDLE_TIMEOUT     | ✅      | ❌      | ✅    |
        | LRU_EVICTION     | ✅      | ❌      | ✅    |
        | SHUTDOWN         | ✅      | ❌      | ✅    |
        | MANUAL           | ✅      | ✅      | ❌    |

    依赖:
        - SemanticBufferManager: 读取 buffer 状态、执行 evict
        - RelayController: 生成摘要
        - on_generate_memory 回调: 发送归档事件给 Librarian

    Examples:
        >>> trigger_manager = TriggerManager(buffer_manager, relay_controller)
        >>> trigger_manager.set_generation_callback(callback)
        >>> await trigger_manager.resolve_topic(identity, FlushReason.IDLE_TIMEOUT)
    """

    def __init__(
        self,
        buffer_manager: "SemanticBufferManager",
        relay_controller: Optional["BaseRelayController"] = None,
    ) -> None:
        """
        初始化 TriggerManager

        Args:
            buffer_manager: SemanticBufferManager 实例（用于读取/修改 buffer 状态）
            relay_controller: RelayController 实例（用于生成摘要，可选）
        """
        self._buffer_manager = buffer_manager
        self._relay_controller = relay_controller
        self._on_generate_memory: Optional[Callable] = None

        logger.info("TriggerManager 初始化完成")

    # ========== 依赖注入 ==========

    def set_generation_callback(self, callback: Callable) -> None:
        """
        注入记忆生成回调函数

        Args:
            callback: 异步回调函数，接收 payload 字典参数
        """
        self._on_generate_memory = callback
        logger.debug("generation_callback 已注入到 TriggerManager")

    def set_relay_controller(self, relay_controller: "BaseRelayController") -> None:
        self._relay_controller = relay_controller

    # ========== 统一调度器 ==========

    async def resolve_topic(
        self,
        topic_id: str,
        trigger_reason: FlushReason,
        wait_for_archive: bool = False,
    ) -> None:
        """统一的话题结算调度器。"""
        buffer = self._buffer_manager.get_buffer(topic_id)
        if not buffer or not buffer.blocks:
            logger.debug("resolve_topic: buffer 为空或不存在，跳过结算")
            return

        # 1. 查表：决策开关
        actions = DECISION_MATRIX.get(trigger_reason)
        if not actions:
            logger.error(f"未知的触发原因: {trigger_reason}")
            return

        need_archive = actions["archive"]
        need_compact = actions["compact"]
        need_evict = actions["evict"]

        logger.info(
            f"resolve_topic: topic_id={topic_id}, "
            f"reason={trigger_reason.value}, "
            f"actions=[Archive={need_archive}, Compact={need_compact}, Evict={need_evict}], "
            f"blocks={len(buffer.blocks)}"
        )

        # 2. 提取数据快照（防止后续操作污染）
        blocks_snapshot = buffer.blocks.copy()
        previous_summary = buffer.state_summary

        # ================= 执行区 =================

        # Action 1: Archive（默认异步非阻塞；shutdown drain 可等待完成）
        if need_archive:
            await self._archive_topic(
                topic_id,
                blocks_snapshot,
                previous_summary,
                trigger_reason,
                buffer.user_id,
                topic_title=buffer.topic_title,
                topic_summary=buffer.topic_summary,
                wait_for_completion=wait_for_archive,
            )

        # Action 2: Compact（同步阻塞）
        if need_compact:
            await self._compact_topic(topic_id, blocks_snapshot, previous_summary)

        # 无论是否 Compact，只要发生结算，旧 Blocks 都必须清空
        buffer.blocks.clear()
        buffer.total_tokens = 0
        buffer.last_update = datetime.now().timestamp()

        # Action 3: Evict（内存清理）
        if need_evict:
            self._buffer_manager.pop_buffer(topic_id)

    # ========== 原子操作 ==========

    async def _archive_topic(
        self,
        topic_id: str,
        blocks_snapshot: List[LogicalBlock],
        state_summary: str,
        reason: FlushReason = FlushReason.IDLE_TIMEOUT,
        user_id: Optional[str] = None,
        wait_for_completion: bool = False,
        topic_title: str = "",
        topic_summary: str = "",
    ) -> None:
        """
        Archive 原子操作：将 blocks 打包发送给 Librarian

        默认保持异步非阻塞；当 wait_for_completion=True 时等待回调执行完成。
        """
        # 过滤 worth_saving=False 的 block
        # worth_saving=None 时保留（gateway 离线或异常时不影响冷链路）
        original_count = len(blocks_snapshot)
        blocks_to_archive = [
            block for block in blocks_snapshot
            if block.worth_saving is not False
        ]
        filtered_count = original_count - len(blocks_to_archive)

        if filtered_count > 0:
            logger.info(
                f"worth_saving 过滤: 原始 {original_count} blocks, "
                f"过滤 {filtered_count} blocks (worth_saving=False), "
                f"保留 {len(blocks_to_archive)} blocks"
            )

        # 仅当有 block 需要归档时才发射事件
        if not blocks_to_archive:
            logger.debug("所有 blocks 被 worth_saving 过滤，跳过 Archive")
            return

        # 构建完整的 payload，包含 reason 以便回调函数正确处理
        payload = ArchivePayload(
            topic_id=topic_id,
            topic_title=topic_title,
            topic_summary=topic_summary,
            user_id=user_id,
            blocks=blocks_to_archive,
            state_summary=state_summary,
            reason=reason,
        )

        # 使用回调函数触发记忆生成
        if self._on_generate_memory:
            if wait_for_completion:
                await self._on_generate_memory(payload)
                logger.info(
                    f"Archive: 已等待记忆生成完成，blocks={len(blocks_to_archive)}"
                )
                return

            # 安全地启动后台任务
            # 在有运行中事件循环时使用 create_task，否则尝试获取或创建循环
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self._on_generate_memory(payload))
            except RuntimeError:
                # 没有运行中的事件循环，创建一个新的来执行任务
                try:
                    asyncio.run(self._on_generate_memory(payload))
                except Exception as e:
                    logger.warning(f"Archive 回调执行失败: {e}")
            logger.info(f"Archive: 通过回调触发记忆生成，blocks={len(blocks_to_archive)}")
        else:
            logger.warning("未配置 generation_callback，跳过 Archive")

    async def _compact_topic(
        self,
        topic_id: str,
        blocks_to_fold: List[LogicalBlock],
        previous_summary: str,
    ) -> None:
        """
        Compact 原子操作：生成 state_summary 并更新 buffer

        特性：同步阻塞（必须等待摘要生成完成）
        """
        if not self._relay_controller:
            logger.warning("RelayController 未注入，跳过 Compact")
            return

        buffer = self._buffer_manager.get_buffer(topic_id)
        if not buffer:
            return

        # 调用 RelayController 生成摘要（已包含 previous_summary 合并逻辑）
        new_summary = self._relay_controller.generate_summary(
            blocks_to_fold=blocks_to_fold,
            previous_summary=previous_summary
        )

        buffer.state_summary = new_summary
        buffer.last_update = datetime.now().timestamp()

        logger.debug(f"Compact: 生成新摘要，长度={len(new_summary)}")


__all__ = [
    "TriggerManager",
    "DECISION_MATRIX",
]
