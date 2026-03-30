"""
HiveMemory - 语义流感知层 / MMU (Semantic Flow Perception Layer / Memory Management Unit)

职责:
    作为短期记忆的 MMU（内存管理单元），管理多话题的生命周期。
    负责话题路由(route)、换入(swap-in)、换出(swap-out)和 LRU 驱逐。

特性:
    - LogicalBlock 作为处理单元（页 / Page）
    - 多话题并发管理（活跃话题池）
    - LRU 驱逐策略
    - URGENT 信号立即 flush
    - 异步空闲超时监控

映射关系 (ShortTermMemory.md):
    TopicManager (MMU) = SemanticBufferManager
    TopicSegment = SemanticBuffer
    Pages = blocks (List[LogicalBlock])

Note:
    Phase 4.5 重构：
    - 移除 Adsorber 和 Relay 依赖（话题路由由 TheEye 完成）
    - 新增 route_and_ingest / swap_out_topic / get_active_topics_snapshots
    - BufferManager 升级为 MMU（含 LRU 驱逐）

参考: ShortTermMemory.md, PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 4.5.0
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from hivememory.core.models import Identity
from hivememory.engines.perception.buffer_manager import SemanticBufferManager
from hivememory.engines.perception.relay_controller import BaseRelayController
from hivememory.engines.perception.trigger_manager import TriggerManager
from hivememory.engines.perception.interfaces import BasePerceptionLayer
from hivememory.engines.perception.models import (
    BufferState,
    FlushReason,
    InteractionPayload,
    LogicalBlock,
    SemanticBuffer,
)
from hivememory.patchouli.config import SemanticFlowPerceptionConfig
from hivememory.utils.token_estimator import estimate_tokens

logger = logging.getLogger(__name__)


class SemanticFlowPerceptionLayer(BasePerceptionLayer):
    """
    语义流感知层 / MMU (Phase 4.5 重构版)

    作为短期记忆的内存管理单元 (MMU)，管理多话题的并发生命周期：
        - 话题路由：根据 TheEye 的 target_topic 将载荷路由到正确话题
        - LRU 驱逐：活跃话题池满时驱逐最久未访问的话题
        - URGENT 信号：write_focus/update_focus 触发立即 flush
        - 空闲超时：长期不活跃的话题自动换出

    架构：
        - BufferManager (MMU): 话题池管理（CRUD + 路由 + LRU）
        - PerceptionLayer: 编排和协调

    Examples:
        >>> config = SemanticFlowPerceptionConfig()
        >>> perception = SemanticFlowPerceptionLayer(
        ...     config=config
        ... )
        >>> perception.set_generation_callback(on_generate_memory)
        >>> perception.route_and_ingest("NEW_TOPIC", payload)
    """

    def __init__(
        self,
        config: SemanticFlowPerceptionConfig,
        relay_controller: BaseRelayController,
    ):
        """
        初始化语义流感知层 (MMU)

        Args:
            config: SemanticFlowPerceptionConfig 配置对象
            relay_controller: 接力控制器 / Page Folding 摘要生成器
        """
        super().__init__()

        self.config = config

        # 空闲超时监控器配置（由基类管理）
        self._idle_timeout_seconds = config.idle_timeout_seconds
        self._scan_interval_seconds = config.scan_interval_seconds

        self._relay_controller = relay_controller

        # BufferManager 作为 MMU（话题管理器）
        self._buffer_manager = SemanticBufferManager(
            max_resident_topics=config.max_resident_topics
        )

        # TriggerManager 负责话题结算调度
        self._trigger_manager = TriggerManager(
            buffer_manager=self._buffer_manager,
            relay_controller=self._relay_controller,
        )

        logger.info("SemanticFlowPerceptionLayer (MMU) 初始化完成")

    def set_generation_callback(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        self._trigger_manager.set_generation_callback(callback)

    # ========== Kernel 模式载荷摄入 (v3.0) ==========

    async def route_and_ingest(
        self,
        topic_id: str,
        payload: InteractionPayload,
    ) -> str:
        """
        MMU 核心方法：路由到指定话题并摄入载荷

        流程:
            - NEW_TOPIC: 检查驱逐 → 创建新话题 → ingest_payload
            - 已有 topic_id: 换入话题 → ingest_payload（不存在则回退到 NEW_TOPIC）

        Args:
            topic_id: 目标话题 ID 或 "NEW_TOPIC"
            payload: Kernel → Perception 的原子传输包

        Returns:
            str: 最终路由到的 topic_id
        """
        # 统一处理：需要创建新话题的情况
        if topic_id == "NEW_TOPIC":
            topic_id = await self.create_new_topic(payload.identity)
        else:
            buffer = self._buffer_manager.get_buffer(topic_id)
            if buffer is None:
                logger.warning(f"话题 {topic_id} 不存在，回退到 NEW_TOPIC")
                topic_id = await self.create_new_topic(payload.identity)

        # 摄入载荷
        await self.ingest_payload(payload, topic_id)

        # 更新最后活跃话题
        self._buffer_manager.set_last_active_topic(topic_id)

        return topic_id

    async def ingest_payload(
        self,
        payload: InteractionPayload,
        topic_id: str,
    ) -> None:
        """
        摄入完整交互载荷

        从 InteractionPayload 直接构建 LogicalBlock。

        流程:
            1. MTPLogParser 清洗 → clean_text + fallback_traces
            2. 构建 LogicalBlock (v3.0 字段)
            3. 信号检查:
               - URGENT (write_focus/update_focus): 添加 block → 立即 flush
               - NORMAL: 直接添加 block

        Args:
            payload: Kernel → Perception 的原子传输包
            topic_id: 目标话题 ID
        """
        from hivememory.patchouli.protocol.mtp_log_parser import MTPLogParser

        # 1. 清洗 MTP 噪音
        clean_text, fallback_traces = MTPLogParser.parse(payload.assistant_message)

        # 优先使用 Kernel 传入的 traces，回退到 parser 解析的
        traces = payload.mtp_traces if payload.mtp_traces else fallback_traces

        # 2. 构建 LogicalBlock (v3.0 字段)
        block = LogicalBlock(
            user_query=payload.user_message,
            rewritten_query=payload.rewritten_query,
            semantic_traces=traces,
            raw_response=payload.assistant_message,
            clean_response=clean_text,
            worth_saving=payload.worth_saving,
            write_focus=payload.write_focus,
            update_focus=payload.update_focus,
        )

        # 2.5 计算 block 的 total_tokens
        block.total_tokens = (
            estimate_tokens(block.user_query)
            + estimate_tokens(block.clean_response)
            + sum(
                estimate_tokens(t.query or "") + estimate_tokens(t.target or "")
                for t in block.semantic_traces
            )
        )

        # 计算 priority
        is_urgent = (payload.write_focus is not None or payload.update_focus is not None)
        # 3. 信号检查
        if is_urgent:
            # URGENT: 添加 block → 立即 flush
            block.priority = "URGENT"
            self._buffer_manager.add_block(topic_id, block)
            reason = (
                FlushReason.MTP_WRITE if payload.write_focus is not None
                else FlushReason.MTP_UPDATE
            )

            # 调用统一调度器
            await self._trigger_manager.resolve_topic(
                topic_id=topic_id,
                trigger_reason=reason,
                mtp_focus=payload.write_focus or payload.update_focus,
            )
        else:
            # NORMAL: 话题路由已由 TheEye 完成，直接添加 block
            self._buffer_manager.add_block(topic_id, block)

        # 4. Page Folding 检查（token 溢出时压缩旧 blocks）
        await self._maybe_fold_pages(topic_id)

        # 重置状态
        self._buffer_manager.update_metadata(
            topic_id, state=BufferState.IDLE
        )

    async def _maybe_fold_pages(self, topic_id: str) -> None:
        """
        Page Folding: token 溢出时触发 Compact 操作
        """
        buffer = self._buffer_manager.get_buffer(topic_id)
        if buffer is None:
            return

        threshold = self.config.fold_token_threshold

        logger.debug(
            f"_maybe_fold_pages: topic_id={topic_id}, "
            f"total_tokens={buffer.total_tokens}, threshold={threshold}, "
            f"blocks_count={len(buffer.blocks)}"
        )

        if buffer.total_tokens <= threshold:
            return

        logger.info(
            f"Token 溢出: topic_id={topic_id}, "
            f"total_tokens={buffer.total_tokens} > threshold={threshold}"
        )

        # 调用统一调度器
        await self._trigger_manager.resolve_topic(
            topic_id=topic_id,
            trigger_reason=FlushReason.TOKEN_OVERFLOW,
        )

    async def manual_trigger(
        self,
        topic_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        手动触发话题结算 (Archive + Compact)

        语义：立即归档 + 生成摘要并保留内存。
        当用户主动请求保存当前对话状态时使用。不同于其他触发原因，
        MANUAL 触发会同时执行 Archive 和 Compact，但不会 Evict 话题。

        Args:
            topic_id: 目标话题 ID。如果为 None，则使用 last_active_topic_id 作为回退。

        Returns:
            Dict: 包含以下键的字典:
                - success: bool - 操作是否成功
                - topic_id: str - 实际操作的话题 ID
                - message: str - 操作结果描述
                - blocks_archived: int - 归档的 block 数量

        Raises:
            ValueError: 如果 topic_id 未指定且没有 last_active_topic_id
        """
        # 解析目标话题
        target_topic_id = topic_id
        if target_topic_id is None:
            target_topic_id = self._buffer_manager.get_last_active_topic()

        if target_topic_id is None:
            raise ValueError(
                "manual_trigger: 未指定 topic_id 且没有活跃话题可回退"
            )

        # 验证话题存在
        buffer = self._buffer_manager.get_buffer(target_topic_id)
        if buffer is None:
            return {
                "success": False,
                "topic_id": target_topic_id,
                "message": f"话题 {target_topic_id} 不存在",
                "blocks_archived": 0,
            }

        if not buffer.blocks:
            return {
                "success": True,
                "topic_id": target_topic_id,
                "message": "话题为空，无需处理",
                "blocks_archived": 0,
            }

        blocks_count = len(buffer.blocks)

        # 调用统一调度器（MANUAL 触发）
        await self._trigger_manager.resolve_topic(
            topic_id=target_topic_id,
            trigger_reason=FlushReason.MANUAL,
        )

        logger.info(
            f"manual_trigger 完成: topic_id={target_topic_id}, "
            f"blocks_archived={blocks_count}"
        )

        return {
            "success": True,
            "topic_id": target_topic_id,
            "message": f"成功归档 {blocks_count} 个 blocks",
            "blocks_archived": blocks_count,
        }

    def get_buffer(
        self,
        topic_id: str,
    ) -> Optional[SemanticBuffer]:
        """
        获取指定 Buffer

        Args:
            topic_id: 话题 ID

        Returns:
            SemanticBuffer: 缓冲区对象，不存在则返回 None
        """
        return self._buffer_manager.get_buffer(topic_id)

    def clear_buffer(
        self,
        topic_id: str,
    ) -> bool:
        """
        清理指定的 Buffer

        Args:
            topic_id: 话题 ID

        Returns:
            bool: 是否成功清理
        """
        # clear_buffer 原意是清空内容，不移除 topic
        cleared = self._buffer_manager.clear_buffer(topic_id)
        if cleared is not None:  # 如果 buffer 存在，clear_buffer 返回 list (可能为空)
            self._buffer_manager.update_metadata(
                topic_id,
                state=BufferState.IDLE,
            )
            return True
        return False

    def list_active_buffers(self) -> List[str]:
        """
        列出所有活跃的 Buffer

        Returns:
            List[str]: topic_id 列表
        """
        return [b.topic_id for b in self._buffer_manager.get_all_buffers()]

    def get_buffer_info(
        self,
        topic_id: str,
    ) -> Dict[str, Any]:
        """
        获取缓冲区信息

        Args:
            topic_id: 话题 ID

        Returns:
            Dict: 缓冲区信息字典
        """
        info = self._buffer_manager.get_buffer_info(topic_id)
        info["mode"] = "semantic_flow"
        return info

    # ========== 话题路由与管理 ==========

    def get_active_topics_snapshots(
        self,
        identity: Optional[Identity] = None,
    ) -> List["TopicSnapshot"]:
        """
        获取活跃话题快照列表（用于 TheEye 路由决策）

        每个快照包含:
        - topic_id: 话题 ID
        - title: 话题标题
        - state_summary: 状态摘要（如果有折叠）
        - last_turn: 最后一轮对话 {"user": "...", "assistant": "..."}

        Args:
            identity: 可选，如果提供则只返回该用户的话题

        Returns:
            List[TopicSnapshot]: 话题快照列表
        """
        from hivememory.engines.perception.models import TopicSnapshot

        # 获取 buffers
        if identity:
            buffers = self._buffer_manager.get_buffers_by_owner(identity)
        else:
            buffers = self._buffer_manager.get_all_buffers()

        snapshots = []
        for buffer in buffers:
            # 只返回有内容的话题
            if not buffer.blocks:
                continue

            # 获取最后一个 block
            last_block = buffer.blocks[-1]
            last_turn = {
                "user": last_block.user_query,
                "assistant": last_block.clean_response,
            }

            snapshot = TopicSnapshot(
                topic_id=buffer.topic_id,
                title=buffer.title,
                state_summary=buffer.state_summary,
                last_turn=last_turn,
                total_tokens=buffer.total_tokens,
            )
            snapshots.append(snapshot)

        return snapshots

    async def prepare_topic(
        self,
        target_topic_id: str,
        new_topic_title: Optional[str],
        new_topic_summary: Optional[str],
        identity: Identity,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        预创建/刷新话题，返回 (topic_id, pool_snapshot, topic_context)

        在 LLM 生成之前调用，将驱逐和创建提前执行：
        - 已有话题: 刷新 last_accessed_at 置顶
        - 新话题: 分配 UUID，保存 title/summary，检查 LRU 驱逐

        Args:
            target_topic_id: "NEW_TOPIC" 或已有 topic_id
            new_topic_title: Gateway 生成的新话题标题
            new_topic_summary: Gateway 生成的新话题摘要
            identity: 用户身份

        Returns:
            (topic_id, pool_snapshot, topic_context)
        """
        if target_topic_id == "NEW_TOPIC":
            topic_id = await self.create_new_topic(
                identity=identity,
                title=new_topic_title,
                summary=new_topic_summary,
            )
        else:
            buffer = self._buffer_manager.get_buffer(target_topic_id)
            if buffer is None:
                logger.warning(f"话题 {target_topic_id} 不存在，回退到创建新话题")
                topic_id = await self.create_new_topic(
                    identity=identity,
                    title=new_topic_title,
                    summary=new_topic_summary,
                )
            else:
                # 已有话题：刷新访问时间（置顶）
                topic_id = target_topic_id

        pool_snapshot = self.get_topic_pool_snapshot(identity)
        topic_context = self.get_topic_context(topic_id, max_recent_blocks=5)

        return topic_id, pool_snapshot, topic_context

    def get_topic_pool_snapshot(self, identity: Identity) -> Dict[str, Any]:
        """
        返回完整话题池状态供前端直接渲染

        Args:
            identity: 用户身份

        Returns:
            Dict: {topics: [...], max_resident_topics, current_count}
        """
        buffers = self._buffer_manager.get_buffers_by_owner(identity)
        buffers_sorted = sorted(
            buffers, key=lambda b: b.last_accessed_at, reverse=True
        )
        topics = [
            {
                "topic_id": buf.topic_id,
                "title": buf.title,
                "state_summary": buf.state_summary or "",
                "block_count": len(buf.blocks),
                "last_accessed_at": buf.last_accessed_at,
                "total_tokens": buf.total_tokens,
            }
            for buf in buffers_sorted
        ]
        return {
            "topics": topics,
            "max_resident_topics": self._buffer_manager.max_resident_topics,
            "current_count": len(buffers),
        }

    def get_topic_context(
        self,
        topic_id: str,
        max_recent_blocks: int = 5,
    ) -> Dict[str, Any]:
        """
        获取话题的完整上下文用于 Prompt 组装

        Args:
            topic_id: 话题 ID
            max_recent_blocks: 返回最近的 N 个 blocks

        Returns:
            Dict: {
                "state_summary": str,
                "blocks": List[LogicalBlock],
                "total_tokens": int,
                "title": str,
            }
        """
        # 获取 buffer
        buffer = self._buffer_manager.get_buffer(topic_id)
        if buffer is None:
            logger.warning(f"话题不存在: topic_id={topic_id}，返回空上下文")
            return {
                "state_summary": "",
                "blocks": [],
                "total_tokens": 0,
                "title": "未知话题",
            }

        # 更新访问时间
        buffer.last_accessed_at = datetime.now().timestamp()

        # 获取最近的 N 个 blocks
        recent_blocks = buffer.blocks[-max_recent_blocks:] if buffer.blocks else []

        return {
            "state_summary": buffer.state_summary,
            "blocks": recent_blocks,
            "total_tokens": buffer.total_tokens,
            "title": buffer.title,
        }

    async def create_new_topic(
        self,
        identity: Identity,
        title: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> str:
        """
        创建新话题（必要时先执行 LRU 驱逐）

        Args:
            identity: 新话题的归属身份
            title: 可选话题标题
            summary: 可选话题摘要

        Returns:
            str: 新创建的 topic_id
        """
        if self._buffer_manager.needs_eviction():
            await self._evict_lru_topic()
        buffer = self._buffer_manager.create_buffer(
            identity=identity,
            title=title or "新建话题",
        )
        if summary:
            buffer.state_summary = summary
        return buffer.topic_id

    async def _evict_lru_topic(self) -> None:
        """
        LRU 驱逐：找到最久未访问的话题，调用统一调度器
        """
        buffer = self._buffer_manager.get_lru_buffer()
        if buffer is None:
            return

        logger.info(
            f"LRU 驱逐话题: topic_id={buffer.topic_id}, "
            f"title={buffer.title}"
        )

        # 调用统一调度器（Archive + Evict）
        await self._trigger_manager.resolve_topic(
            topic_id=buffer.topic_id,
            trigger_reason=FlushReason.LRU_EVICTION,
        )

    def swap_out_topic(
        self, topic_id: str
    ) -> Optional[SemanticBuffer]:
        """
        显式换出指定话题

        Args:
            topic_id: 话题的 ID

        Returns:
            被换出的 SemanticBuffer，不存在返回 None
        """
        return self._buffer_manager.pop_buffer(topic_id)

    def update_topic_title(
        self, topic_id: str, title: str
    ) -> None:
        """
        更新话题标题
        """
        buffer = self._buffer_manager.get_buffer(topic_id)
        if buffer:
            buffer.title = title
            logger.debug(f"话题 {topic_id} 标题更新为: {title}")

    # ========== Idle Hibernate (§5.1) ==========

    async def scan_idle_buffers_now(self) -> List[str]:
        """
        立即执行一次空闲 Buffer 扫描（手动触发）

        Returns:
            List[str]: 被刷新的 topic_id 列表
        """
        flushed_keys = []
        all_buffers = self._buffer_manager.get_all_buffers()

        for buffer in all_buffers:
            if buffer.is_idle(self._idle_timeout_seconds):
                logger.info(
                    f"检测到空闲话题: topic_id={buffer.topic_id}, "
                    f"idle_time={(datetime.now().timestamp() - buffer.last_update):.1f}s"
                )

                # 调用统一调度器（Archive + Evict）
                await self._trigger_manager.resolve_topic(
                    topic_id=buffer.topic_id,
                    trigger_reason=FlushReason.IDLE_TIMEOUT,
                )

                flushed_keys.append(buffer.topic_id)

        return flushed_keys


__all__ = [
    "SemanticFlowPerceptionLayer",
]
