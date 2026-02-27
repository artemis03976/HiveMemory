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
    - 新增 route_and_ingest / swap_out_topic / get_active_topics_menu
    - BufferManager 升级为 MMU（含 LRU 驱逐）

参考: ShortTermMemory.md, PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 4.5.0
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from hivememory.core.models import Identity, StreamMessage
from hivememory.engines.perception.buffer_manager import SemanticBufferManager
from hivememory.engines.perception.interfaces import BasePerceptionLayer
from hivememory.engines.perception.models import (
    BufferState,
    FlushEvent,
    FlushReason,
    InteractionPayload,
    LogicalBlock,
    SemanticBuffer,
    TraceItem,
)
from hivememory.patchouli.config import SemanticFlowPerceptionConfig

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
        ...     config=config,
        ...     on_flush_callback=on_flush
        ... )
        >>> perception.route_and_ingest("NEW_TOPIC", payload)
    """

    def __init__(
        self,
        config: SemanticFlowPerceptionConfig,
        on_flush_callback: Optional[
            Callable[[List[StreamMessage], FlushReason], None]
        ] = None,
        # 以下参数保留向后兼容签名，但不再使用
        adsorber: Optional[Any] = None,
        relay_controller: Optional[Any] = None,
    ):
        """
        初始化语义流感知层 (MMU)

        Args:
            config: SemanticFlowPerceptionConfig 配置对象
            on_flush_callback: Flush 回调函数
            adsorber: (已弃用) 语义吸附器，保留参数兼容性
            relay_controller: (已弃用) 接力控制器，保留参数兼容性
        """
        super().__init__()

        self.config = config
        self.on_flush_callback = on_flush_callback

        # 空闲超时监控器配置（由基类管理）
        self._idle_timeout_seconds = config.idle_timeout_seconds
        self._scan_interval_seconds = config.scan_interval_seconds

        # BufferManager 作为 MMU（话题管理器）
        self._buffer_manager = SemanticBufferManager(
            max_resident_topics=getattr(config, "max_resident_topics", 5)
        )

        logger.info("SemanticFlowPerceptionLayer (MMU) 初始化完成")

    # ========== Kernel 模式载荷摄入 (v3.0) ==========

    def ingest_payload(self, payload: InteractionPayload) -> None:
        """
        摄入 Kernel 递归循环的完整交互载荷

        直接构建 LogicalBlock（使用 v3.0 字段），绕过 StreamParser + Builder。

        流程:
            1. MTPLogParser 清洗 → clean_text + fallback_traces
            2. 构建 LogicalBlock (v3.0 字段)
            3. 信号检查:
               - URGENT (write_focus/update_focus): 添加 block → 立即 flush
               - NORMAL: 直接添加 block（话题路由已由 TheEye 完成）

        Args:
            payload: Kernel → Perception 的原子传输包
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

        # 计算 priority
        is_urgent = (payload.write_focus is not None
                     or payload.update_focus is not None)
        if is_urgent:
            block.priority = "URGENT"

        identity = payload.identity

        # 3. 信号检查
        if is_urgent:
            # URGENT: 添加 block → 立即 flush
            self._buffer_manager.add_block_to_buffer(identity, block)
            reason = (
                FlushReason.MTP_WRITE if payload.write_focus is not None
                else FlushReason.MTP_UPDATE
            )
            buffer = self._buffer_manager.get_buffer(identity)
            flush_event = FlushEvent(
                flush_reason=reason,
                blocks_to_flush=buffer.blocks.copy(),
                write_focus=payload.write_focus,
                update_focus=payload.update_focus,
            )
            self._handle_flush_event(identity, flush_event)
        else:
            # NORMAL: 话题路由已由 TheEye 完成，直接添加 block
            self._buffer_manager.add_block_to_buffer(identity, block)

        # 重置状态
        self._buffer_manager.update_buffer_metadata(
            identity, state=BufferState.IDLE
        )

    def _handle_flush_event(self, identity: Identity, event: FlushEvent) -> None:
        """
        处理 flush 事件

        Args:
            identity: 身份标识
            event: FlushEvent 包含 flush 详情
        """
        logger.info(
            f"Flush buffer {identity.buffer_key}, "
            f"原因: {event.flush_reason.value}, "
            f"blocks: {len(event.blocks_to_flush)}"
        )

        # 1. 清空 buffer
        self._buffer_manager.clear_buffer(identity)

        # 2. 更新 relay_summary（如果有）
        if event.relay_summary:
            self._buffer_manager.update_buffer_metadata(
                identity, relay_summary=event.relay_summary
            )

        # 3. 重置话题核心
        self._buffer_manager.update_buffer_metadata(
            identity, reset_topic_kernel=True
        )

        # 4. 过滤 worth_saving=False 的 block
        # worth_saving=None 时保留（gateway 离线或异常时不影响冷链路）
        original_count = len(event.blocks_to_flush)
        blocks_to_process = [
            block for block in event.blocks_to_flush
            if block.worth_saving is not False
        ]
        filtered_count = original_count - len(blocks_to_process)

        if filtered_count > 0:
            logger.info(
                f"worth_saving 过滤: 原始 {original_count} blocks, "
                f"过滤 {filtered_count} blocks (worth_saving=False), "
                f"保留 {len(blocks_to_process)} blocks"
            )

        # 5. 调用回调（仅当有 block 需要处理时）
        if blocks_to_process:
            try:
                messages = self._blocks_to_messages(blocks_to_process, identity)
                if self._bus:
                    # 通过 SystemBus 发布 flush 事件
                    self._bus.emit(
                        "perception.flushed",
                        messages=messages,
                        reason=event.flush_reason,
                        write_focus=event.write_focus,
                        update_focus=event.update_focus,
                    )
                elif self.on_flush_callback:
                    # Fallback: 直接回调（无 bus 时，如测试环境）
                    self.on_flush_callback(
                        messages, event.flush_reason,
                        write_focus=event.write_focus,
                        update_focus=event.update_focus,
                    )
            except Exception as e:
                logger.error(f"Flush 回调失败: {e}")
        elif (self._bus or self.on_flush_callback) and not blocks_to_process:
            logger.debug(
                f"所有 blocks 被 worth_saving 过滤，跳过 Generation 回调"
            )

    def _blocks_to_messages(
        self,
        blocks: List[LogicalBlock],
        identity: Identity,
    ) -> List[StreamMessage]:
        """
        将 blocks 转换为 stream messages

        Args:
            blocks: 要转换的 blocks
            identity: 身份标识

        Returns:
            stream messages 列表
        """
        messages = []
        for block in blocks:
            messages.extend(block.to_stream_messages(identity))
        return messages

    def flush_buffer(
        self,
        identity: Identity,
        reason: FlushReason = FlushReason.MANUAL,
    ) -> List[StreamMessage]:
        """
        手动刷新 Buffer

        Args:
            identity: 身份标识对象
            reason: 刷新原因

        Returns:
            List[StreamMessage]: 被 Flush 的消息列表
        """
        # 获取最新的 buffer 状态
        buffer = self._buffer_manager.get_buffer(identity)
        if not buffer.blocks:
            return []

        # 创建 flush event
        flush_event = FlushEvent(
            flush_reason=reason,
            blocks_to_flush=buffer.blocks.copy(),
        )

        # 处理 flush
        self._handle_flush_event(identity, flush_event)

        return self._blocks_to_messages(flush_event.blocks_to_flush, identity)

    def get_buffer(
        self,
        identity: Identity,
    ) -> Optional[SemanticBuffer]:
        """
        获取指定 Buffer

        Args:
            identity: 身份标识对象

        Returns:
            SemanticBuffer: 缓冲区对象，不存在则创建
        """
        return self._buffer_manager.get_buffer(identity)

    def clear_buffer(
        self,
        identity: Identity,
    ) -> bool:
        """
        清理指定的 Buffer

        Args:
            identity: 身份标识对象

        Returns:
            bool: 是否成功清理
        """
        cleared = self._buffer_manager.clear_buffer(identity)
        self._buffer_manager.update_buffer_metadata(
            identity,
            reset_topic_kernel=True,
            reset_relay_summary=True,
            state=BufferState.IDLE,
        )
        return len(cleared) > 0 or True  # 总是返回 True 表示操作成功

    def list_active_buffers(self) -> List[str]:
        """
        列出所有活跃的 Buffer

        Returns:
            List[str]: Buffer key 列表
        """
        return self._buffer_manager.list_active_buffers()

    def get_buffer_info(
        self,
        identity: Identity,
    ) -> Dict[str, Any]:
        """
        获取缓冲区信息

        Args:
            identity: 身份标识对象

        Returns:
            Dict: 缓冲区信息字典
        """
        info = self._buffer_manager.get_buffer_info(identity)
        info["mode"] = "semantic_flow"
        info["identity"] = identity
        return info

    # ========== MMU 路由与话题管理 (Phase 4.5) ==========

    def get_active_topics_menu(self) -> List[Dict[str, str]]:
        """
        获取活跃话题菜单，供 TheEye 路由决策使用

        Returns:
            List[Dict]: [{topic_id, title, buffer_key}, ...]
        """
        return self._buffer_manager.get_active_topics_menu()

    def route_and_ingest(
        self,
        topic_id: str,
        payload: InteractionPayload,
    ) -> None:
        """
        MMU 核心方法：路由到指定话题并摄入载荷

        流程:
            - NEW_TOPIC: 检查驱逐 → 创建新话题 → ingest_payload
            - 已有 topic_id: 换入话题 → ingest_payload

        Args:
            topic_id: 目标话题 ID 或 "NEW_TOPIC"
            payload: Kernel → Perception 的原子传输包
        """
        identity = payload.identity

        if topic_id == "NEW_TOPIC":
            # 检查是否需要 LRU 驱逐
            if self._buffer_manager.needs_eviction():
                self._evict_lru_topic()
            # 创建新话题（使用 payload 的 identity）
            self._buffer_manager.create_new_topic(identity)
        else:
            # 按 topic_id 换入已有话题
            routed = self._buffer_manager.route(topic_id)
            if routed is None:
                logger.warning(
                    f"话题 {topic_id} 不存在，回退到 NEW_TOPIC"
                )
                if self._buffer_manager.needs_eviction():
                    self._evict_lru_topic()
                self._buffer_manager.create_new_topic(identity)

        # 摄入载荷
        self.ingest_payload(payload)

    def _evict_lru_topic(self) -> None:
        """
        LRU 驱逐：找到最久未访问的话题，flush 后换出

        驱逐流程:
            1. 找到 LRU 话题
            2. flush 其 buffer（触发 Generation 回调）
            3. 从活跃池移除
        """
        lru = self._buffer_manager.find_lru_topic()
        if lru is None:
            return

        buffer_key, buffer = lru
        logger.info(
            f"LRU 驱逐话题: {buffer_key}, "
            f"title={buffer.title}"
        )

        # flush buffer 内容
        if buffer.blocks:
            parts = buffer_key.split(":")
            if len(parts) == 3:
                evict_identity = Identity(
                    user_id=parts[0],
                    agent_id=parts[1],
                    session_id=parts[2],
                )
                self.flush_buffer(evict_identity, FlushReason.LRU_EVICTION)

        # 从活跃池移除
        self._buffer_manager.swap_out(buffer_key)

    def swap_out_topic(
        self, buffer_key: str
    ) -> Optional[SemanticBuffer]:
        """
        显式换出指定话题

        Args:
            buffer_key: 话题的 buffer key

        Returns:
            被换出的 SemanticBuffer，不存在返回 None
        """
        return self._buffer_manager.swap_out(buffer_key)

    def update_topic_title(
        self, topic_id: str, title: str
    ) -> None:
        """
        更新话题标题

        Args:
            topic_id: 话题 ID (buffer_id)
            title: 新标题
        """
        buf = self._buffer_manager.route(topic_id)
        if buf is not None:
            buf.title = title
            logger.debug(f"话题 {topic_id} 标题更新为: {title}")


__all__ = [
    "SemanticFlowPerceptionLayer",
]
