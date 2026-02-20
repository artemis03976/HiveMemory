"""
HiveMemory - 语义流感知层 (Semantic Flow Perception Layer)

职责:
    使用统一语义流架构的感知层实现。
    负责编排 flush 逻辑，协调 BufferManager、Adsorber 和 Relay。

特性:
    - LogicalBlock 作为处理单元
    - 语义吸附判定
    - Token 溢出检测与接力
    - 异步空闲超时监控

Note:
    v3.0 重构：
    - BufferManager 简化为纯状态管理器
    - Adsorber 和 Relay 变为无状态服务
    - Flush 编排逻辑移至 PerceptionLayer

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from hivememory.core.models import Identity, StreamMessage
from hivememory.engines.perception.buffer_manager import SemanticBufferManager
from hivememory.engines.perception.interfaces import BasePerceptionLayer
from hivememory.engines.perception.relay_controller import RelayController
from hivememory.engines.perception.semantic_adsorber import SemanticBoundaryAdsorber
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
    语义流感知层 (v3.0 重构版)

    使用统一的语义流架构：
        - LogicalBlock 作为处理单元
        - 语义吸附判定
        - Token 溢出检测与接力
        - 异步空闲超时监控

    职责：
        - 解析消息
        - 编排 flush 逻辑（协调 Adsorber 和 Relay）
        - 管理话题核心向量更新
        - 管理空闲超时监控

    架构：
        - BufferManager: 纯状态容器（CRUD 操作）
        - Adsorber: 无状态服务（语义漂移检测）
        - Relay: 无状态服务（Token 溢出检测）
        - PerceptionLayer: 编排和协调

    Examples:
        >>> def on_flush(messages, reason):
        ...     print(f"Flush: {reason}, Messages: {len(messages)}")
        >>>
        >>> config = SemanticFlowPerceptionConfig()
        >>> perception = SemanticFlowPerceptionLayer(
        ...     config=config,
        ...     adsorber=adsorber,
        ...     relay_controller=relay_controller,
        ...     on_flush_callback=on_flush
        ... )
        >>> perception.start_idle_monitor()
        >>>
        >>> perception.add_message("user", "hello", identity)
        >>>
        >>> perception.stop_idle_monitor()
    """

    def __init__(
        self,
        config: SemanticFlowPerceptionConfig,
        adsorber: SemanticBoundaryAdsorber,
        relay_controller: RelayController,
        on_flush_callback: Optional[
            Callable[[List[StreamMessage], FlushReason], None]
        ] = None,
    ):
        """
        初始化语义流感知层

        Args:
            config: SemanticFlowPerceptionConfig 配置对象
            adsorber: 语义吸附器（无状态服务）
            relay_controller: 接力控制器（无状态服务）
            on_flush_callback: Flush 回调函数
                参数: (messages: List[StreamMessage], reason: FlushReason)
        """
        super().__init__()

        self.config = config
        self.adsorber = adsorber
        self.relay_controller = relay_controller
        self.on_flush_callback = on_flush_callback

        # 空闲超时监控器配置（由基类管理）
        self._idle_timeout_seconds = config.idle_timeout_seconds
        self._scan_interval_seconds = config.scan_interval_seconds

        # BufferManager 作为纯状态容器
        self._buffer_manager = SemanticBufferManager()

        logger.info("SemanticFlowPerceptionLayer 初始化完成")

    # ========== Kernel 模式载荷摄入 (v3.0) ==========

    def ingest_payload(self, payload: InteractionPayload) -> None:
        """
        摄入 Kernel 递归循环的完整交互载荷 (BlockBuilder 状态机, §3.2)

        直接构建 LogicalBlock（使用 v3.0 字段），绕过 StreamParser + Builder。

        流程:
            1. MTPLogParser 清洗 → clean_text + fallback_traces
            2. 构建 LogicalBlock (v3.0 字段)
            3. 信号检查:
               - URGENT (write_focus/update_focus): 添加 block → 立即 flush
               - NORMAL: 语义吸附判定 + 溢出接力判定 → 添加 block → 更新话题核心

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
        buffer = self._buffer_manager.get_buffer(identity)

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
            # NORMAL: 语义吸附 + 溢出接力
            flush_event = self.adsorber.should_adsorb(buffer, block)
            if flush_event:
                self._handle_flush_event(identity, flush_event)
                buffer = self._buffer_manager.get_buffer(identity)
            else:
                flush_event = self.relay_controller.should_relay(buffer, block)
                if flush_event:
                    self._handle_flush_event(identity, flush_event)
                    buffer = self._buffer_manager.get_buffer(identity)

            # 添加 block 到 buffer
            self._buffer_manager.add_block_to_buffer(identity, block)

            # 更新话题核心
            new_kernel = self.adsorber.compute_new_topic_kernel(buffer, block)
            if new_kernel:
                self._buffer_manager.update_buffer_metadata(
                    identity, topic_kernel_vector=new_kernel
                )

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
        if self.on_flush_callback and blocks_to_process:
            try:
                messages = self._blocks_to_messages(blocks_to_process, identity)
                self.on_flush_callback(
                    messages, event.flush_reason,
                    write_focus=event.write_focus,
                    update_focus=event.update_focus,
                )
            except Exception as e:
                logger.error(f"Flush 回调失败: {e}")
        elif self.on_flush_callback and not blocks_to_process:
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
        builder = self._buffer_manager.get_builder(identity)

        # 如果 builder 有完成的 block，先添加它
        if builder.is_complete:
            completed_block = builder.build()
            self._buffer_manager.add_block_to_buffer(identity, completed_block)
            self._buffer_manager.reset_builder(identity)

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
        self._buffer_manager.reset_builder(identity)
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


__all__ = [
    "SemanticFlowPerceptionLayer",
]
