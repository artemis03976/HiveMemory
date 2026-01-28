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
from hivememory.engines.perception.stream_parser import UnifiedStreamParser
from hivememory.engines.perception.relay_controller import RelayController
from hivememory.engines.perception.semantic_adsorber import SemanticBoundaryAdsorber
from hivememory.engines.perception.models import (
    BufferState,
    FlushEvent,
    FlushReason,
    LogicalBlock,
    SemanticBuffer,
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
        ...     parser=parser,
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
        parser: UnifiedStreamParser,
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
            parser: 流式解析器
            adsorber: 语义吸附器（无状态服务）
            relay_controller: 接力控制器（无状态服务）
            on_flush_callback: Flush 回调函数
                参数: (messages: List[StreamMessage], reason: FlushReason)
        """
        super().__init__()

        self.config = config
        self.parser = parser
        self.adsorber = adsorber
        self.relay_controller = relay_controller
        self.on_flush_callback = on_flush_callback

        # 空闲超时监控器配置（由基类管理）
        self._idle_timeout_seconds = config.idle_timeout_seconds
        self._scan_interval_seconds = config.scan_interval_seconds

        # BufferManager 作为纯状态容器
        self._buffer_manager = SemanticBufferManager()

        logger.info("SemanticFlowPerceptionLayer 初始化完成")

    def perceive(
        self,
        role: str,
        content: str,
        identity: Identity,
        rewritten_query: Optional[str] = None,
        gateway_intent: Optional[str] = None,
        worth_saving: Optional[bool] = None,
    ) -> None:
        """
        添加消息到感知层

        编排流程：
            1. 解析消息
            2. 获取 buffer 和 builder
            3. 检查是否需要开始新 block
            4. 添加消息到 builder
            5. 如果 block 未完成，返回
            6. Block 完成，构建它
            7. 吸附判断：检查语义漂移
            8. 容量判断：检查 Token 溢出
            9. 将 block 添加到 buffer
            10. 更新话题核心
            11. 重置 builder 和状态

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            identity: 身份标识对象
            rewritten_query: Gateway 重写后的查询（可选）
            gateway_intent: Gateway 意图分类结果（可选）
            worth_saving: Gateway 价值判断（可选）
        """
        # 1. 解析消息
        try:
            raw_message = {"role": role, "content": content}
            stream_message = self.parser.parse_message(raw_message)
            stream_message.identity = identity
        except Exception as e:
            logger.error(f"消息解析失败: {e}")
            return

        # 2. 获取 buffer 和 builder
        buffer = self._buffer_manager.get_buffer(identity)
        builder = self._buffer_manager.get_builder(identity)

        # 3. 检查是否需要开始新 block
        if builder.should_create_new_block(stream_message):
            builder.start(
                rewritten_query=rewritten_query,
                gateway_intent=gateway_intent,
                worth_saving=worth_saving,
            )
            self._buffer_manager.update_buffer_metadata(
                identity, state=BufferState.PROCESSING
            )
            logger.debug(f"为 {identity.buffer_key} 开始新 block")

        # 4. 添加消息到 builder
        builder.add_message(stream_message)

        # 5. 如果 block 未完成，返回
        if not builder.is_complete:
            return

        # 6. Block 完成，构建它
        completed_block = builder.build()

        # 7. 吸附判断：检查语义漂移
        flush_event = self.adsorber.should_adsorb(buffer, completed_block)
        if flush_event:
            self._handle_flush_event(identity, flush_event)
            # flush 后刷新 buffer 引用
            buffer = self._buffer_manager.get_buffer(identity)
        # 8. 容量判断：检查 Token 溢出（仅在未发生语义漂移 Flush 时）
        else:
            flush_event = self.relay_controller.should_relay(buffer, completed_block)
            if flush_event:
                self._handle_flush_event(identity, flush_event)
                buffer = self._buffer_manager.get_buffer(identity)

        # 9. 将 block 添加到 buffer
        self._buffer_manager.add_block_to_buffer(identity, completed_block)

        # 10. 更新话题核心
        new_kernel = self.adsorber.compute_new_topic_kernel(buffer, completed_block)
        if new_kernel:
            self._buffer_manager.update_buffer_metadata(
                identity, topic_kernel_vector=new_kernel
            )

        # 11. 重置 builder 和状态
        self._buffer_manager.reset_builder(identity)
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

        # 4. 调用回调
        if self.on_flush_callback and event.blocks_to_flush:
            try:
                messages = self._blocks_to_messages(event.blocks_to_flush, identity)
                self.on_flush_callback(messages, event.flush_reason)
            except Exception as e:
                logger.error(f"Flush 回调失败: {e}")

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
