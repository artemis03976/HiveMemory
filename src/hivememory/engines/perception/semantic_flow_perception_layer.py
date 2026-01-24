"""
HiveMemory - 语义流感知层 (Semantic Flow Perception Layer)

职责:
    使用统一语义流架构的感知层实现。

特性:
    - LogicalBlock 作为处理单元
    - 语义吸附判定
    - Token 溢出检测与接力
    - 异步空闲超时监控
    - 多框架支持（LangChain、OpenAI、简单文本）

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
import threading
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING

from hivememory.core.models import FlushReason, Identity, StreamMessage, StreamMessageType
from hivememory.engines.perception.interfaces import (
    BasePerceptionLayer,
    StreamParser,
    SemanticAdsorber,
    RelayController,
)
from hivememory.engines.perception.models import (
    LogicalBlock,
    SemanticBuffer,
    BufferState,
)
from hivememory.engines.perception.stream_parser import UnifiedStreamParser
from hivememory.engines.perception.semantic_adsorber import SemanticBoundaryAdsorber
from hivememory.engines.perception.relay_controller import TokenOverflowRelayController

logger = logging.getLogger(__name__)


class SemanticFlowPerceptionLayer(BasePerceptionLayer):
    """
    语义流感知层

    使用统一的语义流架构：
        - LogicalBlock 作为处理单元
        - 语义吸附判定
        - Token 溢出检测与接力
        - 异步空闲超时监控

    职责：
        - 管理所有 SemanticBuffer（按 user_id:agent_id:session_id 组织）
        - 协调 Parser、Adsorber、Relay
        - 处理消息流并触发 Flush
        - 提供 Buffer 查询和管理接口

    Examples:
        >>> def on_flush(blocks, reason):
        ...     print(f"Flush: {reason}, Blocks: {len(blocks)}")
        >>>
        >>> perception = SemanticFlowPerceptionLayer(on_flush_callback=on_flush)
        >>> perception.start_idle_monitor()  # 启动异步空闲监控
        >>>
        >>> # 添加消息（使用 BasePerceptionLayer 接口）
        >>> perception.add_message("user", "hello", identity)
        >>>
        >>> perception.stop_idle_monitor()  # 停止监控
    """

    def __init__(
        self,
        parser: Optional[StreamParser] = None,
        adsorber: Optional[SemanticAdsorber] = None,
        relay_controller: Optional[RelayController] = None,
        on_flush_callback: Optional[
            Callable[[List[StreamMessage], FlushReason], None]
        ] = None,
        idle_timeout_seconds: int = 900,
        scan_interval_seconds: int = 30,
    ):
        """
        初始化语义流感知层

        Args:
            parser: 流式解析器（可选，使用默认）
            adsorber: 语义吸附器（可选，使用默认）
            relay_controller: 接力控制器（可选，使用默认）
            on_flush_callback: Flush 回调函数（统一接口）
                参数: (messages: List[StreamMessage], reason: FlushReason)
            idle_timeout_seconds: 空闲超时时间（秒），默认 900（15 分钟）
            scan_interval_seconds: 扫描间隔（秒），默认 30
        """
        super().__init__()

        self.parser = parser or UnifiedStreamParser()
        self.adsorber = adsorber or SemanticBoundaryAdsorber()
        self.relay_controller = relay_controller or TokenOverflowRelayController()
        self.on_flush_callback = on_flush_callback

        # 空闲超时监控器配置（由基类管理）
        self._idle_timeout_seconds = idle_timeout_seconds
        self._scan_interval_seconds = scan_interval_seconds

        # Buffer 池管理
        self._buffers: Dict[str, SemanticBuffer] = {}
        self._lock = threading.RLock()

        logger.info("SemanticFlowPerceptionLayer 初始化完成")

    # ========== BasePerceptionLayer 接口实现 ==========

    def add_message(
        self,
        role: str,
        content: str,
        identity: Identity,
        rewritten_query: Optional[str] = None,
        gateway_intent: Optional[str] = None,
        worth_saving: Optional[bool] = None,
    ) -> None:
        """
        添加消息到感知层（BasePerceptionLayer 统一接口）

        处理流程：
            1. 解析原始消息为 StreamMessage
            2. 获取或创建 SemanticBuffer
            3. 判断是否需要创建 LogicalBlock
            4. 添加消息到当前 Block
            5. 检查 Block 是否闭合
            6. 语义吸附判定（是否触发 Flush）

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            identity: 身份标识对象
            rewritten_query: Gateway 重写后的查询（可选）
            gateway_intent: Gateway 意图分类结果（可选）
            worth_saving: Gateway 价值判断（可选）

        Examples:
            >>> from hivememory.core.models import Identity
            >>> identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
            >>> perception.add_message("user", "hello", identity)
        """
        with self._lock:
            # 1. 解析消息
            try:
                raw_message = {"role": role, "content": content}
                stream_message = self.parser.parse_message(raw_message)
                # 更新身份信息
                stream_message.identity = Identity(
                    user_id=identity.user_id,
                    agent_id=identity.agent_id,
                    session_id=identity.session_id or ""
                )
                # 更新 Gateway 信息
                stream_message.rewritten_query = rewritten_query
                stream_message.gateway_intent = gateway_intent
                stream_message.worth_saving = worth_saving
            except Exception as e:
                logger.error(f"消息解析失败: {e}")
                return

            logger.debug(
                f"解析消息: {stream_message.message_type} - "
                f"{stream_message.content[:50]}..."
            )

            # 2. 获取或创建 Buffer
            buffer = self.get_buffer(identity)

            # 3. 判断是否需要创建新 Block
            if self.parser.should_create_new_block(stream_message):
                buffer.current_block = LogicalBlock()
                buffer.state = BufferState.PROCESSING
                logger.debug(f"创建新 LogicalBlock: {buffer.current_block.block_id}")

            # 4. 添加到当前 Block
            if buffer.current_block:
                buffer.current_block.add_stream_message(stream_message)

            # 5. 检查是否完成
            if buffer.current_block and buffer.current_block.is_complete:
                completed_block = buffer.current_block
                buffer.current_block = None
                buffer.state = BufferState.IDLE

                # 6. 多方面检测（Token溢出、语义漂移等），如有需要则先 Flush 旧 Buffer
                # 注意：此时新 Block 尚未加入 Buffer
                self._check_and_flush(buffer, completed_block)

                # 7. 将完整 Block 加入缓冲区（此时 Buffer 可能已被清空）
                buffer.add_block(completed_block)

                # 8. 更新话题核心
                try:
                    self.adsorber.update_topic_kernel(buffer, completed_block)
                except Exception as e:
                    logger.warning(f"更新话题核心失败: {e}")

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
        with self._lock:
            buffer_key = identity.buffer_key
            buffer = self.get_buffer(identity)
            if not buffer:
                logger.debug(f"Buffer 不存在: {buffer_key}")
                return []

            # 将当前 Block 加入（如果存在且完整）
            if buffer.current_block and buffer.current_block.is_complete:
                buffer.add_block(buffer.current_block)
                buffer.current_block = None

            if not buffer.blocks:
                return []

            blocks = self._flush(buffer, reason)
            # 转换为 StreamMessage
            messages = []
            for block in blocks:
                messages.extend(
                    block.to_stream_messages(
                        identity=Identity(
                            user_id=buffer.user_id,
                            agent_id=buffer.agent_id,
                            session_id=buffer.session_id,
                        )
                    )
                )
            return messages

    def get_buffer(
        self,
        identity: Identity,
    ) -> Optional[SemanticBuffer]:
        """
        获取指定 Buffer

        Args:
            identity: 身份标识对象

        Returns:
            SemanticBuffer: 缓冲区对象，不存在返回 None
        """
        key = identity.buffer_key

        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = SemanticBuffer(
                    user_id=identity.user_id,
                    agent_id=identity.agent_id,
                    session_id=identity.session_id or key.split(":")[-1],
                )
                logger.debug(f"创建新 Buffer: {key}")

            return self._buffers.get(key)

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

        Examples:
            >>> from hivememory.core.models import Identity
            >>> identity = Identity(user_id="user1", agent_id="agent1", session_id="sess1")
            >>> success = perception.clear_buffer(identity)
            >>> print(f"清理{'成功' if success else '失败'}")
        """
        with self._lock:
            buffer_key = identity.buffer_key
            buffer = self.get_buffer(identity)
            if buffer:
                # 先处理未完成的 Block
                buffer.clear()
                logger.info(f"清理 Buffer: {buffer_key}")
                return True
            else:
                logger.debug(f"Buffer 不存在，无需清理: {buffer_key}")
                return False

    def list_active_buffers(self) -> List[str]:
        """
        列出所有活跃的 Buffer

        Returns:
            List[str]: Buffer key 列表

        Examples:
            >>> buffers = perception.list_active_buffers()
            >>> print(f"当前有 {len(buffers)} 个活跃 Buffer")
        """
        with self._lock:
            return list(self._buffers.keys())

    def get_buffer_info(
        self,
        identity: Identity,
    ) -> Dict[str, Any]:
        """
        获取缓冲区信息（BasePerceptionLayer 统一接口）

        Args:
            identity: 身份标识对象

        Returns:
            Dict: 缓冲区信息字典
        """
        buffer = self.get_buffer(identity)
        if buffer:
            return {
                "exists": True,
                "mode": "semantic_flow",
                "buffer_id": buffer.buffer_id,
                "user_id": buffer.user_id,
                "agent_id": buffer.agent_id,
                "session_id": buffer.session_id,
                "block_count": len(buffer.blocks),
                "total_tokens": buffer.total_tokens,
                "state": buffer.state.value,
                "has_current_block": buffer.current_block is not None,
                "relay_summary": buffer.relay_summary,
            }
        else:
            return {
                "exists": False,
                "mode": "semantic_flow",
                "user_id": identity.user_id,
                "agent_id": identity.agent_id,
                "session_id": identity.session_id,
            }

    # ========== 内部方法 ==========

    def _check_and_flush(
        self,
        buffer: SemanticBuffer,
        new_block: LogicalBlock
    ) -> Optional[List[LogicalBlock]]:
        """
        检查是否需要 Flush（在新 Block 加入 Buffer 之前调用）

        检测顺序：
            1. 语义吸附判定（Adsorber：语义漂移）
            2. Token 溢出检测（RelayController）

        空闲超时检测由 BasePerceptionLayer.start_idle_monitor() 异步处理。

        如果需要 Flush，先清空旧 Buffer，新 Block 将作为新 Buffer 的第一个元素。

        Args:
            buffer: 当前语义缓冲区
            new_block: 即将加入的新 Block（尚未加入 buffer.blocks）

        Returns:
            Optional[List[LogicalBlock]]: Flush 的 Block 列表，如果未触发则返回 None
        """

        # 1. 语义吸附判定（语义漂移）
        should_adsorb, flush_reason = self.adsorber.should_adsorb(new_block, buffer)

        if not should_adsorb:
            # 语义吸附判定未通过，触发 Flush
            result = self._flush(buffer, flush_reason)
            # 重置话题核心，新 Block 将开启新话题
            self.adsorber.reset_topic_kernel(buffer)
            return result

        # 2. 通过语义吸附判定后，进行 Token 溢出检测
        if self.relay_controller.should_trigger_relay(buffer, new_block):
            # 触发 Token 溢出接力
            result = self._flush(buffer, FlushReason.TOKEN_OVERFLOW)
            # 重置话题核心，新 Block 将开启新话题，但保有上一次的话题摘要
            self.adsorber.reset_topic_kernel(buffer)
            return result

        # 默认吸附，不触发 Flush
        return None

    def _flush(
        self,
        buffer: SemanticBuffer,
        reason: FlushReason,
    ) -> List[LogicalBlock]:
        """
        执行 Flush，清空 Buffer
        内部将 LogicalBlock 转换为 StreamMessage 后调用回调函数

        Args:
            buffer: 当前语义缓冲区
            reason: Flush 原因

        Returns:
            List[LogicalBlock]: Flush 的 Block 列表
        """
        blocks_to_process = buffer.blocks.copy()

        logger.info(
            f"感知层触发 Flush，Buffer ID={buffer.buffer_id}, "
            f"原因: {reason.value}, "
            f"Block 数量: {len(blocks_to_process)}"
        )

        # 清空 Buffer
        buffer.blocks.clear()
        buffer.total_tokens = 0

        # 如果是 Token 溢出，生成接力摘要
        if reason == FlushReason.TOKEN_OVERFLOW:
            try:
                summary = self.relay_controller.generate_summary(blocks_to_process)
                buffer.relay_summary = summary
                logger.debug(f"接力摘要: {summary}")
            except Exception as e:
                logger.warning(f"生成接力摘要失败: {e}")

        # 调用回调（转换为 StreamMessage）
        if self.on_flush_callback:
            try:
                # 转换 LogicalBlock -> StreamMessage
                messages = []
                for block in blocks_to_process:
                    messages.extend(
                        block.to_stream_messages(
                            identity=Identity(
                                user_id=buffer.user_id,
                                agent_id=buffer.agent_id,
                                session_id=buffer.session_id,
                            )
                        )
                    )
                self.on_flush_callback(messages, reason)
            except Exception as e:
                logger.error(f"Flush 回调执行失败: {e}")

        return blocks_to_process


__all__ = [
    "SemanticFlowPerceptionLayer",
]
