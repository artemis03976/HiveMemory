"""
HiveMemory - 语义流感知层 (Semantic Flow Perception Layer)

职责:
    使用统一语义流架构的感知层实现。

特性:
    - LogicalBlock 作为处理单元
    - 语义吸附判定
    - Token 溢出检测与接力
    - 多框架支持（LangChain、OpenAI、简单文本）

参考: PROJECT.md 2.3.1 节

作者: HiveMemory Team
版本: 1.0.0
"""

import logging
import threading
from typing import Dict, List, Optional, Any, Callable

from hivememory.core.models import ConversationMessage, FlushReason
from hivememory.perception.interfaces import (
    BasePerceptionLayer,
    StreamParser,
    SemanticAdsorber,
    RelayController,
)
from hivememory.perception.models import (
    StreamMessage,
    LogicalBlock,
    SemanticBuffer,
    BufferState,
)
from hivememory.perception.stream_parser import UnifiedStreamParser
from hivememory.perception.semantic_adsorber import SemanticBoundaryAdsorber
from hivememory.perception.relay_controller import TokenOverflowRelayController

logger = logging.getLogger(__name__)


class SemanticFlowPerceptionLayer(BasePerceptionLayer):
    """
    语义流感知层

    使用统一的语义流架构：
        - LogicalBlock 作为处理单元
        - 语义吸附判定
        - Token 溢出检测与接力

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
        >>>
        >>> # 添加消息（使用 BasePerceptionLayer 接口）
        >>> perception.add_message("user", "hello", "user1", "agent1", "sess1")
    """

    def __init__(
        self,
        parser: Optional[StreamParser] = None,
        adsorber: Optional[SemanticAdsorber] = None,
        relay_controller: Optional[RelayController] = None,
        on_flush_callback: Optional[
            Callable[[List[ConversationMessage], FlushReason], None]
        ] = None,
    ):
        """
        初始化语义流感知层

        Args:
            parser: 流式解析器（可选，使用默认）
            adsorber: 语义吸附器（可选，使用默认）
            relay_controller: 接力控制器（可选，使用默认）
            on_flush_callback: Flush 回调函数（统一接口）
                参数: (messages: List[ConversationMessage], reason: FlushReason)
        """

        self.parser = parser or UnifiedStreamParser()
        self.adsorber = adsorber or SemanticBoundaryAdsorber()
        self.relay_controller = relay_controller or TokenOverflowRelayController()
        self.on_flush_callback = on_flush_callback

        # Buffer 池管理
        self._buffers: Dict[str, SemanticBuffer] = {}
        self._lock = threading.RLock()

        logger.info("SemanticFlowPerceptionLayer 初始化完成")

    def _get_buffer_key(self, user_id: str, agent_id: str, session_id: str) -> str:
        """生成 Buffer 唯一键"""
        return f"{user_id}:{agent_id}:{session_id}"

    # ========== BasePerceptionLayer 接口实现 ==========

    def add_message(
        self,
        role: str,
        content: str,
        user_id: str,
        agent_id: str,
        session_id: str,
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
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Examples:
            >>> perception.add_message("user", "hello", "user1", "agent1", "sess1")
        """
        with self._lock:
            # 1. 解析消息
            try:
                raw_message = {"role": role, "content": content}
                stream_message = self.parser.parse_message(raw_message)
            except Exception as e:
                logger.error(f"消息解析失败: {e}")
                return

            logger.debug(
                f"解析消息: {stream_message.message_type} - "
                f"{stream_message.content[:50]}..."
            )

            # 2. 获取或创建 Buffer
            buffer = self.get_buffer(user_id, agent_id, session_id)

            # 3. 判断是否需要创建新 Block
            if self.parser.should_create_new_block(stream_message):
                # 保存上一个 Block（如果存在且完整）
                if buffer.current_block and buffer.current_block.is_complete:
                    buffer.add_block(buffer.current_block)

                # 创建新 Block
                buffer.current_block = LogicalBlock()
                buffer.state = BufferState.PROCESSING
                logger.debug(f"创建新 LogicalBlock: {buffer.current_block.block_id}")

            # 4. 添加到当前 Block
            if buffer.current_block:
                buffer.current_block.add_stream_message(stream_message)

            # 5. 检查是否完成
            if buffer.current_block and buffer.current_block.is_complete:
                # 将完整 Block 加入缓冲区
                buffer.add_block(buffer.current_block)
                buffer.current_block = None
                buffer.state = BufferState.IDLE

                # 6. 语义吸附判定
                self._check_and_flush(buffer)

    def flush_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
        reason: FlushReason = FlushReason.MANUAL,
    ) -> None:
        """
        手动刷新 Buffer

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID
            reason: 刷新原因
        """
        with self._lock:
            buffer = self.get_buffer(user_id, agent_id, session_id)
            if not buffer:
                logger.debug(f"Buffer 不存在: {user_id}:{agent_id}:{session_id}")
                return

            # 将当前 Block 加入（如果存在且完整）
            if buffer.current_block and buffer.current_block.is_complete:
                buffer.add_block(buffer.current_block)
                buffer.current_block = None

            if not buffer.blocks:
                return

            blocks = self._flush(buffer, reason)

    def get_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> Optional[SemanticBuffer]:
        """
        获取指定 Buffer

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Returns:
            SemanticBuffer: 缓冲区对象，不存在返回 None
        """
        key = self._get_buffer_key(user_id, agent_id, session_id)
        
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = SemanticBuffer(
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                )
                logger.debug(f"创建新 Buffer: {key}")
                
            return self._buffers.get(key)

    def clear_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str
    ) -> bool:
        """
        清理指定的 Buffer

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Returns:
            bool: 是否成功清理

        Examples:
            >>> success = perception.clear_buffer("user1", "agent1", "sess1")
            >>> print(f"清理{'成功' if success else '失败'}")
        """
        with self._lock:
            buffer = self.get_buffer(user_id, agent_id, session_id)
            if buffer:
                # 先处理未完成的 Block
                buffer.clear()
                logger.info(f"清理 Buffer: {user_id}:{agent_id}:{session_id}")
                return True
            else:
                logger.debug(f"Buffer 不存在，无需清理: {user_id}:{agent_id}:{session_id}")
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
        user_id: str,
        agent_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        获取缓冲区信息（BasePerceptionLayer 统一接口）

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Returns:
            Dict: 缓冲区信息字典
        """
        buffer = self.get_buffer(user_id, agent_id, session_id)
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
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
            }

    # ========== 内部方法 ==========

    def _check_and_flush(self, buffer: SemanticBuffer) -> Optional[List[LogicalBlock]]:
        """检查并触发 Flush"""
        if not buffer.blocks:
            return None

        latest_block = buffer.blocks[-1]

        # 检查是否需要 Flush
        should_adsorb, flush_reason = self.adsorber.should_adsorb(latest_block, buffer)

        if not should_adsorb:
            # 触发 Flush
            return self._flush(buffer, flush_reason)

        # 更新话题核心
        try:
            self.adsorber.update_topic_kernel(buffer, latest_block)
        except Exception as e:
            logger.warning(f"更新话题核心失败: {e}")

        return None

    def _flush(
        self,
        buffer: SemanticBuffer,
        reason: FlushReason,
    ) -> List[LogicalBlock]:
        """
        执行 Flush

        内部将 LogicalBlock 转换为 ConversationMessage 后调用回调函数
        """
        blocks_to_process = buffer.blocks.copy()

        logger.info(
            f"感知层触发 Flush: {buffer.buffer_id}, "
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

        # 调用回调（转换为 ConversationMessage）
        if self.on_flush_callback:
            try:
                # 转换 LogicalBlock -> ConversationMessage
                messages = []
                for block in blocks_to_process:
                    messages.extend(
                        block.to_conversation_messages(
                            user_id=buffer.user_id,
                            agent_id=buffer.agent_id,
                            session_id=buffer.session_id,
                        )
                    )
                self.on_flush_callback(messages, reason)
            except Exception as e:
                logger.error(f"Flush 回调执行失败: {e}")

        return blocks_to_process


__all__ = [
    "SemanticFlowPerceptionLayer",
]
