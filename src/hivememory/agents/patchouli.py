"""
Patchouli - Librarian Agent (图书管理员智能体)

职责:
    1. 监听对话流
    2. 价值评估 (过滤无用信息)
    3. 提取结构化记忆原子
    4. 存储到向量数据库
    5. 管理感知层（统一接口）
    6. 记忆检索（Stage 2 - 待实现）
    7. 生命周期管理（Stage 3 - 待实现）

架构:
    - SimplePerceptionLayer: 简单缓冲策略（三重触发）
    - SemanticFlowPerceptionLayer: 语义流策略（统一语义流）

参考: PROJECT.md 4.0 - 4.2 节

作者: HiveMemory Team
版本: 0.4.0 (感知层重构)
"""

import logging
from typing import List, Optional, Dict, Any

from hivememory.core.models import MemoryAtom, ConversationMessage, FlushReason
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.generation.orchestrator import MemoryOrchestrator
from hivememory.perception.interfaces import BasePerceptionLayer

logger = logging.getLogger(__name__)


class PatchouliAgent:
    """
    Patchouli - HiveMemory 的图书管理员

    性格特征 (Persona):
        - 极度挑剔: 只记录有价值的信息,拒绝噪音
        - 结构化强迫症: 致力于将混乱的对话转化为规范的JSON
        - 客观中立: 以第三人称视角记录事实

    使用示例:
        >>> from hivememory.memory.storage import QdrantMemoryStore
        >>> storage = QdrantMemoryStore()
        >>>
        >>> # 使用简单感知层
        >>> patchouli = PatchouliAgent(storage=storage)
        >>>
        >>> # 使用语义流感知层（默认）
        >>> patchouli = PatchouliAgent(storage=storage, enable_semantic_flow=True)
        >>>
        >>> # 添加消息
        >>> patchouli.add_message("user", "帮我写贪吃蛇游戏", "user123", "agent1")
        >>> patchouli.add_message("assistant", "好的，我来实现...", "user123", "agent1")
    """

    def __init__(
        self,
        storage: Optional[QdrantMemoryStore] = None,
        enable_semantic_flow: bool = True,
    ):
        """
        初始化 Patchouli Agent

        Args:
            storage: Qdrant 存储实例（可选，自动创建）
            enable_semantic_flow: 是否使用语义流感知层（默认 True）

        Examples:
            >>> # 使用简单感知层
            >>> patchouli = PatchouliAgent()
            >>>
            >>> # 使用语义流感知层（默认）
            >>> patchouli = PatchouliAgent(enable_semantic_flow=True)
        """
        self.storage = storage or QdrantMemoryStore()

        # 感知层配置
        self.enable_semantic_flow = enable_semantic_flow
        self.perception_layer: Optional[BasePerceptionLayer] = None

        # 初始化感知层
        self._init_perception_layer()

        # 初始化记忆生成编排器（Stage 1: 记忆生成核心逻辑）
        self.orchestrator = MemoryOrchestrator(storage=self.storage)

        logger.info(
            f"Patchouli Agent 初始化完成 "
            f"(perception_layer_mode={'semantic_flow' if enable_semantic_flow else 'simple'})"
        )
    
    def _init_perception_layer(self) -> None:
        """
        初始化感知层

        架构模式:
            - SimplePerceptionLayer: 简单缓冲策略（三重触发）
            - SemanticFlowPerceptionLayer: 语义流策略（统一语义流）
        
        """
        try:
            if self.enable_semantic_flow:
                from hivememory.perception import SemanticFlowPerceptionLayer

                self.perception_layer = SemanticFlowPerceptionLayer(
                    on_flush_callback=self._on_perception_flush
                )
                logger.debug("语义流感知层已初始化")

            else:
                from hivememory.perception import SimplePerceptionLayer

                self.perception_layer = SimplePerceptionLayer(
                    on_flush_callback=self._on_perception_flush
                )
                logger.debug("简单感知层已初始化")

        except ImportError as e:
            logger.error(f"无法导入感知层: {e}")
            raise

    # ========== 感知层 API ==========

    def add_message(
        self,
        role: str,
        content: str,
        user_id: str,
        agent_id: str = "default",
        session_id: Optional[str] = None,
    ) -> None:
        """
        添加消息到感知层

        Args:
            role: 角色 (user/assistant/system)
            content: 消息内容
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID（可选）

        Examples:
            >>> patchouli.add_message("user", "你好", "user123", "chatbot")
            >>> patchouli.add_message("assistant", "你好！有什么可以帮你的吗？", "user123", "chatbot")
        """
        if not self.perception_layer:
            raise RuntimeError("感知层未初始化")

        self.perception_layer.add_message(
            role=role,
            content=content,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id or f"{user_id}_{agent_id}",
        )

        logger.debug(f"向感知层添加消息: {role} - {content[:50]}...")

    def flush_perception(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> None:
        """
        手动触发感知层 Flush

        Args:
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Examples:
            >>> patchouli.flush_perception("user123", "chatbot", "session_456")
        """
        if not self.perception_layer:
            raise RuntimeError("感知层未初始化")

        self.perception_layer.flush_buffer(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            reason=FlushReason.MANUAL,
        )

    def _on_perception_flush(
        self,
        messages: List[ConversationMessage],
        reason: FlushReason,
    ) -> None:
        """
        感知层 Flush 回调（统一接口）
        将感知层生成的消息传递给编排器处理

        Args:
            messages: ConversationMessage 列表
            reason: Flush 原因
        """
        try:
            # 从消息中提取上下文
            if not messages:
                logger.warning("空消息列表，跳过处理")
                return

            # TODO: 获取正确的 user_id 和 agent_id
            first_msg = messages[0]
            user_id = getattr(first_msg, "user_id", "unknown")
            agent_id = "perception_layer"

            logger.info(f"Patchouli 开始处理 {len(messages)} 条消息...")

            # 调用编排器处理
            memories = self.orchestrator.process(
                messages=messages,
            )
            
            logger.info(f"Patchouli 处理完成")
            if memories:
                logger.info(f"✓ 成功提取 {len(memories)} 条记忆")
            else:
                logger.info("未提取到记忆（对话可能无价值或被过滤）")   

        except Exception as e:
            logger.error(f"感知层 Flush 处理失败: {e}", exc_info=True)

    def get_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> Optional[Any]:
        """
        获取缓冲区对象

        Args:
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            Buffer 实例（SimpleBuffer 或 SemanticBuffer）
        """
        if not self.perception_layer:
            return None

        return self.perception_layer.get_buffer(user_id, agent_id, session_id)

    def clear_buffer(self, user_id: str, agent_id: str, session_id: str) -> bool:
        """
        清理指定的 Buffer

        Args:
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            bool: 是否成功清理

        Examples:
            >>> patchouli.clear_buffer("user123", "chatbot", "session_456")
            True
        """
        if not self.perception_layer:
            return False

        return self.perception_layer.clear_buffer(user_id, agent_id, session_id)

    def get_buffer_info(self, user_id: str, agent_id: str, session_id: str) -> Dict[str, Any]:
        """
        获取 Buffer 信息（统一接口）

        Args:
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            Dict: Buffer 信息字典

        Examples:
            >>> info = patchouli.get_buffer_info("user123", "chatbot", "session_456")
            >>> print(f"消息数量: {info.get('block_count', info.get('message_count', 0))}")
        """
        if not self.perception_layer:
            return {"exists": False, "mode": "none"}

        return self.perception_layer.get_buffer_info(user_id, agent_id, session_id)

    def list_active_buffers(self) -> List[str]:
        """
        列出所有活跃的 Buffer

        Returns:
            List[str]: Buffer key 列表

        Examples:
            >>> buffers = patchouli.list_active_buffers()
            >>> print(f"当前有 {len(buffers)} 个活跃 Buffer")
        """
        if not self.perception_layer:
            return []

        return self.perception_layer.list_active_buffers()


# 便捷函数
def create_patchouli_agent(
    storage: Optional[QdrantMemoryStore] = None,
    enable_semantic_flow: bool = True,
) -> PatchouliAgent:
    """
    创建 Patchouli Agent 实例

    Args:
        storage: Qdrant 存储实例（可选）
        enable_semantic_flow: 是否使用语义流感知层（默认 True）

    Returns:
        PatchouliAgent: Agent 实例

    Examples:
        >>> # 使用简单感知层（默认）
        >>> patchouli = create_patchouli_agent()
        >>>
        >>> # 使用语义流感知层
        >>> patchouli = create_patchouli_agent(enable_semantic_flow=True)
    """
    return PatchouliAgent(
        storage=storage,
        enable_semantic_flow=enable_semantic_flow,
    )


__all__ = [
    "PatchouliAgent",
    "create_patchouli_agent",
]
