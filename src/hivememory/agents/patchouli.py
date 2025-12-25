"""
Patchouli - Librarian Agent (图书管理员智能体)

这是模块化重构后的高层 API，保持向后兼容。

职责:
    1. 监听对话流
    2. 价值评估 (过滤无用信息)
    3. 提取结构化记忆原子
    4. 存储到向量数据库
    5. 管理对话 Buffer（Stage 1）
    6. 记忆检索（Stage 2 - 待实现）
    7. 生命周期管理（Stage 3 - 待实现）

内部实现:
    使用 hivememory.generation 模块的编排器和缓冲器。

参考: PROJECT.md 4.0 - 4.2 节

作者: HiveMemory Team
版本: 0.3.0 (添加 Buffer 管理)
"""

import logging
import threading
from typing import List, Optional, Dict, Any

from hivememory.core.models import MemoryAtom, ConversationMessage
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.generation.orchestrator import MemoryOrchestrator
from hivememory.generation.buffer import ConversationBuffer

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
        >>> patchouli = PatchouliAgent(storage=storage)
        >>>
        >>> memories = patchouli.process_transcript(
        ...     messages=[
        ...         ConversationMessage(role="user", content="写快排"),
        ...         ConversationMessage(role="assistant", content="代码...")
        ...     ],
        ...     user_id="user123",
        ...     agent_id="agent456"
        ... )
        >>> print(f"提取了 {len(memories)} 条记忆")

    注意:
        该类现在是对 hivememory.generation.MemoryOrchestrator 的薄封装。
    """

    def __init__(self, storage: Optional[QdrantMemoryStore] = None):
        """
        初始化 Patchouli Agent

        Args:
            storage: Qdrant 存储实例（可选，自动创建）

        Examples:
            >>> # 使用默认存储
            >>> patchouli = PatchouliAgent()
            >>>
            >>> # 使用自定义存储
            >>> custom_storage = QdrantMemoryStore()
            >>> patchouli = PatchouliAgent(storage=custom_storage)
        """
        self.storage = storage or QdrantMemoryStore()

        # 初始化编排器（Stage 1: 记忆生成核心逻辑）
        self.orchestrator = MemoryOrchestrator(storage=self.storage)

        # Buffer 池管理（全局单例，避免重复创建）
        self._buffers: Dict[str, ConversationBuffer] = {}
        self._buffer_lock = threading.Lock()

        logger.info("Patchouli Agent 初始化完成")

    def process_transcript(
        self,
        messages: List[ConversationMessage],
        user_id: str,
        agent_id: str = "default_agent",
    ) -> List[MemoryAtom]:
        """
        处理对话片段,提取记忆原子

        该方法保持与原版本完全兼容的接口。

        工作流程:
            1. 价值评估 (Gating) → 过滤无价值对话
            2. LLM 提取 → 生成结构化草稿
            3. 查重检测 → CREATE/UPDATE/TOUCH
            4. 记忆构建 → MemoryAtom
            5. 持久化 → Qdrant

        Args:
            messages: 对话消息列表
            user_id: 用户ID
            agent_id: Agent ID（默认 "default_agent"）

        Returns:
            List[MemoryAtom]: 提取的记忆原子列表

        Examples:
            >>> messages = [
            ...     ConversationMessage(
            ...         role="user",
            ...         content="帮我写一个Python快排算法",
            ...         user_id="user123",
            ...         session_id="sess456"
            ...     ),
            ...     ConversationMessage(
            ...         role="assistant",
            ...         content="好的，这是快排实现：\n```python\n...\n```",
            ...         user_id="user123",
            ...         session_id="sess456"
            ...     )
            ... ]
            >>>
            >>> memories = patchouli.process_transcript(
            ...     messages=messages,
            ...     user_id="user123",
            ...     agent_id="worker_agent"
            ... )
            >>>
            >>> for memory in memories:
            ...     print(f"✓ {memory.index.title}")
        """
        if not messages:
            logger.warning("空消息列表，跳过处理")
            return []

        logger.info(f"Patchouli 开始处理 {len(messages)} 条消息...")

        # 调用编排器处理
        memories = self.orchestrator.process(
            messages=messages,
            user_id=user_id,
            agent_id=agent_id,
        )

        if memories:
            logger.info(f"✓ 成功提取 {len(memories)} 条记忆")
        else:
            logger.info("未提取到记忆（对话可能无价值或被过滤）")

        return memories

    def get_or_create_buffer(
        self,
        user_id: str,
        agent_id: str,
        session_id: str,
        **buffer_kwargs
    ) -> ConversationBuffer:
        """
        获取或创建 ConversationBuffer（全局单例复用）

        该方法确保同一 (user_id, agent_id, session_id) 组合只创建一个 Buffer 实例，
        避免重复创建导致的性能浪费。

        Args:
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID
            **buffer_kwargs: 额外的 Buffer 配置参数（如 trigger_manager, on_flush_callback）

        Returns:
            ConversationBuffer: Buffer 实例（复用已有或创建新的）

        Examples:
            >>> patchouli = PatchouliAgent()
            >>> buffer = patchouli.get_or_create_buffer(
            ...     user_id="user123",
            ...     agent_id="chatbot",
            ...     session_id="session_456"
            ... )
            >>> buffer.add_message("user", "你好")
            >>> buffer.add_message("assistant", "你好！有什么可以帮你的？")
        """
        buffer_key = f"{user_id}:{agent_id}:{session_id}"

        with self._buffer_lock:
            if buffer_key not in self._buffers:
                logger.info(f"创建新 Buffer: {buffer_key}")
                self._buffers[buffer_key] = ConversationBuffer(
                    orchestrator=self.orchestrator,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    **buffer_kwargs
                )
            else:
                logger.debug(f"复用已有 Buffer: {buffer_key}")

            return self._buffers[buffer_key]

    def clear_buffer(self, user_id: str, agent_id: str, session_id: str) -> bool:
        """
        清理指定的 Buffer

        Args:
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            bool: 是否成功清理（True=存在并已清理，False=不存在）

        Examples:
            >>> patchouli.clear_buffer("user123", "chatbot", "session_456")
            True
        """
        buffer_key = f"{user_id}:{agent_id}:{session_id}"

        with self._buffer_lock:
            if buffer_key in self._buffers:
                del self._buffers[buffer_key]
                logger.info(f"清理 Buffer: {buffer_key}")
                return True
            else:
                logger.debug(f"Buffer 不存在，无需清理: {buffer_key}")
                return False

    def flush_buffer(self, user_id: str, agent_id: str, session_id: str) -> List[MemoryAtom]:
        """
        手动触发 Buffer 刷新（提取记忆）

        Args:
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            List[MemoryAtom]: 提取的记忆列表

        Examples:
            >>> # 手动触发记忆提取
            >>> memories = patchouli.flush_buffer("user123", "chatbot", "session_456")
            >>> print(f"提取了 {len(memories)} 条记忆")
        """
        buffer_key = f"{user_id}:{agent_id}:{session_id}"

        with self._buffer_lock:
            if buffer_key in self._buffers:
                buffer = self._buffers[buffer_key]
                logger.info(f"手动触发 Buffer 刷新: {buffer_key}")
                return buffer.flush()
            else:
                logger.warning(f"Buffer 不存在，无法刷新: {buffer_key}")
                return []

    def get_buffer_info(self, user_id: str, agent_id: str, session_id: str) -> Dict[str, Any]:
        """
        获取 Buffer 信息

        Args:
            user_id: 用户 ID
            agent_id: Agent ID
            session_id: 会话 ID

        Returns:
            Dict: Buffer 信息字典

        Examples:
            >>> info = patchouli.get_buffer_info("user123", "chatbot", "session_456")
            >>> print(f"消息数量: {info['message_count']}")
        """
        buffer_key = f"{user_id}:{agent_id}:{session_id}"

        with self._buffer_lock:
            if buffer_key in self._buffers:
                buffer = self._buffers[buffer_key]
                return {
                    "exists": True,
                    "message_count": len(buffer.messages),
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "session_id": session_id
                }
            else:
                return {
                    "exists": False,
                    "message_count": 0,
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "session_id": session_id
                }

    def list_active_buffers(self) -> List[str]:
        """
        列出所有活跃的 Buffer

        Returns:
            List[str]: Buffer key 列表

        Examples:
            >>> buffers = patchouli.list_active_buffers()
            >>> print(f"当前有 {len(buffers)} 个活跃 Buffer")
        """
        with self._buffer_lock:
            return list(self._buffers.keys())


# 便捷函数
def create_patchouli_agent(storage: Optional[QdrantMemoryStore] = None) -> PatchouliAgent:
    """
    创建 Patchouli Agent 实例

    Args:
        storage: Qdrant 存储实例（可选）

    Returns:
        PatchouliAgent: Agent 实例

    Examples:
        >>> patchouli = create_patchouli_agent()
    """
    return PatchouliAgent(storage=storage)


__all__ = [
    "PatchouliAgent",
    "create_patchouli_agent",
]
