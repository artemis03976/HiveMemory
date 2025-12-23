"""
Patchouli - Librarian Agent (图书管理员智能体)

这是模块化重构后的高层 API，保持向后兼容。

职责:
    1. 监听对话流
    2. 价值评估 (过滤无用信息)
    3. 提取结构化记忆原子
    4. 存储到向量数据库

内部实现:
    使用 hivememory.generation 模块的编排器和缓冲器。

参考: PROJECT.md 4.0 - 4.2 节

作者: HiveMemory Team
版本: 0.2.0 (模块化重构)
"""

import logging
from typing import List, Optional

from hivememory.core.models import MemoryAtom, ConversationMessage
from hivememory.memory.storage import QdrantMemoryStore
from hivememory.generation.orchestrator import MemoryOrchestrator

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
        原始实现已备份为 patchouli_legacy.py。
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

        # 初始化编排器（核心逻辑）
        self.orchestrator = MemoryOrchestrator(storage=self.storage)

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
