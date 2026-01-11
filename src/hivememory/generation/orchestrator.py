"""
HiveMemory - 记忆生成编排器 (Memory Generation Orchestrator)

职责:
    协调所有组件，执行完整的记忆生成流程。

工作流程:
    Step 1: 价值评估 (Gating) → Pass/Drop
    Step 2: LLM 提取 → ExtractedMemoryDraft
    Step 3: 查重检测 → CREATE/UPDATE/TOUCH
    Step 4: 记忆原子构建 → MemoryAtom
    Step 5: 持久化 → Qdrant

作者: HiveMemory Team
版本: 0.1.0
"""

import logging
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from hivememory.core.config import MemoryGenerationConfig

from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType
from hivememory.generation.models import ConversationMessage, ExtractedMemoryDraft
from hivememory.generation.interfaces import (
    ValueGater,
    MemoryExtractor,
    Deduplicator,
    DuplicateDecision,
)
from hivememory.generation.gating import create_default_gater
from hivememory.generation.extractor import create_default_extractor
from hivememory.generation.deduplicator import create_default_deduplicator

logger = logging.getLogger(__name__)


class MemoryGenerationOrchestrator:
    """
    记忆生成编排器

    协调价值评估、LLM 提取、查重、存储等所有步骤。

    Examples:
        >>> from hivememory.memory.storage import QdrantMemoryStore
        >>> storage = QdrantMemoryStore()
        >>> orchestrator = MemoryGenerationOrchestrator(storage=storage)
        >>> memories = orchestrator.process(messages, user_id="u1", agent_id="a1")
    """

    def __init__(
        self,
        storage,  # QdrantMemoryStore
        gater: Optional[ValueGater] = None,
        extractor: Optional[MemoryExtractor] = None,
        deduplicator: Optional[Deduplicator] = None,
        config: Optional["MemoryGenerationConfig"] = None,
    ):
        """
        初始化编排器

        Args:
            storage: 向量存储实例
            gater: 价值评估器（可选，使用配置）
            extractor: 记忆提取器（可选，使用配置）
            deduplicator: 查重器（可选，使用配置）
            config: 记忆配置（可选，用于创建组件）

        Examples:
            >>> # 使用默认配置
            >>> orchestrator = MemoryGenerationOrchestrator(storage=storage)
            >>>
            >>> # 使用自定义配置
            >>> from hivememory.core.config import MemoryGenerationConfig
            >>> config = MemoryGenerationConfig()
            >>> orchestrator = MemoryGenerationOrchestrator(storage=storage, config=config)
        """
        self.storage = storage

        # 使用传入的配置或加载默认配置
        if config is None:
            from hivememory.core.config import MemoryGenerationConfig
            config = MemoryGenerationConfig()

        # 如果组件未提供，使用配置创建
        if gater is None:
            self.gater = create_default_gater(config.gater)
        else:
            self.gater = gater

        if extractor is None:
            self.extractor = create_default_extractor(config.extractor)
        else:
            self.extractor = extractor

        if deduplicator is None:
            self.deduplicator = create_default_deduplicator(
                storage, config.deduplicator
            )
        else:
            self.deduplicator = deduplicator

        logger.info("MemoryGenerationOrchestrator 初始化完成")

    def process(
        self,
        messages: List[ConversationMessage],
    ) -> List[MemoryAtom]:
        """
        处理对话片段，提取记忆原子

        完整流程:
            1. 价值评估 → 过滤无价值对话
            2. LLM 提取 → 生成结构化草稿
            3. 查重检测 → 判断 CREATE/UPDATE/TOUCH
            4. 记忆构建 → MemoryAtom
            5. 持久化 → Qdrant

        Args:
            messages: 对话消息列表

        Returns:
            List[MemoryAtom]: 提取的记忆原子列表

        Examples:
            >>> memories = orchestrator.process(
            ...     messages=[
            ...         ConversationMessage(role="user", content="写快排"),
            ...         ConversationMessage(role="assistant", content="代码...")
            ...     ],
            ... )
            >>> len(memories)
            1
        """
        if not messages:
            logger.debug("空消息列表，跳过处理")
            return []
        
        user_id = messages[0].user_id
        agent_id = messages[0].agent_id
        session_id = messages[0].session_id

        logger.info(f"开始处理 {len(messages)} 条消息...")

        # ========== Step 1: 价值评估 ==========
        logger.debug("Step 1: 价值评估...")
        has_value = self.gater.evaluate(messages)

        if not has_value:
            logger.info("对话无长期价值，跳过提取")
            return []

        # ========== Step 2: LLM 提取 ==========
        logger.debug("Step 2: LLM 提取...")

        # 格式化对话
        transcript = self._format_transcript(messages)

        # 调用提取器
        draft = self.extractor.extract(
            transcript=transcript,
            metadata={
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

        if not draft or not draft.has_value:
            logger.info("LLM 判断对话无价值，跳过存储")
            return []

        # ========== Step 3: 查重检测 ==========
        logger.debug("Step 3: 查重检测...")

        decision, existing_memory = self.deduplicator.check_duplicate(draft)

        # 根据决策执行操作
        if decision == DuplicateDecision.TOUCH:
            # 仅更新访问时间
            logger.info("记忆重复，更新访问时间")
            self.storage.update_access_info(existing_memory.id)
            return [existing_memory]

        elif decision == DuplicateDecision.UPDATE:
            # 知识演化合并
            logger.info("记忆演化，合并内容")
            merged_memory = self.deduplicator.merge_memory(existing_memory, draft)

            # 重新生成向量
            self._save_memory(merged_memory)
            return [merged_memory]

        elif decision == DuplicateDecision.CREATE:
            # 创建新记忆
            logger.info("创建新记忆")
            memory = self._draft_to_memory(draft, user_id, agent_id, session_id)

            # 持久化
            self._save_memory(memory)
            return [memory]

        else:  # DISCARD
            logger.info("低质量重复，丢弃")
            return []

    def _format_transcript(self, messages: List[ConversationMessage]) -> str:
        """
        格式化对话为文本

        Args:
            messages: 对话消息列表

        Returns:
            str: 格式化的对话文本

        Examples:
            >>> transcript = orchestrator._format_transcript(messages)
            >>> print(transcript)
            👤 User: 你好
            🤖 Assistant: 你好！
        """
        lines = []
        for msg in messages:
            role_display = {
                "user": "👤 User",
                "assistant": "🤖 Assistant",
                "system": "⚙️ System"
            }.get(msg.role, msg.role)

            lines.append(f"{role_display}: {msg.content}")

        return "\n".join(lines)

    def _draft_to_memory(
        self,
        draft: ExtractedMemoryDraft,
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> MemoryAtom:
        """
        将草稿转换为完整的 MemoryAtom

        Args:
            draft: 提取的草稿
            user_id: 用户ID
            agent_id: Agent ID
            session_id: 会话ID

        Returns:
            MemoryAtom: 记忆原子对象

        Examples:
            >>> memory = orchestrator._draft_to_memory(draft, "u1", "a1", "s1")
            >>> memory.index.title
            "Python 快排算法"
        """
        # 映射字符串类型到枚举
        try:
            mem_type = MemoryType(draft.memory_type)
        except ValueError:
            logger.warning(f"未知的记忆类型: {draft.memory_type}, 使用 FACT")
            mem_type = MemoryType.FACT

        return MemoryAtom(
            meta=MetaData(
                source_agent_id=agent_id,
                user_id=user_id,
                session_id=session_id,
                confidence_score=draft.confidence_score,
            ),
            index=IndexLayer(
                title=draft.title,
                summary=draft.summary,
                tags=draft.tags,
                memory_type=mem_type,
            ),
            payload=PayloadLayer(
                content=draft.content,
            ),
        )

    def _save_memory(self, memory: MemoryAtom) -> None:
        """
        保存记忆到向量数据库

        Args:
            memory: MemoryAtom 对象

        Raises:
            Exception: 存储失败时抛出

        Examples:
            >>> orchestrator._save_memory(memory)
        """
        try:
            self.storage.upsert_memory(memory)
            logger.info(f"✓ 记忆已存储: '{memory.index.title}' (ID: {memory.id})")

        except Exception as e:
            logger.error(f"存储记忆失败: {e}", exc_info=True)
            raise


__all__ = [
    "MemoryOrchestrator",
]
