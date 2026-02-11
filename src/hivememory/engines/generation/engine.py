"""
HiveMemory - 记忆生成编排器 (Memory Generation Orchestrator)

职责:
    协调所有组件，执行完整的记忆生成流程。

工作流程:
    Step 1: LLM 提取 → ExtractedMemoryDraft
    Step 2: 查重检测 → CREATE/UPDATE/TOUCH
    Step 3: 记忆原子构建 → MemoryAtom
    Step 4: 持久化 → Qdrant

作者: HiveMemory Team
版本: 0.2.0
"""

import re
import logging
from typing import Dict, List, Optional
from datetime import datetime

from hivememory.infrastructure.storage import QdrantMemoryStore
from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType, StreamMessage, Identity
from hivememory.engines.generation.models import ExtractedMemoryDraft
from hivememory.engines.generation.interfaces import (
    BaseMemoryExtractor,
    BaseDeduplicator,
    DuplicateDecision,
)

logger = logging.getLogger(__name__)

# MTP 别名系统: MemoryType -> 别名前缀映射 (Section 2.3.1)
MEMORY_TYPE_ALIAS_PREFIX: Dict[str, str] = {
    "CODE_SNIPPET": "code",
    "FACT": "fact",
    "URL_RESOURCE": "url",
    "REFLECTION": "ref",
    "USER_PROFILE": "user",
    "WORK_IN_PROGRESS": "wip",
}


class MemoryGenerationEngine:
    """
    记忆生成引擎

    协调LLM 提取、查重、存储等所有步骤。

    遵循显式依赖注入原则：所有子组件必须通过构造函数传入，
    不在内部实例化依赖项。

    Examples:
        >>> from hivememory.engines.generation import create_default_generation_engine
        >>> engine = create_default_generation_engine(storage=storage)
        >>>
        >>> # 高级：手动注入组件
        >>> orchestrator = MemoryGenerationOrchestrator(
        ...     storage=storage,
        ...     extractor=my_extractor,
        ...     deduplicator=my_deduplicator,
        ... )
    """

    def __init__(
        self,
        storage: QdrantMemoryStore,
        extractor: BaseMemoryExtractor,
        deduplicator: BaseDeduplicator,
    ):
        """
        初始化编排器

        Args:
            storage: 向量存储实例
            extractor: 记忆提取器（必需）
            deduplicator: 查重器（必需）

        Note:
            所有组件参数都是必需的。
        """
        self.storage = storage
        self.extractor = extractor
        self.deduplicator = deduplicator

        logger.info("MemoryGenerationOrchestrator 初始化完成")

    def process(
        self,
        messages: List[StreamMessage],
    ) -> List[MemoryAtom]:
        """
        处理对话片段，提取记忆原子

        完整流程:
            1. LLM 提取 → 生成结构化草稿
            2. 查重检测 → 判断 CREATE/UPDATE/TOUCH
            3. 记忆构建 → MemoryAtom
            4. 持久化 → Qdrant

        Args:
            messages: 对话消息列表

        Returns:
            List[MemoryAtom]: 提取的记忆原子列表

        Examples:
            >>> memories = orchestrator.process(
            ...     messages=[
            ...         StreamMessage(role="user", content="写快排"),
            ...         StreamMessage(role="assistant", content="代码...")
            ...     ],
            ... )
            >>> len(memories)
            1
        """
        if not messages:
            logger.debug("空消息列表，跳过处理")
            return []
        
        identity = messages[0].identity
        user_id = identity.user_id
        agent_id = identity.agent_id
        session_id = identity.session_id

        logger.info(f"开始处理 {len(messages)} 条消息...")

        # ========== Step 1: LLM 提取 ==========
        logger.debug("Step 1: LLM 提取...")

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

        # ========== Step 2: 查重检测 ==========
        logger.debug("Step 2: 查重检测...")

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
            memory = self._draft_to_memory(draft, identity)

            # 持久化
            self._save_memory(memory)
            return [memory]

        else:  # DISCARD
            logger.info("低质量重复，丢弃")
            return []

    def _format_transcript(self, messages: List[StreamMessage]) -> str:
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
        identity: Identity,
    ) -> MemoryAtom:
        """
        将草稿转换为完整的 MemoryAtom

        Args:
            draft: 提取的草稿
            identity: 身份标识

        Returns:
            MemoryAtom: 记忆原子对象

        Examples:
            >>> identity = Identity(user_id="u1", agent_id="a1", session_id="s1")
            >>> memory = orchestrator._draft_to_memory(draft, identity)
            >>> memory.index.title
            "Python 快排算法"
        """
        # 映射字符串类型到枚举
        try:
            mem_type = MemoryType(draft.memory_type)
        except ValueError:
            logger.warning(f"未知的记忆类型: {draft.memory_type}, 使用 FACT")
            mem_type = MemoryType.FACT

        # 构建 MTP 别名 (Section 2.3)
        alias = self._build_alias(
            memory_type=draft.memory_type,
            alias_suffix=draft.alias_suffix,
            title=draft.title,
        )

        return MemoryAtom(
            meta=MetaData(
                source_agent_id=identity.agent_id,
                user_id=identity.user_id,
                session_id=identity.session_id,
                confidence_score=draft.confidence_score,
            ),
            index=IndexLayer(
                title=draft.title,
                summary=draft.summary,
                tags=draft.tags,
                memory_type=mem_type,
                alias=alias,
            ),
            payload=PayloadLayer(
                content=draft.content,
            ),
        )

    @staticmethod
    def _build_alias(
        memory_type: str,
        alias_suffix: str,
        title: str,
    ) -> Optional[str]:
        """
        构建完整的 MTP 别名 (Section 2.3.1)

        策略:
            1. 从 MEMORY_TYPE_ALIAS_PREFIX 取前缀
            2. 优先使用 LLM 生成的 alias_suffix
            3. alias_suffix 为空时从 title 派生 fallback suffix
            4. 清洗并验证最终别名格式

        Args:
            memory_type: 记忆类型字符串 (e.g. "CODE_SNIPPET")
            alias_suffix: LLM 生成的别名后缀 (可能为空)
            title: 记忆标题 (用于 fallback)

        Returns:
            完整别名 (e.g. "code_quicksort_impl"), 或 None
        """
        prefix = MEMORY_TYPE_ALIAS_PREFIX.get(memory_type, "mem")

        # 确定 suffix: 优先使用 LLM 生成的，否则从 title 派生
        suffix = alias_suffix.strip() if alias_suffix else ""
        if not suffix:
            suffix = title.lower().strip()
            suffix = re.sub(r'[^a-z0-9\s_]', '', suffix)
            suffix = re.sub(r'\s+', '_', suffix)
            suffix = re.sub(r'_+', '_', suffix).strip('_')

        if not suffix:
            return None

        # 清洗 suffix: 确保 snake_case 合规
        suffix = re.sub(r'[^a-z0-9_]', '', suffix.lower())
        suffix = re.sub(r'_+', '_', suffix).strip('_')
        suffix = suffix[:40]

        return f"{prefix}_{suffix}"

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
    "MemoryGenerationEngine",
]
