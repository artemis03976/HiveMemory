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
from hivememory.core.models import MemoryAtom, MetaData, IndexLayer, PayloadLayer, MemoryType, Identity
from hivememory.engines.generation.models import ExtractedMemoryDraft, GenerationRequest, GenerationContext, WriteFocus, UpdateFocus, MergeResult
from hivememory.engines.generation.generation_transcript_builder import GenerationTranscriptBuilder
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
    "AGENT_PROFILE": "agent",
}


class MemoryGenerationEngine:
    """
    记忆生成引擎

    协调 LLM 提取、查重、存储等所有步骤。

    Examples:
        >>> from hivememory.engines.generation import MemoryGenerationEngine
        >>>
        >>> engine = MemoryGenerationEngine(
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
        初始化引擎

        Args:
            storage: 向量存储实例
            extractor: 记忆提取器（必需）
            deduplicator: 查重器（必需）

        """
        self.storage = storage
        self.extractor = extractor
        self.deduplicator = deduplicator

        logger.info("MemoryGenerationEngine 初始化完成")

    def process(self, request: GenerationRequest) -> List[MemoryAtom]:
        """
        处理对话片段，提取记忆原子 (三模式)

        Mode A (被动观察): request.write_focus=None, request.update_focus=None
        Mode B (主动响应): request.is_write=True (WRITE 指令触发)
        Mode C (合并更新): request.is_update=True (UPDATE 指令触发)

        Args:
            request: GenerationRequest 对象

        Returns:
            List[MemoryAtom]: 提取的记忆原子列表
        """
        if not request.has_context and not request.is_write and not request.is_update:
            logger.debug("空生成上下文且无 write_focus/update_focus，跳过处理")
            return []

        # 路由到对应模式
        if request.is_update:
            return self._process_mode_c(request)
        elif request.is_write:
            return self._process_mode_b(request)
        else:
            return self._process_mode_a(request)

    def _process_mode_a(self, request: GenerationRequest) -> List[MemoryAtom]:
        """
        Mode A: 被动观察模式 (默认)

        从对话中被动提取有价值的记忆。
        """
        identity = request.identity

        logger.info(f"[Mode A] 开始处理...")

        # Step 1: 渲染 transcript（Phase 3 优先路径）
        transcript = self._render_transcript(request)
        if not transcript:
            return []

        draft = self.extractor.extract(
            transcript=transcript,
            metadata={}
        )

        if not draft or not draft.has_value:
            logger.info("[Mode A] LLM 判断对话无价值，跳过存储")
            return []

        # Step 2-4: 查重 → 构建 → 持久化
        return self._dedup_and_persist(draft, identity)

    def _process_mode_b(self, request: GenerationRequest) -> List[MemoryAtom]:
        """
        Mode B: 主动响应模式 (WRITE 指令触发)

        Agent 明确要求保存，以 WriteFocus 为核心，对话历史为背景。
        包含 fallback 机制：LLM 提取失败时直接从 WriteFocus 构建草稿。
        """
        focus = request.write_focus
        identity = request.identity

        logger.info(f"[Mode B] WRITE 主动响应: content='{focus.content[:50]}...'")

        # 格式化背景对话；空上下文时由 _render_transcript() 统一返回占位文本
        transcript = self._render_transcript(request)

        # Step 1: LLM 提取 (Mode B prompt)
        draft = self.extractor.extract(
            transcript=transcript,
            metadata={
                "mode": "write",
                "write_content": focus.content,
                "write_reason": focus.reason or "(未提供)",
            }
        )

        # Fallback: LLM 失败时直接从 WriteFocus 构建草稿 (保底入库)
        if draft is None:
            logger.warning("[Mode B] LLM 提取失败，启用 fallback 直接构建草稿")
            draft = self._build_fallback_draft(focus)

        # Step 2-4: 查重 → 构建 → 持久化
        return self._dedup_and_persist(draft, identity)

    def _build_fallback_draft(self, focus: WriteFocus) -> ExtractedMemoryDraft:
        """
        从 WriteFocus 直接构建 fallback 草稿

        当 LLM 提取失败时，保证 WRITE 内容不丢失。
        """
        title = focus.title or focus.content[:50]
        summary = focus.reason or title
        if len(summary) < 10:
            summary = summary + " — " + focus.content[:50]
        return ExtractedMemoryDraft(
            title=title,
            summary=summary,
            tags=["mtp_write"],
            memory_type="FACT",
            content=focus.content,
            confidence_score=1.0,
            has_value=True,
            alias_suffix="",
        )

    def _process_mode_c(self, request: GenerationRequest) -> List[MemoryAtom]:
        """
        Mode C: 合并更新模式 (UPDATE 指令触发)

        Agent 请求修改已有记忆，以 UpdateFocus 为核心。
        LLM 执行智能合并，生成新内容和变更日志。
        包含 fallback 机制：LLM 合并失败时直接拼接。
        """
        uf = request.update_focus
        identity = request.identity

        # 从内存索引中获取原始记忆
        existing = uf.existing_memory

        if existing is None:
            logger.error("[Mode C] existing_memory 未注入，无法执行 UPDATE")
            return []

        logger.info(
            f"[Mode C] UPDATE 合并: alias='{uf.target_alias}', "
            f"instruction='{uf.instruction[:50]}...'"
        )

        # 格式化对话上下文；空上下文时由 _render_transcript() 统一返回占位文本
        transcript = self._render_transcript(request)

        # Step 1: 调用 extractor.merge() (Mode C Merge Prompt)
        merge_result = self.extractor.merge(
            old_content=existing.payload.content,
            metadata={
                "mode": "update",
                "instruction": uf.instruction,
                "new_content": uf.content or "",
                "memory_title": existing.index.title,
                "memory_alias": existing.index.alias or uf.target_alias,
                "transcript": transcript,
            }
        )

        # Fallback: LLM 合并失败时直接拼接
        if merge_result is None:
            logger.warning("[Mode C] LLM 合并失败，启用 fallback")
            merge_result = self._build_update_fallback(uf, existing)

        # Step 2: 版本历史 + 更新 + 持久化
        return self._apply_update(existing, merge_result)

    def _build_update_fallback(
        self, uf: UpdateFocus, existing: MemoryAtom
    ) -> MergeResult:
        """
        UPDATE fallback: LLM 合并失败时的保底策略

        - 有 content: 追加到旧内容末尾
        - 仅有 instruction: 保留旧内容不变，changelog 记录 instruction
        """
        if uf.content:
            new_content = (
                f"{existing.payload.content}\n\n"
                f"## 更新 ({datetime.now().strftime('%Y-%m-%d')})\n"
                f"{uf.content}"
            )
            changelog = f"Fallback 追加: {uf.instruction[:80]}"
        else:
            new_content = existing.payload.content
            changelog = f"Fallback (无变更): {uf.instruction[:80]}"

        return MergeResult(new_content=new_content, changelog=changelog)

    def _apply_update(
        self, memory: MemoryAtom, result: MergeResult
    ) -> List[MemoryAtom]:
        """
        执行版本历史追踪 + 内容更新 + 持久化

        1. Push old content → artifacts.full_history
        2. 更新 history_summary
        3. 覆盖 payload.content
        4. 更新 meta (updated_at, confidence, version)
        5. 持久化 (重新生成向量)
        """
        now = datetime.now()

        # 1. Push History: 旧内容压入 artifacts.full_history
        history_item = {
            "timestamp": now.isoformat(),
            "content": memory.payload.content,
            "reason": result.changelog,
        }
        memory.payload.artifacts.full_history.append(history_item)

        # 2. 更新 history_summary (简化版本记录)
        summary_line = f"{now.strftime('%Y-%m-%d')}: {result.changelog}"
        memory.payload.history_summary.append(summary_line)

        # 3. Update Head: 覆盖 payload.content
        memory.payload.content = result.new_content

        # 4. 更新 meta
        memory.meta.updated_at = now
        memory.meta.confidence_score = 1.0  # Agent 主动修改
        memory.meta.version += 1

        # 5. 持久化 (重新生成向量)
        self._save_memory(memory)

        logger.info(
            f"[Mode C] UPDATE 完成: '{memory.index.title}' "
            f"v{memory.meta.version}, changelog='{result.changelog}'"
        )
        return [memory]

    def _dedup_and_persist(
        self,
        draft: ExtractedMemoryDraft,
        identity: Identity,
    ) -> List[MemoryAtom]:
        """
        查重 → 构建 → 持久化 (Mode A/B 共用)
        """

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

    def _render_transcript(self, request: GenerationRequest) -> str:
        """
        统一 transcript 渲染入口。

        Returns:
            str: 渲染后的 transcript 文本，无有效内容时返回 "(无背景对话)"
        """
        if not request.context.turns and not request.context.state_summary:
            return "(无背景对话)"
        builder = GenerationTranscriptBuilder()
        return builder.build_transcript(request.context)

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
            >>> identity = Identity(user_id="u1", agent_id="a1")
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
                session_id=None,  # session_id 已从 Identity 中移除
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
