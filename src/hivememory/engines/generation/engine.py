"""
HiveMemory - 记忆生成编排器 (Memory Generation Orchestrator)

职责:
    协调所有组件，执行完整的记忆生成流程。

工作流程:
    Step 1: LLM 提取 → ExtractedMemoryDraft
    Step 2: 查重检测 → CREATE/UPDATE/TOUCH
    Step 3: 记忆原子构建 → MemoryAtom
    Step 4: 返回 GenerationOutcome，外围 Familiar 负责持久化

作者: HiveMemory Team
版本: 0.2.0
"""

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

from hivememory.core.errors import WorkspaceMismatchError
from hivememory.core.models import (
    IdentityScope,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryLifecycleState,
    MemoryType,
    MetaData,
    PayloadLayer,
    UpdateFocus,
    WriteFocus,
)
from hivememory.core.models.provenance import (
    MemoryProvenance,
    normalize_contributing_agent_ids,
)
from hivememory.engines.generation.interfaces import (
    BaseDeduplicator,
    BaseMemoryExtractor,
)
from hivememory.engines.generation.models import (
    DuplicateDecision,
    ExtractedMemoryDraft,
    GenerationOutcome,
    GenerationRequest,
    MergeResult,
    provenance_from_actor,
    system_settlement_provenance,
)
from hivememory.utils.time import require_utc, utc_now

if TYPE_CHECKING:
    from hivememory.patchouli.memory_library.stores import MidTermMemoryStore

logger = logging.getLogger(__name__)

# MTP 别名系统: MemoryType -> 别名前缀映射 (Section 2.3.1)
MEMORY_TYPE_ALIAS_PREFIX: dict[str, str] = {
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
        mid_term: "MidTermMemoryStore",
        extractor: BaseMemoryExtractor,
        deduplicator: BaseDeduplicator,
    ):
        self._mid_term = mid_term
        self.extractor = extractor
        self.deduplicator = deduplicator
        logger.info("MemoryGenerationEngine 初始化完成")

    async def process(
        self,
        request: GenerationRequest,
        *,
        identity_scope: IdentityScope,
        now: datetime | None = None,
    ) -> list[GenerationOutcome]:
        """
        处理对话片段，提取记忆原子 (三模式)

        Mode A (被动结算): request.write_focus=None, request.update_focus=None
            没有具体 Agent 作为操作来源主体，来源记录使用保留 system；
        Mode B (主动响应): request.is_write=True (WRITE 指令触发)
            以提交操作的 actor Agent 为来源；
        Mode C (合并更新): request.is_update=True (UPDATE 指令触发)
            保留已有 metadata 的来源字段。

        主动模式的贡献者先记录发起 Agent 本身，再合并上下文轮次贡献者；
        SETTLE 的贡献者来自上下文轮次身份。版本演化（dedup 合并、Mode C）
        会把本次贡献者并入已有集合，而来源字段只记录 provenance 事实，
        不参与读取授权。

        Args:
            request: GenerationRequest 对象
            identity_scope: 唯一的身份/ownership 来源

        Returns:
            List[GenerationOutcome]: 结构化生成结果列表
        """
        if not request.has_context and not request.is_write and not request.is_update:
            logger.debug("空生成上下文且无 write_focus/update_focus，跳过处理")
            return []

        # 提交边界时点：由 Familiar 传入的同一 UTC now；缺省取当前 UTC。
        commit_now = require_utc(now) if now is not None else utc_now()

        # 路由到对应模式
        if request.is_update:
            return await self._process_mode_c(request, identity_scope, now=commit_now)
        elif request.is_write:
            return await self._process_mode_b(request, identity_scope, now=commit_now)
        else:
            return await self._process_mode_a(request, identity_scope, now=commit_now)

    async def _process_mode_a(
        self,
        request: GenerationRequest,
        identity_scope: IdentityScope,
        *,
        now: datetime,
    ) -> list[GenerationOutcome]:
        """
        Mode A: 被动结算模式 (默认)

        从对话中被动提取有价值的记忆。当前唯一生产入口是话题 SETTLE：
        结算没有具体 Agent 作为操作来源主体，来源记录使用保留
        ``SYSTEM_AGENT_ID``，实际参与内容的 Agent 进入贡献者集合。
        """
        logger.info("[Mode A] 开始处理...")

        # Step 1: 渲染 transcript（Phase 3 优先路径）
        transcript = self._render_transcript(request)
        if not transcript:
            return []

        draft = self.extractor.extract(transcript=transcript, metadata={})

        if not draft or not draft.has_value:
            logger.info("[Mode A] LLM 判断对话无价值，跳过存储")
            return []

        # Step 2-4: 查重 → 构建/更新 → 返回 outcome
        return await self._dedup_and_resolve(
            draft,
            identity_scope,
            system_settlement_provenance(request.context),
            now=now,
        )

    async def _process_mode_b(
        self,
        request: GenerationRequest,
        identity_scope: IdentityScope,
        *,
        now: datetime,
    ) -> list[GenerationOutcome]:
        """
        Mode B: 主动响应模式 (WRITE 指令触发)

        Agent 明确要求保存，以 WriteFocus 为核心，对话历史为背景。
        包含 fallback 机制：LLM 提取失败时直接从 WriteFocus 构建草稿。
        """
        focus = request.write_focus

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
            },
        )

        # Fallback: LLM 失败时直接从 WriteFocus 构建草稿 (保底入库)
        if draft is None:
            logger.warning("[Mode B] LLM 提取失败，启用 fallback 直接构建草稿")
            draft = self._build_fallback_draft(focus)

        # Step 2-4: 查重 → 构建/更新 → 返回 outcome
        return await self._dedup_and_resolve(
            draft,
            identity_scope,
            provenance_from_actor(identity_scope, request.context),
            now=now,
        )

    def _build_fallback_draft(self, focus: WriteFocus) -> ExtractedMemoryDraft:
        """
        从 WriteFocus 直接构建 fallback 草稿

        当 LLM 提取失败时，保证 WRITE 内容不丢失。
        """
        title = focus.title or focus.content[:50]
        summary = focus.reason or title
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

    async def _process_mode_c(
        self,
        request: GenerationRequest,
        identity_scope: IdentityScope,
        *,
        now: datetime,
    ) -> list[GenerationOutcome]:
        """
        Mode C: 合并更新模式 (UPDATE 指令触发)

        Agent 请求修改已有记忆，以 UpdateFocus 为核心。
        LLM 执行智能合并，生成新内容和变更日志。
        包含 fallback 机制：LLM 合并失败时直接拼接。
        """
        uf = request.update_focus
        # 从内存索引中获取原始记忆
        existing = request.existing_memory

        if existing is None:
            logger.error("[Mode C] existing_memory 未注入，无法执行 UPDATE")
            return []
        if existing.workspace_identity != identity_scope.workspace_identity:
            raise WorkspaceMismatchError(
                details={
                    "memory_id": str(existing.id),
                    "memory_workspace_id": existing.workspace_identity.workspace_id,
                    "request_workspace_id": identity_scope.workspace_identity.workspace_id,
                }
            )

        logger.info(
            f"[Mode C] UPDATE 合并: alias='{uf.base_alias}', "
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
                "memory_alias": existing.index.alias or uf.base_alias,
                "transcript": transcript,
            },
        )

        # Fallback: LLM 合并失败时直接拼接
        if merge_result is None:
            logger.warning("[Mode C] LLM 合并失败，启用 fallback")
            merge_result = self._build_update_fallback(uf, existing, now=now)

        # Step 2: 版本历史 + 更新，持久化由 Familiar 负责；
        # 主动 UPDATE 以发起 Agent 及上下文贡献者演化已有记忆的贡献者集合。
        return self._apply_update(
            existing,
            merge_result,
            provenance=provenance_from_actor(identity_scope, request.context),
        )

    def _build_update_fallback(
        self, uf: UpdateFocus, existing: MemoryAtom, *, now: datetime
    ) -> MergeResult:
        """
        UPDATE fallback: LLM 合并失败时的保底策略

        - 有 content: 追加到旧内容末尾
        - 仅有 instruction: 保留旧内容不变，changelog 记录 instruction
        """
        if uf.content:
            new_content = (
                f"{existing.payload.content}\n\n"
                f"## 更新 ({now.strftime('%Y-%m-%d')})\n"
                f"{uf.content}"
            )
            changelog = f"Fallback 追加: {uf.instruction[:80]}"
        else:
            new_content = existing.payload.content
            changelog = f"Fallback (无变更): {uf.instruction[:80]}"

        return MergeResult(new_content=new_content, changelog=changelog)

    def _apply_update(
        self,
        memory: MemoryAtom,
        result: MergeResult,
        *,
        provenance: MemoryProvenance,
        dedup_draft: ExtractedMemoryDraft | None = None,
    ) -> list[GenerationOutcome]:
        """
        执行内容修订（纯计算）。持久化、版本分配与内容时间由 Familiar 在
        提交边界统一决定；本方法只做：

        1. 捕获变更前完整原子（深拷贝，供版本记录与 embedding 输入对比）
        2. 按需刷新 dedup index
        3. 覆盖 payload.content
        4. 并入本次生成的贡献者集合

        ``meta.version``/``updated_at``/``decay_anchor_at``/``confidence_score``
        的赋值全部发生在 Familiar 的提交边界（使用同一个 ``now``）。
        """
        # §3.2：实际内容相同的重复更新不创建新版本。合并结果与现有内容一致
        # 时降级为 TOUCH 纯决策——不合并贡献者、不刷新 dedup index、不改任何
        # 字段；访问统计由 Familiar 经受限 patch_payload 处理。
        if result.new_content == memory.payload.content:
            logger.info(
                "[Mode C] UPDATE 合并结果与现有内容一致，降级为 TOUCH: '%s'",
                memory.index.title,
            )
            return [
                GenerationOutcome(
                    atom=memory,
                    duplicate_decision=DuplicateDecision.TOUCH,
                )
            ]

        # 深拷贝保留修改前的完整原子；嵌套对象不与候选共享引用。
        before_snapshot = memory.model_copy(deep=True)

        if dedup_draft is not None:
            self._merge_dedup_index(memory, dedup_draft)

        # Update Head: 覆盖 payload.content
        memory.payload.content = result.new_content

        # 版本演化引入新的内容贡献者：把本次来源裁定中的贡献者并入已有集合
        # （去重并保持首次出现顺序）。settle 贡献者因此能进入已有 Memory 及其
        # 后续 Version Artifact；来源主体字段按约定保留不改写。
        memory.meta.provenance.contributing_agent_ids = normalize_contributing_agent_ids(
            [*memory.meta.provenance.contributing_agent_ids, *provenance.contributing_agent_ids]
        )

        logger.info(
            f"[Mode C] UPDATE 内容已合并（待提交边界分配版本与内容时间）: "
            f"'{memory.index.title}', changelog='{result.changelog}'"
        )

        return [
            GenerationOutcome(
                atom=memory,
                duplicate_decision=DuplicateDecision.UPDATE,
                memory_before_snapshot=before_snapshot,
                changelog=result.changelog,
            )
        ]

    async def _dedup_and_resolve(
        self,
        draft: ExtractedMemoryDraft,
        identity_scope: IdentityScope,
        provenance: MemoryProvenance,
        *,
        now: datetime,
    ) -> list[GenerationOutcome]:
        """
        查重 → 构建/演化决策 (Mode A/B 共用)

        Args:
            draft: 提取的草稿
            identity_scope: 已验证的 Workspace ownership 来源
            provenance: 本次生成的来源裁定（操作来源主体与内容贡献者）
        """
        query_text = f"{draft.title} {draft.summary}"
        candidates = await self._mid_term.search(
            identity_scope,
            query=query_text,
            top_k=1,
            filters=None,
            mode="dense",
        )

        decision, existing_memory = self.deduplicator.check_duplicate(draft, candidates)

        # 根据决策执行操作
        if decision == DuplicateDecision.TOUCH:
            # 内容相同的重复访问：引擎保持纯决策，不改任何字段（MVL-0 冻结）。
            # 访问计数的推进与持久化由 Familiar 经受限 patch_payload 完成。
            logger.info("记忆重复，访问状态由 Familiar 经受限 patch 更新")

            return [
                GenerationOutcome(
                    atom=existing_memory,
                    duplicate_decision=DuplicateDecision.TOUCH,
                )
            ]

        elif decision == DuplicateDecision.UPDATE:
            logger.info("记忆演化，覆盖当前版本内容")

            # TODO: 如轻量覆盖路径效果不足，可在这里引入强合并路径：
            # 将 draft 与 existing_memory 交给 extractor.merge()，生成一个完整的新 head。
            # 不要回退到字符串追加，历史展示应由 MemoryVersionArtifact 链路负责。
            merge_result = MergeResult(
                new_content=draft.content,
                changelog=f"Dedup update: {draft.summary[:120]}",
            )
            return self._apply_update(
                existing_memory,
                merge_result,
                provenance=provenance,
                dedup_draft=draft,
            )

        elif decision == DuplicateDecision.CREATE:
            logger.info("创建新记忆")

            memory = self._draft_to_memory(draft, identity_scope, provenance, now=now)

            return [
                GenerationOutcome(
                    atom=memory,
                    duplicate_decision=DuplicateDecision.CREATE,
                )
            ]

        else:  # DISCARD
            logger.info("低质量重复，丢弃")
            return [
                GenerationOutcome(
                    atom=None,
                    duplicate_decision=DuplicateDecision.DISCARD,
                    message="Low-quality duplicate, discarded.",
                )
            ]

    def _merge_dedup_index(
        self,
        existing: MemoryAtom,
        draft: ExtractedMemoryDraft,
    ) -> None:
        """
        用 dedup 草稿刷新检索层；历史版本由 MemoryVersionArtifact 记录。
        """
        merged_tags = []
        for tag in [*existing.index.tags, *draft.tags]:
            normalized = tag.lower().strip()
            if normalized and normalized not in merged_tags:
                merged_tags.append(normalized)
        existing.index.title = draft.title
        existing.index.summary = draft.summary
        existing.index.tags = merged_tags[:5]
        try:
            existing.index.memory_type = MemoryType(draft.memory_type)
        except ValueError:
            logger.warning(f"未知的记忆类型: {draft.memory_type}, 保留原类型")

    def _render_transcript(self, request: GenerationRequest) -> str:
        """
        统一 transcript 渲染入口。

        Returns:
            str: 渲染后的 transcript 文本，无有效内容时返回 "(无背景对话)"
        """
        if not request.context.turns and not request.context.state_summary:
            return "(无背景对话)"
        # 延迟导入：prompts 在依赖层级上高于 engines，模块加载期 import 会形成
        # engines.generation ↔ prompts.transcript 循环（见 PendingAtomRuntimeDesign §6.2）。
        from hivememory.prompts.transcript import GenerationTranscriptBuilder

        builder = GenerationTranscriptBuilder()
        return builder.build_transcript(request.context)

    def _draft_to_memory(
        self,
        draft: ExtractedMemoryDraft,
        identity_scope: IdentityScope,
        provenance: MemoryProvenance,
        *,
        now: datetime,
    ) -> MemoryAtom:
        """
        将草稿转换为完整的 MemoryAtom

        Args:
            draft: 提取的草稿
            identity_scope: 已验证的 Workspace ownership 来源
            provenance: 本次生成的来源裁定；来源字段只记录 provenance，
                不参与读取授权

        Returns:
            MemoryAtom: 记忆原子对象

        Examples:
            >>> memory = orchestrator._draft_to_memory(draft, identity_scope, provenance)
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

        # 创建时点 = 提交边界 now（Familiar 传入）：created/updated/decay 同值。
        return MemoryAtom(
            meta=MetaData(
                workspace_identity=identity_scope.workspace_identity,
                provenance=provenance,
                access_policy=MemoryAccessPolicy.public(),
                created_at=now,
                updated_at=now,
                lifecycle=MemoryLifecycleState(
                    confidence_score=draft.confidence_score,
                    decay_anchor_at=now,
                ),
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
    ) -> str | None:
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
            suffix = re.sub(r"[^a-z0-9\s_]", "", suffix)
            suffix = re.sub(r"\s+", "_", suffix)
            suffix = re.sub(r"_+", "_", suffix).strip("_")

        if not suffix:
            return None

        # 清洗 suffix: 确保 snake_case 合规
        suffix = re.sub(r"[^a-z0-9_]", "", suffix.lower())
        suffix = re.sub(r"_+", "_", suffix).strip("_")
        suffix = suffix[:40]

        return f"{prefix}_{suffix}"


__all__ = [
    "MemoryGenerationEngine",
]
