"""
HiveMemory - 记忆生成编排器 (Memory Generation Orchestrator)

职责:
    协调所有组件，执行完整的记忆生成流程。

工作流程:
    Step 1: LLM 提取 → ExtractedMemoryDraft（话题结算可以提取多条）
    Step 2: 逐条查重判断 → 新建、触碰已有记忆，或新建并提名合并
    Step 3: 记忆原子构建 → MemoryAtom
    Step 4: 返回 GenerationOutcome，外围 Familiar 负责持久化
"""

import logging
from collections.abc import Collection
from datetime import datetime
from typing import TYPE_CHECKING

from hivememory.core.errors import ModelProviderError, WorkspaceMismatchError
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryLifecycleState,
    MemoryType,
    MetaData,
    PayloadLayer,
    UpdateFocus,
    WorkspaceIdentity,
    WriteFocus,
)
from hivememory.core.models.provenance import (
    MemoryProvenance,
    normalize_contributing_agent_ids,
)
from hivememory.engines.generation.alias import AliasGenerator
from hivememory.engines.generation.interfaces import (
    BaseDeduplicator,
    BaseMemoryExtractor,
)
from hivememory.engines.generation.models import (
    DuplicateDecision,
    ExtractedMemoryDraft,
    GenerationAction,
    GenerationOutcome,
    GenerationRequest,
    RevisionResult,
    provenance_from_actor,
    system_settlement_provenance,
)
from hivememory.engines.memory_compiler import MemoryCompiler, MemoryCompileTarget
from hivememory.utils.time import require_utc, utc_now

if TYPE_CHECKING:
    from hivememory.patchouli.memory_library.stores import MidTermMemoryStore

logger = logging.getLogger(__name__)

_compiler = MemoryCompiler()


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
        alias_generator: AliasGenerator | None = None,
    ):
        """
        Args:
            mid_term: 中期记忆存储（查重检索与 alias 唯一性查询）
            extractor: LLM 提取器
            deduplicator: 查重器
            alias_generator: Workspace 内唯一的别名生成器；缺省以 ``mid_term``
                作为占用查询端口构造
        """
        self._mid_term = mid_term
        self.extractor = extractor
        self.deduplicator = deduplicator
        self._alias_generator = alias_generator or AliasGenerator(mid_term)
        logger.info("MemoryGenerationEngine 初始化完成")

    async def process(
        self,
        request: GenerationRequest,
        *,
        belong_to: WorkspaceIdentity,
        from_actor: ActorIdentity,
        now: datetime | None = None,
    ) -> list[GenerationOutcome]:
        """
        处理对话片段，提取记忆原子 (三模式)

        Mode A (被动结算): request.write_focus=None, request.update_focus=None
            没有具体 Agent 作为操作来源主体，来源记录使用保留 system；
        Mode B (主动响应): request.is_write=True (WRITE 指令触发)
            以提交操作的 actor Agent 为来源；
        Mode C (修订): request.is_update=True (UPDATE 指令触发)
            保留已有 metadata 的来源字段。

        主动模式的贡献者先记录发起 Agent 本身，再合并上下文轮次贡献者；
        SETTLE 的贡献者来自上下文轮次身份。查重判为相似时先新建独立记忆，
        贡献者在后续合并任务中并入保留对象；Mode C 随内容修订并入贡献者。
        来源字段只记录 provenance 事实，不参与读取授权。

        Args:
            request: GenerationRequest 对象
            belong_to: 记忆的 Workspace 归属
            from_actor: 本次生成的发起者，也是查重时的可见性主体

        Returns:
            List[GenerationOutcome]: 结构化生成结果列表
        """
        if not request.has_context and not request.is_write and not request.is_update:
            logger.debug("空生成上下文且无 write_focus/update_focus，跳过处理")
            return []

        # 提交边界时点：由 Familiar 传入的同一 UTC now；缺省取当前 UTC。
        commit_now = require_utc(now) if now is not None else utc_now()

        # 路由到对应模式；focus 在分派处收窄后交给对应模式
        if request.update_focus is not None:
            return await self._process_mode_c(
                request, request.update_focus, belong_to, from_actor=from_actor, now=commit_now
            )
        if request.write_focus is not None:
            return await self._process_mode_b(
                request, request.write_focus, belong_to, from_actor=from_actor, now=commit_now
            )
        return await self._process_mode_a(request, belong_to, from_actor=from_actor, now=commit_now)

    async def _process_mode_a(
        self,
        request: GenerationRequest,
        belong_to: WorkspaceIdentity,
        *,
        from_actor: ActorIdentity,
        now: datetime,
    ) -> list[GenerationOutcome]:
        """
        Mode A: 被动结算模式 (默认)

        从对话中被动提取有价值的记忆。当前唯一生产入口是话题 SETTLE：
        结算没有具体 Agent 作为操作来源主体，来源记录使用保留
        ``SYSTEM_AGENT_ID``，实际参与内容的 Agent 进入贡献者集合。

        一段话题可以提取出多条彼此独立的草稿，每条各自查重并产出结果，
        顺序与草稿一致。同一批草稿尚未落库，别名生成器查不到彼此的占用，
        因此本批已分配的别名随后续草稿传入。

        模型判断无价值或输出无法解析时没有产出；提取器的提供商失败（含输出
        被截断）向上传播，由生成任务按失败收尾，不被记录为没有产出。

        查重命中本话题主动写入结算到的记忆时丢弃草稿：被动提取会从同一份材料
        里重新推导出 agent 已经写入的内容，并入后会以较新的被动草稿为骨架改写它。
        """
        logger.info("[Mode A] 开始处理...")

        # Step 1: 渲染 transcript
        transcript = self._render_transcript(request)
        if not transcript:
            return []

        drafts = [draft for draft in await self.extractor.extract(transcript) if draft.has_value]
        if not drafts:
            logger.info("[Mode A] LLM 判断对话无价值，跳过存储")
            return []

        # Step 2-4: 逐条查重 → 构建 → 返回 outcome
        provenance = system_settlement_provenance(request.context)
        outcomes: list[GenerationOutcome] = []
        taken_aliases: set[str] = set()
        for draft in drafts:
            resolved = await self._dedup_and_resolve(
                draft,
                belong_to,
                provenance,
                from_actor=from_actor,
                now=now,
                taken_aliases=taken_aliases,
                dropped_on_hit=request.topic_write_memory_ids,
            )
            outcomes.extend(resolved)
            taken_aliases.update(
                outcome.atom.index.alias
                for outcome in resolved
                if outcome.atom is not None and outcome.atom.index.alias
            )
        return outcomes

    async def _process_mode_b(
        self,
        request: GenerationRequest,
        focus: WriteFocus,
        belong_to: WorkspaceIdentity,
        *,
        from_actor: ActorIdentity,
        now: datetime,
    ) -> list[GenerationOutcome]:
        """
        Mode B: 主动响应模式 (WRITE 指令触发)

        Agent 明确要求保存，以 WriteFocus 为核心，对话历史为背景。
        包含 fallback 机制：提供商失败或输出无法解析时直接从 WriteFocus 构建
        草稿，明确的保存意图不因 Librarian LLM 暂时失败而丢失；程序错误不降级。

        相似命中本话题其他主动写入结算到的记忆时照常新建、不提名合并：同一话题里
        agent 分开写的记忆是有意分开的。
        """
        logger.info("[Mode B] WRITE 主动响应: content_length=%d", len(focus.content))

        # 格式化背景对话；空上下文时由 _render_transcript() 统一返回占位文本
        transcript = self._render_transcript(request)

        # Step 1: LLM 提取 (Mode B prompt)
        try:
            draft = await self.extractor.extract_write(
                transcript, content=focus.content, reason=focus.reason, title=focus.title
            )
        except ModelProviderError as error:
            logger.warning("[Mode B] LLM 提供商失败: %s", error)
            draft = None

        # Fallback: LLM 失败时直接从 WriteFocus 构建草稿 (保底入库)
        if draft is None:
            logger.warning("[Mode B] LLM 提取失败，启用 fallback 直接构建草稿")
            draft = self._build_fallback_draft(focus)

        # Step 2-4: 查重 → 构建/更新 → 返回 outcome
        return await self._dedup_and_resolve(
            draft,
            belong_to,
            provenance_from_actor(from_actor, request.context),
            from_actor=from_actor,
            now=now,
            kept_apart_from=request.topic_write_memory_ids,
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
        uf: UpdateFocus,
        belong_to: WorkspaceIdentity,
        *,
        from_actor: ActorIdentity,
        now: datetime,
    ) -> list[GenerationOutcome]:
        """
        Mode C: 修订模式 (UPDATE 指令触发)

        Agent 请求修改已有记忆，以 UpdateFocus 为核心。
        LLM 执行修订，生成新内容和变更日志。
        包含 fallback 机制：提供商失败或输出无法解析时直接拼接；程序错误不降级。
        """
        # 从内存索引中获取原始记忆
        existing = request.existing_memory

        if existing is None:
            logger.error("[Mode C] existing_memory 未注入，无法执行 UPDATE")
            return []
        if existing.workspace_identity != belong_to:
            raise WorkspaceMismatchError(
                details={
                    "memory_id": str(existing.id),
                    "memory_workspace_id": existing.workspace_identity.workspace_id,
                    "request_workspace_id": belong_to.workspace_id,
                }
            )

        logger.info(
            "[Mode C] UPDATE 修订: alias=%s, instruction_length=%d",
            uf.base_alias,
            len(uf.instruction),
        )

        # 格式化对话上下文；空上下文时由 _render_transcript() 统一返回占位文本
        transcript = self._render_transcript(request)

        # Step 1: 调用 extractor.revise() (Mode C Update Prompt)
        try:
            revision_result = await self.extractor.revise(
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
        except ModelProviderError as error:
            logger.warning("[Mode C] LLM 提供商失败: %s", error)
            revision_result = None

        # Fallback: LLM 修订失败时直接拼接
        if revision_result is None:
            logger.warning("[Mode C] LLM 修订失败，启用 fallback")
            revision_result = self._build_update_fallback(uf, existing, now=now)

        # Step 2: 版本历史 + 更新，持久化由 Familiar 负责；
        # 主动 UPDATE 以发起 Agent 及上下文贡献者演化已有记忆的贡献者集合。
        return self._apply_update(
            existing,
            revision_result,
            provenance=provenance_from_actor(from_actor, request.context),
        )

    def _build_update_fallback(
        self, uf: UpdateFocus, existing: MemoryAtom, *, now: datetime
    ) -> RevisionResult:
        """
        UPDATE fallback: LLM 修订失败时的保底策略

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

        return RevisionResult(new_content=new_content, changelog=changelog)

    async def merge(self, retained: MemoryAtom, absorbed: MemoryAtom) -> GenerationOutcome | None:
        """纯计算两条记忆的融合内容，不修改输入；版本与持久化由 Familiar 提交。

        来源主体、别名与访问状态始终沿用保留记忆。
        """
        result = await self.extractor.merge(retained, absorbed)
        if result is None:
            return None
        try:
            memory_type = MemoryType(result.memory_type)
        except ValueError:
            memory_type = retained.index.memory_type
        # 普通记忆不能经模型输出变为 Profile；Profile 的创建有独立入口。
        if memory_type == MemoryType.AGENT_PROFILE:
            return None
        index = IndexLayer(
            title=result.title,
            summary=result.summary,
            tags=result.tags,
            memory_type=memory_type,
            alias=retained.index.alias,
        )
        memory = retained.model_copy(deep=True)
        memory.index = index
        memory.payload.content = result.content
        memory.meta.provenance.contributing_agent_ids = normalize_contributing_agent_ids(
            [
                *retained.meta.provenance.contributing_agent_ids,
                *absorbed.meta.provenance.contributing_agent_ids,
            ]
        )
        return GenerationOutcome(
            atom=memory,
            action=GenerationAction.UPDATE,
            memory_before_snapshot=retained.model_copy(deep=True),
            changelog=result.changelog,
        )

    def _apply_update(
        self,
        memory: MemoryAtom,
        result: RevisionResult,
        *,
        provenance: MemoryProvenance,
    ) -> list[GenerationOutcome]:
        """
        执行内容修订（纯计算）。持久化、版本分配与内容时间由 Familiar 在
        提交边界统一决定；本方法只做：

        1. 捕获变更前完整原子（深拷贝，供版本记录与 embedding 输入对比）
        2. 覆盖 payload.content
        3. 并入本次生成的贡献者集合

        ``meta.version``/``updated_at``/``decay_anchor_at``/``confidence_score``
        的赋值全部发生在 Familiar 的提交边界（使用同一个 ``now``）。
        """
        # 实际内容相同的重复更新不创建新版本。修订结果与现有内容一致
        # 时返回 TOUCH 动作——不合并贡献者、不改任何
        # 字段；访问统计由 Familiar 经受限 patch_payload 处理。
        if result.new_content == memory.payload.content:
            logger.info(
                "[Mode C] UPDATE 修订结果与现有内容一致，降级为 TOUCH: memory_id=%s",
                memory.id,
            )
            return [
                GenerationOutcome(
                    atom=memory,
                    action=GenerationAction.TOUCH,
                )
            ]

        # 深拷贝保留修改前的完整原子；嵌套对象不与候选共享引用。
        before_snapshot = memory.model_copy(deep=True)

        # Update Head: 覆盖 payload.content
        memory.payload.content = result.new_content

        # 版本演化引入新的内容贡献者：把本次来源裁定中的贡献者并入已有集合
        # （去重并保持首次出现顺序）。settle 贡献者因此能进入已有 Memory 及其
        # 后续 Version Artifact；来源主体字段按约定保留不改写。
        memory.meta.provenance.contributing_agent_ids = normalize_contributing_agent_ids(
            [*memory.meta.provenance.contributing_agent_ids, *provenance.contributing_agent_ids]
        )

        logger.info(
            "[Mode C] UPDATE 内容已修订（待提交边界分配版本与内容时间）: memory_id=%s",
            memory.id,
        )

        return [
            GenerationOutcome(
                atom=memory,
                action=GenerationAction.UPDATE,
                memory_before_snapshot=before_snapshot,
                changelog=result.changelog,
            )
        ]

    async def _dedup_and_resolve(
        self,
        draft: ExtractedMemoryDraft,
        belong_to: WorkspaceIdentity,
        provenance: MemoryProvenance,
        *,
        from_actor: ActorIdentity,
        now: datetime,
        taken_aliases: Collection[str] = (),
        dropped_on_hit: Collection[str] = (),
        kept_apart_from: Collection[str] = (),
    ) -> list[GenerationOutcome]:
        """
        查重 → 新建、触碰或提名合并 (Mode A/B 共用)

        Args:
            draft: 提取的草稿
            belong_to: 已验证的 Workspace ownership 来源
            provenance: 本次生成的来源裁定（操作来源主体与内容贡献者）
            taken_aliases: 同一批中已分配、尚未落库的别名，新建时视为已占用
            dropped_on_hit: 记忆 ID；查重为 IDENTICAL 或 SIMILAR 且命中其中一条时
                丢弃草稿，不新建、不触碰、不提名合并
            kept_apart_from: 记忆 ID；查重为 SIMILAR 且命中其中一条时照常新建，
                但不提名合并
        """
        # 已有记忆的稠密向量由 DENSE_EMBEDDING 编译文本编码；查询用同一目标编译草稿，
        # 两边文本形态一致，内容相同的草稿才能达到 IDENTICAL 的相似度。别名不进入
        # 编译文本，查询时还没有分配。
        query_text = _compiler.compile(
            self._draft_to_memory(draft, belong_to, provenance, alias=None, now=now),
            MemoryCompileTarget.DENSE_EMBEDDING,
        ).text
        candidates = await self._mid_term.search(
            belong_to,
            query=query_text,
            top_k=1,
            filters=None,
            mode="dense",
            from_actor=from_actor,
        )

        decision, existing_memory = self.deduplicator.check_duplicate(draft, candidates)

        if (
            decision in (DuplicateDecision.IDENTICAL, DuplicateDecision.SIMILAR)
            and existing_memory is not None
            and str(existing_memory.id) in dropped_on_hit
        ):
            logger.info(
                "草稿命中本话题主动写入结算到的记忆，丢弃: memory_id=%s, decision=%s, score=%.3f",
                existing_memory.id,
                decision.value,
                candidates[0]["score"],
            )
            return []

        # IDENTICAL/SIMILAR 由查重器保证携带命中的已有记忆。
        if decision == DuplicateDecision.IDENTICAL:
            existing_memory = _require_matched_memory(decision, existing_memory)
            # 内容相同的重复访问：引擎保持纯决策，不改任何字段。
            # 访问计数的推进与持久化由 Familiar 经受限 patch_payload 完成。
            logger.info("记忆重复，访问状态由 Familiar 经受限 patch 更新")

            return [
                GenerationOutcome(
                    atom=existing_memory,
                    action=GenerationAction.TOUCH,
                )
            ]

        logger.info("创建新记忆")

        memory = self._draft_to_memory(draft, belong_to, provenance, alias=None, now=now)
        # 构建 Workspace 内唯一的 MTP 别名（唯一性第一层：生成侧先查询空闲候选）；
        # 无法构造有效别名时回退到以这条记忆的 id 命名的别名，因此先构造原子。
        memory.index.alias = await self._alias_generator.generate(
            workspace_identity=belong_to,
            memory_type=draft.memory_type,
            alias_suffix=draft.alias_suffix,
            title=draft.title,
            memory_id=memory.id,
            taken=taken_aliases,
        )

        # 相似时也先保留独立的新记忆；策略不同或任一方是 Profile 时不提名合并，
        # 避免扩大可见范围或覆盖承载结构化配置的 Profile；命中 kept_apart_from
        # 中的记忆时也不提名。
        merge_target_id = None
        if decision == DuplicateDecision.SIMILAR:
            existing_memory = _require_matched_memory(decision, existing_memory)
            if (
                existing_memory.meta.access_policy == memory.meta.access_policy
                and existing_memory.index.memory_type != MemoryType.AGENT_PROFILE
                and memory.index.memory_type != MemoryType.AGENT_PROFILE
                and str(existing_memory.id) not in kept_apart_from
            ):
                merge_target_id = str(existing_memory.id)

        return [
            GenerationOutcome(
                atom=memory,
                action=GenerationAction.CREATE,
                merge_target_id=merge_target_id,
            )
        ]

    def _render_transcript(self, request: GenerationRequest) -> str:
        """
        统一 transcript 渲染入口。

        Returns:
            str: 渲染后的 transcript 文本，无有效内容时返回 "(无背景对话)"
        """
        if not request.context.turns and not request.context.state_summary:
            return "(无背景对话)"
        # 延迟导入：prompts 在依赖层级上高于 engines，模块加载期 import 会形成
        # engines.generation ↔ prompts.transcript 循环。
        from hivememory.prompts.transcript import GenerationTranscriptBuilder

        builder = GenerationTranscriptBuilder()
        return builder.build_transcript(request.context)

    def _draft_to_memory(
        self,
        draft: ExtractedMemoryDraft,
        belong_to: WorkspaceIdentity,
        provenance: MemoryProvenance,
        *,
        alias: str | None,
        now: datetime,
    ) -> MemoryAtom:
        """
        将草稿转换为完整的 MemoryAtom

        Args:
            draft: 提取的草稿
            belong_to: 已验证的 Workspace ownership 来源
            provenance: 本次生成的来源裁定；来源字段只记录 provenance，
                不参与读取授权
            alias: 已由 ``AliasGenerator`` 确认 Workspace 内空闲的别名

        Returns:
            MemoryAtom: 记忆原子对象

        Examples:
            >>> memory = engine._draft_to_memory(
            ...     draft, belong_to, provenance, alias="code_quicksort", now=now
            ... )
            >>> memory.index.title
            "Python 快排算法"
        """
        # 映射字符串类型到枚举
        try:
            mem_type = MemoryType(draft.memory_type)
        except ValueError:
            logger.warning("未知的记忆类型: %s, 使用 FACT", draft.memory_type)
            mem_type = MemoryType.FACT

        # 创建时点 = 提交边界 now（Familiar 传入）：created/updated/decay 同值。
        return MemoryAtom(
            meta=MetaData(
                workspace_identity=belong_to,
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


def _require_matched_memory(
    decision: DuplicateDecision,
    existing_memory: MemoryAtom | None,
) -> MemoryAtom:
    """IDENTICAL/SIMILAR 判断必须携带命中的已有记忆；违反查重器契约时显式失败。"""
    if existing_memory is None:
        raise RuntimeError(f"查重判断 {decision.value} 未返回命中的已有记忆")
    return existing_memory


__all__ = [
    "MemoryGenerationEngine",
]
