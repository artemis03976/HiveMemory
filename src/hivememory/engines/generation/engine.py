"""
HiveMemory - 记忆生成编排器 (Memory Generation Orchestrator)

职责:
    协调所有组件，执行完整的记忆生成流程。

工作流程:
    Step 1: LLM 提取 → ExtractedMemoryDraft（话题结算可以提取多条）
    Step 2: 逐条查重判断 → 新建、触碰已有记忆，或新建并提名合并
    Step 3: 记忆原子构建 → MemoryAtom
    Step 4: 返回 GenerationOutcome，外围 Familiar 负责持久化

话题结算选馆长时，Step 1-2 由馆长的探索循环代替：草稿全部新建，不经过查重；对已有记忆的
修订成为以馆长读到的版本为底的 UPDATE。

修订（馆长、Mode C）与合并都以编辑集改已有记忆：标题与摘要只做片段替换，标签逐个增删。

WRITE、修订与合并的模型输出不合格时，带着上一次的输出与被拒的说明再请模型一次
（``_ask_with_one_retry``）。馆长在自己的循环里把错误交回模型，话题提取的输出不合格时没有产出。
"""

import logging
from collections.abc import Awaitable, Callable, Collection
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import ValidationError

from hivememory.core.errors import ModelProviderError, WorkspaceMismatchError
from hivememory.core.models import (
    HISTORICAL_MEMORY_TYPES,
    ActorIdentity,
    CitedSegment,
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
from hivememory.engines.generation.edits import (
    EditSetError,
    InvalidOutputError,
    OutputRetry,
    apply_edit_set,
    describe_validation_error,
    parse_edit_set,
)
from hivememory.engines.generation.interfaces import (
    BaseDeduplicator,
    BaseMemoryExtractor,
)
from hivememory.engines.generation.librarian import Librarian
from hivememory.engines.generation.models import (
    AppendEdit,
    DuplicateDecision,
    ExtractedMemoryDraft,
    GenerationAction,
    GenerationContext,
    GenerationOutcome,
    GenerationRequest,
    MemoryEditSet,
    provenance_from_actor,
    system_settlement_provenance,
)
from hivememory.engines.memory_compiler import MemoryCompiler, MemoryCompileTarget
from hivememory.utils.json_parser import JSONParseError, parse_llm_json
from hivememory.utils.time import require_utc, utc_now

if TYPE_CHECKING:
    from hivememory.memory_library.stores import MidTermMemoryStore

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
        ...     mid_term=mid_term,
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
        librarian: Librarian | None = None,
    ):
        """
        Args:
            mid_term: 中期记忆存储（查重检索与 alias 唯一性查询）
            extractor: LLM 提取器
            deduplicator: 查重器
            alias_generator: Workspace 内唯一的别名生成器；缺省以 ``mid_term``
                作为占用查询端口构造
            librarian: 话题结算选馆长时注入；为 None 时结算走一次性生成（Mode A）
        """
        self._mid_term = mid_term
        self.extractor = extractor
        self.deduplicator = deduplicator
        self._alias_generator = alias_generator or AliasGenerator(mid_term)
        self._librarian = librarian
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
            注入了馆长时改由馆长结算，来源裁定相同；
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
        if self._librarian is not None:
            return await self._process_librarian(
                request, self._librarian, belong_to, from_actor=from_actor, now=commit_now
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

        查重命中本话题写入过的记忆时丢弃草稿：被动提取会从同一份材料里重新推导出
        agent 已经写入的内容，并入后会以较新的被动草稿为骨架改写它；话题再次沉淀时
        材料仍是全部交互，此前沉淀新建的记忆同样会被复述。

        每条草稿引用的段换算为事件区间，随新建的记忆交给提交。
        """
        logger.debug("[Mode A] 开始处理...")

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
                cited_segments=_cite(request.context, draft.source_segments),
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

    async def _process_librarian(
        self,
        request: GenerationRequest,
        librarian: Librarian,
        belong_to: WorkspaceIdentity,
        *,
        from_actor: ActorIdentity,
        now: datetime,
    ) -> list[GenerationOutcome]:
        """话题结算的馆长方式：馆长先看本话题的写入与库中相关记忆，再暂存草稿与修订。

        馆长在写之前已经查看过两者，暂存的草稿全部按新建提交，不查重、不触碰、
        不提名合并；残余的重叠接受。来源裁定、别名分配与段引用与 Mode A 相同。
        修订以馆长读到的版本为底应用编辑集，成为 UPDATE；结算的贡献者并入那条记忆。
        """
        result = await librarian.settle(
            self._render_transcript(request),
            topic_write_memory_ids=request.topic_write_memory_ids,
            belong_to=belong_to,
            from_actor=from_actor,
        )
        provenance = system_settlement_provenance(request.context)
        outcomes: list[GenerationOutcome] = []
        taken_aliases: set[str] = set()
        for draft in result.drafts:
            memory = self._draft_to_memory(draft, belong_to, provenance, alias=None, now=now)
            alias = await self._alias_generator.generate(
                workspace_identity=belong_to,
                memory_type=draft.memory_type,
                alias_suffix=draft.alias_suffix,
                title=draft.title,
                memory_id=memory.id,
                taken=taken_aliases,
            )
            memory.index.alias = alias
            taken_aliases.add(alias)
            outcomes.append(
                GenerationOutcome(
                    atom=memory,
                    action=GenerationAction.CREATE,
                    cited_segments=_cite(request.context, draft.source_segments),
                )
            )
        outcomes.extend(
            _revision_outcome(
                revision.base,
                revision.revised,
                revision.edit_set,
                contributors=provenance.contributing_agent_ids,
                cited_segments=_cite(request.context, revision.edit_set.source_segments),
            )
            for revision in result.revisions
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

        Agent 明确要求保存，以 WriteFocus 为核心，对话历史为背景。模型的输出不能用作草稿时
        带着错误重试一次；仍不能用、提供商失败或没有模型时直接从 WriteFocus 构建兜底草稿，
        明确的保存意图不因模型暂时失败而丢失；程序错误不降级。兜底草稿没有模型引用的段。

        相似命中本话题写入过的记忆时照常新建、不提名合并：同一话题里 agent 分开写的
        记忆是有意分开的。
        """
        logger.debug("[Mode B] WRITE 主动响应: content_length=%d", len(focus.content))

        draft = await self._write_draft(request, focus)
        if draft is None:
            draft = self._build_fallback_draft(focus)

        # Step 2-4: 查重 → 构建/更新 → 返回 outcome
        return await self._dedup_and_resolve(
            draft,
            belong_to,
            provenance_from_actor(from_actor, request.context),
            from_actor=from_actor,
            now=now,
            cited_segments=_cite(request.context, draft.source_segments),
            kept_apart_from=request.topic_write_memory_ids,
        )

    async def _write_draft(
        self, request: GenerationRequest, focus: WriteFocus
    ) -> ExtractedMemoryDraft | None:
        """请模型把 WRITE 写成草稿；需要兜底时返回 None，并记下原因。

        明确的保存意图强制入库：草稿的 has_value 为真、置信度为 1.0，不采用模型的判断。
        """
        # 空上下文时由 _render_transcript() 统一返回占位文本
        transcript = self._render_transcript(request)

        async def ask(retry: OutputRetry | None) -> str | None:
            return await self.extractor.extract_write(
                transcript,
                content=focus.content,
                reason=focus.reason,
                title=focus.title,
                retry=retry,
            )

        try:
            draft = await _ask_with_one_retry(ask, _parse_write_draft)
        except ModelProviderError as error:
            logger.warning(
                "[Mode B] 写入改用兜底草稿: reason=model_provider_failed, error_class=%s",
                type(error).__name__,
            )
            return None
        except InvalidOutputError:
            logger.warning("[Mode B] 写入改用兜底草稿: reason=invalid_output")
            return None
        if draft is None:
            logger.warning("[Mode B] 写入改用兜底草稿: reason=no_model")
            return None
        draft.has_value = True
        draft.confidence_score = 1.0
        return draft

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

        模型按指令与新材料写出对目标记忆的编辑集，以 ``existing_memory`` 为底应用；不合格时
        带着错误重试一次。空编辑集表示不需要改，降为 TOUCH。

        仍不合格、提供商失败或没有模型时回退：带内容就把内容作为带日期的更新段追加到正文，
        只有指令就不改（TOUCH）。UPDATE 的 ACK 表示意图已被接纳，内容不因模型没能写成
        编辑集而丢失；回退只改正文、不改索引，没有模型引用的段。程序错误不降级。
        """
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

        logger.debug(
            "[Mode C] UPDATE 修订: alias=%s, instruction_length=%d",
            uf.base_alias,
            len(uf.instruction),
        )

        edit_set, revised = await self._revise(request, uf, existing) or _update_fallback(
            uf, existing, now=now
        )
        if edit_set.is_empty:
            logger.info("[Mode C] UPDATE 不需要改，降级为 TOUCH: memory_id=%s", existing.id)
            return [GenerationOutcome(atom=existing, action=GenerationAction.TOUCH)]

        # 主动 UPDATE 以发起 Agent 及上下文贡献者演化已有记忆的贡献者集合。
        provenance = provenance_from_actor(from_actor, request.context)
        return [
            _revision_outcome(
                existing,
                revised,
                edit_set,
                contributors=provenance.contributing_agent_ids,
                cited_segments=_cite(request.context, edit_set.source_segments),
            )
        ]

    async def _revise(
        self, request: GenerationRequest, uf: UpdateFocus, existing: MemoryAtom
    ) -> tuple[MemoryEditSet, MemoryAtom] | None:
        """请模型写出通过校验的编辑集；需要回退时返回 None，并记下原因。"""
        transcript = self._render_transcript(request)

        async def ask(retry: OutputRetry | None) -> str | None:
            return await self.extractor.revise(
                existing,
                instruction=uf.instruction,
                new_content=uf.content,
                transcript=transcript,
                retry=retry,
            )

        try:
            accepted = await _ask_with_one_retry(
                ask, lambda output: _accept(output, existing, adds_tags_only=False)
            )
        except ModelProviderError as error:
            logger.warning(
                "[Mode C] 修订改用回退: memory_id=%s, reason=model_provider_failed, "
                "error_class=%s",
                existing.id,
                type(error).__name__,
            )
            return None
        except EditSetError:
            logger.warning(
                "[Mode C] 修订改用回退: memory_id=%s, reason=invalid_edit_set", existing.id
            )
            return None
        if accepted is None:
            logger.warning("[Mode C] 修订改用回退: memory_id=%s, reason=no_model", existing.id)
        return accepted

    async def merge(self, retained: MemoryAtom, absorbed: MemoryAtom) -> GenerationOutcome | None:
        """以保留记忆为底的修订：把被吸收记忆独有的内容写成对保留记忆的编辑集，纯计算。

        标签只增加；类型、来源主体、别名与访问状态沿用保留记忆，
        贡献者取两边的并集。空编辑集表示被吸收的内容保留记忆都已有：内容不变，仍产出一个
        记下这次吸收的版本。不修改输入；版本与持久化由 Familiar 提交。未启用模型时没有产出。

        Raises:
            EditSetError: 带着错误重试一次后仍不能应用
        """

        async def ask(retry: OutputRetry | None) -> str | None:
            return await self.extractor.merge(retained, absorbed, retry=retry)

        accepted = await _ask_with_one_retry(
            ask, lambda output: _accept(output, retained, adds_tags_only=True)
        )
        if accepted is None:
            return None
        edit_set, merged = accepted
        return _revision_outcome(
            retained,
            merged,
            edit_set,
            contributors=absorbed.meta.provenance.contributing_agent_ids,
            cited_segments=[],
        )

    async def _dedup_and_resolve(
        self,
        draft: ExtractedMemoryDraft,
        belong_to: WorkspaceIdentity,
        provenance: MemoryProvenance,
        *,
        from_actor: ActorIdentity,
        now: datetime,
        cited_segments: list[CitedSegment],
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
            cited_segments: 草稿引用的段，只随新建交给提交；触碰没有新版本，不带
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
                "草稿命中本话题写入过的记忆，丢弃: memory_id=%s, decision=%s, score=%.3f",
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

        # 相似时也先保留独立的新记忆；策略不同时不提名合并，避免扩大可见范围；
        # 命中 kept_apart_from 中的记忆时也不提名。
        merge_target_id = None
        if decision == DuplicateDecision.SIMILAR:
            existing_memory = _require_matched_memory(decision, existing_memory)
            if (
                existing_memory.meta.access_policy == memory.meta.access_policy
                and str(existing_memory.id) not in kept_apart_from
            ):
                merge_target_id = str(existing_memory.id)

        return [
            GenerationOutcome(
                atom=memory,
                action=GenerationAction.CREATE,
                merge_target_id=merge_target_id,
                cited_segments=cited_segments,
            )
        ]

    def _render_transcript(self, request: GenerationRequest) -> str:
        """
        统一 transcript 渲染入口。

        Returns:
            str: 渲染后的 transcript 文本，无有效内容时返回 "(无背景对话)"
        """
        if not request.context.turns and not request.context.fold_summary:
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
            ...     draft, belong_to, provenance, alias="fact_quicksort", now=now
            ... )
            >>> memory.index.title
            "Python 快排算法"
        """
        mem_type = _producible_memory_type(draft.memory_type)
        if mem_type is None:
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


async def _ask_with_one_retry[T](
    ask: Callable[[OutputRetry | None], Awaitable[str | None]],
    accept: Callable[[str], T],
) -> T | None:
    """请模型输出一次并交给 ``accept``；被拒时带着上一次的输出与说明再请一次。

    WRITE、修订与合并没有工具循环，一次带错误的重试是能给模型的最小反馈。重试发生在同一次
    生成之内、提交之前，不重复任何写入。没有模型时返回 None。

    Raises:
        InvalidOutputError: 第二次的输出仍被拒绝
    """
    output = await ask(None)
    if output is None:
        return None
    try:
        return accept(output)
    except InvalidOutputError as rejection:
        logger.info("模型输出不合格，带着错误重试一次")
        retried = await ask((output, str(rejection)))
    return accept(retried or "")


def _parse_write_draft(output: str) -> ExtractedMemoryDraft:
    """把 WRITE 的模型输出解析为草稿；说明与编辑集的解析写法相同，是英文的原因与字段路径。

    Raises:
        InvalidOutputError: 输出不是一个 JSON 对象，或不符合草稿的结构
    """
    try:
        data = parse_llm_json(output)
    except JSONParseError as error:
        raise InvalidOutputError("the output is not a JSON object") from error
    if not isinstance(data, dict):
        raise InvalidOutputError("the output is not a JSON object")
    try:
        return ExtractedMemoryDraft.model_validate(data)
    except ValidationError as error:
        raise InvalidOutputError(describe_validation_error(error)) from error


def _accept(
    output: str, base: MemoryAtom, *, adds_tags_only: bool
) -> tuple[MemoryEditSet, MemoryAtom]:
    """解析模型输出并以 ``base`` 为底应用；空编辑集表示不需要改。

    合并的编辑集只增加标签：被吸收记忆的内容补进来，不拿走保留记忆原有的标签。
    """
    edit_set = parse_edit_set(output)
    if adds_tags_only and edit_set.remove_tags:
        raise EditSetError("a merge only adds tags; remove_tags must be empty")
    if edit_set.is_empty:
        return edit_set, base.model_copy(deep=True)
    return edit_set, apply_edit_set(base, edit_set)


def _revision_outcome(
    base: MemoryAtom,
    revised: MemoryAtom,
    edit_set: MemoryEditSet,
    *,
    contributors: Collection[str],
    cited_segments: list[CitedSegment],
) -> GenerationOutcome:
    """把以 ``base`` 为底应用编辑集得到的 ``revised`` 包成待提交的修订（纯计算）。

    本次的贡献者并入已有集合，来源主体按约定保留不改写。版本、内容时间与置信度在
    Familiar 的提交边界赋值；提交前基础的版本变了时，Familiar 以携带的编辑集重放到当前版本上。
    """
    revised.meta.provenance.contributing_agent_ids = normalize_contributing_agent_ids(
        [*base.meta.provenance.contributing_agent_ids, *contributors]
    )
    return GenerationOutcome(
        atom=revised,
        action=GenerationAction.UPDATE,
        memory_before_snapshot=base.model_copy(deep=True),
        changelog=edit_set.changelog,
        cited_segments=cited_segments,
        edit_set=edit_set,
    )


def _update_fallback(
    uf: UpdateFocus, existing: MemoryAtom, *, now: datetime
) -> tuple[MemoryEditSet, MemoryAtom]:
    """UPDATE 的回退：带内容就追加一段带日期的更新，只有指令就是空编辑集（不改）。"""
    if not uf.content:
        unchanged = MemoryEditSet(changelog=f"Fallback (无变更): {uf.instruction[:80]}")
        return unchanged, existing.model_copy(deep=True)
    section = f"## 更新 ({now.strftime('%Y-%m-%d')})\n{uf.content}"
    appended = MemoryEditSet(
        edits=[AppendEdit(op="append", text=section)],
        changelog=f"Fallback 追加: {uf.instruction[:80]}",
    )
    return appended, apply_edit_set(existing, appended)


def _cite(context: GenerationContext, labels: list[str]) -> list[CitedSegment]:
    """把模型引用的段标签换算为事件区间；材料中不存在的标签丢弃并记一条日志。"""
    cited = context.cited_segments(labels)
    # 一个标签只对应一段，去重后的标签数与换算出的段数之差就是不存在的标签数。
    dropped = len(set(labels)) - len(cited)
    if dropped:
        logger.info("模型引用了材料中不存在的段，已丢弃: count=%d", dropped)
    return cited


def _producible_memory_type(value: str) -> MemoryType | None:
    """模型给出的类型名对应可以产生的类型时返回它，未知名与只作历史取值的类型返回 None。

    旧类型名留在枚举中只为解码历史，模型照旧写出它们时不能借此产生旧类型。
    """
    try:
        memory_type = MemoryType(value)
    except ValueError:
        return None
    if memory_type in HISTORICAL_MEMORY_TYPES:
        return None
    return memory_type


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
