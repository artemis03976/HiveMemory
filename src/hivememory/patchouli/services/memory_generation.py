"""Patchouli 记忆生成使魔。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import ValidationError

from hivememory.components.cancellation import join_shielded
from hivememory.components.serial_gate import KeyedSerialGate
from hivememory.core.errors import (
    InvalidMemoryFieldError,
    MemoryMergeAbortedError,
    MemoryUpdateBaseChangedError,
    ResourceNotFoundError,
    WorkspaceMismatchError,
)
from hivememory.core.memory_access import memory_is_readable
from hivememory.core.models import (
    ActorIdentity,
    IndexLayer,
    MemoryAccessPolicy,
    MemoryAtom,
    MemoryLifecycleState,
    MemoryType,
    MetaData,
    PayloadLayer,
    PendingAtomResolution,
    PendingAtomSettlement,
    WorkspaceIdentity,
    WorkspaceMemoryKey,
)
from hivememory.core.models.artifact import (
    ArtifactRef,
    ArtifactType,
    CitedSegment,
    MemoryEventLog,
    MemoryEventType,
    snapshot_memory_atom,
)
from hivememory.core.models.provenance import MemoryProvenance
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.engines.artifacts.memory import MemoryCreationBundle
from hivememory.engines.generation.alias import fallback_alias
from hivememory.engines.generation.edits import EditSetError, rebase_edit_set
from hivememory.engines.generation.models import (
    GenerationAction,
    GenerationContext,
    GenerationOutcome,
    GenerationRequest,
    MergeFocus,
)
from hivememory.engines.memory_compiler import embedding_inputs_changed
from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationResult,
    MemoryGenerationSource,
    MemoryGenerationTaskSpec,
)
from hivememory.patchouli.services.source_artifacts import SourceArtifactRecorder
from hivememory.utils.time import require_utc, utc_now

if TYPE_CHECKING:
    from hivememory.engines.artifacts.engine import ArtifactEngine
    from hivememory.engines.generation.engine import MemoryGenerationEngine
    from hivememory.memory_library.library import MemoryLibrary

logger = logging.getLogger(__name__)


class MemoryGenerationFamiliar:
    """记忆的唯一写入者：执行生成与合并，也承接管理页的手动创建与编辑。

    所有内容提交都经 ``_commit_outcome``：版本记录先于写入，版本与内容时间在提交边界推进。
    交互原文与附件的快照由 ``SourceArtifactRecorder`` 保存，它们是生成的来源，不是记忆。
    """

    def __init__(
        self,
        *,
        generation_engine: MemoryGenerationEngine,
        memory_library: MemoryLibrary,
        artifact_engine: ArtifactEngine | None = None,
        asset_reader: WorkspaceAssetReaderPort | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        from hivememory.engines.artifacts.engine import ArtifactEngine

        self._generation_engine = generation_engine
        self._mid_term = memory_library.mid_term
        self._artifact_engine = artifact_engine or ArtifactEngine.noop()
        # 交互原文与附件的快照：生成的来源，不是记忆。
        self._sources = SourceArtifactRecorder(
            artifact_engine=self._artifact_engine, asset_reader=asset_reader
        )
        # 完整内容写入路径的提交时点来源（局部注入，非全局时钟）。
        self._now = now or utc_now
        # 同一 Workspace 的合并依次执行，见 _run_merge。
        self._merge_gate = KeyedSerialGate[WorkspaceIdentity]()
        # 合并成功后“被吸收的记忆 → 吸收它的记忆”，只在进程内：之后以被吸收者为目标的合并
        # 沿它改指仍然存在的那条。重启后为空，改指不到的合并按目标缺失中止。
        self._absorbed_into: dict[UUID, UUID] = {}

        logger.info("MemoryGenerationFamiliar 初始化完成")

    # ============ 入口 ============

    async def execute(
        self,
        spec: MemoryGenerationTaskSpec,
        *,
        on_commit_started: Callable[[], None],
    ) -> list[MemoryGenerationResult]:
        """执行统一生成任务规范，只返回结果不发布事件。

        任务分两段：提交之前保存来源交互 Artifact、调用模型、查重与重验，除交互
        Artifact 外不写任何东西；提交写版本记录、中期存储与附件。开始提交之前同步调用
        ``on_commit_started``，提交这一段不被取消打断，写完后再传出取消：在提交之前被
        中止的任务可以从头重跑而不留下写了一半的记忆。
        """
        if spec.request.merge_focus is not None:
            return await self._run_merge(
                spec, spec.request.merge_focus, on_commit_started=on_commit_started
            )
        interaction_ref = await self._sources.capture_interaction(
            spec.interaction_input, spec.belong_to
        )
        return await self._run_generation(
            spec,
            interaction_ref=interaction_ref,
            on_commit_started=on_commit_started,
        )

    @staticmethod
    async def _commit_uninterrupted[T](
        commit: Coroutine[Any, Any, T], on_commit_started: Callable[[], None]
    ) -> T:
        """登记进入提交后执行提交；期间收到的取消等提交结束后再传出。"""
        on_commit_started()
        return await join_shielded(asyncio.create_task(commit))

    # ============ 生成：结算、WRITE 与 UPDATE ============

    async def _run_generation(
        self,
        spec: MemoryGenerationTaskSpec,
        *,
        interaction_ref: ArtifactRef | None,
        on_commit_started: Callable[[], None],
    ) -> list[MemoryGenerationResult]:
        """
        执行 compute -> commit -> artifacts -> persist 流水线。

        提交边界在本方法入口取一次 UTC now，随后传给引擎（内容日期）与
        全部字段/Artifact 赋值；TOUCH 走受限 patch，CREATE/UPDATE 走完整
        ``upsert``。版本记录是内容提交成功的前置条件。
        """
        commit_now = require_utc(self._now())

        # Step 1：纯计算，GenerationEngine 不负责持久化与版本分配。
        outcomes = await self._generation_engine.process(
            await self._with_current_update_base(spec),
            belong_to=spec.belong_to,
            from_actor=spec.from_actor,
            now=commit_now,
        )
        committable: list[GenerationOutcome] = []
        for outcome in outcomes:
            if outcome.action == GenerationAction.UPDATE:
                checked = await self._committable_revision(spec, outcome)
                if checked is None:
                    continue
                outcome = checked
            committable.append(outcome)
        return await self._commit_uninterrupted(
            self._commit_generation(spec, committable, commit_now, interaction_ref),
            on_commit_started,
        )

    async def _with_current_update_base(self, spec: MemoryGenerationTaskSpec) -> GenerationRequest:
        """主动 UPDATE 以执行时读到的版本为底计算编辑集；其余任务的请求原样交给引擎。

        任务可能排队很久：agent 的指令作用在此刻的内容上，排队期间别处的修改模型也看得到，
        提交前的重放因此只需覆盖模型调用期间的修改。基础已删除或对发起者不可读时不修订：
        发起者不能修订它读不到的记忆，写回也会让已删除的记忆重新出现。
        """
        focus = spec.request.update_focus
        if focus is None:
            return spec.request
        current = await self._read_readable(spec, UUID(focus.base_uuid))
        if current is None:
            raise ResourceNotFoundError(
                "修订的基础记忆已不存在或不可读", details={"memory_id": focus.base_uuid}
            )
        return spec.request.model_copy(update={"existing_memory": current})

    async def _committable_revision(
        self, spec: MemoryGenerationTaskSpec, outcome: GenerationOutcome
    ) -> GenerationOutcome | None:
        """修订提交前对上基础记忆的当前版本，返回可以提交的修订；丢弃时返回 None。

        对不上时按路径对调用方的承诺处理：

        - 主动 UPDATE 对应 agent 的一次请求，失败要让 agent 知道：基础缺失或不可读以
          ``ResourceNotFoundError``、编辑集重放不了以 ``MemoryUpdateBaseChangedError`` 让任务失败；
        - 馆长的修订来自一次结算，同一次结算还有别的结果，一条失败不应连累其余：丢弃这一条，
          记一行 warning。
        """
        base = outcome.memory_before_snapshot
        if base is None:
            raise RuntimeError("修订结果缺少修改前快照")
        from_update = spec.request.update_focus is not None
        try:
            checked = await self._onto_current_base(spec, outcome)
        except EditSetError as error:
            if from_update:
                raise MemoryUpdateBaseChangedError(
                    details={"memory_id": str(base.id), "base_version": base.meta.version}
                ) from error
            logger.warning("馆长的修订在提交前丢弃: memory_id=%s, reason=base_changed", base.id)
            return None
        if checked is None:
            if from_update:
                raise ResourceNotFoundError(
                    "修订的基础记忆已不存在或不可读", details={"memory_id": str(base.id)}
                )
            logger.warning("馆长的修订在提交前丢弃: memory_id=%s, reason=missing", base.id)
        return checked

    async def _commit_generation(
        self,
        spec: MemoryGenerationTaskSpec,
        outcomes: list[GenerationOutcome],
        commit_now: datetime,
        interaction_ref: ArtifactRef | None,
    ) -> list[MemoryGenerationResult]:
        memories = [outcome.atom for outcome in outcomes if outcome.atom is not None]
        logger.info(
            "提取出 %d 条记忆: topic_id=%s, intent_id=%s",
            len(memories),
            spec.topic_id,
            spec.intent_id,
        )

        # Step 2/3：按 outcome 依次完成提交字段分配、artifact 挂载与持久化。
        for outcome in outcomes:
            if outcome.atom is None:
                continue
            await self._commit_outcome(
                outcome,
                belong_to=spec.belong_to,
                now=commit_now,
                interaction_ref=interaction_ref,
                gen_context=spec.request.context,
                creation_source=spec.source.creation_artifact_intent,
                update_source=spec.source.version_update_source,
            )

        # Step 4：只有确实产生 Memory CREATE/UPDATE 时才对 topic
        # bindings 做附件 Artifact promotion；TOUCH 与纯上传/选择
        # 不提升。失败沿 best-effort 语义记录 warning，不回滚本轮结果。
        if any(
            outcome.action in {GenerationAction.CREATE, GenerationAction.UPDATE}
            for outcome in outcomes
        ):
            bindings = (
                spec.interaction_input.asset_bindings if spec.interaction_input is not None else ()
            )
            await self._sources.promote_attachments(bindings, belong_to=spec.belong_to)

        # 只有 artifact 与持久化均完成后，才把 Engine outcome 收缩为跨域事实；
        # settlement 随该结果交给控制面独立发布。
        return [self._build_generation_result(spec, outcome) for outcome in outcomes]

    def _build_generation_result(
        self,
        spec: MemoryGenerationTaskSpec,
        outcome: GenerationOutcome,
    ) -> MemoryGenerationResult:
        atom = outcome.atom
        canonical_alias = atom.get_alias() if atom is not None else None
        canonical_uuid = str(atom.id) if atom is not None else None
        merge_nomination = None
        if outcome.merge_target_id is not None and canonical_uuid is not None:
            # 新记忆已经完成持久化；WRITE 的首次结算仍为 CREATED，
            # 句柄身份只交给后续合并任务用于成功后的改指。
            is_write = spec.source == MemoryGenerationSource.WRITE
            merge_nomination = MergeFocus(
                retained_memory_id=outcome.merge_target_id,
                absorbed_memory_id=canonical_uuid,
                pending_alias=spec.pending_alias if is_write else None,
                intent_id=spec.intent_id if is_write else None,
            )
        return MemoryGenerationResult(
            action=outcome.action,
            canonical_alias=canonical_alias,
            canonical_uuid=canonical_uuid,
            merge_nomination=merge_nomination,
            settlement=self._build_settlement(
                spec,
                outcome.action,
                canonical_alias=canonical_alias,
                canonical_uuid=canonical_uuid,
            ),
        )

    def _build_settlement(
        self,
        spec: MemoryGenerationTaskSpec,
        action: GenerationAction,
        *,
        canonical_alias: str | None,
        canonical_uuid: str | None,
    ) -> PendingAtomSettlement | None:
        if not spec.intent_id or not spec.pending_alias:
            return None
        resolution = self._resolution_for(action)
        return PendingAtomSettlement(
            pending_alias=spec.pending_alias,
            intent_id=spec.intent_id,
            resolution=resolution,
            canonical_alias=canonical_alias,
            canonical_uuid=canonical_uuid,
            message=(f"Pending atom '{spec.pending_alias}' settled as " f"{resolution.value}."),
        )

    def _resolution_for(
        self,
        action: GenerationAction,
    ) -> PendingAtomResolution:
        if action == GenerationAction.CREATE:
            return PendingAtomResolution.CREATED
        if action == GenerationAction.TOUCH:
            return PendingAtomResolution.TOUCHED
        return PendingAtomResolution.UPDATED

    # ============ 合并 ============

    async def _run_merge(
        self,
        spec: MemoryGenerationTaskSpec,
        focus: MergeFocus,
        *,
        on_commit_started: Callable[[], None],
    ) -> list[MemoryGenerationResult]:
        """以保留记忆为底写入被吸收记忆独有的内容，再删除被吸收记忆；失败最多留下重复。

        同一 Workspace 的合并依次执行，从读取两条记忆到写回、删除都在门内：两个合并不会
        同时写同一条保留记忆，也不会在模型调用期间删掉对方的目标。几次提名串成一串时，
        无论哪个先执行，内容都依次进入同一条记忆。保留记忆已被另一次合并吸收时，沿吸收关系
        改指仍然存在的那条。编辑集带着错误重试一次后仍不能应用时中止，两条都保留。
        """
        async with self._merge_gate.hold(spec.belong_to):
            retained = await self._mid_term.get(
                spec.belong_to,
                self._surviving_memory_id(UUID(focus.retained_memory_id)),
                from_actor=spec.from_actor,
            )
            absorbed = await self._mid_term.get(
                spec.belong_to, UUID(focus.absorbed_memory_id), from_actor=spec.from_actor
            )
            if retained is None or absorbed is None:
                raise MemoryMergeAbortedError(details={"reason": "memory_missing"})
            self._require_mergeable(retained, absorbed)

            try:
                outcome = await self._generation_engine.merge(retained, absorbed)
            except EditSetError as error:
                raise MemoryMergeAbortedError(details={"reason": "model_output_invalid"}) from error
            if outcome is None:
                raise MemoryMergeAbortedError(details={"reason": "model_output_invalid"})

            # 门只串行合并；模型调用期间仍可能有修订、删除或策略变更，写回前对上两条的当前版本。
            try:
                merged = await self._onto_current_base(spec, outcome)
            except EditSetError as error:
                raise MemoryMergeAbortedError(
                    details={"reason": "version_changed", "memory_id": str(retained.id)}
                ) from error
            current_absorbed = await self._read_readable(spec, absorbed.id)
            if merged is None or current_absorbed is None:
                missing = retained if merged is None else absorbed
                raise MemoryMergeAbortedError(
                    details={"reason": "memory_missing", "memory_id": str(missing.id)}
                )
            if current_absorbed.meta.version != absorbed.meta.version:
                # 被吸收的内容已经不同，按旧内容算出的编辑集不再代表它，不重放。
                raise MemoryMergeAbortedError(
                    details={"reason": "version_changed", "memory_id": str(absorbed.id)}
                )
            atom = merged.atom
            if atom is None:
                raise RuntimeError("合并计算未返回待提交的记忆")
            self._require_mergeable(atom, current_absorbed)
            for ref in absorbed.payload.artifacts.refs:
                if ref.artifact_type == ArtifactType.INTERACTION:
                    self._append_artifact_ref_once(atom, ref)
            return await self._commit_uninterrupted(
                self._commit_merge(spec, focus, merged, retained, absorbed), on_commit_started
            )

    def _surviving_memory_id(self, memory_id: UUID) -> UUID:
        """沿进程内的吸收关系找到吸收了 ``memory_id`` 的那条；没有被吸收过时就是它自己。"""
        surviving = memory_id
        while surviving in self._absorbed_into:
            surviving = self._absorbed_into[surviving]
        if surviving != memory_id:
            logger.info(
                "合并目标已被吸收，改指仍然存在的记忆: memory_id=%s, redirected_to=%s",
                memory_id,
                surviving,
            )
        return surviving

    async def _commit_merge(
        self,
        spec: MemoryGenerationTaskSpec,
        focus: MergeFocus,
        outcome: GenerationOutcome,
        retained: MemoryAtom,
        absorbed: MemoryAtom,
    ) -> list[MemoryGenerationResult]:
        await self._commit_outcome(
            outcome,
            belong_to=spec.belong_to,
            now=require_utc(self._now()),
            interaction_ref=None,
            gen_context=spec.request.context,
            creation_source=spec.source.creation_artifact_intent,
            update_source=spec.source.version_update_source,
            source_artifact_refs=absorbed.payload.artifacts.refs,
        )

        await self._mid_term.delete_by_key(
            WorkspaceMemoryKey(workspace_identity=spec.belong_to, memory_id=absorbed.id)
        )
        self._absorbed_into[absorbed.id] = retained.id
        settlement = None
        if focus.pending_alias is not None and focus.intent_id is not None:
            settlement = PendingAtomSettlement(
                pending_alias=focus.pending_alias,
                intent_id=focus.intent_id,
                resolution=PendingAtomResolution.MERGED,
                canonical_alias=retained.get_alias(),
                canonical_uuid=str(retained.id),
            )
        return [
            MemoryGenerationResult(
                # 合并写回保留记忆的新版本，对控制面而言是一次修订。
                action=GenerationAction.UPDATE,
                canonical_alias=retained.get_alias(),
                canonical_uuid=str(retained.id),
                settlement=settlement,
                absorbed_memory_uuid=str(absorbed.id),
            )
        ]

    @staticmethod
    def _require_mergeable(retained: MemoryAtom, absorbed: MemoryAtom) -> None:
        """不同策略不能融合；同一条记忆也不能吸收自身。"""
        if retained.id == absorbed.id or retained.meta.access_policy != absorbed.meta.access_policy:
            raise MemoryMergeAbortedError(details={"reason": "not_mergeable"})

    # ============ 基础变化规则 ============

    async def _onto_current_base(
        self, spec: MemoryGenerationTaskSpec, outcome: GenerationOutcome
    ) -> GenerationOutcome | None:
        """把修订或合并的结果对上基础记忆此刻的版本，返回可以提交的结果。

        馆长的修订、主动 UPDATE 与合并的保留记忆共用这一条规则：基础的内容版本没变就照原样
        提交；变了就把编辑集重放到当前版本上，期间别处的修改只要没碰到编辑引用的原文就得以
        保留。访问统计与读取策略的变动不推进内容版本，结果沿用此刻的值，不被整条写回改回去。
        基础已删除或对发起者不可读时返回 None；怎样失败由各条路径决定。

        Raises:
            EditSetError: 版本变了，编辑引用的原文在当前版本中已经对不上
        """
        atom = outcome.atom
        base = outcome.memory_before_snapshot
        if atom is None or base is None or outcome.edit_set is None:
            raise RuntimeError("修订结果缺少待提交的记忆、修改前快照或编辑集")
        current = await self._read_readable(spec, base.id)
        if current is None:
            return None
        if current.meta.version != base.meta.version:
            # 重放的结果以当前版本为底，访问统计与读取策略也就是此刻的值。
            rebased = _rebased(outcome, current)
            logger.info(
                "修订的基础已变，编辑集已重放到当前版本: memory_id=%s, base_version=%d, "
                "current_version=%d",
                base.id,
                base.meta.version,
                current.meta.version,
            )
            return rebased
        atom.meta.lifecycle = current.meta.lifecycle.model_copy(deep=True)
        atom.meta.access_policy = current.meta.access_policy.model_copy(deep=True)
        return outcome

    async def _read_readable(
        self, spec: MemoryGenerationTaskSpec, memory_id: UUID
    ) -> MemoryAtom | None:
        """按复合归属键读取一条记忆；已删除或对发起者不可读时返回 None。"""
        current = await self._mid_term.get_by_key(
            WorkspaceMemoryKey(workspace_identity=spec.belong_to, memory_id=memory_id)
        )
        if current is None or not memory_is_readable(
            current, workspace_identity=spec.belong_to, actor_identity=spec.from_actor
        ):
            return None
        return current

    # ============ 管理页的手动创建与编辑 ============

    async def create_external_memory(
        self,
        belong_to: WorkspaceIdentity,
        *,
        from_actor: ActorIdentity,
        memory_type: MemoryType,
        title: str,
        summary: str = "",
        content: str = "",
        tags: list[str] | None = None,
        alias: str | None = None,
        access_policy: MemoryAccessPolicy | None = None,
    ) -> MemoryAtom:
        """以外部提交的材料创建记忆原子并持久化。

        原子的身份、时间戳、来源记录与生命周期都在这里确定：创建时点取提交
        边界的同一时刻（created/updated/decay 同值），来源记录取本次发起者，
        只作展示、不参与可见性授权；没有给出读取策略时为 PUBLIC；没有给出
        别名时以 ``mem_<memory id>`` 命名。字段非法以
        ``InvalidMemoryFieldError`` 拒绝。
        """
        try:
            index = IndexLayer(
                title=title,
                summary=summary,
                tags=tags or [],
                memory_type=memory_type,
                alias=alias,
            )
            payload = PayloadLayer(content=content)
        except ValidationError as exc:
            raise InvalidMemoryFieldError.from_validation_error(exc) from exc
        commit_now = require_utc(self._now())
        atom = MemoryAtom(
            meta=MetaData(
                workspace_identity=belong_to,
                provenance=MemoryProvenance(
                    source_agent_id=from_actor.agent_id,
                    source_team_id=from_actor.team_id,
                ),
                access_policy=(
                    access_policy if access_policy is not None else MemoryAccessPolicy.public()
                ),
                created_at=commit_now,
                updated_at=commit_now,
                lifecycle=MemoryLifecycleState(decay_anchor_at=commit_now),
            ),
            index=index,
            payload=payload,
        )
        if atom.index.alias is None:
            # 没有给出别名时与生成路径一致，以这条记忆的 id 命名并落库，保证可按别名寻址。
            atom.index.alias = fallback_alias(atom.id)
        await self._commit_manual(
            GenerationOutcome(atom=atom, action=GenerationAction.CREATE),
            belong_to=belong_to,
            commit_now=commit_now,
        )
        return atom

    async def update_external_memory(
        self,
        memory_id: UUID,
        *,
        belong_to: WorkspaceIdentity,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        alias: str | None = None,
        tags: list[str] | None = None,
        access_policy: MemoryAccessPolicy | None = None,
    ) -> MemoryAtom | None:
        """对外部手动的记忆编辑进行持久化处理。

        内容字段有变化时提交新版本；读取策略有变化时随后单独替换，不推进版本、
        不写版本记录。策略必须在内容之后写：内容提交写回的是本次开始时读到的
        整条原子，先写的策略会被它改回旧值。两步之间没有原子保证，策略写入
        失败时内容已经提交，重新提交同一份编辑只会补上策略。
        """
        key = WorkspaceMemoryKey(workspace_identity=belong_to, memory_id=memory_id)
        atom = await self._mid_term.get_by_key(key)
        if atom is None:
            return None

        commit_now = require_utc(self._now())
        # 修改前的完整原子：版本记录的 snapshot_before 与向量输入比较都以它为准。
        before_snapshot = atom.model_copy(deep=True)
        changed_fields = self._apply_external_update(
            atom,
            title=title,
            summary=summary,
            content=content,
            alias=alias,
            tags=tags,
        )
        if changed_fields:
            edit = GenerationOutcome(
                atom=atom,
                action=GenerationAction.UPDATE,
                memory_before_snapshot=before_snapshot,
                changelog=_manual_changelog(changed_fields),
            )
            await self._commit_manual(edit, belong_to=belong_to, commit_now=commit_now)
        else:
            # 实际内容相同的重复更新不创建新版本。
            logger.debug("外部编辑未产生字段变化，跳过版本提交: %s", atom.id)

        if access_policy is None or access_policy == atom.meta.access_policy:
            return atom
        return await self._mid_term.patch_payload(key, {"meta.access_policy": access_policy})

    @staticmethod
    def _apply_external_update(
        atom: MemoryAtom,
        *,
        title: str | None,
        summary: str | None,
        content: str | None,
        alias: str | None,
        tags: list[str] | None,
    ) -> list[str]:
        """按字段应用外部编辑，返回实际发生变化的字段。

        赋值经模型校验与规范化（去首尾空白、空白 alias 转 None、tags 规范
        化），变化判断比较规范化后的值：实际内容相同的重复更新不创建
        新版本。非法取值以 ``InvalidMemoryFieldError`` 拒绝。
        """
        edits = (
            (atom.index, "title", title),
            (atom.index, "summary", summary),
            (atom.payload, "content", content),
            (atom.index, "alias", alias),
            (atom.index, "tags", tags),
        )
        changed_fields: list[str] = []
        for layer, field, value in edits:
            if value is None:
                continue
            before = getattr(layer, field)
            try:
                setattr(layer, field, value)
            except ValidationError as exc:
                raise InvalidMemoryFieldError.from_validation_error(exc) from exc
            if getattr(layer, field) != before:
                changed_fields.append(field)
        return changed_fields

    async def _commit_manual(
        self, outcome: GenerationOutcome, *, belong_to: WorkspaceIdentity, commit_now: datetime
    ) -> None:
        """手动创建与编辑与生成走同一条提交：版本记录先于写入，版本与内容时间在此推进。"""
        await self._commit_outcome(
            outcome,
            belong_to=belong_to,
            now=commit_now,
            interaction_ref=None,
            gen_context=GenerationContext(),
            creation_source="MANUAL",
            update_source="MANUAL_EDIT",
        )

    # ============ 提交：版本记录、Artifact 与写入 ============

    async def _commit_outcome(
        self,
        outcome: GenerationOutcome,
        *,
        belong_to: WorkspaceIdentity,
        now: datetime,
        interaction_ref: ArtifactRef | None,
        gen_context: GenerationContext,
        creation_source: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"],
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"],
        source_artifact_refs: list[ArtifactRef] | None = None,
    ) -> None:
        """单个 outcome 的提交流水线：字段分配 → artifact 挂载 → 持久化。

        - TOUCH：不改内容字段与版本，经受限 ``patch_payload`` 只推进访问
          计数与 ``last_accessed_at``。
        - CREATE/UPDATE：版本与内容时间在此提交边界分配；版本记录写入是
          内容提交成功的前置条件（builder 失败直接传播，不静默降级）。
        """
        atom = outcome.atom
        if atom is None:  # 调用方已跳过空 atom；此处守卫保证类型收窄。
            return
        action = outcome.action

        if action == GenerationAction.TOUCH:
            key = WorkspaceMemoryKey(
                workspace_identity=belong_to,
                memory_id=atom.id,
            )
            patched = await self._mid_term.patch_payload(
                key,
                {
                    "meta.lifecycle.access_count": atom.meta.lifecycle.access_count + 1,
                    "meta.lifecycle.last_accessed_at": now,
                },
            )
            if patched is None:
                logger.warning("TOUCH 目标记忆已不存在，跳过访问统计: memory_id=%s", atom.id)
                return
            return

        if action == GenerationAction.UPDATE:
            # 引擎是纯计算，outcome.atom 携带的即提交前版本；内容修订在此
            # 提交边界推进一次版本与内容时间，衰减基准随之推进。
            atom.meta.version += 1
            atom.meta.updated_at = now
            atom.meta.lifecycle.decay_anchor_at = now
            atom.meta.lifecycle.confidence_score = 1.0

        await self._attach_memory_artifact(
            atom=atom,
            action=action,
            memory_before_snapshot=outcome.memory_before_snapshot,
            changelog=outcome.changelog,
            gen_context=gen_context,
            interaction_ref=interaction_ref,
            creation_source=creation_source,
            update_source=update_source,
            now=now,
            source_artifact_refs=source_artifact_refs,
            cited_segments=outcome.cited_segments,
        )

        # 新建没有上一版向量可保留；修订与合并写回只在向量输入文本变化时重算。
        before = outcome.memory_before_snapshot
        recompute = before is None or embedding_inputs_changed(before, atom)
        await self._mid_term.upsert(atom, recompute_vectors=recompute)
        logger.info(
            "记忆已存储: memory_id=%s, action=%s, recompute_vectors=%s",
            atom.id,
            action.value,
            recompute,
        )

    async def _attach_memory_artifact(
        self,
        *,
        atom: MemoryAtom,
        action: GenerationAction,
        memory_before_snapshot: MemoryAtom | None,
        changelog: str | None,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None,
        creation_source: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"],
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"] = "UPDATE",
        now: datetime | None = None,
        source_artifact_refs: list[ArtifactRef] | None = None,
        cited_segments: list[CitedSegment] | None = None,
    ) -> None:
        """为单个已生成或外部编辑的 MemoryAtom 挂载 artifact。

        ``now`` 为提交边界时点：版本记录 ``changed_at`` 与事件 ``at`` 使用
        同一时点。全部内容提交（外部创建/编辑与生成
        CREATE/UPDATE）都经过这里，alias 唯一性在写 Artifact 之前检查：
        alias 冲突时不留下孤立版本记录（upsert 仍作为不变量兜底重验）。
        ``cited_segments`` 只写进这一版的版本记录，外部编辑与合并没有。
        """
        await self._mid_term.ensure_alias_available(atom)
        commit_now = require_utc(now) if now is not None else require_utc(self._now())

        src_refs = list(source_artifact_refs or ())
        if interaction_ref is not None:
            src_refs.append(interaction_ref)
        if action == GenerationAction.CREATE:
            bundle = await self._build_creation_artifacts(
                atom=atom,
                gen_context=gen_context,
                source_artifact_refs=src_refs,
                cited_segments=cited_segments or [],
                creation_source=creation_source,
                now=commit_now,
            )

            atom.payload.artifacts.events.append(
                MemoryEventLog(
                    event_type=MemoryEventType.CREATED,
                    at=commit_now,
                    artifact_refs=bundle.refs,
                )
            )

        elif action == GenerationAction.UPDATE:
            version_ref = await self._build_update_artifact(
                atom=atom,
                memory_before_snapshot=memory_before_snapshot,
                changelog=changelog,
                source_artifact_refs=src_refs,
                cited_segments=cited_segments or [],
                update_source=update_source,
                now=commit_now,
            )

            atom.payload.artifacts.events.append(
                MemoryEventLog(
                    event_type=MemoryEventType.VERSIONED,
                    at=commit_now,
                    artifact_refs=[version_ref] if version_ref else [],
                    note=changelog,
                )
            )

        self._append_artifact_ref_once(atom, interaction_ref)

    async def _build_creation_artifacts(
        self,
        *,
        atom: MemoryAtom,
        gen_context: GenerationContext,
        source_artifact_refs: list[ArtifactRef],
        cited_segments: list[CitedSegment],
        creation_source: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"],
        now: datetime,
    ) -> MemoryCreationBundle:
        """构建 v1 版本记录与创建 Artifact；失败直接传播：没有版本历史就不提交内容。"""
        bundle = await self._artifact_engine.memory.build_for_create(
            memory=atom,
            context=gen_context,
            source_intent=creation_source,
            source_artifact_refs=source_artifact_refs,
            cited_segments=cited_segments,
            now=now,
        )
        if bundle.initial_version_ref is None:
            raise RuntimeError(
                f"版本存储未产生 v1 版本记录，拒绝提交无历史内容: {atom.id}；"
                "请检查 patchouli.artifacts 配置（memory 组件必须启用）"
            )

        for ref in bundle.refs:
            self._append_artifact_ref_once(atom, ref)

        return bundle

    async def _build_update_artifact(
        self,
        *,
        atom: MemoryAtom,
        memory_before_snapshot: MemoryAtom | None,
        changelog: str | None,
        source_artifact_refs: list[ArtifactRef],
        cited_segments: list[CitedSegment],
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"],
        now: datetime,
    ) -> ArtifactRef:
        """构建版本记录；失败直接传播：版本记录是提交成功的前置条件。"""
        snapshot_before = (
            snapshot_memory_atom(memory_before_snapshot)
            if memory_before_snapshot is not None
            else None
        )
        version_ref = await self._artifact_engine.memory.build_for_update(
            memory_after=atom,
            snapshot_before=snapshot_before,
            update_source=update_source,
            changelog=changelog,
            source_artifact_refs=source_artifact_refs,
            cited_segments=cited_segments,
            now=now,
        )
        if version_ref is None:
            raise RuntimeError(
                f"版本存储未产生版本记录，拒绝提交无历史内容: {atom.id}；"
                "请检查 patchouli.artifacts 配置（memory 组件必须启用）"
            )

        self._append_artifact_ref_once(atom, version_ref)

        return version_ref

    @staticmethod
    def _append_artifact_ref_once(atom: MemoryAtom, ref: ArtifactRef | None) -> None:
        if ref is None:
            return
        if ref.workspace_identity != atom.workspace_identity:
            raise WorkspaceMismatchError(details={"artifact_id": ref.artifact_id})
        refs = atom.payload.artifacts.refs
        exists = any(
            existing.artifact_id == ref.artifact_id and existing.artifact_type == ref.artifact_type
            for existing in refs
        )
        if not exists:
            refs.append(ref)


__all__ = ["MemoryGenerationFamiliar"]


def _rebased(outcome: GenerationOutcome, current: MemoryAtom) -> GenerationOutcome:
    """把修订或合并携带的编辑集重放到当前版本上，以当前版本为底提交。

    Raises:
        EditSetError: 编辑引用的原文在当前版本中已经对不上
    """
    if outcome.atom is None or outcome.edit_set is None:
        raise RuntimeError("待重放的结果缺少修订后的记忆或编辑集")
    return outcome.model_copy(
        update={
            "atom": rebase_edit_set(outcome.atom, outcome.edit_set, current),
            "memory_before_snapshot": current.model_copy(deep=True),
        }
    )


def _manual_changelog(changed_fields: list[str]) -> str:
    if not changed_fields:
        return "Manual edit: metadata refreshed"
    return f"Manual edit: {', '.join(changed_fields)}"
