"""Patchouli 记忆生成使魔。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import ValidationError

from hivememory.core.errors import (
    InvalidMemoryFieldError,
    WorkspaceDomainError,
    WorkspaceMismatchError,
)
from hivememory.core.models import (
    IdentityScope,
    MemoryAtom,
    MemoryType,
    PendingAtomResolution,
    PendingAtomSettlement,
    WorkspaceMemoryKey,
    require_identity_scope,
)
from hivememory.core.models.artifact import (
    ArtifactRef,
    MemoryEventLog,
    MemoryEventType,
    snapshot_memory_atom,
)
from hivememory.core.models.workspace_asset import TopicAssetBinding
from hivememory.engines.artifacts.memory import MemoryCreationBundle
from hivememory.engines.generation.models import (
    DuplicateDecision,
    GenerationContext,
    GenerationOutcome,
)
from hivememory.patchouli.control.memory_generation.models import (
    InteractionArtifactInput,
    MemoryGenerationResult,
    MemoryGenerationSource,
    MemoryGenerationTaskSpec,
)
from hivememory.system.runtime.workspace.ports import WorkspaceAssetReaderPort
from hivememory.utils.time import require_utc, utc_now

if TYPE_CHECKING:
    from hivememory.engines.artifacts.engine import ArtifactEngine
    from hivememory.engines.generation.engine import MemoryGenerationEngine
    from hivememory.patchouli.memory_library.library import MemoryLibrary

logger = logging.getLogger(__name__)


class MemoryGenerationFamiliar:
    """记忆生成数据面，负责执行生成、挂载 artifact 并写入中期记忆库。"""

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
        # W1-F：附件 promotion 的只读 reader（assembler 经 PatchouliRuntime
        # 注入进程级唯一 Store）；ref 失效或 Store 关闭时 best-effort 降级。
        self._asset_reader = asset_reader
        # A2-P 时间边界：完整内容写入路径的提交时点来源（局部注入，非全局时钟）。
        self._now = now or utc_now

        logger.info("MemoryGenerationFamiliar 初始化完成")

    @staticmethod
    def _embedding_inputs_changed(
        before: MemoryAtom | dict | None,
        after: MemoryAtom,
    ) -> bool:
        """判断一次完整提交是否改变了 embedding 输入。

        embedding 只编译 ``index.title/memory_type/tags/summary``；仅
        ``payload.agent_config``、关系或 lifecycle 变化时无需重算向量。
        ``before`` 可为完整原子或其 canonical JSON dict。
        """
        if before is None:
            return True
        if isinstance(before, MemoryAtom):
            before_index = before.index
            old: tuple[str, str, list[str], str] = (
                before_index.title,
                before_index.memory_type.value,
                sorted(before_index.tags or []),
                before_index.summary,
            )
        elif isinstance(before, dict):
            raw_index: dict[str, Any] = before.get("index") or {}
            raw_type = raw_index.get("memory_type")
            memory_type = str(getattr(raw_type, "value", raw_type))
            old = (
                str(raw_index.get("title")),
                memory_type,
                sorted(str(tag) for tag in (raw_index.get("tags") or [])),
                str(raw_index.get("summary")),
            )
        else:
            return True
        new = (
            after.index.title,
            after.index.memory_type.value,
            sorted(after.index.tags or []),
            after.index.summary,
        )
        return old != new

    async def execute(
        self,
        spec: MemoryGenerationTaskSpec,
    ) -> list[MemoryGenerationResult]:
        """
        执行统一生成任务规范，只返回结果不发布事件。
        """
        interaction_ref = await self._capture_interaction_artifact(
            spec.interaction_input,
            spec.identity_scope,
        )
        return await self._run_generation(
            spec,
            interaction_ref=interaction_ref,
        )

    async def create_external_memory(
        self,
        identity_scope: IdentityScope,
        atom: MemoryAtom,
    ) -> MemoryAtom:
        """
        对外部创建的记忆原子进行持久化处理。
        """
        identity_scope = require_identity_scope(identity_scope)
        if atom.workspace_identity != identity_scope.workspace_identity:
            raise WorkspaceMismatchError(details={"memory_id": str(atom.id)})
        commit_now = require_utc(self._now())
        # 提交边界决定创建时点：created/updated/decay 同值（M0.1）。
        atom.meta.created_at = commit_now
        atom.meta.updated_at = commit_now
        atom.meta.lifecycle.decay_anchor_at = commit_now
        await self._attach_memory_artifact(
            atom=atom,
            decision=DuplicateDecision.CREATE,
            memory_before_snapshot=None,
            changelog=None,
            gen_context=GenerationContext(),
            interaction_ref=None,
            creation_source="MANUAL",
            now=commit_now,
        )
        await self._mid_term.upsert(atom, recompute_vectors=True)
        return atom

    async def update_external_memory(
        self,
        memory_id: UUID,
        *,
        identity_scope: IdentityScope,
        title: str | None = None,
        summary: str | None = None,
        content: str | None = None,
        alias: str | None = None,
        tags: list[str] | None = None,
        agent_config: dict | None = None,
    ) -> MemoryAtom | None:
        """
        对外部手动的记忆编辑进行持久化处理。
        """
        identity_scope = require_identity_scope(identity_scope)
        atom = await self._mid_term.get_for_mutation(identity_scope, memory_id)
        if atom is None:
            return None

        commit_now = require_utc(self._now())
        # 修改前完整原子 canonical JSON；版本记录 snapshot_before 直接嵌入。
        before_snapshot = snapshot_memory_atom(atom)
        changed_fields = self._apply_external_update(
            atom,
            title=title,
            summary=summary,
            content=content,
            alias=alias,
            tags=tags,
            agent_config=agent_config,
        )
        if not changed_fields:
            # A2-P §3.2：实际内容相同的重复更新不创建新版本。
            logger.info("外部编辑未产生字段变化，跳过版本提交: %s", atom.id)
            return atom
        # 提交边界分配内容时间/版本/衰减基准/置信度（M0.1/M0.2：UPDATE 重置 1.0）。
        atom.meta.version += 1
        atom.meta.updated_at = commit_now
        atom.meta.lifecycle.decay_anchor_at = commit_now
        atom.meta.lifecycle.confidence_score = 1.0

        await self._attach_memory_artifact(
            atom=atom,
            decision=DuplicateDecision.UPDATE,
            memory_before_snapshot=before_snapshot,
            changelog=_manual_changelog(changed_fields),
            gen_context=GenerationContext(),
            interaction_ref=None,
            creation_source="MANUAL",
            update_source="MANUAL_EDIT",
            now=commit_now,
        )
        recompute = self._embedding_inputs_changed(before_snapshot, atom)
        await self._mid_term.upsert(atom, recompute_vectors=recompute)
        return atom

    @staticmethod
    def _apply_external_update(
        atom: MemoryAtom,
        *,
        title: str | None,
        summary: str | None,
        content: str | None,
        alias: str | None,
        tags: list[str] | None,
        agent_config: dict | None,
    ) -> list[str]:
        """按字段应用外部编辑，返回实际发生变化的字段。

        赋值经模型校验与规范化（去首尾空白、空白 alias 转 None、tags 规范
        化），变化判断比较规范化后的值——§3.2：实际内容相同的重复更新不创建
        新版本。非法取值以 ``InvalidMemoryFieldError`` 拒绝。
        """
        edits = (
            (atom.index, "title", title),
            (atom.index, "summary", summary),
            (atom.payload, "content", content),
            (atom.index, "alias", alias),
            (atom.index, "tags", tags),
            (atom.payload, "agent_config", agent_config),
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

    async def _run_generation(
        self,
        spec: MemoryGenerationTaskSpec,
        interaction_ref: ArtifactRef | None = None,
    ) -> list[MemoryGenerationResult]:
        """
        执行 compute -> commit -> artifacts -> persist 流水线。

        提交边界在本方法入口取一次 UTC now，随后传给引擎（内容日期）与
        全部字段/Artifact 赋值；TOUCH 走受限 patch，CREATE/UPDATE 走完整
        ``upsert``。版本记录是内容提交成功的前置条件（M0.3）。
        """
        commit_now = require_utc(self._now())

        # Step 1：纯计算，GenerationEngine 不负责持久化与版本分配。
        outcomes = await self._generation_engine.process(
            spec.request,
            identity_scope=spec.identity_scope,
            now=commit_now,
        )

        memories = [outcome.atom for outcome in outcomes if outcome.atom is not None]
        logger.info(f"Extracted {len(memories)} memories" if memories else "No memories extracted")

        # Step 2/3：按 outcome 依次完成提交字段分配、artifact 挂载与持久化。
        for outcome in outcomes:
            if outcome.duplicate_decision == DuplicateDecision.DISCARD or outcome.atom is None:
                continue
            await self._commit_outcome(
                outcome,
                identity_scope=spec.identity_scope,
                now=commit_now,
                interaction_ref=interaction_ref,
                gen_context=spec.request.context,
                creation_source=spec.source.creation_artifact_intent,
                update_source=spec.source.version_update_source,
            )

        # Step 4（W1-F）：只有确实产生 Memory CREATE/UPDATE 时才对 topic
        # bindings 做附件 Artifact promotion；TOUCH/DISCARD 与纯上传/选择
        # 不提升。失败沿 best-effort 语义记录 warning，不回滚本轮结果。
        if any(
            outcome.duplicate_decision in {DuplicateDecision.CREATE, DuplicateDecision.UPDATE}
            for outcome in outcomes
        ):
            bindings = (
                spec.interaction_input.asset_bindings if spec.interaction_input is not None else ()
            )
            await self._promote_attachment_bindings(
                bindings,
                identity_scope=spec.identity_scope,
            )

        # 只有 artifact 与持久化均完成后，才把 Engine outcome 收缩为跨域事实；
        # settlement 随该结果交给控制面独立发布。
        return [self._build_generation_result(spec, outcome) for outcome in outcomes]

    async def _promote_attachment_bindings(
        self,
        bindings: tuple[TopicAssetBinding, ...],
        *,
        identity_scope: IdentityScope,
    ) -> None:
        """沿 binding.asset_ref 提升附件 DocumentArtifact（best-effort）。

        每个 binding 固定为：acquire READY representation → 构建
        DocumentArtifact → 释放 lease。ref 已 remove、Store 已关闭或写入
        失败时跳过该 binding 并记录结构化 warning；已提交的 binding 保持
        不变，不回滚 Interaction/Memory 结果（计划 11.3 节）。promotion
        retry 复用现有 generation operation identity 与同一 binding payload。
        """
        if not bindings:
            return
        for binding in bindings:
            try:
                await self._promote_single_binding(
                    binding,
                    identity_scope=identity_scope,
                )
            except WorkspaceDomainError as exc:
                logger.warning(
                    "附件 promotion 跳过: asset_id=%s, code=%s",
                    binding.asset_id,
                    exc.code,
                )
            except Exception as exc:
                logger.warning(
                    "附件 promotion 写入失败: asset_id=%s, error=%s",
                    binding.asset_id,
                    type(exc).__name__,
                    exc_info=True,
                )

    async def _promote_single_binding(
        self,
        binding: TopicAssetBinding,
        *,
        identity_scope: IdentityScope,
    ) -> None:
        if self._asset_reader is None:
            return
        lease = self._asset_reader.acquire_ready_representation(
            identity_scope,
            binding.asset_ref,
        )
        try:
            representation = lease.representation
            content_format = ""
            if isinstance(representation.content_object, Mapping):
                content_format = str(representation.content_object.get("format") or "")
            source_type = "markdown" if content_format == "markdown" else "file"
            mime_type = "text/markdown" if content_format == "markdown" else "text/plain"
            # 冻结映射：source asset/representation 标识 + revision +
            # producer/version 全部钉进 source_uri，content_hash 单列，
            # 使提升产物自身锁定来源版本（ADR-0003 消费版本冻结不变量）。
            source_uri = (
                f"attachment://{binding.asset_id}"
                f"#{representation.representation_id}"
                f"?revision={representation.revision}"
                f"&producer={representation.producer}"
                f"&producer_version={representation.producer_version}"
            )
            await self._artifact_engine.document.build_and_store(
                source_type=source_type,
                source_uri=source_uri,
                content_hash=representation.content_hash,
                retrieved_at=datetime.now(UTC),
                workspace_identity=identity_scope.workspace_identity,
                mime_type=mime_type,
                title=f"attachment:{binding.asset_id}",
            )
        finally:
            self._asset_reader.release_representation_lease(lease.lease_id)

    def _build_generation_result(
        self,
        spec: MemoryGenerationTaskSpec,
        outcome: GenerationOutcome,
    ) -> MemoryGenerationResult:
        atom = outcome.atom
        canonical_alias = atom.get_alias() if atom is not None else None
        canonical_uuid = str(atom.id) if atom is not None else None
        return MemoryGenerationResult(
            canonical_alias=canonical_alias,
            canonical_uuid=canonical_uuid,
            settlement=self._build_settlement(
                spec,
                outcome.duplicate_decision,
                canonical_alias=canonical_alias,
                canonical_uuid=canonical_uuid,
            ),
        )

    def _build_settlement(
        self,
        spec: MemoryGenerationTaskSpec,
        decision: DuplicateDecision,
        *,
        canonical_alias: str | None,
        canonical_uuid: str | None,
    ) -> PendingAtomSettlement | None:
        if not spec.intent_id or not spec.pending_alias:
            return None
        resolution = self._resolution_for(spec, decision)
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
        spec: MemoryGenerationTaskSpec,
        decision: DuplicateDecision,
    ) -> PendingAtomResolution:
        if decision == DuplicateDecision.CREATE:
            return PendingAtomResolution.CREATED
        if decision == DuplicateDecision.TOUCH:
            return PendingAtomResolution.TOUCHED
        if decision == DuplicateDecision.UPDATE:
            if spec.source == MemoryGenerationSource.UPDATE:
                return PendingAtomResolution.UPDATED
            return PendingAtomResolution.MERGED
        return PendingAtomResolution.DISCARDED

    async def _capture_interaction_artifact(
        self,
        interaction_input: InteractionArtifactInput | None,
        identity_scope: IdentityScope,
    ) -> ArtifactRef | None:
        """
        构建原始交互 artifact。
        """
        if interaction_input is None:
            return None
        if not interaction_input.blocks:
            return None
        try:
            return await self._artifact_engine.interaction.build_and_store(
                topic_id=interaction_input.topic_id,
                topic_title=interaction_input.topic_title,
                topic_summary=interaction_input.topic_summary,
                blocks=interaction_input.blocks,
                identity_scope=identity_scope,
            )
        except Exception:
            logger.warning("Failed to build interaction artifact", exc_info=True)
            return None

    async def _commit_outcome(
        self,
        outcome: GenerationOutcome,
        *,
        identity_scope: IdentityScope,
        now: datetime,
        interaction_ref: ArtifactRef | None,
        gen_context: GenerationContext,
        creation_source: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"],
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"],
    ) -> None:
        """单个 outcome 的提交流水线：字段分配 → artifact 挂载 → 持久化。

        - TOUCH：不改内容字段与版本，经受限 ``patch_payload`` 只推进访问
          计数与 ``last_accessed_at``（MVL-0 M0.5 决定 ②）。
        - CREATE/UPDATE：版本与内容时间在此提交边界分配；版本记录写入是
          内容提交成功的前置条件（builder 失败直接传播，不静默降级）。
        """
        atom = outcome.atom
        if atom is None:  # 调用方已跳过 DISCARD/空 atom；此处守卫保证类型收窄。
            return
        decision = outcome.duplicate_decision

        if decision == DuplicateDecision.TOUCH:
            key = WorkspaceMemoryKey(
                workspace_identity=identity_scope.workspace_identity,
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
                logger.warning("TOUCH 目标记忆已不存在，跳过访问统计: %s", atom.id)
                return
            return

        if decision == DuplicateDecision.UPDATE:
            # 引擎是纯计算，outcome.atom 携带的即提交前版本；内容修订在此
            # 提交边界推进一次版本与内容时间，衰减基准随之推进（M0.1/M0.2）。
            atom.meta.version += 1
            atom.meta.updated_at = now
            atom.meta.lifecycle.decay_anchor_at = now
            atom.meta.lifecycle.confidence_score = 1.0

        await self._attach_memory_artifact(
            atom=atom,
            decision=decision,
            memory_before_snapshot=outcome.memory_before_snapshot,
            changelog=outcome.changelog,
            gen_context=gen_context,
            interaction_ref=interaction_ref,
            creation_source=creation_source,
            update_source=update_source,
            now=now,
        )

        recompute = decision == DuplicateDecision.CREATE or self._embedding_inputs_changed(
            outcome.memory_before_snapshot, atom
        )
        await self._mid_term.upsert(atom, recompute_vectors=recompute)
        logger.info(
            f"记忆已存储 '{atom.index.title}' (ID: {atom.id}, "
            f"decision={decision.value}, recompute_vectors={recompute})"
        )

    async def _attach_memory_artifact(
        self,
        *,
        atom: MemoryAtom,
        decision: DuplicateDecision,
        memory_before_snapshot: MemoryAtom | dict | None,
        changelog: str | None,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None,
        creation_source: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"],
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"] = "UPDATE",
        now: datetime | None = None,
    ) -> None:
        """为单个已生成或外部编辑的 MemoryAtom 挂载 artifact。

        ``now`` 为提交边界时点：版本记录 ``changed_at`` 与事件 ``at`` 使用
        同一时点（A2-P 时间边界 §2.3）。全部内容提交（外部创建/编辑与生成
        CREATE/UPDATE）都经过这里，类型相关的内容约束在写 Artifact 之前检查。
        """
        _require_content_invariants(atom)
        commit_now = require_utc(now) if now is not None else require_utc(self._now())

        src_refs = [interaction_ref] if interaction_ref else []
        if decision == DuplicateDecision.CREATE:
            bundle = await self._build_creation_artifacts(
                atom=atom,
                gen_context=gen_context,
                source_artifact_refs=src_refs,
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

        elif decision == DuplicateDecision.UPDATE:
            version_ref = await self._build_update_artifact(
                atom=atom,
                memory_before_snapshot=memory_before_snapshot,
                changelog=changelog,
                source_artifact_refs=src_refs,
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
        creation_source: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"],
        now: datetime,
    ) -> MemoryCreationBundle:
        """构建 v1 版本记录与创建 Artifact；失败直接传播（M0.3：无历史不提交）。"""
        bundle = await self._artifact_engine.memory.build_for_create(
            memory=atom,
            context=gen_context,
            source_intent=creation_source,
            source_artifact_refs=source_artifact_refs,
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
        memory_before_snapshot: MemoryAtom | dict | None,
        changelog: str | None,
        source_artifact_refs: list[ArtifactRef],
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"],
        now: datetime,
    ) -> ArtifactRef:
        """构建版本记录；失败直接传播（M0.3：版本记录是提交成功的前置条件）。"""
        snapshot_before: dict | None = None
        if isinstance(memory_before_snapshot, MemoryAtom):
            snapshot_before = snapshot_memory_atom(memory_before_snapshot)
        elif isinstance(memory_before_snapshot, dict):
            snapshot_before = memory_before_snapshot
        version_ref = await self._artifact_engine.memory.build_for_update(
            memory_after=atom,
            snapshot_before=snapshot_before,
            update_source=update_source,
            changelog=changelog,
            source_artifact_refs=source_artifact_refs,
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


def _require_content_invariants(atom: MemoryAtom) -> None:
    """内容提交的类型相关约束：AGENT_PROFILE 必须带 alias（按 alias 寻址与 CALL）。

    约束放在提交边界而不是模型：模型约束同样作用于读取解码，收紧会让既有
    不合规记录无法读取；提交边界只拒绝新的写入。
    """
    if atom.index.memory_type == MemoryType.AGENT_PROFILE and not atom.index.alias:
        raise InvalidMemoryFieldError("AGENT_PROFILE 必须设置 alias")


def _manual_changelog(changed_fields: list[str]) -> str:
    if not changed_fields:
        return "Manual edit: metadata refreshed"
    return f"Manual edit: {', '.join(changed_fields)}"
