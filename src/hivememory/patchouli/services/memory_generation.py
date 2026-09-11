"""Patchouli 记忆生成使魔。"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from hivememory.core.errors import WorkspaceDomainError, WorkspaceMismatchError
from hivememory.core.models import (
    IdentityScope,
    MemoryAtom,
    PendingAtomResolution,
    PendingAtomSettlement,
    require_identity_scope,
)
from hivememory.core.models.artifact import (
    ArtifactRef,
    MemoryEventLog,
    MemoryEventType,
    MemoryVersionSnapshot,
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
    ) -> None:
        from hivememory.engines.artifacts.engine import ArtifactEngine

        self._generation_engine = generation_engine
        self._mid_term = memory_library.mid_term
        self._artifact_engine = artifact_engine or ArtifactEngine.noop()
        # W1-F：附件 promotion 的只读 reader（assembler 经 PatchouliRuntime
        # 注入进程级唯一 Store）；ref 失效或 Store 关闭时 best-effort 降级。
        self._asset_reader = asset_reader

        logger.info("MemoryGenerationFamiliar 初始化完成")

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
        await self._attach_memory_artifact(
            atom=atom,
            decision=DuplicateDecision.CREATE,
            memory_before_snapshot=None,
            changelog=None,
            gen_context=GenerationContext(),
            interaction_ref=None,
            creation_source="MANUAL",
        )
        await self._mid_term.upsert(atom)
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

        before_snapshot = MemoryVersionSnapshot.from_memory_atom(atom)
        changed_fields = self._apply_external_update(
            atom,
            title=title,
            summary=summary,
            content=content,
            alias=alias,
            tags=tags,
            agent_config=agent_config,
        )
        atom.meta.updated_at = datetime.now(UTC)
        atom.meta.version += 1

        await self._attach_memory_artifact(
            atom=atom,
            decision=DuplicateDecision.UPDATE,
            memory_before_snapshot=before_snapshot,
            changelog=_manual_changelog(changed_fields),
            gen_context=GenerationContext(),
            interaction_ref=None,
            creation_source="MANUAL",
            update_source="MANUAL_EDIT",
        )
        await self._mid_term.upsert(atom)
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
        changed_fields: list[str] = []
        if title is not None:
            atom.index.title = title
            changed_fields.append("title")
        if summary is not None:
            atom.index.summary = summary
            changed_fields.append("summary")
        if content is not None:
            atom.payload.content = content
            changed_fields.append("content")
        if alias is not None:
            atom.index.alias = alias or None
            changed_fields.append("alias")
        if tags is not None:
            atom.index.tags = tags
            changed_fields.append("tags")
        if agent_config is not None:
            atom.payload.artifacts.agent_config = agent_config
            changed_fields.append("agent_config")
        return changed_fields

    async def _run_generation(
        self,
        spec: MemoryGenerationTaskSpec,
        interaction_ref: ArtifactRef | None = None,
    ) -> list[MemoryGenerationResult]:
        """
        执行 compute -> artifacts -> persist 三步流水线。
        """
        # Step 1：纯计算，GenerationEngine 不负责持久化。
        outcomes = await self._generation_engine.process(
            spec.request,
            identity_scope=spec.identity_scope,
        )

        memories = [outcome.atom for outcome in outcomes if outcome.atom is not None]
        logger.info(f"Extracted {len(memories)} memories" if memories else "No memories extracted")

        # Step 2：构建 artifact，并在第一次写库前挂载到 MemoryAtom。
        await self._attach_memory_artifacts(
            outcomes,
            spec.request.context,
            interaction_ref,
            creation_source=spec.source.creation_artifact_intent,
            update_source=spec.source.version_update_source,
        )

        # Step 3：写入 CREATE/UPDATE 结果。
        for outcome in outcomes:
            if outcome.duplicate_decision != DuplicateDecision.DISCARD and outcome.atom is not None:
                try:
                    await self._mid_term.upsert(outcome.atom)
                    logger.info(
                        f"记忆已存储 '{outcome.atom.index.title}' " f"(ID: {outcome.atom.id})"
                    )
                except Exception as exc:
                    logger.error(f"存储记忆失败: {exc}", exc_info=True)
                    raise

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

    async def _attach_memory_artifacts(
        self,
        outcomes: list[GenerationOutcome],
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None,
        *,
        creation_source: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"],
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"] = "UPDATE",
    ) -> None:
        """
        构建 artifact 并挂载 refs/events，不负责发布事件。
        """
        for outcome in outcomes:
            atom = outcome.atom
            if atom is None:
                continue
            await self._attach_memory_artifact(
                atom=atom,
                decision=outcome.duplicate_decision,
                memory_before_snapshot=outcome.memory_before_snapshot,
                changelog=outcome.changelog,
                gen_context=gen_context,
                interaction_ref=interaction_ref,
                creation_source=creation_source,
                update_source=update_source,
            )

    async def _attach_memory_artifact(
        self,
        *,
        atom: MemoryAtom,
        decision: DuplicateDecision,
        memory_before_snapshot: MemoryVersionSnapshot | None,
        changelog: str | None,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None,
        creation_source: Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"],
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"] = "UPDATE",
    ) -> None:
        """为单个已生成或外部编辑的 MemoryAtom 挂载 artifact。"""

        src_refs = [interaction_ref] if interaction_ref else []
        if decision == DuplicateDecision.CREATE:
            bundle = await self._build_creation_artifacts(
                atom=atom,
                gen_context=gen_context,
                source_artifact_refs=src_refs,
                creation_source=creation_source,
            )

            atom.payload.artifacts.events.append(
                MemoryEventLog(
                    event_type=MemoryEventType.CREATED,
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
            )

            atom.payload.artifacts.events.append(
                MemoryEventLog(
                    event_type=MemoryEventType.VERSIONED,
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
    ) -> MemoryCreationBundle:
        try:
            bundle = await self._artifact_engine.memory.build_for_create(
                memory=atom,
                context=gen_context,
                source_intent=creation_source,
                source_artifact_refs=source_artifact_refs,
            )
        except Exception:
            logger.warning(
                f"Failed to build creation artifacts for {getattr(atom, 'id', '?')}",
                exc_info=True,
            )
            return MemoryCreationBundle()

        for ref in bundle.refs:
            self._append_artifact_ref_once(atom, ref)

        return bundle

    async def _build_update_artifact(
        self,
        *,
        atom: MemoryAtom,
        memory_before_snapshot: MemoryVersionSnapshot | None,
        changelog: str | None,
        source_artifact_refs: list[ArtifactRef],
        update_source: Literal["UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"],
    ) -> ArtifactRef | None:
        try:
            version_ref = await self._artifact_engine.memory.build_for_update(
                memory_after=atom,
                snapshot_before=memory_before_snapshot,
                update_source=update_source,
                changelog=changelog,
                source_artifact_refs=source_artifact_refs,
            )
        except Exception:
            logger.warning(
                f"Failed to build version artifact for {getattr(atom, 'id', '?')}",
                exc_info=True,
            )
            return None

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


def _manual_changelog(changed_fields: list[str]) -> str:
    if not changed_fields:
        return "Manual edit: metadata refreshed"
    return f"Manual edit: {', '.join(changed_fields)}"
