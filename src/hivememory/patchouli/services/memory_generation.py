"""Patchouli 记忆生成使魔。"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from hivememory.core.models import MemoryAtom, PendingAtomResolution, PendingAtomSettlement
from hivememory.core.models.artifact import (
    ArtifactRef,
    MemoryEventLog,
    MemoryEventType,
    MemoryVersionSnapshot,
)
from hivememory.engines.artifacts.memory import MemoryCreationBundle
from hivememory.engines.generation.models import (
    DuplicateDecision,
    GenerationContext,
    GenerationOutcome,
)
from hivememory.patchouli.runtime.memory_tasks import (
    InteractionArtifactInput,
    MemoryGenerationResult,
    MemoryGenerationSource,
    MemoryGenerationTaskSpec,
)

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
    ) -> None:
        from hivememory.engines.artifacts.engine import ArtifactEngine

        self._generation_engine = generation_engine
        self._mid_term = memory_library.mid_term
        self._artifact_engine = artifact_engine or ArtifactEngine.noop()

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
        )
        return await self._run_generation(spec, interaction_ref=interaction_ref)

    async def create_external_memory(self, atom: MemoryAtom) -> MemoryAtom:
        """
        对外部创建的记忆原子进行持久化处理。
        """
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
        atom = await self._mid_term.get(memory_id)
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
        outcomes = await self._generation_engine.process(spec.request)

        memories = [outcome.atom for outcome in outcomes if outcome.atom is not None]
        logger.info(
            f"Extracted {len(memories)} memories"
            if memories
            else "No memories extracted"
        )

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
            if (
                outcome.duplicate_decision != DuplicateDecision.DISCARD
                and outcome.atom is not None
            ):
                try:
                    await self._mid_term.upsert(outcome.atom)
                    logger.info(
                        f"记忆已存储 '{outcome.atom.index.title}' "
                        f"(ID: {outcome.atom.id})"
                    )
                except Exception as exc:
                    logger.error(f"存储记忆失败: {exc}", exc_info=True)
                    raise

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
            message=(
                f"Pending atom '{spec.pending_alias}' settled as "
                f"{resolution.value}."
            ),
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
        update_source: Literal[
            "UPDATE", "MERGE", "MANUAL_EDIT", "SYSTEM_REWRITE"
        ] = "UPDATE",
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
        refs = atom.payload.artifacts.refs
        exists = any(
            existing.artifact_id == ref.artifact_id
            and existing.artifact_type == ref.artifact_type
            for existing in refs
        )
        if not exists:
            refs.append(ref)


__all__ = ["MemoryGenerationFamiliar"]


def _manual_changelog(changed_fields: list[str]) -> str:
    if not changed_fields:
        return "Manual edit: metadata refreshed"
    return f"Manual edit: {', '.join(changed_fields)}"
