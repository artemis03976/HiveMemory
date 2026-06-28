"""Patchouli 记忆生成使魔。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from hivememory.core.models import PendingAtomResolution, PendingAtomSettlement
from hivememory.core.models.artifact import ArtifactRef, MemoryProvenance
from hivememory.engines.generation.models import DuplicateDecision, GenerationContext
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
        generation_engine: "MemoryGenerationEngine",
        memory_library: "MemoryLibrary",
        artifact_engine: "ArtifactEngine | None" = None,
    ) -> None:
        self._generation_engine = generation_engine
        self._mid_term = memory_library.mid_term
        self._artifact_engine = artifact_engine
        logger.info("MemoryGenerationFamiliar 初始化完成")

    async def execute(
        self,
        spec: MemoryGenerationTaskSpec,
    ) -> List[MemoryGenerationResult]:
        """执行统一生成任务规范，只返回结果不发布事件。"""
        interaction_ref = await self._build_interaction_artifact(
            spec.interaction_input,
        )
        return await self._run_generation(spec, interaction_ref=interaction_ref)

    async def _run_generation(
        self,
        spec: MemoryGenerationTaskSpec,
        interaction_ref: ArtifactRef | None = None,
    ) -> List[MemoryGenerationResult]:
        """执行 compute -> artifacts -> persist 三步流水线。"""
        request = spec.request
        source_intent = spec.source_intent

        # Step 1：纯计算，GenerationEngine 不负责持久化。
        outcomes = await self._generation_engine.process(request)
        results = [
            self._build_generation_result(spec, outcome)
            for outcome in outcomes
        ]

        memories = [result.atom for result in results if result.atom is not None]
        logger.info(
            f"Extracted {len(memories)} memories"
            if memories
            else "No memories extracted"
        )

        # Step 2：构建 artifact，并在第一次写库前挂载到 MemoryAtom。
        await self._build_memory_artifacts(
            results,
            request.context,
            source_intent,
            interaction_ref,
        )

        # Step 3：写入 CREATE/UPDATE 结果；settlement 由控制面统一发布。
        for result in results:
            if (
                result.duplicate_decision != DuplicateDecision.DISCARD
                and result.atom is not None
            ):
                try:
                    await self._mid_term.upsert(result.atom)
                    logger.info(
                        f"记忆已存储 '{result.atom.index.title}' "
                        f"(ID: {result.atom.id})"
                    )
                except Exception as exc:
                    logger.error(f"存储记忆失败: {exc}", exc_info=True)
                    raise

        return results

    def _build_generation_result(
        self,
        spec: MemoryGenerationTaskSpec,
        outcome,
    ) -> MemoryGenerationResult:
        atom = outcome.atom
        canonical_alias = atom.get_alias() if atom is not None else None
        canonical_uuid = str(atom.id) if atom is not None else None
        return MemoryGenerationResult(
            intent_id=spec.intent_id,
            pending_alias=spec.pending_alias,
            atom=atom,
            canonical_alias=canonical_alias,
            canonical_uuid=canonical_uuid,
            duplicate_decision=outcome.duplicate_decision,
            memory_before_snapshot=outcome.memory_before_snapshot,
            changelog=outcome.changelog,
            settlement=self._build_settlement(
                spec,
                outcome.duplicate_decision,
                canonical_alias=canonical_alias,
                canonical_uuid=canonical_uuid,
            ),
            message=outcome.message,
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

    async def _build_interaction_artifact(
        self,
        interaction_input: InteractionArtifactInput | None,
    ) -> ArtifactRef | None:
        """构建原始交互 artifact。"""
        if self._artifact_engine is None or interaction_input is None:
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

    async def _build_memory_artifacts(
        self,
        results: List[MemoryGenerationResult],
        gen_context: GenerationContext,
        source_intent: str,
        interaction_ref: ArtifactRef | None,
    ) -> None:
        """构建 artifact 并挂载 refs/provenance，不负责发布事件。"""
        if not self._artifact_engine:
            if interaction_ref:
                for result in results:
                    if (
                        result.atom is not None
                        and result.duplicate_decision
                        in (DuplicateDecision.CREATE, DuplicateDecision.UPDATE)
                    ):
                        result.atom.payload.artifacts.refs.append(interaction_ref)
            return

        builder = self._artifact_engine.memory
        src_refs = [interaction_ref] if interaction_ref else []

        for result in results:
            atom = result.atom
            if atom is None:
                continue
            try:
                if result.duplicate_decision == DuplicateDecision.CREATE:
                    bundle = await builder.build_for_create(
                        memory=atom,
                        context=gen_context,
                        source_intent=source_intent,
                        source_artifact_refs=src_refs,
                    )
                    atom.payload.artifacts.refs.extend(
                        [bundle.initial_version_ref, bundle.creation_ref]
                    )
                    if interaction_ref:
                        atom.payload.artifacts.refs.append(interaction_ref)
                    atom.payload.artifacts.provenance.append(
                        MemoryProvenance(
                            action="created",
                            source_intent=source_intent,
                            source_artifacts=src_refs,
                        )
                    )

                elif result.duplicate_decision == DuplicateDecision.UPDATE:
                    version_ref = await builder.build_for_update(
                        memory_after=atom,
                        snapshot_before=result.memory_before_snapshot,
                        update_source="UPDATE",
                        changelog=result.changelog,
                        source_artifact_refs=src_refs,
                    )
                    if interaction_ref:
                        atom.payload.artifacts.refs.append(interaction_ref)
                    atom.payload.artifacts.refs.append(version_ref)
                    atom.payload.artifacts.provenance.append(
                        MemoryProvenance(
                            action="updated",
                            source_intent=source_intent,
                            source_artifacts=src_refs,
                        )
                    )

            except Exception:
                logger.warning(
                    f"Failed to build memory artifacts for {getattr(atom, 'id', '?')}",
                    exc_info=True,
                )


__all__ = ["MemoryGenerationFamiliar"]
