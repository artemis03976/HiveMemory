"""Patchouli 记忆生成使魔。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List
from uuid import UUID

from hivememory.core.models.artifact import ArtifactRef, MemoryProvenance
from hivememory.core.models.pending import (
    PendingAtomMaterializeTask,
    UpdateFocus,
    WriteFocus,
)
from hivememory.engines.generation.models import (
    DuplicateDecision,
    GenerationContext,
    GenerationRequest,
    MemoryGenerationResult,
)

if TYPE_CHECKING:
    from hivememory.engines.artifacts.engine import ArtifactEngine
    from hivememory.engines.generation.engine import MemoryGenerationEngine
    from hivememory.patchouli.memory_library.stores import MidTermMemoryStore

logger = logging.getLogger(__name__)


class MemoryGenerationFamiliar:
    """记忆生成数据面，负责执行生成、挂载 artifact 并写入中期记忆库。"""

    def __init__(
        self,
        *,
        generation_engine: "MemoryGenerationEngine",
        mid_term: "MidTermMemoryStore",
        artifact_engine: "ArtifactEngine | None" = None,
    ) -> None:
        self._generation_engine = generation_engine
        self._mid_term = mid_term
        self._artifact_engine = artifact_engine
        logger.info("MemoryGenerationFamiliar 初始化完成")

    async def run_archive_generation(
        self,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None = None,
    ) -> List[MemoryGenerationResult]:
        """执行被动 ARCHIVE 生成链路。"""
        logger.info(f"Memory generation archive task: {len(gen_context.turns)} turns")
        return await self._run_generation(
            GenerationRequest(context=gen_context),
            source_intent="ARCHIVE",
            interaction_ref=interaction_ref,
        )

    async def run_active_generation(
        self,
        task: PendingAtomMaterializeTask,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None = None,
    ) -> List[MemoryGenerationResult]:
        """执行单个 MTP WRITE/UPDATE 主动生成链路。"""
        if task.source_verb == "WRITE":
            return await self._run_mode_b(task, gen_context, interaction_ref)
        return await self._run_mode_c(task, gen_context, interaction_ref)

    async def execute(
        self,
        request: GenerationRequest | Any,
        *,
        source_intent: str = "WRITE",
        interaction_ref: ArtifactRef | None = None,
    ) -> List[MemoryGenerationResult]:
        """过渡期通用执行入口，后续由 MemoryGenerationTaskSpec 固化协议。"""
        if isinstance(request, GenerationRequest):
            return await self._run_generation(
                request,
                source_intent=source_intent,
                interaction_ref=interaction_ref,
            )

        generation_request = getattr(request, "request", None)
        if isinstance(generation_request, GenerationRequest):
            return await self._run_generation(
                generation_request,
                source_intent=getattr(request, "source_intent", source_intent),
                interaction_ref=getattr(request, "interaction_ref", interaction_ref),
            )

        raise TypeError("MemoryGenerationFamiliar.execute requires a GenerationRequest")

    async def _run_mode_b(
        self,
        task: PendingAtomMaterializeTask,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None = None,
    ) -> List[MemoryGenerationResult]:
        """Mode B：将 MTP WRITE 请求转换为 GenerationRequest。"""
        focus = task.focus
        assert isinstance(focus, WriteFocus)
        logger.info(f"Mode B WRITE: content='{focus.content[:50]}...'")
        request = GenerationRequest(
            context=gen_context,
            write_focus=focus,
            identity=task.identity,
            intent_id=task.intent_id,
            pending_alias=task.pending_alias,
        )
        return await self._run_generation(
            request,
            source_intent="WRITE",
            interaction_ref=interaction_ref,
        )

    async def _run_mode_c(
        self,
        task: PendingAtomMaterializeTask,
        gen_context: GenerationContext,
        interaction_ref: ArtifactRef | None = None,
    ) -> List[MemoryGenerationResult]:
        """Mode C：加载 UPDATE 目标记忆，并转换为 GenerationRequest。"""
        focus = task.focus
        assert isinstance(focus, UpdateFocus)
        logger.info(f"Mode C UPDATE: alias='{focus.base_alias}'")

        existing = await self._mid_term.get(UUID(focus.base_uuid))
        if existing is None:
            logger.error(f"UPDATE target memory not found: {focus.base_uuid}")
            raise RuntimeError(f"UPDATE target memory not found: {focus.base_uuid}")

        request = GenerationRequest(
            context=gen_context,
            update_focus=focus,
            existing_memory=existing,
            identity=task.identity,
            intent_id=task.intent_id,
            pending_alias=task.pending_alias,
        )
        return await self._run_generation(
            request,
            source_intent="UPDATE",
            interaction_ref=interaction_ref,
        )

    async def _run_generation(
        self,
        request: GenerationRequest,
        source_intent: str = "WRITE",
        interaction_ref: ArtifactRef | None = None,
    ) -> List[MemoryGenerationResult]:
        """执行 compute -> artifacts -> persist 三步流水线，只返回结果不发布事件。"""
        # Step 1：纯计算，GenerationEngine 不再负责持久化。
        results = await self._generation_engine.process(request)

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
                result.duplicate_decision
                in (DuplicateDecision.CREATE, DuplicateDecision.UPDATE)
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
                        changelog=(
                            atom.payload.artifacts.full_history[-1].get("reason")
                            if atom.payload.artifacts.full_history
                            else None
                        ),
                        source_artifact_refs=src_refs,
                    )
                    if atom.payload.artifacts.full_history:
                        atom.payload.artifacts.full_history[-1]["artifact_refs"] = [
                            version_ref.model_dump(mode="json")
                        ]
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
