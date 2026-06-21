"""Patchouli 记忆生成入口协调器。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List
from uuid import UUID

from hivememory.core.models.artifact import ArtifactRef
from hivememory.core.models.pending import PendingAtomMaterializeTask, UpdateFocus, WriteFocus
from hivememory.engines.generation.models import GenerationRequest
from hivememory.engines.perception.models import ArchivePayload
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.memory_tasks import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskSpec,
)
from hivememory.prompts.transcript import GenerationTranscriptBuilder

if TYPE_CHECKING:
    from hivememory.engines.artifacts.engine import ArtifactEngine

logger = logging.getLogger(__name__)


class MemoryGenerationCoordinator:
    """将原始生成请求归一化为 MemoryGenerationTaskSpec。"""

    def __init__(
        self,
        *,
        bus: Any,
        artifact_engine: "ArtifactEngine | None" = None,
    ) -> None:
        self._bus = bus
        self._artifact_engine = artifact_engine
        self._transcript_builder = GenerationTranscriptBuilder()

    async def submit_archive(self, payload: ArchivePayload) -> MemoryGenerationTask | None:
        """将感知层 ArchivePayload 转为 ARCHIVE 任务规范。"""
        gen_context = self._transcript_builder.build_context(
            payload.blocks,
            state_summary=payload.state_summary,
        )
        if not gen_context.turns:
            logger.warning("空对话轮次，跳过被动生成")
            return None

        interaction_ref = await self._build_interaction_ref(
            topic_id=payload.topic_id,
            topic_title=payload.topic_title,
            topic_summary=payload.topic_summary,
            blocks=payload.blocks,
        )
        spec = MemoryGenerationTaskSpec(
            topic_id=payload.topic_id,
            label=payload.topic_id,
            source=MemoryGenerationSource.ARCHIVE,
            request=GenerationRequest(context=gen_context),
            source_intent="ARCHIVE",
            interaction_ref=interaction_ref,
        )
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION,
            spec,
        )

    async def submit_active(
        self,
        tasks: List[PendingAtomMaterializeTask],
        topic_id: str,
    ) -> List[MemoryGenerationTask]:
        """将 MTP WRITE/UPDATE 请求转为主动生成任务规范。"""
        if not tasks:
            return []

        topic_data = await self._bus.request(
            PatchouliLocalRoutes.TOPIC_GET_SHORT_TERM,
            topic_id,
        )
        blocks = topic_data.recent_blocks(5) if topic_data is not None else []
        state_summary = topic_data.state_summary if topic_data is not None else ""
        gen_context = self._transcript_builder.build_context(
            blocks,
            state_summary=state_summary,
        )
        interaction_ref = await self._build_interaction_ref(
            topic_id=topic_id,
            topic_title=topic_data.topic_title if topic_data is not None else "",
            topic_summary=topic_data.topic_summary if topic_data is not None else "",
            blocks=blocks,
        )

        specs = [
            await self._build_active_spec(
                task,
                topic_id=topic_id,
                gen_context=gen_context,
                interaction_ref=interaction_ref,
            )
            for task in tasks
        ]
        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY,
            specs,
        )

    async def _build_active_spec(
        self,
        task: PendingAtomMaterializeTask,
        *,
        topic_id: str,
        gen_context,
        interaction_ref: ArtifactRef | None,
    ) -> MemoryGenerationTaskSpec:
        source = MemoryGenerationSource(task.source_verb)
        focus = task.focus
        if source == MemoryGenerationSource.WRITE:
            assert isinstance(focus, WriteFocus)
            request = GenerationRequest(
                context=gen_context,
                write_focus=focus,
                identity=task.identity,
                intent_id=task.intent_id,
                pending_alias=task.pending_alias,
            )
            source_intent = "WRITE"
        elif source == MemoryGenerationSource.UPDATE:
            assert isinstance(focus, UpdateFocus)
            existing = await self._bus.request(
                PatchouliLocalRoutes.MEMORY_GET,
                UUID(focus.base_uuid),
            )
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
            source_intent = "UPDATE"
        else:
            raise ValueError(f"Unsupported active generation source: {source}")

        return MemoryGenerationTaskSpec(
            topic_id=topic_id,
            label=task.pending_alias,
            source=source,
            request=request,
            source_intent=source_intent,
            interaction_ref=interaction_ref,
            pending_alias=task.pending_alias,
        )

    async def _build_interaction_ref(
        self,
        *,
        topic_id: str,
        topic_title: str,
        topic_summary: str,
        blocks: List[Any],
    ) -> ArtifactRef | None:
        """构建交互 artifact；失败不阻断生成链路。"""
        if self._artifact_engine is None or not blocks:
            return None
        try:
            return await self._artifact_engine.interaction.build_and_store(
                topic_id=topic_id,
                topic_title=topic_title,
                topic_summary=topic_summary,
                blocks=blocks,
            )
        except Exception:
            logger.warning("InteractionArtifact 写入失败，继续生成流程", exc_info=True)
            return None


__all__ = ["MemoryGenerationCoordinator"]
