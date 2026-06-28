"""Patchouli memory generation entrypoint coordinator."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, List
from uuid import UUID

from hivememory.core.models.pending import PendingAtomMaterializeTask, UpdateFocus, WriteFocus
from hivememory.engines.generation.models import GenerationRequest
from hivememory.engines.perception.models import TopicMaterializeTask
from hivememory.patchouli.contracts.local_events import PatchouliLocalEvents
from hivememory.patchouli.contracts.local_routes import PatchouliLocalRoutes
from hivememory.patchouli.runtime.memory_tasks import (
    InteractionArtifactInput,
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskSpec,
)
from hivememory.prompts.transcript import GenerationTranscriptBuilder

if TYPE_CHECKING:
    from hivememory.engines.perception.models import LogicalBlock

logger = logging.getLogger(__name__)


class SpecBuildError(RuntimeError):
    """Raised when one active generation spec cannot be built."""


class MemoryGenerationCoordinator:
    """将原始生成请求归一化为 MemoryGenerationTaskSpec。"""

    def __init__(
        self,
        *,
        bus: Any,
    ) -> None:
        self._bus = bus
        self._transcript_builder = GenerationTranscriptBuilder()

    async def submit_settlement(self, payload: TopicMaterializeTask) -> MemoryGenerationTask | None:
        """将感知层 TopicMaterializeTask 转为 SETTLEMENT 任务规范。"""
        gen_context = self._transcript_builder.build_context(
            payload.blocks,
            state_summary=payload.state_summary,
        )
        if not gen_context.turns:
            logger.warning("空对话轮次，跳过被动生成")
            return None

        spec = MemoryGenerationTaskSpec(
            topic_id=payload.topic_id,
            label=payload.topic_id,
            source=MemoryGenerationSource.ARCHIVE,
            request=GenerationRequest(context=gen_context),
            source_intent="SETTLEMENT",
            interaction_input=self._build_interaction_input(
                topic_id=payload.topic_id,
                topic_title=payload.topic_title,
                topic_summary=payload.topic_summary,
                blocks=payload.blocks,
            ),
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
            PatchouliLocalRoutes.TOPIC_GET,
            topic_id,
        )
        blocks = topic_data.recent_blocks(5) if topic_data is not None else []
        state_summary = topic_data.state_summary if topic_data is not None else ""
        gen_context = self._transcript_builder.build_context(
            blocks,
            state_summary=state_summary,
        )
        interaction_input = self._build_interaction_input(
            topic_id=topic_id,
            topic_title=topic_data.topic_title if topic_data is not None else "",
            topic_summary=topic_data.topic_summary if topic_data is not None else "",
            blocks=blocks,
        )

        # 并行构建任务规范，防止 UPDATE 任务的 IO 操作阻塞
        raw_specs = await asyncio.gather(
            *[
                self._try_build_active_spec(
                    task,
                    topic_id=topic_id,
                    gen_context=gen_context,
                    interaction_input=interaction_input,
                )
                for task in tasks
            ]
        )
        # 过滤构建失败的任务
        specs = [spec for spec in raw_specs if spec is not None]
        if not specs:
            return []

        return await self._bus.request(
            PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY,
            specs,
        )

    async def _try_build_active_spec(
        self,
        task: PendingAtomMaterializeTask,
        *,
        topic_id: str,
        gen_context,
        interaction_input: InteractionArtifactInput | None,
    ) -> MemoryGenerationTaskSpec | None:
        try:
            return await self._build_active_spec(
                task,
                topic_id=topic_id,
                gen_context=gen_context,
                interaction_input=interaction_input,
            )
        except SpecBuildError as exc:
            logger.error(
                "Active spec build failed, skipping task: pending_alias=%s, err=%s",
                task.pending_alias,
                exc,
            )
            await self._publish_pending_atom_failed(task.pending_alias)
            return None

    async def _build_active_spec(
        self,
        task: PendingAtomMaterializeTask,
        *,
        topic_id: str,
        gen_context,
        interaction_input: InteractionArtifactInput | None,
    ) -> MemoryGenerationTaskSpec:
        source = MemoryGenerationSource(task.source_verb)
        focus = task.focus
        if source == MemoryGenerationSource.WRITE:
            assert isinstance(focus, WriteFocus)
            request = GenerationRequest(
                context=gen_context,
                write_focus=focus,
                identity=task.identity,
            )
            source_intent = "WRITE"
        elif source == MemoryGenerationSource.UPDATE:
            assert isinstance(focus, UpdateFocus)
            try:
                base_uuid = UUID(focus.base_uuid)
            except ValueError as exc:
                raise SpecBuildError(
                    f"UPDATE target memory UUID is invalid: {focus.base_uuid}"
                ) from exc

            existing = await self._bus.request(
                PatchouliLocalRoutes.MEMORY_GET,
                base_uuid,
            )
            if existing is None:
                logger.error(f"UPDATE target memory not found: {focus.base_uuid}")
                raise SpecBuildError(f"UPDATE target memory not found: {focus.base_uuid}")
            request = GenerationRequest(
                context=gen_context,
                update_focus=focus,
                existing_memory=existing,
                identity=task.identity,
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
            interaction_input=interaction_input,
            intent_id=task.intent_id,
            pending_alias=task.pending_alias,
        )

    def _build_interaction_input(
        self,
        *,
        topic_id: str,
        topic_title: str,
        topic_summary: str,
        blocks: List["LogicalBlock"],
    ) -> InteractionArtifactInput | None:
        """将原始交互数据冻结为生成数据平面的交互输入。"""
        if not blocks:
            return None
        return InteractionArtifactInput(
            topic_id=topic_id,
            topic_title=topic_title,
            topic_summary=topic_summary,
            blocks=tuple(blocks),
        )

    async def _publish_pending_atom_failed(self, pending_alias: str) -> None:
        """
        发布待处理 PendingAtom 的失败事件。
        """
        try:
            await self._bus.publish(
                PatchouliLocalEvents.PENDING_ATOM_FAILED,
                pending_alias=pending_alias,
            )
        except Exception as pub_err:
            logger.warning(f"FAILED event publish error: {pub_err}")


__all__ = ["MemoryGenerationCoordinator"]
