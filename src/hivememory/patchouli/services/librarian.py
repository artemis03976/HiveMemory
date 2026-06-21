"""Patchouli 馆长核心。

当前过渡期仅保留主动生成入口。感知代理职责已迁入 PerceptionFamiliar，
生命周期维护职责已迁入 LifecycleFamiliar。后续 4.8 第 4-6 步会继续
将主动生成入口迁入 MemoryGenerationCoordinator / MemoryGenerationFamiliar。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from hivememory.core.models.pending import PendingAtomMaterializeTask
from hivememory.engines.generation.models import GenerationContext
from hivememory.patchouli.runtime.memory_tasks import MemoryGenerationTask
from hivememory.patchouli.services.memory_generation_tasks import (
    MemoryGenerationTaskController,
)
from hivememory.prompts.transcript import GenerationTranscriptBuilder

if TYPE_CHECKING:
    from hivememory.engines.artifacts.engine import ArtifactEngine
    from hivememory.patchouli.services.retrieval import RetrievalFamiliar

logger = logging.getLogger(__name__)


class LibrarianCore:
    """过渡期馆长核心，仅保留主动记忆生成编排。"""

    def __init__(
        self,
        bus: Optional[Any] = None,
        retrieval_familiar: Optional["RetrievalFamiliar"] = None,
        task_controller: Optional[MemoryGenerationTaskController] = None,
        artifact_engine: Optional["ArtifactEngine"] = None,
    ) -> None:
        self._bus = bus
        self._retrieval_familiar = retrieval_familiar
        self._artifact_engine = artifact_engine
        self._memory_task_controller = task_controller

        logger.info("LibrarianCore 初始化完成")

    async def run_active_generation(
        self,
        tasks: List[PendingAtomMaterializeTask],
        topic_id: str,
    ) -> List[MemoryGenerationTask]:
        """启动 MTP WRITE/UPDATE 主动记忆生成任务。"""
        if not tasks:
            return []

        topic_data = None
        if self._retrieval_familiar is not None:
            topic_data = self._retrieval_familiar.get_short_term_topic(topic_id)
        blocks = topic_data.recent_blocks(5) if topic_data is not None else []
        state_summary = topic_data.state_summary if topic_data is not None else ""

        interaction_ref = None
        if self._artifact_engine and blocks:
            try:
                interaction_ref = await self._artifact_engine.interaction.build_and_store(
                    topic_id=topic_id,
                    topic_title=topic_data.topic_title if topic_data is not None else "",
                    topic_summary=topic_data.topic_summary if topic_data is not None else "",
                    blocks=blocks,
                )
            except Exception:
                logger.warning("InteractionArtifact 写入失败，继续生成流程", exc_info=True)

        gen_context = self._build_generation_context(blocks, state_summary)
        return await self._memory_task_controller.run_active_generation(
            tasks,
            topic_id,
            gen_context=gen_context,
            interaction_ref=interaction_ref,
        )

    def _build_generation_context(
        self,
        blocks: List[Any],
        state_summary: str = "",
    ) -> GenerationContext:
        """根据短期话题 blocks 构建结构化生成上下文。"""
        builder = GenerationTranscriptBuilder()
        return builder.build_context(blocks, state_summary=state_summary)


__all__ = [
    "LibrarianCore",
]
