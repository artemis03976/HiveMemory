"""InteractionArtifactBuilder - 从 LogicalBlock[] 构建 raw interaction artifact。"""

from datetime import datetime
from typing import Sequence

from hivememory.core.models import LogicalBlock
from hivememory.core.models.artifact import (
    ArtifactRef,
    InteractionArtifact,
    InteractionTurnSnapshot,
)
from hivememory.patchouli.memory_library import ArtifactStore
from hivememory.system.config.patchouli import ArtifactComponentConfig


class InteractionArtifactBuilder:
    """
    只读取 LogicalBlock.turn，不读取 GenerationContext。
    不写入 memory id / alias / source intent / capture policy。
    """

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    async def build_and_store(
        self,
        *,
        topic_id: str,
        topic_title: str = "",
        topic_summary: str = "",
        blocks: Sequence[LogicalBlock],
    ) -> ArtifactRef | None:
        artifact = InteractionArtifact(
            topic_id=topic_id,
            topic_title=topic_title,
            topic_summary=topic_summary,
            turns=[_snapshot(b) for b in blocks],
            captured_at=datetime.now(),
        )
        return await self._store.put(artifact)


class NoOpInteractionArtifactBuilder:
    async def build_and_store(
        self,
        *,
        topic_id: str,
        topic_title: str = "",
        topic_summary: str = "",
        blocks: Sequence[LogicalBlock],
    ) -> ArtifactRef | None:
        return None


def create_interaction_builder(
    config: ArtifactComponentConfig,
    store: ArtifactStore | None,
) -> InteractionArtifactBuilder | NoOpInteractionArtifactBuilder:
    if store is None or not config.enabled:
        return NoOpInteractionArtifactBuilder()
    return InteractionArtifactBuilder(store)


def _snapshot(block: LogicalBlock) -> InteractionTurnSnapshot:
    t = block.turn
    return InteractionTurnSnapshot(
        block_id=block.block_id,
        turn_id=t.turn_id,
        created_at=block.created_at,
        user_id=t.identity.user_id,
        agent_id=t.identity.agent_id,
        team_id=t.identity.team_id,
        user_query=t.user_query,
        rewritten_query=t.rewritten_query,
        assistant_final_text=t.assistant_final_text,
        turn_events=[e.model_dump() for e in t.turn_events],
        actions=[a.model_dump() for a in t.actions],
        semantic_traces=[s.model_dump() for s in t.semantic_traces],
    )
