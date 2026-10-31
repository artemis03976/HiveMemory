"""InteractionArtifactBuilder - 从 LogicalBlock[] 与折叠记录构建 raw interaction artifact。"""

from collections.abc import Sequence

from hivememory.config.patchouli import ArtifactComponentConfig
from hivememory.core.models import LogicalBlock, TopicFold, WorkspaceIdentity
from hivememory.core.models.artifact import (
    ArtifactRef,
    InteractionArtifact,
    InteractionFoldSnapshot,
    InteractionTurnSnapshot,
)
from hivememory.patchouli.memory_library import ArtifactStore
from hivememory.utils.time import utc_now


class InteractionArtifactBuilder:
    """
    只读取 LogicalBlock.turn 与话题的折叠记录，不读取 GenerationContext。
    不写入 memory id / alias / source intent / capture policy。

    来源 provenance 按 block 粒度由 ``InteractionTurnSnapshot.actor_identity``
    记录，不设置顶层 Agent 来源字段。
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
        folds: Sequence[TopicFold],
        belong_to: WorkspaceIdentity,
    ) -> ArtifactRef | None:
        """以资源归属保存交互与折叠记录，内容参与者身份保留在各轮快照中。"""
        artifact = InteractionArtifact(
            workspace_identity=belong_to,
            topic_id=topic_id,
            topic_title=topic_title,
            topic_summary=topic_summary,
            turns=[_snapshot(b) for b in blocks],
            folds=[_fold_snapshot(fold) for fold in folds],
            captured_at=utc_now(),
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
        folds: Sequence[TopicFold],
        belong_to: WorkspaceIdentity,
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
        # actor 三元组收敛为单一 ActorIdentity；随 turn 冻结，写入后不再解释
        actor_identity=t.identity,
        outcome=t.outcome,
        turn_events=[e.model_dump() for e in t.turn_events],
    )


def _fold_snapshot(fold: TopicFold) -> InteractionFoldSnapshot:
    return InteractionFoldSnapshot(
        folded_block_ids=list(fold.folded_block_ids),
        summary=fold.summary,
        folded_at=fold.folded_at,
    )
