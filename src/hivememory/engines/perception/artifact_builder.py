"""
InteractionArtifactBuilder - Phase 2

从 LogicalBlock 序列构建 InteractionArtifact，
不嵌入任何记忆归属信息（memory_id / source_intent / capture_policy）。
"""

from datetime import datetime
from typing import Sequence

from hivememory.core.models.artifact import InteractionArtifact, InteractionTurnSnapshot
from hivememory.engines.perception.models import LogicalBlock


class InteractionArtifactBuilder:
    def build(
        self,
        *,
        topic_id: str,
        topic_title: str = "",
        topic_summary: str = "",
        blocks: Sequence[LogicalBlock],
    ) -> InteractionArtifact:
        turns = [self._snapshot(block) for block in blocks]
        return InteractionArtifact(
            topic_id=topic_id,
            topic_title=topic_title,
            topic_summary=topic_summary,
            turns=turns,
            captured_at=datetime.now(),
        )

    @staticmethod
    def _snapshot(block: LogicalBlock) -> InteractionTurnSnapshot:
        turn = block.turn
        return InteractionTurnSnapshot(
            block_id=block.block_id,
            turn_id=turn.turn_id,
            created_at=block.created_at,
            user_id=turn.identity.user_id,
            agent_id=turn.identity.agent_id,
            team_id=turn.identity.team_id,
            user_query=turn.user_query,
            rewritten_query=turn.rewritten_query,
            assistant_final_text=turn.assistant_final_text,
            turn_events=[e.model_dump() for e in turn.turn_events],
            actions=[a.model_dump() for a in turn.actions],
            semantic_traces=[t.model_dump() for t in turn.semantic_traces],
        )
