"""Read models exposed by MemoryLibrary stores."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models import TopicSnapshot
from hivememory.engines.perception.models import LogicalBlock
from hivememory.patchouli.memory_library.buffer import BufferState


class TopicData(BaseModel):
    """Immutable read view of a short-term topic buffer.

    This is the public data shape for short-term storage reads. Callers should
    consume TopicData instead of receiving the mutable SemanticBuffer entity.
    """

    topic_id: str
    user_id: str
    current_agent_id: str = "default"
    topic_title: str
    topic_summary: str = ""
    state_summary: str = ""
    blocks: tuple[LogicalBlock, ...] = Field(default_factory=tuple)
    state: BufferState = BufferState.IDLE
    last_update: float
    last_accessed_at: float
    total_tokens: int = 0
    # 最近一次 run 使用的模型展示名，从 SemanticBuffer.model_used 读取
    model_used: str = ""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, use_enum_values=True)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def is_empty(self) -> bool:
        return not self.blocks

    def recent_blocks(self, limit: int) -> list[LogicalBlock]:
        if limit <= 0:
            return []
        return list(self.blocks[-limit:])

    def is_idle(self, timeout_seconds: int) -> bool:
        return (datetime.now().timestamp() - self.last_update) > timeout_seconds

    def to_topic_snapshot(self) -> TopicSnapshot:
        last_turn: Optional[dict[str, str]] = None
        if self.blocks:
            last_block = self.blocks[-1]
            last_turn = {
                "user": last_block.user_query,
                "assistant": last_block.assistant_final_text,
            }
        return TopicSnapshot(
            topic_id=self.topic_id,
            topic_title=self.topic_title,
            topic_summary=self.topic_summary,
            state_summary=self.state_summary,
            last_turn=last_turn,
            total_tokens=self.total_tokens,
            block_count=self.block_count,
            last_accessed_at=self.last_accessed_at,
            model_used=self.model_used,
        )


from dataclasses import dataclass
from typing import Optional


@dataclass
class ArtifactIntegrityResult:
    artifact_id: str
    ok: bool
    stored_hash: Optional[str] = None
    actual_hash: Optional[str] = None


@dataclass(frozen=True)
class StorageHealthComponent:
    name: str
    healthy: bool
    required: bool = True
    detail: Optional[str] = None


@dataclass(frozen=True)
class StorageHealthReport:
    components: tuple[StorageHealthComponent, ...]

    @property
    def healthy(self) -> bool:
        return all(
            component.healthy
            for component in self.components
            if component.required
        )


__all__ = [
    "TopicData",
    "ArtifactIntegrityResult",
    "StorageHealthComponent",
    "StorageHealthReport",
]
