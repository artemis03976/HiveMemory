"""依赖中立的话题只读领域模型。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models.interaction import TurnRecord
from hivememory.core.models.workspace import WorkspaceIdentity, WorkspaceTopicKey


class BufferState(str, Enum):
    """短期话题缓冲区状态。"""

    IDLE = "idle"
    PROCESSING = "processing"
    FLUSHING = "flushing"


class TopicLastTurn(BaseModel):
    """话题快照中的最后一轮展示值。"""

    user: str = ""
    assistant: str = ""

    model_config = ConfigDict(frozen=True)


class TopicSnapshot(BaseModel):
    """供 Gateway 路由和前端话题池使用的只读快照。"""

    topic_id: str
    workspace_identity: WorkspaceIdentity
    topic_title: str
    topic_summary: str = ""
    state_summary: str = ""
    last_turn: TopicLastTurn | None = None
    total_tokens: int = 0
    block_count: int = 0
    last_accessed_at: float = 0.0
    model_used: str = ""

    model_config = ConfigDict(frozen=True, use_enum_values=True)


class LogicalBlock(BaseModel):
    """感知层写入短期话题的不可变逻辑块。"""

    block_id: str = Field(default_factory=lambda: uuid4().hex)
    turn: TurnRecord = Field(default_factory=TurnRecord)
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    total_tokens: int = 0
    gateway_intent: str | None = None
    worth_saving: bool | None = None

    model_config = ConfigDict(frozen=True, use_enum_values=True, extra="forbid")

    @property
    def is_complete(self) -> bool:
        return bool(self.turn.user_query) and bool(
            self.turn.assistant_final_text
            or self.turn.turn_events
            or self.turn.actions
        )

    @property
    def anchor_text(self) -> str:
        return self.turn.anchor_text

    @property
    def identity(self):
        return self.turn.identity

    @property
    def rewritten_query(self) -> str | None:
        return self.turn.rewritten_query

    @property
    def user_query(self) -> str:
        return self.turn.user_query

    @property
    def assistant_final_text(self) -> str:
        return self.turn.assistant_final_text

    @property
    def turn_events(self):
        return self.turn.turn_events

    @property
    def actions(self):
        return self.turn.actions

    @property
    def semantic_traces(self):
        return self.turn.semantic_traces


class TopicData(BaseModel):
    """短期话题缓冲区的完整不可变读取对象。"""

    topic_id: str
    workspace_identity: WorkspaceIdentity
    current_agent_id: str = "default"
    topic_title: str
    topic_summary: str = ""
    state_summary: str = ""
    blocks: tuple[LogicalBlock, ...] = Field(default_factory=tuple)
    state: BufferState = BufferState.IDLE
    last_update: float
    last_accessed_at: float
    total_tokens: int = 0
    model_used: str = ""

    model_config = ConfigDict(frozen=True, use_enum_values=False)

    @property
    def topic_key(self) -> WorkspaceTopicKey:
        """返回已由短期 Store 验证的复合 Topic 键。"""
        return WorkspaceTopicKey(
            owner_user_id=self.workspace_identity.owner_user_id,
            workspace_id=self.workspace_identity.workspace_id,
            topic_id=self.topic_id,
        )

    @property
    def user_id(self) -> str:
        """兼容展示旧 owner 字段；资源寻址必须使用 workspace_identity。"""
        return self.workspace_identity.owner_user_id

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def is_empty(self) -> bool:
        return not self.blocks

    def recent_blocks(self, limit: int) -> tuple[LogicalBlock, ...]:
        if limit <= 0:
            return ()
        return self.blocks[-limit:]

    def is_idle(self, timeout_seconds: int) -> bool:
        return (datetime.now().timestamp() - self.last_update) > timeout_seconds

    def to_topic_snapshot(self) -> TopicSnapshot:
        last_turn = None
        if self.blocks:
            last_block = self.blocks[-1]
            last_turn = TopicLastTurn(
                user=last_block.user_query,
                assistant=last_block.assistant_final_text,
            )
        return TopicSnapshot(
            topic_id=self.topic_id,
            workspace_identity=self.workspace_identity,
            topic_title=self.topic_title,
            topic_summary=self.topic_summary,
            state_summary=self.state_summary,
            last_turn=last_turn,
            total_tokens=self.total_tokens,
            block_count=self.block_count,
            last_accessed_at=self.last_accessed_at,
            model_used=self.model_used,
        )


__all__ = [
    "BufferState",
    "LogicalBlock",
    "TopicData",
    "TopicLastTurn",
    "TopicSnapshot",
]
