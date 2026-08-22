"""短期记忆缓冲区实体模型。

SemanticBuffer 是短期记忆的核心存储单元（TopicSegment），
由 ShortTermMemoryStore 持有和管理。
"""

from __future__ import annotations

from datetime import datetime
from typing import List
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

from hivememory.core.models import BufferState, LogicalBlock, WorkspaceIdentity, WorkspaceTopicKey


class SemanticBuffer(BaseModel):
    """
    话题段 (TopicSegment) — 短期记忆的独立工作区

    MMU 架构中的核心数据容器，代表一个独立的讨论线程。
    每个 TopicSegment 拥有绝对纯净的上下文隔离。

    映射关系 (ShortTermMemory.md §2.2):
        TopicSegment = SemanticBuffer
        Pages = blocks (List[LogicalBlock])
    """
    topic_id: str = Field(default_factory=lambda: str(uuid4()), description="话题唯一标识")
    workspace_identity: WorkspaceIdentity

    current_agent_id: str = Field(default="default", description="当前话题挂载的活跃 Agent 别名")

    topic_title: str = Field(default="新建话题", description="话题标题")
    topic_summary: str = Field(default="", description="话题展示摘要")
    state_summary: str = Field(default="", description="页折叠后的状态摘要")

    blocks: List[LogicalBlock] = Field(default_factory=list)

    state: BufferState = BufferState.IDLE
    last_update: float = Field(default_factory=lambda: datetime.now().timestamp())
    last_accessed_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    total_tokens: int = 0
    # 最近一次 run 实际使用的模型展示名（来自 ModelRegistry，经 InteractionPayload 写入）
    model_used: str = Field(default="", description="最近 run 使用的模型展示名")

    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=True)

    @property
    def topic_key(self) -> WorkspaceTopicKey:
        """返回此 buffer 的权威复合键。"""
        return WorkspaceTopicKey(
            owner_user_id=self.workspace_identity.owner_user_id,
            workspace_id=self.workspace_identity.workspace_id,
            topic_id=self.topic_id,
        )

    @property
    def user_id(self) -> str:
        """兼容旧展示字段；内部存储不再以裸 user_id 寻址。"""
        return self.workspace_identity.owner_user_id

    def clear(self) -> None:
        self.blocks.clear()
        self.total_tokens = 0
        self.state = BufferState.IDLE
        self.last_update = datetime.now().timestamp()

    def get_block_count(self) -> int:
        return len(self.blocks)

    @property
    def has_blocks(self) -> bool:
        """是否存在原始 block（窄语义，不等价于 has_content）。"""
        return bool(self.blocks)

    @property
    def has_content(self) -> bool:
        """是否存在可参与路由与生命周期判断的内容（原始 blocks 或非空白折叠摘要）。"""
        return bool(self.blocks) or bool(self.state_summary.strip())

    def get_topic_summary(self) -> str:
        if self.blocks:
            user_queries = [b.anchor_text for b in self.blocks if b.anchor_text]
            if user_queries:
                return f"包含 {len(user_queries)} 个用户查询"
            return f"{len(self.blocks)} 个 Block"
        if self.state_summary.strip():
            return "已折叠历史内容"
        return "空缓冲区"

    def is_idle(self, timeout_seconds: int = 900) -> bool:
        return (datetime.now().timestamp() - self.last_update) > timeout_seconds


__all__ = ["SemanticBuffer"]
