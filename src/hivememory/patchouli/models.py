"""Public Patchouli service contract models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hivememory.core.models import AgentProfile, Identity, TopicSnapshot
from hivememory.core.protocol.models import AgentRunContext, EyeGazeResult


@dataclass
class StreamPrelude:
    """Data emitted before streaming Agent tokens."""

    topic_id: str
    is_new_topic: bool
    pool_topics: List[TopicSnapshot]
    memory_refs: List[Any]
    max_resident_topics: int = 0

    @property
    def pool_snapshot(self) -> Dict[str, Any]:
        """兼容旧前端包格式；新代码应直接使用 pool_topics。"""
        return {
            "topics": [topic.model_dump(mode="json") for topic in self.pool_topics],
            "max_resident_topics": self.max_resident_topics,
            "current_count": len(self.pool_topics),
        }


@dataclass
class PreparedAgentRun:
    """Complete context prepared by Patchouli for one Agent run."""

    agent_run_context: AgentRunContext
    gaze_result: EyeGazeResult
    stream_prelude: StreamPrelude
    generation_options: Optional[Dict[str, Any]] = field(default=None)

    @property
    def identity(self) -> Identity:
        return self.agent_run_context.identity

    @property
    def agent_id(self) -> str:
        return self.agent_run_context.identity.agent_id

    @property
    def topic_id(self) -> str:
        return self.agent_run_context.topic_id

    @property
    def user_message(self) -> str:
        return self.agent_run_context.user_message

    @property
    def agent_profile(self) -> AgentProfile:
        return self.agent_run_context.agent_profile
