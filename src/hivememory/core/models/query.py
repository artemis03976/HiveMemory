"""检索过滤条件：跨子系统共享的结构化业务过滤模型（不含授权语义）。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models.memory import MemoryType


class QueryFilters(BaseModel):
    """不含授权语义的结构化业务过滤条件。"""

    memory_type: MemoryType | None = None
    # 匹配 meta.provenance.contributing_agent_ids 贡献者集合（可检出"参与
    # 过但未收尾"的 Agent），并保留 meta.provenance.source_agent_id 分支兼容
    # 无贡献者集合的记录。
    source_agent_id: str | None = None
    time_range: tuple[datetime, datetime] | None = None
    tags: list[str] = Field(default_factory=list)
    min_confidence: float = 0.0

    model_config = ConfigDict(extra="forbid")

    def is_empty(self) -> bool:
        return (
            self.memory_type is None
            and self.source_agent_id is None
            and self.time_range is None
            and len(self.tags) == 0
            and self.min_confidence == 0.0
        )


__all__ = ["QueryFilters"]
