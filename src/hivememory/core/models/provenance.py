"""Memory 来源 provenance 值对象（依赖中立的数据定义）。

从 generation 层迁移至 core：``MemoryProvenance`` 的字段与归一化规则是
Memory/Artifact 共用的领域事实，不依赖任何生成上下文。上下文相关的构造
逻辑（从 ``GenerationContext``/``IdentityScope`` 构建）仍留在 generation 层。
"""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hivememory.core.constants import SYSTEM_AGENT_ID

__all__ = ["MemoryProvenance", "normalize_contributing_agent_ids"]


def normalize_contributing_agent_ids(value: Iterable[str]) -> tuple[str, ...]:
    """归一化内容贡献者集合：去重并保持首次出现顺序。

    贡献者表达"哪些具体 Agent 的工作产出了内容"，不表达资产归属或授权
    目标。保留 ``SYSTEM_AGENT_ID`` 表示"没有具体 Agent 作为操作来源主体"，
    不是内容贡献者，与空白标识一并丢弃。
    """
    normalized: list[str] = []
    for agent_id in value:
        stripped = agent_id.strip() if isinstance(agent_id, str) else ""
        if not stripped or stripped == SYSTEM_AGENT_ID:
            continue
        if stripped not in normalized:
            normalized.append(stripped)
    return tuple(normalized)


class MemoryProvenance(BaseModel):
    """
    一次记忆写入的来源裁定

    区分两个正交语义：
    - 操作来源主体：``source_agent_id`` / ``source_team_id``，回答"谁触发了
      这次写入"；被动结算等没有具体 Agent 的操作使用保留 ``SYSTEM_AGENT_ID``；
    - 内容贡献者：``contributing_agent_ids``，回答"哪些具体 Agent 的工作
      产出了内容"。

    两个字段都只记录 provenance 事实，不参与读取授权。
    """

    source_agent_id: str = Field(..., min_length=1)
    source_team_id: str | None = None
    contributing_agent_ids: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("contributing_agent_ids")
    @classmethod
    def _normalize_contributors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """去重并保持首次出现顺序；system 不是内容贡献者。"""
        return normalize_contributing_agent_ids(value)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
