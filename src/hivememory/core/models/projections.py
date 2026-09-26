"""跨子系统的资源读取结果投影。

``ResolvedAgentProfile`` 是 Patchouli Profile 读取 backing（``GET_AGENT_PROFILE``）
的返回值，按"共享 DTO 放在依赖中立契约"的规则定义在 core，避免 Patchouli 与
Workspace 为取得 DTO 互导对方的实现。Memory 侧不提供裁剪投影：canonical 读取
与历史记录使用完整 MemoryAtom（A2-P §3.3 引用隔离）。
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hivememory.core.models.agent import AgentProfile
from hivememory.core.models.memory import MemoryAccessPolicy


class ResolvedAgentProfile(BaseModel):
    """一次 Profile 定义解析的 backing 结果（A2 §8 D-3）。

    ``profile`` 是解析出的 ``AgentProfile`` 独立副本（``agent_id`` 取自源原子
    alias）；``access_policy`` / ``source_memory_id`` / ``source_version`` 是
    源原子的可见性依据与关联信息，只在 workspace→Patchouli 的进程内 backing
    契约上流动，供 Profile 解析缓存做命中授权与失效对账——可见性不进
    AgentProfile 模型，能力面对 actor 只交付 ``profile``。

    builtin Profile（default / omni_doll）没有源原子，三项关联字段均为
    ``None``；不伪造 source atom，也不把常量 alias 当作授权依据。
    """

    profile: AgentProfile = Field(description="解析出的 AgentProfile 独立副本")
    access_policy: MemoryAccessPolicy | None = Field(
        default=None, description="源原子的读取策略；builtin 为 None"
    )
    source_memory_id: UUID | None = Field(default=None, description="源原子 UUID")
    source_version: int | None = Field(
        default=None, description="源原子内容版本（仅诊断，不作为当前性依据）"
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_source_consistency(self) -> ResolvedAgentProfile:
        """源原子关联要么完整（atom 来源）要么全缺（builtin），不允许半截结果。"""
        source_fields = (self.access_policy, self.source_memory_id, self.source_version)
        present = [value is not None for value in source_fields]
        if any(present) and not all(present):
            raise ValueError(
                "ResolvedAgentProfile 的 access_policy/source_memory_id/source_version "
                "必须同时提供（atom 来源）或同时缺省（builtin）"
            )
        return self

    @property
    def is_builtin(self) -> bool:
        """是否为无源原子的 builtin Profile。"""
        return self.source_memory_id is None


__all__ = [
    "ResolvedAgentProfile",
]
