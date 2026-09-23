"""跨子系统的不可变资源快照投影。

``ProfileSnapshot`` 由 Patchouli application 的 Profile 读取用例产出；按
"共享 DTO 放在依赖中立契约"的规则定义在 core，避免 Patchouli 与 Workspace
为取得 DTO 互导对方的 runtime 实现。Memory 侧不再提供裁剪投影：canonical
读取与历史记录使用完整 MemoryAtom（A2-P §3.3 引用隔离）。

快照与 canonical 对象解耦：携带 source identity/revision 供 freshness
校验，后续 canonical 变化不改变已取出的快照，也不表示快照仍然新鲜。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models.agent import AgentProfile


class ProfileSnapshot(BaseModel):
    """一次授权 Agent Profile 读取的快照投影。

    ``profile`` 是 canonical ``AgentProfile`` 的 copy-on-read 深拷贝：读取
    方每次获得独立副本，修改不影响 canonical 或其他调用方。内建 profile
    （default/omni_doll）以 ``source_kind="builtin"`` 显式标识，不把常量
    alias 当作跨 Workspace 授权旁路。
    """

    agent_alias: str | None = Field(default=None, description="请求的正式 alias；内建为 None")
    profile: AgentProfile = Field(description="Profile 投影（copy-on-read 深拷贝）")
    source_kind: Literal["builtin", "atom"]
    source_atom_uuid: str | None = Field(default=None, description="来源 atom UUID")
    source_revision: int | None = Field(default=None, description="来源 atom 版本")

    model_config = ConfigDict(frozen=True)


__all__ = [
    "ProfileSnapshot",
]
