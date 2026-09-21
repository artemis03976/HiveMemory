"""跨子系统的不可变资源快照投影。

``ProfileSnapshot`` 已由 Patchouli application 的 Profile 读取用例产出；
``MemorySnapshot`` 是同一快照契约的 Memory 侧形状，自 WRX-3 起随派生
cache/检索快照语义接入消费（父计划 5.2/5.3/7.6 节）。按"共享 DTO 放在依赖中立契约"的规则定义在 core，
避免 Patchouli 与 Workspace 为取得 DTO 互导对方的 runtime 实现。

快照与 canonical 对象解耦：携带 source identity/revision 供 freshness
校验，后续 canonical 变化不改变已取出的快照，也不表示快照仍然新鲜。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hivememory.core.models.agent import AgentProfile
from hivememory.core.models.memory import MemoryAccessPolicy, MemoryAtom
from hivememory.core.models.workspace import WorkspaceIdentity


class MemorySnapshot(BaseModel):
    """一次授权 Memory 读取的不可变快照投影。

    携带 canonical identity（``memory_id`` + ``workspace_identity``）、
    source revision（``meta.version``，乐观锁版本）与内容投影。
    """

    memory_id: str = Field(description="canonical Memory UUID（全局寻址 ID）")
    workspace_identity: WorkspaceIdentity = Field(description="资源归属 Workspace")
    alias: str = Field(description="Workspace 分区内的正式别名")
    memory_type: str = Field(description="MemoryType 值")
    title: str | None = None
    summary: str | None = None
    content: str = Field(description="payload 内容全文")
    tags: tuple[str, ...] = ()
    source_agent_id: str = Field(description="来源 provenance（非授权字段）")
    visibility: str = Field(description="MemoryAccessPolicy 可见性值")
    confidence_score: float = 0.0
    created_at: datetime
    updated_at: datetime
    source_revision: int = Field(description="canonical source 版本（meta.version）")

    @classmethod
    def from_atom(cls, atom: MemoryAtom) -> MemorySnapshot:
        """从已通过授权校验的 canonical atom 构建快照投影。"""
        policy: MemoryAccessPolicy = atom.meta.access_policy
        index = atom.index
        return cls(
            memory_id=str(atom.id),
            workspace_identity=atom.meta.workspace_identity,
            alias=atom.get_alias(),
            memory_type=index.memory_type.value,
            title=index.title,
            summary=index.summary,
            content=atom.payload.content,
            tags=tuple(index.tags or ()),
            source_agent_id=atom.meta.source_agent_id,
            visibility=policy.visibility.value,
            confidence_score=atom.meta.confidence_score,
            created_at=atom.meta.created_at,
            updated_at=atom.meta.updated_at,
            source_revision=atom.meta.version,
        )

    model_config = ConfigDict(frozen=True)


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
    "MemorySnapshot",
    "ProfileSnapshot",
]
