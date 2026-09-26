"""派生缓存的显式资源 key 类型（A2 §3）。

两个 cache 的资源 key 只由 Workspace 归属坐标与资源引用组成，不包含
Actor 的 user/agent/team/session 投影：同一 Workspace 的多个 Actor 复用
同一资源条目，每次交付时按资源 policy 分别授权。``WorkspaceIdentity`` 自身
保留 owner 坐标，UUID 与 alias 只在 Workspace 分区内寻址，不提供跨分区
的成功旁路。
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from hivememory.core.models import WorkspaceIdentity


@dataclass(frozen=True, slots=True)
class AtomIdKey:
    """Atom cache 的主键：Workspace 分区内的 memory UUID。"""

    workspace: WorkspaceIdentity
    memory_id: UUID


@dataclass(frozen=True, slots=True)
class AtomAliasKey:
    """Atom cache 的 alias 索引键：Workspace 分区内的规范化 alias。"""

    workspace: WorkspaceIdentity
    alias: str


@dataclass(frozen=True, slots=True)
class ProfileKey:
    """Profile 解析缓存键：Workspace 分区内的 agent alias（即 agent_id）。"""

    workspace: WorkspaceIdentity
    agent_alias: str


__all__ = ["AtomAliasKey", "AtomIdKey", "ProfileKey"]
