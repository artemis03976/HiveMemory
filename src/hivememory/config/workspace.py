"""Workspace 运行时配置：派生缓存容量与会话容量。

缓存采用全局 LRU 容量，不承诺 Workspace 独立配额；负缓存首版不启用。
缓存容量只影响命中率与内存占用，当前性由失效事件协作保证，不以 TTL 代替一致性。
会话按 Workspace 数量与单会话轮数限额，达到上限时显式拒绝，不淘汰历史。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["WorkspaceCacheConfig", "WorkspaceConfig", "WorkspaceSessionConfig"]


class WorkspaceCacheConfig(BaseModel):
    """workspace 派生缓存容量。"""

    atom_capacity: int = Field(default=4096, ge=1, description="完整原子缓存的全局 LRU 容量")
    profile_capacity: int = Field(
        default=256, ge=1, description="Profile 解析结果缓存的全局 LRU 容量"
    )

    model_config = ConfigDict(extra="forbid")


class WorkspaceSessionConfig(BaseModel):
    """会话容量按 Workspace 与单个会话限制，用尽时不淘汰历史。"""

    max_sessions_per_workspace: int = Field(default=128, ge=1)
    max_turns_per_session: int = Field(default=512, ge=1)

    model_config = ConfigDict(extra="forbid")


class WorkspaceConfig(BaseModel):
    """workspace 运行时根配置。"""

    sessions: WorkspaceSessionConfig = Field(default_factory=WorkspaceSessionConfig)
    cache: WorkspaceCacheConfig = Field(default_factory=WorkspaceCacheConfig)

    model_config = ConfigDict(extra="forbid")
