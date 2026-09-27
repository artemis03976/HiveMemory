"""Workspace 运行时配置：读取能力派生缓存的容量（A2 §3.2 / §8.2）。

全局 LRU 容量，不承诺 Workspace 独立配额；负缓存首版不启用，无对应配置。
容量只影响命中率与内存占用，不影响正确性——当前性由失效事件协作保证，
不以 TTL 代替写入一致性。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["WorkspaceCacheConfig", "WorkspaceConfig"]


class WorkspaceCacheConfig(BaseModel):
    """workspace 派生缓存容量。"""

    atom_capacity: int = Field(default=4096, ge=1, description="完整原子缓存的全局 LRU 容量")
    profile_capacity: int = Field(
        default=256, ge=1, description="Profile 解析结果缓存的全局 LRU 容量"
    )

    model_config = ConfigDict(extra="forbid")


class WorkspaceConfig(BaseModel):
    """workspace 运行时根配置。"""

    cache: WorkspaceCacheConfig = Field(default_factory=WorkspaceCacheConfig)

    model_config = ConfigDict(extra="forbid")
