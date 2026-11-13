"""Workspace 运行时配置：派生缓存容量、会话容量、任务进程关闭等待与写入句柄登记。

缓存采用全局 LRU 容量，不承诺 Workspace 独立配额；不缓存缺失或不可见的结果。
缓存容量只影响命中率与内存占用，当前性由失效事件协作保证，不以 TTL 代替一致性。
会话按 Workspace 数量与单会话轮数限额，达到上限时显式拒绝，不淘汰历史。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "WorkspaceCacheConfig",
    "WorkspaceConfig",
    "WorkspaceProcessConfig",
    "WorkspaceSessionConfig",
    "WorkspaceWriteIntentsConfig",
]


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


class WorkspaceProcessConfig(BaseModel):
    """任务进程配置：System 关闭时等待交付方自行收口的上限，超时后兜底关闭。"""

    shutdown_wait_seconds: float = Field(default=10.0, ge=0)

    model_config = ConfigDict(extra="forbid")


class WorkspaceWriteIntentsConfig(BaseModel):
    """写入句柄登记：正常关闭时写盘、启动时读回，登记满保留天数的句柄删除。"""

    record_file: str = Field(
        default=".hivememory/write_intents.json",
        description="写入句柄登记写盘的文件，下次启动时读回；相对路径按工作目录解析",
    )
    retention_days: int = Field(
        default=7,
        ge=1,
        description="写入句柄自登记起保留的天数，之后删除，读取时按已过期交付",
    )

    model_config = ConfigDict(extra="forbid")


class WorkspaceConfig(BaseModel):
    """workspace 运行时根配置。"""

    sessions: WorkspaceSessionConfig = Field(default_factory=WorkspaceSessionConfig)
    process: WorkspaceProcessConfig = Field(default_factory=WorkspaceProcessConfig)
    cache: WorkspaceCacheConfig = Field(default_factory=WorkspaceCacheConfig)
    write_intents: WorkspaceWriteIntentsConfig = Field(default_factory=WorkspaceWriteIntentsConfig)

    model_config = ConfigDict(extra="forbid")
