"""正式记忆变更的失效通知，只携带资源坐标而不携带内容。"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from hivememory.core.models.identity import WorkspaceIdentity


class MemoryChangeEvent(BaseModel):
    """一次 canonical 提交尝试对应的读取投影失效通知。

    通知不证明持久化成功；primary 失败或 secondary 部分提交时仍需清除旧投影。
    """

    belong_to: WorkspaceIdentity
    memory_id: UUID
    operation: Literal["upsert", "patch", "delete"]
    model_config = ConfigDict(frozen=True)


__all__ = ["MemoryChangeEvent"]
