"""Workspace 基础设施 DTO。

本模块只保留 Workspace 基础设施自有的 DTO（canonical 失效描述等）。
跨边界的业务/结果 DTO 由对应领域契约承载；``MemorySnapshot``/
``ProfileSnapshot`` 是 Patchouli application 产出、未来 Workspace cache
消费的共享投影，按"DTO 放在依赖中立共享契约"的规则定义在
``hivememory.core.models.projections``（父计划 4.1/5.7.1 节）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hivememory.core.models import WorkspaceIdentity


@dataclass(frozen=True)
class CanonicalResourceChange:
    """一次 canonical mutation 的失效通知描述（``ResourceInvalidationPort`` 参数）。

    ``kind`` 覆盖父计划 6.2 节的失效来源；派生 cache 实现据此精确失效，
    不得把通知解释为"全量清空"或授权证明。
    """

    kind: Literal[
        "memory_created",
        "memory_updated",
        "memory_deleted",
        "memory_archived",
        "memory_revived",
        "profile_changed",
        "asset_representation_changed",
    ]
    workspace_identity: WorkspaceIdentity
    memory_id: str | None = None
    alias: str | None = None
    source_revision: int | None = None


__all__ = ["CanonicalResourceChange"]
