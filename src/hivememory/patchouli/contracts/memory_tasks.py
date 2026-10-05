"""记忆生成任务的对外只读快照契约（依赖中立的公共契约）。

``MemoryGenerationTask`` 及其状态/来源枚举是 Patchouli 向 System/workspace
能力层公开的任务观察结果，按"共享 DTO 放在依赖中立契约"的规则上收到
``patchouli.contracts``（A2 §8.2）：能力层只依赖本模块，不导入控制面实现。
由通用 work queue 结果投影快照的工厂逻辑依赖 ``WorkState`` / ``TaskOutcome``，
留在 ``patchouli.control.memory_generation.models``。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from hivememory.core.models import ActorIdentity, WorkspaceIdentity


class MemoryGenerationTaskStatus(str, Enum):
    """对外记忆生成任务的生命周期状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class MemoryGenerationSource(str, Enum):
    """触发记忆生成的领域操作来源。"""

    WRITE = "WRITE"
    UPDATE = "UPDATE"
    SETTLE = "SETTLE"

    @property
    def creation_artifact_intent(
        self,
    ) -> Literal["ARCHIVE", "WRITE", "IMPORT", "MANUAL", "SYSTEM"]:
        """映射新建记忆制品使用的来源意图。"""

        if self == MemoryGenerationSource.SETTLE:
            return "SYSTEM"
        if self == MemoryGenerationSource.WRITE:
            return "WRITE"
        return "SYSTEM"

    @property
    def version_update_source(
        self,
    ) -> Literal["UPDATE"]:
        """映射记忆版本更新使用的来源类型。"""

        return "UPDATE"


@dataclass(frozen=True)
class MemoryGenerationTask:
    """单个记忆生成任务的对外只读快照。

    快照创建后不会原地更新。调用方需要通过控制器重新查询以获取新状态，不能
    把曾经取得的实例视为可观察的运行时句柄。
    """

    task_id: str
    topic_id: str
    label: str
    source: MemoryGenerationSource
    from_actor: ActorIdentity
    belong_to: WorkspaceIdentity
    pending_alias: str | None = None
    status: MemoryGenerationTaskStatus = MemoryGenerationTaskStatus.PENDING
    canonical_alias: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested: bool = False
    cancel_reason: str | None = None

    def as_failed(
        self,
        error: str,
        *,
        finished_at: datetime | None = None,
    ) -> MemoryGenerationTask:
        """从当前快照派生失败快照。"""

        return replace(
            self,
            status=MemoryGenerationTaskStatus.FAILED,
            error=error,
            finished_at=finished_at,
        )

    def with_cancel_request(self, reason: str) -> MemoryGenerationTask:
        """从当前快照派生已收到取消请求的快照。"""

        return replace(
            self,
            cancel_requested=True,
            cancel_reason=reason,
        )

    @property
    def cancelled(self) -> bool:
        """判断任务是否已收到取消请求或已经进入取消终态。"""

        return self.cancel_requested or self.status == MemoryGenerationTaskStatus.CANCELLED


__all__ = [
    "MemoryGenerationSource",
    "MemoryGenerationTask",
    "MemoryGenerationTaskStatus",
]
