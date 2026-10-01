"""任务进程的 RuntimeEvent 领域投影（``chat.run.*``）。

事件发生时机由进程编排显式决定；本模块只把进程记录的阶段与终态投影为
可观测事件，不修改进程记录。scope、关联上下文、payload 安全转换与
best-effort 边界统一由 :class:`RuntimeEventPublisher` 负责。
"""

from __future__ import annotations

from hivememory.components.events.publisher import RuntimeEventPublisher, Severity
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors import WorkspaceDomainError
from hivememory.workspace.process.table import (
    CancelResult,
    ProcessOutcome,
    ProcessPhase,
    ProcessRecord,
)

# 运行中的阶段对外投影为事件 status；进入终态后直接使用进程终态。
_RUNNING_STATUS = {
    ProcessPhase.CREATED: "created",
    ProcessPhase.GATEWAY: "preparing",
    ProcessPhase.PREPARE: "preparing",
    ProcessPhase.ACTOR: "streaming",
    ProcessPhase.FINALIZE: "finalizing",
    ProcessPhase.TERMINAL: "terminal",
}


def _event_status(record: ProcessRecord) -> str:
    if record.outcome is not ProcessOutcome.RUNNING:
        return str(record.outcome.value)
    return _RUNNING_STATUS[record.phase]


class TaskProcessEventEmitter:
    """把任务进程的生命周期投影为 ``chat.run.*`` 可观测事件（事件类型沿用既有 wire 名称）。"""

    def __init__(self, publisher: RuntimeEventPublisher) -> None:
        # 来源标签沿用迁移前的 wire 取值，前端展示不受生产端迁移影响。
        self._publisher = publisher.scoped(
            subsystem="system",
            component="chat_application_service",
        )

    def for_process(
        self,
        record: ProcessRecord,
        *,
        trace_id: str | None = None,
    ) -> BoundProcessEvents:
        """绑定一次进程的稳定关联字段（身份坐标取自进程创建时冻结的 scope）。"""
        return BoundProcessEvents(
            record,
            self._publisher.bind(
                task_type="foreground",
                trace_id=trace_id,
                process_id=record.process_id,
                workspace_id=record.identity_scope.workspace_identity.workspace_id,
                agent_id=record.identity_scope.actor_identity.agent_id,
            ),
        )

    def cancel_requested(self, result: CancelResult, *, workspace_id: str) -> None:
        """停止请求的即时判定；进程不存在（含跨 scope）时同样发布。"""
        self._publisher.bind(
            process_id=result.process_id,
            workspace_id=workspace_id,
        ).emit(
            RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED,
            status=result.status,
            reason=result.reason,
            data={"cancelled": result.cancelled},
        )


class BoundProcessEvents:
    """绑定一次进程的 ``chat.run.*`` 发布；status/reason 在发布时读取进程记录。"""

    def __init__(self, record: ProcessRecord, publisher: RuntimeEventPublisher) -> None:
        self._record = record
        self._publisher = publisher

    def bind_topic(self, topic_id: str) -> None:
        """prepare 返回后，此后的事件都关联本轮 Topic。"""
        self._publisher = self._publisher.bind(topic_id=topic_id)

    def created(self) -> None:
        self._emit(RuntimeEventType.CHAT_RUN_CREATED)

    def status(self) -> None:
        self._emit(RuntimeEventType.CHAT_RUN_STATUS)

    def command_completed(self, *, command_id: str) -> None:
        self._emit(RuntimeEventType.CHAT_RUN_COMPLETED, data={"command_id": command_id})

    def completed(self, *, memory_task_ids: list[str]) -> None:
        self._emit(
            RuntimeEventType.CHAT_RUN_COMPLETED,
            data={"memory_task_ids": memory_task_ids},
        )

    def cancelled(self, *, phase: ProcessPhase | None = None) -> None:
        """进程被取消；``phase`` 是停止请求生效的阶段，Actor 自行报告取消时为空。"""
        self._emit(
            RuntimeEventType.CHAT_RUN_CANCELLED,
            data={"phase": phase.value} if phase is not None else None,
        )

    def closed_before_terminal(self) -> None:
        """交付方在进程发布终态前关闭（如客户端断流），按取消收口。"""
        self._emit(
            RuntimeEventType.CHAT_RUN_CANCELLED,
            message="Task process stream closed before terminal event.",
            data={"close_reason": self._record.stop_reason or "stream_closed"},
        )

    def failed(self, error: Exception | None = None) -> None:
        """进程失败；``error`` 为空表示 Actor 自行报告失败。

        公共事件只携带 Workspace 领域错误的安全错误码，不写入异常正文。
        """
        if error is None:
            message = None
        elif isinstance(error, WorkspaceDomainError):
            message = error.code
        else:
            message = "Task process failed."
        self._emit(RuntimeEventType.CHAT_RUN_FAILED, severity="error", message=message)

    def _emit(
        self,
        event_type: RuntimeEventType,
        *,
        severity: Severity = "info",
        message: str | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        self._publisher.emit(
            event_type,
            status=_event_status(self._record),
            reason=self._record.stop_reason,
            severity=severity,
            message=message,
            data=data,
        )


__all__ = ["BoundProcessEvents", "TaskProcessEventEmitter"]
