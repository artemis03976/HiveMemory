"""任务进程的 RuntimeEvent 领域投影（``chat.run.*``）。

事件发生时机由进程编排显式决定；本模块只把进程记录的阶段与终态投影为
可观测事件，不修改进程记录。scope、关联上下文、payload 安全转换与
best-effort 边界统一由 :class:`RuntimeEventPublisher` 负责。

:class:`BoundProcessEvents` 持有绑定了进程关联字段的发布器，不回指进程
记录——发布时需要的状态（phase/outcome/stop_reason）由调用方显式传入，
避免事件发布器与进程记录互相引用。
"""

from __future__ import annotations

from hivememory.components.events.publisher import RuntimeEventPublisher, Severity
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors import WorkspaceDomainError
from hivememory.core.models import ExecutionLabels
from hivememory.workspace.contracts import CPUFailureReason
from hivememory.workspace.process.table import (
    SYSTEM_SHUTDOWN_STOP_REASON,
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
        # 来源标签标明事件由 workspace 的任务进程产生。
        self._publisher = publisher.scoped(
            subsystem="workspace",
            component="task_process",
        )

    def for_process(
        self,
        *,
        process_id: str,
        labels: ExecutionLabels,
        trace_id: str | None = None,
    ) -> BoundProcessEvents:
        """绑定一次进程的稳定关联字段。

        ``labels`` 是注册入口在认证成功后创建的观测标签，只作为事件的关联
        字段，不等于授权或分区，也不携带身份数据。
        """
        return BoundProcessEvents(
            self._publisher.bind(
                task_type="foreground",
                trace_id=trace_id,
                process_id=process_id,
                workspace_id=labels.workspace_id,
                agent_id=labels.agent_id,
            )
        )

    def cancel_requested(self, result: CancelResult) -> None:
        """停止请求找不到可控进程时的即时判定。

        进程不存在与请求方无权控制同样呈现为 ``not_found``；事件只带请求中的
        ``process_id``，不带其他关联字段。
        """
        self._publisher.bind(process_id=result.process_id).emit(
            RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED,
            status=result.status,
            reason=result.reason,
            data={"cancelled": result.cancelled},
        )


class BoundProcessEvents:
    """绑定一次进程的 ``chat.run.*`` 发布；不回指进程记录。

    发布时的进程状态（status/reason）由调用方显式传入：本类只经注册入口
    写入进程记录的 ``events`` 字段、由进程编排与控制面读取，二者单向
    依赖，不做互相引用。
    """

    def __init__(self, publisher: RuntimeEventPublisher) -> None:
        self._publisher = publisher

    def bind_topic(self, topic_id: str) -> None:
        """prepare 返回后，此后的事件都关联本轮 Topic。"""
        self._publisher = self._publisher.bind(topic_id=topic_id)

    def created(self, record: ProcessRecord) -> None:
        self._emit(record, RuntimeEventType.CHAT_RUN_CREATED)

    def cancel_requested(self, result: CancelResult) -> None:
        """停止请求的即时判定（经进程绑定的稳定关联字段发布）。"""
        self._publisher.emit(
            RuntimeEventType.CHAT_RUN_CANCEL_REQUESTED,
            status=result.status,
            reason=result.reason,
            data={"cancelled": result.cancelled},
        )

    def status(self, record: ProcessRecord) -> None:
        self._emit(record, RuntimeEventType.CHAT_RUN_STATUS)

    def command_completed(self, record: ProcessRecord, *, command_id: str) -> None:
        self._emit(
            record,
            RuntimeEventType.CHAT_RUN_COMPLETED,
            data={"command_id": command_id},
        )

    def completed(self, record: ProcessRecord) -> None:
        self._emit(
            record,
            RuntimeEventType.CHAT_RUN_COMPLETED,
        )

    def cancelled(self, record: ProcessRecord, *, phase: ProcessPhase | None = None) -> None:
        """进程被取消；``phase`` 是停止请求生效的阶段，Actor 自行报告取消时为空。"""
        self._emit(
            record,
            RuntimeEventType.CHAT_RUN_CANCELLED,
            data={"phase": phase.value} if phase is not None else None,
        )

    def closed_before_terminal(self, record: ProcessRecord) -> None:
        """进程在记录终态之前就被关闭，按取消收口。

        只有两种来源：进程从未运行就被关闭，以及系统关闭等待超时后硬取消
        进程 task。交付方离开走普通的停止路径，不使用这种形态。硬取消只来自
        系统关闭，停止原因缺失时（拒绝停止的 finalize 阶段）按系统关闭记录。
        """
        self._emit(
            record,
            RuntimeEventType.CHAT_RUN_CANCELLED,
            message="Task process closed before terminal event.",
            data={"close_reason": record.stop_reason or SYSTEM_SHUTDOWN_STOP_REASON},
        )

    def failed(
        self,
        record: ProcessRecord,
        error: Exception | None = None,
        *,
        failure_reason: CPUFailureReason | None = None,
    ) -> None:
        """进程失败；``error`` 为空表示 Actor 自行报告失败。

        公共事件只携带 CPU 失败原因码或 Workspace 安全错误码，不写入异常正文。
        """
        if error is None:
            message = failure_reason.value if failure_reason is not None else None
        elif isinstance(error, WorkspaceDomainError):
            message = error.code
        else:
            message = "Task process failed."
        self._emit(
            record,
            RuntimeEventType.CHAT_RUN_FAILED,
            severity="error",
            message=message,
        )

    def _emit(
        self,
        record: ProcessRecord,
        event_type: RuntimeEventType,
        *,
        severity: Severity = "info",
        message: str | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        self._publisher.emit(
            event_type,
            status=_event_status(record),
            reason=record.stop_reason,
            severity=severity,
            message=message,
            data=data,
        )


__all__ = ["BoundProcessEvents", "TaskProcessEventEmitter"]
