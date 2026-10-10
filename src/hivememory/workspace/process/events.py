"""任务进程的 RuntimeEvent 领域投影（``chat.run.*``）。

事件发生时机由进程编排显式决定；本模块只把进程记录的阶段与终态投影为
可观测事件，不修改进程记录。scope、关联上下文、payload 安全转换与
best-effort 边界统一由 :class:`RuntimeEventPublisher` 负责。

:class:`BoundProcessEvents` 持有只读观测标签及其绑定发布器，不回指进程
记录——发布时需要的状态（phase/outcome/stop_reason）由调用方显式传入，
避免事件发布器与进程记录互相引用。
"""

from __future__ import annotations

from hivememory.components.events.publisher import RuntimeEventPublisher, Severity
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.errors import WorkspaceDomainError
from hivememory.core.models import ExecutionLabels
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
        *,
        process_id: str,
        labels: ExecutionLabels,
        trace_id: str | None = None,
    ) -> BoundProcessEvents:
        """绑定一次进程的稳定关联字段。

        ``labels`` 由注册入口在认证成功后绑定一次：事件与 CPU 输入清单
        共用这组只读字符串，不等于授权或分区，也不携带身份数据。
        """
        return BoundProcessEvents(
            self._publisher.bind(
                task_type="foreground",
                trace_id=trace_id,
                process_id=process_id,
                workspace_id=labels.workspace_id,
                agent_id=labels.agent_id,
            ),
            labels=labels,
        )

    def cancel_requested(
        self,
        result: CancelResult,
        *,
        workspace_id: str | None = None,
    ) -> None:
        """停止请求的即时判定；进程不存在或不可控时同样发布。

        ``workspace_id`` 是观测标签：进程存在时取注册时绑定的标签，否则
        可传请求方 context 的驻留 workspace 摘要（经认证网关的诊断查询）。
        """
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
    """绑定一次进程的 ``chat.run.*`` 发布；不回指进程记录。

    发布时的进程状态（status/reason）由调用方显式传入：本类只经注册入口
    写入进程记录的 ``events`` 字段、由进程编排与控制面读取，二者单向
    依赖，不做互相引用。``labels`` 是注册时绑定的唯一标签载体，也交给
    CPU 输入清单，标签不参与授权。
    """

    def __init__(self, publisher: RuntimeEventPublisher, *, labels: ExecutionLabels) -> None:
        self._publisher = publisher
        self._labels = labels

    @property
    def labels(self) -> ExecutionLabels:
        """注册时绑定的只读观测标签，供事件与 CPU 输入清单共同使用。"""
        return self._labels

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

    def completed(self, record: ProcessRecord, *, memory_task_ids: list[str]) -> None:
        self._emit(
            record,
            RuntimeEventType.CHAT_RUN_COMPLETED,
            data={"memory_task_ids": memory_task_ids},
        )

    def cancelled(self, record: ProcessRecord, *, phase: ProcessPhase | None = None) -> None:
        """进程被取消；``phase`` 是停止请求生效的阶段，Actor 自行报告取消时为空。"""
        self._emit(
            record,
            RuntimeEventType.CHAT_RUN_CANCELLED,
            data={"phase": phase.value} if phase is not None else None,
        )

    def closed_before_terminal(self, record: ProcessRecord) -> None:
        """交付方在进程发布终态前关闭（如客户端断流），按取消收口。"""
        self._emit(
            record,
            RuntimeEventType.CHAT_RUN_CANCELLED,
            message="Task process stream closed before terminal event.",
            data={"close_reason": record.stop_reason or "stream_closed"},
        )

    def failed(self, record: ProcessRecord, error: Exception | None = None) -> None:
        """进程失败；``error`` 为空表示 Actor 自行报告失败。

        公共事件只携带 Workspace 领域错误的安全错误码，不写入异常正文。
        """
        if error is None:
            message = None
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
