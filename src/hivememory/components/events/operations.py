"""基于 RuntimeEvent 的子系统操作可复用观测器。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any, Literal, TypeVar

from hivememory.components.events.bus import NullRuntimeEventSink
from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.contracts.runtime_events import RuntimeEventType

T = TypeVar("T")


class RuntimeOperationObserver:
    """为单次子系统操作发出 STARTED/COMPLETED/FAILED 事件。"""

    def __init__(
        self,
        publisher: RuntimeEventPublisher | None = None,
        *,
        subsystem: str,
        component: str,
        operation_key: str,
        operation_name: str | None = None,
        operation_kind: str | None = None,
        task_type: Literal["foreground", "background"] | None = "background",
    ) -> None:
        self._events = (
            (publisher or RuntimeEventPublisher(NullRuntimeEventSink()))
            .scoped(subsystem=subsystem, source=operation_key, component=component)
            .bind(task_type=task_type)
        )
        self._operation_key = operation_key
        self._operation_name = operation_name or operation_key.rsplit(".", 1)[-1]
        self._operation_kind = operation_kind

    async def observe(
        self,
        run: Callable[[], Awaitable[T]],
        *,
        started_data: dict[str, Any] | None = None,
        summarize: Callable[[T], dict[str, Any]] | None = None,
        completed_status: Callable[[T], str] | str = "completed",
        completed_severity: (
            Callable[[T], Literal["debug", "info", "warning", "error"]]
            | Literal["debug", "info", "warning", "error"]
        ) = "info",
        failed_data: Callable[[BaseException], dict[str, Any]] | None = None,
    ) -> T:
        # RuntimeOperationObserver 只封装旁路观测事件，不驱动业务流程或重试策略。
        start_time = monotonic()
        self.emit_started(data=started_data)
        try:
            result = await run()
        except Exception as exc:
            self.emit_failed(
                exc,
                duration_ms=(monotonic() - start_time) * 1000,
                data=failed_data(exc) if failed_data is not None else None,
            )
            raise

        status = completed_status(result) if callable(completed_status) else completed_status
        severity = (
            completed_severity(result) if callable(completed_severity) else completed_severity
        )
        self.emit_completed(
            status=status,
            severity=severity,
            duration_ms=(monotonic() - start_time) * 1000,
            data=summarize(result) if summarize is not None else None,
        )
        return result

    def emit_started(self, *, data: dict[str, Any] | None = None) -> None:
        self._emit(
            RuntimeEventType.SUBSYSTEM_OPERATION_STARTED,
            status="started",
            data=data,
        )

    def emit_completed(
        self,
        *,
        status: str = "completed",
        severity: Literal["debug", "info", "warning", "error"] = "info",
        duration_ms: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self._emit(
            RuntimeEventType.SUBSYSTEM_OPERATION_COMPLETED,
            status=status,
            severity=severity,
            duration_ms=duration_ms,
            data=data,
        )

    def emit_failed(
        self,
        exc: BaseException,
        *,
        duration_ms: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """失败事件只带原因码与异常类名；异常由 ``observe`` 重新抛出，正文留给调用方记录。"""
        self._emit(
            RuntimeEventType.SUBSYSTEM_OPERATION_FAILED,
            status="failed",
            severity="error",
            reason="operation_failed",
            duration_ms=duration_ms,
            data={
                "success": False,
                **(data or {}),
                "error_class": type(exc).__name__,
            },
        )

    def _emit(
        self,
        event_type: RuntimeEventType,
        *,
        status: str,
        severity: Literal["debug", "info", "warning", "error"] = "info",
        reason: str | None = None,
        duration_ms: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "operation_key": self._operation_key,
            "operation_name": self._operation_name,
            "operation_kind": self._operation_kind,
            "duration_ms": duration_ms,
            **(data or {}),
        }
        self._events.emit(event_type, status=status, severity=severity, reason=reason, data=payload)


__all__ = ["RuntimeOperationObserver"]
