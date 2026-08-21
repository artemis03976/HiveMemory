"""RuntimeEvent 生产端的统一发布入口。"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Literal

from pydantic import BaseModel

from hivememory.system.contracts.runtime_events import RuntimeEvent, RuntimeEventType
from hivememory.system.runtime.events import RuntimeEventSink, safe_runtime_event_value

logger = logging.getLogger(__name__)

type Severity = Literal["debug", "info", "warning", "error"]
type TaskType = Literal["foreground", "background"]
type RuntimeEventData = BaseModel | Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RuntimeEventContext:
    task_type: TaskType | None = None
    trace_id: str | None = None
    generation_id: str | None = None
    interaction_id: str | None = None
    agent_run_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    frame_id: str | None = None
    topic_id: str | None = None
    atom_id: str | None = None
    workspace_id: str | None = None


class RuntimeEventPublisher:
    """集中构造并 best-effort 发布 RuntimeEvent。"""

    def __init__(
        self,
        sink: RuntimeEventSink,
        *,
        subsystem: str | None = None,
        source: str | None = None,
        component: str | None = None,
        context: RuntimeEventContext | None = None,
    ) -> None:
        self._sink = sink
        self._subsystem = subsystem
        self._source = source
        self._component = component
        self._context = context or RuntimeEventContext()

    def scoped(
        self,
        *,
        subsystem: str | None = None,
        source: str | None = None,
        component: str | None = None,
    ) -> RuntimeEventPublisher:
        return RuntimeEventPublisher(
            self._sink,
            subsystem=subsystem or self._subsystem,
            source=source or self._source,
            component=component or self._component,
            context=self._context,
        )

    def bind(
        self,
        *,
        task_type: TaskType | None = None,
        trace_id: str | None = None,
        generation_id: str | None = None,
        interaction_id: str | None = None,
        agent_run_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        frame_id: str | None = None,
        topic_id: str | None = None,
        atom_id: str | None = None,
        workspace_id: str | None = None,
    ) -> RuntimeEventPublisher:
        updates = {
            key: value
            for key, value in {
                "task_type": task_type,
                "trace_id": trace_id,
                "generation_id": generation_id,
                "interaction_id": interaction_id,
                "agent_run_id": agent_run_id,
                "task_id": task_id,
                "agent_id": agent_id,
                "frame_id": frame_id,
                "topic_id": topic_id,
                "atom_id": atom_id,
                "workspace_id": workspace_id,
            }.items()
            if value is not None
        }
        return RuntimeEventPublisher(
            self._sink,
            subsystem=self._subsystem,
            source=self._source,
            component=self._component,
            context=replace(self._context, **updates),
        )

    def emit(
        self,
        event_type: RuntimeEventType,
        *,
        status: str | None = None,
        severity: Severity = "info",
        reason: str | None = None,
        message: str | None = None,
        data: RuntimeEventData | None = None,
    ) -> None:
        try:
            event_data = self._prepare_data(data)
            self._sink.emit(
                RuntimeEvent(
                    event_type=event_type,
                    subsystem=self._subsystem,
                    source=self._source or self._subsystem,
                    component=self._component,
                    severity=severity,
                    status=status,
                    reason=reason,
                    message=message,
                    data=event_data,
                    **asdict(self._context),
                )
            )
        except Exception:
            logger.warning("RuntimeEventPublisher emit failed", exc_info=True)

    @staticmethod
    def _prepare_data(data: RuntimeEventData | None) -> dict[str, object]:
        if data is None:
            return {}
        if isinstance(data, BaseModel):
            value = data.model_dump(mode="json")
        elif isinstance(data, Mapping):
            value = dict(data)
        else:
            raise TypeError(f"Unsupported RuntimeEvent payload: {type(data)!r}")
        prepared = safe_runtime_event_value(value)
        if not isinstance(prepared, dict):
            raise TypeError("RuntimeEvent payload must resolve to an object.")
        return prepared


__all__ = [
    "RuntimeEventContext",
    "RuntimeEventData",
    "RuntimeEventPublisher",
    "Severity",
    "TaskType",
]
