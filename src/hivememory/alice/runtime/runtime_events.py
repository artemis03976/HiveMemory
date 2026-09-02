"""Alice Agent run 的 RuntimeEvent 领域投影。"""

from __future__ import annotations

from hivememory.core.protocol.models import AgentRunResult
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.publisher import RuntimeEventPublisher


class AgentRunEventEmitter:
    """把 Agent run 生命周期投影为全局可观测性事件。"""

    def __init__(self, publisher: RuntimeEventPublisher) -> None:
        self._publisher = publisher

    def for_run(
        self,
        *,
        agent_run_id: str,
        generation_id: str | None,
        topic_id: str | None,
        agent_id: str | None,
        workspace_id: str | None = None,
    ) -> BoundAgentRunEvents:
        return BoundAgentRunEvents(
            self._publisher.bind(
                task_type="foreground",
                agent_run_id=agent_run_id,
                generation_id=generation_id,
                interaction_id=generation_id,
                topic_id=topic_id,
                agent_id=agent_id,
                workspace_id=workspace_id,
            )
        )


class BoundAgentRunEvents:
    """绑定一次 run 的稳定关联字段，只负责可观测性发布。"""

    def __init__(self, publisher: RuntimeEventPublisher) -> None:
        self._publisher = publisher

    def started(self) -> None:
        self._publisher.emit(
            RuntimeEventType.AGENT_RUN_STARTED,
            status="started",
        )

    def completed(self, result: AgentRunResult) -> None:
        self._publisher.emit(
            RuntimeEventType.AGENT_RUN_COMPLETED,
            status=str(result.status),
            data=self._terminal_data(result),
        )

    def cancelled(
        self,
        result: AgentRunResult | None = None,
        *,
        message: str | None = None,
        close_reason: str | None = None,
    ) -> None:
        data = self._terminal_data(result) if result is not None else {}
        if close_reason is not None:
            data["close_reason"] = close_reason
        self._publisher.emit(
            RuntimeEventType.AGENT_RUN_CANCELLED,
            status="cancelled",
            message=message,
            data=data,
        )

    def failed(
        self,
        result: AgentRunResult | None = None,
        *,
        message: str | None = None,
        reason: str | None = None,
    ) -> None:
        self._publisher.emit(
            RuntimeEventType.AGENT_RUN_FAILED,
            status="failed",
            severity="error",
            reason=reason,
            message=message,
            data=self._terminal_data(result) if result is not None else None,
        )

    @staticmethod
    def _terminal_data(result: AgentRunResult) -> dict[str, object]:
        return {
            "mtp_iterations": result.mtp_iterations,
            "total_iterations": result.total_iterations,
            "materialize_task_count": len(result.materialize_tasks),
        }


__all__ = ["AgentRunEventEmitter", "BoundAgentRunEvents"]
