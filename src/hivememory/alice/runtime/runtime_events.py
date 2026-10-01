"""Alice Agent run 的 RuntimeEvent 领域投影。"""

from __future__ import annotations

from dataclasses import dataclass

from hivememory.components.events.publisher import RuntimeEventPublisher
from hivememory.core.contracts.runtime_events import RuntimeEventType


@dataclass(frozen=True, slots=True)
class AgentRunStats:
    """``agent.run.*`` 终态事件的观测统计，由调用方从 frame 进度取得。

    Alice 的专属统计（MTP 迭代次数等）不进入 CPU 中立的执行结果，只在
    这里用于可观测性发布；载荷键名沿用既有 wire 名称。
    """

    mtp_iterations: int
    total_iterations: int
    materialize_task_count: int


class AgentRunEventEmitter:
    """把 Agent run 生命周期投影为全局可观测性事件。"""

    def __init__(self, publisher: RuntimeEventPublisher) -> None:
        self._publisher = publisher

    def for_run(
        self,
        *,
        agent_run_id: str,
        process_id: str | None,
        topic_id: str | None,
        agent_id: str | None,
        workspace_id: str | None = None,
    ) -> BoundAgentRunEvents:
        return BoundAgentRunEvents(
            self._publisher.bind(
                task_type="foreground",
                agent_run_id=agent_run_id,
                process_id=process_id,
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

    def completed(self, stats: AgentRunStats) -> None:
        self._publisher.emit(
            RuntimeEventType.AGENT_RUN_COMPLETED,
            status="completed",
            data=self._terminal_data(stats),
        )

    def cancelled(
        self,
        stats: AgentRunStats | None = None,
        *,
        message: str | None = None,
        close_reason: str | None = None,
    ) -> None:
        data = self._terminal_data(stats) if stats is not None else {}
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
        stats: AgentRunStats | None = None,
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
            data=self._terminal_data(stats) if stats is not None else None,
        )

    @staticmethod
    def _terminal_data(stats: AgentRunStats) -> dict[str, object]:
        return {
            "mtp_iterations": stats.mtp_iterations,
            "total_iterations": stats.total_iterations,
            "materialize_task_count": stats.materialize_task_count,
        }


__all__ = ["AgentRunEventEmitter", "AgentRunStats", "BoundAgentRunEvents"]
