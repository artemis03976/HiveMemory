"""帧执行结果到 CPU 中立结果的投影与运行终态观测。"""

from hivememory.agent_runtime import (
    ExecutionFrame,
    FrameExecutionResult,
    FrameExecutionStatus,
    InputTooLargeError,
    ModelContextWindowExceededError,
    TurnContextBudgetExceededError,
)
from hivememory.alice.context.view import CompactionFailedError
from hivememory.alice.events import AgentRunStats, BoundAgentRunEvents
from hivememory.core.protocol.models import SessionCompactionCheckpoint
from hivememory.workspace.contracts import CPUExecutionResult, CPUExecutionStatus, CPUFailureReason


def assemble_execution_result(
    frame: ExecutionFrame,
    engine_result: FrameExecutionResult,
    checkpoint: SessionCompactionCheckpoint | None = None,
) -> CPUExecutionResult:
    """把执行层终态与轮次事件投影为 CPU 中立的执行结果。"""
    if engine_result.status == FrameExecutionStatus.CANCELLED:
        run_status = CPUExecutionStatus.CANCELLED
    elif engine_result.status == FrameExecutionStatus.COMPLETED:
        run_status = CPUExecutionStatus.COMPLETED
    else:
        run_status = CPUExecutionStatus.FAILED
    failure_reason = None
    if run_status == CPUExecutionStatus.FAILED:
        if isinstance(engine_result.error, InputTooLargeError):
            failure_reason = CPUFailureReason.INPUT_TOO_LARGE
        elif isinstance(engine_result.error, CompactionFailedError):
            failure_reason = CPUFailureReason.COMPACTION_FAILED
        elif isinstance(engine_result.error, TurnContextBudgetExceededError):
            failure_reason = CPUFailureReason.TURN_CONTEXT_BUDGET_EXCEEDED
        elif isinstance(engine_result.error, ModelContextWindowExceededError):
            failure_reason = CPUFailureReason.MODEL_CONTEXT_WINDOW_EXCEEDED
    progress = frame.progress
    return CPUExecutionResult(
        status=run_status,
        failure_reason=failure_reason,
        compaction_checkpoint=checkpoint,
        final_text="".join(progress.text_segments),
        turn_events=progress.turn_events,
        model_used=progress.model_used,
    )


def stats_for(frame: ExecutionFrame) -> AgentRunStats:
    """从 frame 进度取得 ``agent.run.*`` 终态事件的观测统计。"""
    progress = frame.progress
    return AgentRunStats(
        mtp_iterations=max(0, progress.iteration - 1),
        total_iterations=progress.iteration,
    )


def publish_terminal(
    run_events: BoundAgentRunEvents,
    result: CPUExecutionResult,
    stats: AgentRunStats,
) -> None:
    if result.status == CPUExecutionStatus.CANCELLED.value:
        run_events.cancelled(stats)
    elif result.status == CPUExecutionStatus.FAILED.value:
        run_events.failed(stats)
    else:
        run_events.completed(stats)
