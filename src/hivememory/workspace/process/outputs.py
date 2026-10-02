"""任务进程的阶段产出与两种交付形态。

编排骨架只产出类型化的阶段产出（:data:`ProcessOutput`），不关心调用方是
流式还是非流式：流式交付由 :func:`stream_events` 把产出投影为 SSE 事件，
非流式交付只取终态产出组装 :data:`NonStreamingResult`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from hivememory.core.models import MemoryAtom
from hivememory.core.protocol.gateway import CommandExecutionResult
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUExecutionResult, CPUInputManifest

# ========== 阶段产出 ==========


@dataclass(frozen=True, slots=True)
class ProcessStarted:
    """进程已登记进进程表，``process_id`` 从此可用于停止请求。"""


@dataclass(frozen=True, slots=True)
class InputsAllocated:
    """CPU 分配成功且进入 Actor 前未被停止。"""

    prepared: PreparedAgentRun
    manifest: CPUInputManifest


@dataclass(frozen=True, slots=True)
class ActorEvent:
    """流式 Actor 的交互输出（token、MTP/CALL 等），原样转交。"""

    event: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Finalizing:
    """Actor 正常完成，进程已进入 finalize（此后拒绝取消）。"""


@dataclass(frozen=True, slots=True)
class CommandCompleted:
    """终态：Gateway 命中系统指令，进程在 Gateway 阶段短路完成。"""

    command_result: CommandExecutionResult


@dataclass(frozen=True, slots=True)
class RunCompleted:
    """终态：Actor 完成且 finalize 已接管本轮交互。"""

    execution_result: CPUExecutionResult
    memory_task_ids: list[str]
    pool_topics: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RunCancelled:
    """终态：停止请求生效，或 Actor 自行报告取消（此时带执行结果）。"""

    reason: str
    execution_result: CPUExecutionResult | None = None


@dataclass(frozen=True, slots=True)
class RunFailed:
    """终态：Actor 自行报告失败。"""

    execution_result: CPUExecutionResult


@dataclass(frozen=True, slots=True)
class ProcessFailed:
    """终态：编排途中出现异常（非流式交付原样上抛）。"""

    error: Exception


type TerminalOutput = CommandCompleted | RunCompleted | RunCancelled | RunFailed | ProcessFailed
type ProcessOutput = ProcessStarted | InputsAllocated | ActorEvent | Finalizing | TerminalOutput


# ========== 非流式交付 ==========


@dataclass(frozen=True, kw_only=True)
class NonStreamingCommandOutcome:
    """非流式交付的系统指令终态。"""

    kind: Literal["command"] = "command"
    command_execution_result: CommandExecutionResult


@dataclass(frozen=True, kw_only=True)
class NonStreamingAgentOutcome:
    """非流式交付的 Agent 运行终态。"""

    kind: Literal["agent"] = "agent"
    execution_result: CPUExecutionResult


type NonStreamingResult = NonStreamingCommandOutcome | NonStreamingAgentOutcome


# ========== 流式交付 ==========


def stream_events(output: ProcessOutput, *, process_id: str) -> list[dict[str, Any]]:
    """把一项阶段产出投影为流式事件（``ProcessFailed`` 由调用方翻译）。"""
    match output:
        case ProcessStarted():
            return [{"event": "process_id", "data": {"process_id": process_id}}]
        case InputsAllocated(prepared=prepared, manifest=manifest):
            return [
                {
                    "event": "topic_info",
                    "data": {
                        "topic_id": prepared.topic_id,
                        "is_new": prepared.is_new_topic,
                        "pool_topics": [
                            topic.model_dump(mode="json") for topic in prepared.pool_topics
                        ],
                    },
                },
                {
                    "event": "memory_refs",
                    "data": {
                        "memories": [_memory_ref_from_atom(memory) for memory in manifest.memories]
                    },
                },
            ]
        case ActorEvent(event=event):
            return [event]
        case Finalizing():
            return [
                {
                    "event": "run_status",
                    "data": {"process_id": process_id, "status": "finalizing"},
                }
            ]
        case CommandCompleted(command_result=command_result):
            return [
                {
                    "event": "command_result",
                    "data": command_result.model_dump(mode="json"),
                },
                _command_done(process_id, command_result),
            ]
        case RunCompleted():
            return [
                {
                    "event": "done",
                    "data": {
                        "process_id": process_id,
                        **_done_result_fields(output.execution_result),
                        "status": "completed",
                        "stopped": False,
                        "reason": None,
                        "memory_task_ids": output.memory_task_ids,
                        "pool_topics": output.pool_topics,
                    },
                }
            ]
        case RunCancelled(reason=reason, execution_result=execution_result):
            base = _done_result_fields(execution_result) if execution_result is not None else {}
            return [_stopped_done(process_id, base, status="cancelled", reason=reason)]
        case RunFailed(execution_result=execution_result):
            return [
                _stopped_done(
                    process_id,
                    _done_result_fields(execution_result),
                    status="failed",
                    reason="agent_run_failed",
                )
            ]
    raise TypeError(f"没有流式投影的阶段产出: {type(output).__name__}")


# 只服务于封口交互记录的执行结果字段，不下发给流式交付方。
_SEALING_ONLY_FIELDS = frozenset({"turn_events", "materialize_tasks"})


def _done_result_fields(result: CPUExecutionResult) -> dict[str, Any]:
    """``done`` 事件携带的执行结果字段（不含封口专用字段）。"""
    return result.model_dump(exclude=set(_SEALING_ONLY_FIELDS))


def _stopped_done(
    process_id: str,
    base: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "event": "done",
        "data": {
            **base,
            "process_id": process_id,
            "status": status,
            "stopped": True,
            "reason": reason,
            "memory_task_ids": [],
        },
    }


def _command_done(process_id: str, command_result: CommandExecutionResult) -> dict[str, Any]:
    return {
        "event": "done",
        "data": {
            "process_id": process_id,
            "final_text": command_result.message,
            "status": "completed",
            "stopped": False,
            "reason": None,
            "memory_task_ids": [],
            "pool_topics": [],
        },
    }


def _memory_ref_from_atom(memory: MemoryAtom) -> dict[str, Any]:
    """把 MemoryAtom 投影为前端引用列表使用的扁平结构。"""

    memory_type = memory.index.memory_type
    return {
        "id": str(memory.id),
        "title": memory.index.title,
        "summary": memory.index.summary,
        "memory_type": (memory_type.value if hasattr(memory_type, "value") else str(memory_type)),
        "tags": list(memory.index.tags),
        "alias": memory.index.alias,
        "content": memory.payload.content,
        "created_at": memory.meta.created_at,
        "updated_at": memory.meta.updated_at,
        "confidence_score": memory.meta.lifecycle.confidence_score,
        "vitality_score": memory.meta.lifecycle.vitality_score,
        "user_id": memory.workspace_identity.owner_user_id,
        "access_count": memory.meta.lifecycle.access_count,
    }


__all__ = [
    "ActorEvent",
    "CommandCompleted",
    "Finalizing",
    "InputsAllocated",
    "NonStreamingAgentOutcome",
    "NonStreamingCommandOutcome",
    "NonStreamingResult",
    "ProcessFailed",
    "ProcessOutput",
    "ProcessStarted",
    "RunCancelled",
    "RunCompleted",
    "RunFailed",
    "TerminalOutput",
    "stream_events",
]
