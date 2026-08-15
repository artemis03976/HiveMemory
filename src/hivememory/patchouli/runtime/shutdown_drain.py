"""Shutdown drain observability summaries for PatchouliRuntime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from hivememory.patchouli.control.memory_generation.models import (
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)

_GENERATION_SUMMARY_KEYS = (
    "requested",
    "found",
    "missing",
    "completed",
    "failed",
    "cancelled",
    "pending",
    "running",
    "timed_out",
)


def build_shutdown_generation_summary(
    tasks: list[MemoryGenerationTask | None],
) -> dict[str, int]:
    """从 shutdown 等待返回的任务快照构造观测计数。

    shutdown 的 ``wait_all(timeout=...)`` 只等待调用时尚未终结的任务，因此窗口
    结束后仍为 PENDING/RUNNING 的快照就是本次 drain 的超时项，无需再为通用
    wait API 引入一份带 ``timed_out`` 标记的包装模型。
    """

    found = [task for task in tasks if task is not None]
    pending = sum(
        task.status == MemoryGenerationTaskStatus.PENDING for task in found
    )
    running = sum(
        task.status == MemoryGenerationTaskStatus.RUNNING for task in found
    )
    return {
        "requested": len(tasks),
        "found": len(found),
        "missing": len(tasks) - len(found),
        "completed": sum(
            task.status == MemoryGenerationTaskStatus.COMPLETED for task in found
        ),
        "failed": sum(
            task.status == MemoryGenerationTaskStatus.FAILED for task in found
        ),
        "cancelled": sum(
            task.status == MemoryGenerationTaskStatus.CANCELLED for task in found
        ),
        "pending": pending,
        "running": running,
        "timed_out": pending + running,
    }


def summarize_shutdown_drain_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "reentrant": result["reentrant"],
        "success": result["success"],
        "perception": summarize_shutdown_drain_perception(result["perception"]),
        "generation": summarize_shutdown_drain_generation(result["generation"]),
        "generation_cancelled_after_timeout": result.get(
            "generation_cancelled_after_timeout",
            0,
        ),
    }


def shutdown_drain_completed_status(result: dict[str, Any]) -> str:
    generation_result = result["generation"]
    if generation_result["timed_out"] > 0:
        return "completed_with_timeout"
    return "completed"


def shutdown_drain_completed_severity(
    result: dict[str, Any],
) -> Literal["debug", "info", "warning", "error"]:
    generation_result = result["generation"]
    if generation_result["timed_out"] > 0:
        return "warning"
    return "info"


def summarize_shutdown_drain_failure(exc: BaseException) -> dict[str, Any]:
    return {
        "reentrant": False,
        "perception": None,
        "generation": None,
    }


def summarize_shutdown_drain_perception(perception_result: Any) -> dict[str, Any]:
    if isinstance(perception_result, dict):
        flushed_topics = perception_result.get("flushed_topics") or []
        skipped_topics = perception_result.get("skipped_topics") or []
        return {
            "success": perception_result.get("success"),
            "trigger_reason": perception_result.get("trigger_reason"),
            "flushed_topic_count": len(flushed_topics),
            "skipped_topic_count": len(skipped_topics),
            "archived_blocks": perception_result.get("archived_blocks"),
        }
    return {
        "success": getattr(perception_result, "success", None),
        "trigger_reason": getattr(perception_result, "trigger_reason", None),
        "flushed_topic_count": len(
            getattr(perception_result, "flushed_topics", []) or []
        ),
        "skipped_topic_count": len(
            getattr(perception_result, "skipped_topics", []) or []
        ),
        "archived_blocks": getattr(perception_result, "archived_blocks", None),
    }


def summarize_shutdown_drain_generation(
    generation_result: Mapping[str, int],
) -> dict[str, int]:
    return {key: generation_result[key] for key in _GENERATION_SUMMARY_KEYS}


__all__ = [
    "build_shutdown_generation_summary",
    "shutdown_drain_completed_severity",
    "shutdown_drain_completed_status",
    "summarize_shutdown_drain_failure",
    "summarize_shutdown_drain_generation",
    "summarize_shutdown_drain_perception",
    "summarize_shutdown_drain_result",
]
