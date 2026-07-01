"""Shutdown drain observability summaries for PatchouliRuntime."""

from __future__ import annotations

from typing import Any, Literal

from hivememory.patchouli.runtime.memory_tasks import MemoryGenerationTaskWaitSummary


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
    if generation_result.timed_out > 0:
        return "completed_with_timeout"
    return "completed"


def shutdown_drain_completed_severity(
    result: dict[str, Any],
) -> Literal["debug", "info", "warning", "error"]:
    generation_result = result["generation"]
    if generation_result.timed_out > 0:
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
    generation_result: MemoryGenerationTaskWaitSummary,
) -> dict[str, int]:
    return {
        "requested": generation_result.requested,
        "found": generation_result.found,
        "missing": generation_result.missing,
        "completed": generation_result.completed,
        "failed": generation_result.failed,
        "cancelled": generation_result.cancelled,
        "pending": generation_result.pending,
        "running": generation_result.running,
        "timed_out": generation_result.timed_out,
    }


__all__ = [
    "shutdown_drain_completed_severity",
    "shutdown_drain_completed_status",
    "summarize_shutdown_drain_failure",
    "summarize_shutdown_drain_generation",
    "summarize_shutdown_drain_perception",
    "summarize_shutdown_drain_result",
]
