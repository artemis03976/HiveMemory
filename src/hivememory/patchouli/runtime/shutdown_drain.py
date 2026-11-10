"""PatchouliRuntime 关停排空与启动读回保存工作的观测性摘要。"""

from __future__ import annotations

from typing import Any, Literal

from hivememory.patchouli.control.memory_generation.saved_work import (
    RestoredWorkReport,
    SavedWorkReport,
)

type _Severity = Literal["debug", "info", "warning", "error"]


def summarize_shutdown_drain_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "reentrant": result["reentrant"],
        "success": result["success"],
        "perception": summarize_shutdown_drain_perception(result["perception"]),
        "generation": summarize_saved_work(result["generation"]),
    }


def shutdown_drain_completed_status(result: dict[str, Any]) -> str:
    if not result["success"]:
        return "completed_without_saved_work"
    return "completed"


def shutdown_drain_completed_severity(result: dict[str, Any]) -> _Severity:
    # 没写出保存文件时，这些工作随进程结束丢失。
    if not result["success"]:
        return "error"
    return "info"


def summarize_shutdown_drain_failure(exc: BaseException) -> dict[str, Any]:
    return {
        "reentrant": False,
        "perception": None,
        "generation": None,
    }


def summarize_shutdown_drain_perception(perception_result: Any) -> dict[str, Any]:
    if isinstance(perception_result, dict):
        settled_topics = perception_result.get("settled_topic_ids") or []
        generation_skipped_topics = perception_result.get("generation_skipped_topic_ids") or []
        return {
            "settled_topic_count": len(settled_topics),
            "generation_skipped_topic_count": len(generation_skipped_topics),
            "resident_block_count": perception_result.get("resident_block_count"),
        }
    generation_skipped_topics = getattr(perception_result, "generation_skipped_topic_ids", []) or []
    return {
        "settled_topic_count": len(getattr(perception_result, "settled_topic_ids", []) or []),
        "generation_skipped_topic_count": len(generation_skipped_topics),
        "resident_block_count": getattr(perception_result, "resident_block_count", None),
    }


def summarize_saved_work(report: SavedWorkReport) -> dict[str, Any]:
    return {
        "saved": report.saved,
        "saved_by_source": dict(report.saved_by_source),
        "waited_for_commit": report.waited_for_commit,
        "file_written": report.file_written,
    }


def summarize_restore_result(report: RestoredWorkReport) -> dict[str, Any]:
    return {
        "read": report.read,
        "resubmitted": report.resubmitted,
        "rejected": report.rejected,
        "rejected_file_written": report.rejected_file is not None,
    }


def restore_completed_status(report: RestoredWorkReport) -> str:
    if report.rejected_file is not None:
        return "completed_with_rejections"
    return "completed"


def restore_completed_severity(report: RestoredWorkReport) -> _Severity:
    if report.rejected_file is not None:
        return "error"
    return "info"


__all__ = [
    "restore_completed_severity",
    "restore_completed_status",
    "shutdown_drain_completed_severity",
    "shutdown_drain_completed_status",
    "summarize_restore_result",
    "summarize_saved_work",
    "summarize_shutdown_drain_failure",
    "summarize_shutdown_drain_perception",
    "summarize_shutdown_drain_result",
]
