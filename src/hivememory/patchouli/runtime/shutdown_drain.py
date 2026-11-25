"""PatchouliRuntime 关停排空与启动读回保存工作、话题池的观测性摘要。"""

from __future__ import annotations

from typing import Any, Literal

from hivememory.patchouli.control.memory_generation.saved_work import (
    RestoredWorkReport,
    SavedWorkReport,
)
from hivememory.patchouli.runtime.models import TopicShutdownFlushReport
from hivememory.patchouli.runtime.topic_pool import RestoredTopicPoolReport, SavedTopicPoolReport

type _Severity = Literal["debug", "info", "warning", "error"]


def summarize_shutdown_drain_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "reentrant": result["reentrant"],
        "success": result["success"],
        "perception": summarize_shutdown_drain_perception(result["perception"]),
        "generation": summarize_saved_work(result["generation"]),
        "topic_pool": summarize_saved_topic_pool(result["topic_pool"]),
    }


def shutdown_drain_completed_status(result: dict[str, Any]) -> str:
    if not result["success"]:
        return "completed_without_saved_work"
    return "completed"


def shutdown_drain_completed_severity(result: dict[str, Any]) -> _Severity:
    # 没写出保存文件或话题池快照时，这些工作或话题随进程结束丢失。
    if not result["success"]:
        return "error"
    return "info"


def summarize_shutdown_drain_failure(exc: BaseException) -> dict[str, Any]:
    return {
        "reentrant": False,
        "perception": None,
        "generation": None,
        "topic_pool": None,
    }


def summarize_shutdown_drain_perception(report: TopicShutdownFlushReport) -> dict[str, Any]:
    return {
        "settled_topic_count": len(report.settled_topic_ids),
        "retired_topic_count": len(report.retired_topic_ids),
        "failed_topic_count": len(report.failed_topic_ids),
        "resident_block_count": report.resident_block_count,
    }


def summarize_saved_work(report: SavedWorkReport) -> dict[str, Any]:
    return {
        "saved": report.saved,
        "saved_by_source": dict(report.saved_by_source),
        "waited_for_commit": report.waited_for_commit,
        "file_written": report.file_written,
    }


def summarize_saved_topic_pool(report: SavedTopicPoolReport) -> dict[str, Any]:
    return {"saved": report.saved, "file_written": report.file_written}


def summarize_restore_result(report: RestoredWorkReport) -> dict[str, Any]:
    return {
        "read": report.read,
        "resubmitted": report.resubmitted,
        "rejected": report.rejected,
        "rejected_file_written": report.rejected_file is not None,
    }


def summarize_topic_pool_restore(report: RestoredTopicPoolReport) -> dict[str, Any]:
    return {
        "read": report.read,
        "restored": report.restored,
        "rejected": report.rejected,
        "rejected_file_written": report.rejected_file is not None,
    }


def restore_completed_status(report: RestoredWorkReport | RestoredTopicPoolReport) -> str:
    if report.rejected_file is not None:
        return "completed_with_rejections"
    return "completed"


def restore_completed_severity(report: RestoredWorkReport | RestoredTopicPoolReport) -> _Severity:
    if report.rejected_file is not None:
        return "error"
    return "info"


__all__ = [
    "restore_completed_severity",
    "restore_completed_status",
    "shutdown_drain_completed_severity",
    "shutdown_drain_completed_status",
    "summarize_restore_result",
    "summarize_saved_topic_pool",
    "summarize_saved_work",
    "summarize_shutdown_drain_failure",
    "summarize_shutdown_drain_perception",
    "summarize_shutdown_drain_result",
    "summarize_topic_pool_restore",
]
