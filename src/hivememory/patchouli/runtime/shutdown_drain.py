"""PatchouliRuntime 关停排空、启动读回保存工作与话题池的观测性摘要。"""

from __future__ import annotations

from typing import Any, Literal

from hivememory.patchouli.control.memory_generation.saved_work import (
    RestoredWorkReport,
    SavedWorkReport,
)
from hivememory.patchouli.runtime.models import RestoredTopicPoolReport

type _Severity = Literal["debug", "info", "warning", "error"]


def summarize_shutdown_drain_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "reentrant": result["reentrant"],
        "success": result["success"],
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
        "generation": None,
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


def summarize_topic_pool_restore(report: RestoredTopicPoolReport) -> dict[str, Any]:
    return {
        "restored": report.restored,
        "retired": report.retired,
        "rejected": report.rejected,
    }


def restore_completed_status(report: RestoredWorkReport) -> str:
    if report.rejected_file is not None:
        return "completed_with_rejections"
    return "completed"


def restore_completed_severity(report: RestoredWorkReport) -> _Severity:
    if report.rejected_file is not None:
        return "error"
    return "info"


def topic_pool_restore_completed_status(report: RestoredTopicPoolReport) -> str:
    if report.rejected:
        return "completed_with_rejections"
    return "completed"


def topic_pool_restore_completed_severity(report: RestoredTopicPoolReport) -> _Severity:
    if report.rejected:
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
    "summarize_shutdown_drain_result",
    "summarize_topic_pool_restore",
    "topic_pool_restore_completed_severity",
    "topic_pool_restore_completed_status",
]
