"""已结束的记忆生成任务的历史文件。

对话页的记忆生成任务列表读控制器的任务快照。已结束的任务在正常关闭时写进这个文件，下次
启动时读回，列表因此不随重启清空。读回后即删除文件，与保存的生成工作相同：进程崩溃时这段
时间的历史随之丢失，下次启动不会读回一份过时的历史。文件无法解析或格式版本不认识时改名
保留，历史以空表开始。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from hivememory.config.files import rejected_path, write_atomically
from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.patchouli.contracts.memory_tasks import (
    MemoryGenerationSource,
    MemoryGenerationTask,
    MemoryGenerationTaskStatus,
)

logger = logging.getLogger(__name__)

TASK_HISTORY_FORMAT_VERSION = 1


def write_task_history(path: Path, tasks: list[MemoryGenerationTask]) -> None:
    """把已结束的任务快照原子写入 ``path``；调用方保证 ``tasks`` 非空。"""
    document = {
        "format_version": TASK_HISTORY_FORMAT_VERSION,
        "tasks": [_encode(task) for task in tasks],
    }
    write_atomically(path, json.dumps(document, ensure_ascii=False).encode("utf-8"))


def take_task_history(path: Path) -> list[MemoryGenerationTask]:
    """读出并删除历史文件，返回其中的任务快照；文件不存在或读不回时返回空表。"""
    if not path.exists():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("task history document must be an object")
        if document.get("format_version") != TASK_HISTORY_FORMAT_VERSION:
            raise ValueError(f"unknown task history format: {document.get('format_version')!r}")
        items = document.get("tasks")
        if not isinstance(items, list):
            raise ValueError("task history document has no tasks list")
        tasks = [_decode(item) for item in items]
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
        rejected = rejected_path(path)
        path.replace(rejected)
        logger.error(
            "读回记忆生成任务历史失败: reason=task_history_unreadable, rejected_file=%s",
            rejected,
            exc_info=True,
        )
        return []
    path.unlink()
    return tasks


def _encode(task: MemoryGenerationTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "topic_id": task.topic_id,
        "label": task.label,
        "source": task.source.value,
        "from_actor": task.from_actor.model_dump(mode="json"),
        "belong_to": task.belong_to.model_dump(mode="json"),
        "pending_alias": task.pending_alias,
        "status": task.status.value,
        "canonical_alias": task.canonical_alias,
        "error": task.error,
        "created_at": task.created_at.isoformat(),
        "started_at": _encode_time(task.started_at),
        "finished_at": _encode_time(task.finished_at),
        "cancel_requested": task.cancel_requested,
        "cancel_reason": task.cancel_reason,
    }


def _decode(item: dict[str, object]) -> MemoryGenerationTask:
    return MemoryGenerationTask(
        task_id=str(item["task_id"]),
        topic_id=str(item["topic_id"]),
        label=str(item["label"]),
        source=MemoryGenerationSource(item["source"]),
        from_actor=ActorIdentity.model_validate(item["from_actor"]),
        belong_to=WorkspaceIdentity.model_validate(item["belong_to"]),
        pending_alias=_optional_text(item["pending_alias"]),
        status=MemoryGenerationTaskStatus(item["status"]),
        canonical_alias=_optional_text(item["canonical_alias"]),
        error=_optional_text(item["error"]),
        created_at=datetime.fromisoformat(str(item["created_at"])),
        started_at=_decode_time(item["started_at"]),
        finished_at=_decode_time(item["finished_at"]),
        cancel_requested=bool(item["cancel_requested"]),
        cancel_reason=_optional_text(item["cancel_reason"]),
    )


def _encode_time(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _decode_time(value: object) -> datetime | None:
    return datetime.fromisoformat(str(value)) if value is not None else None


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None else None


__all__ = ["take_task_history", "write_task_history"]
