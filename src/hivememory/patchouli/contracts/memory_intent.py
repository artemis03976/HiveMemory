"""Patchouli 提交边界的公共输入与结果契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MemoryIntent:
    """中立的记忆意图参数：由 Patchouli 决定生成、更新、合并或丢弃。

    ``kind="write"`` 需要 ``content``；``kind="update"`` 需要 ``instruction``
    以及 ``base_alias`` + ``base_uuid``（UPDATE 的业务前提是调用方已持有
    当前 base 的坐标）。``intent_id`` 是幂等键，缺省生成。
    """

    kind: Literal["write", "update"]
    topic_id: str
    content: str | None = None
    title: str | None = None
    reason: str | None = None
    instruction: str | None = None
    base_alias: str | None = None
    base_uuid: str | None = None
    intent_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "write":
            if not self.content:
                raise ValueError("write 意图需要 content")
            if self.instruction or self.base_alias or self.base_uuid:
                raise ValueError("write 意图不得携带 update 字段")
        elif self.kind == "update":
            if not self.instruction:
                raise ValueError("update 意图需要 instruction")
            if not self.base_alias or not self.base_uuid:
                raise ValueError("update 意图需要 base_alias 与 base_uuid")
        else:
            raise ValueError(f"不支持的意图类型: {self.kind!r}")
        if not isinstance(self.topic_id, str) or not self.topic_id.strip():
            raise ValueError("topic_id 不能为空")


@dataclass(frozen=True)
class MemoryIntentSubmissionResult:
    """意图提交回执，说明"是否被接纳"；接纳不等于已应用。

    ``accepted=False`` 表示生成 admission 未接纳（确定性拒绝或批量响应
    缺失）；应用进度经任务观察 API 以同一 task_id 查询。
    """

    accepted: bool
    task_id: str | None = None
    task_status: str | None = None
    detail: str | None = None


__all__ = ["MemoryIntent", "MemoryIntentSubmissionResult"]
