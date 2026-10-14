"""Patchouli 提交边界的公共输入与结果契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hivememory.patchouli.contracts.memory_tasks import MemoryGenerationAdmissionStatus


@dataclass(frozen=True)
class MemoryIntent:
    """中立的记忆意图参数：由 Patchouli 决定生成、更新、合并或丢弃。

    ``intent_id`` 与 ``pending_alias`` 是 workspace 登记写入意图时产生的标识，
    Patchouli 原样使用：结算事件按它们回到登记的同一句柄，``intent_id`` 同时是
    生成任务的幂等键。``kind="write"`` 需要 ``content``；``kind="update"`` 需要
    ``instruction`` 以及 ``base_alias`` + ``base_uuid``（UPDATE 的业务前提是调用方
    已持有当前 base 的坐标）。
    """

    kind: Literal["write", "update"]
    topic_id: str
    intent_id: str
    pending_alias: str
    content: str | None = None
    title: str | None = None
    reason: str | None = None
    instruction: str | None = None
    base_alias: str | None = None
    base_uuid: str | None = None

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
        for name in ("topic_id", "intent_id", "pending_alias"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} 不能为空")


@dataclass(frozen=True)
class MemoryIntentSubmissionResult:
    """意图提交回执，说明是否被生成队列接纳；接纳不等于已经生成。

    ``REJECTED`` 时意图已结算为 FAILED；``UNKNOWN`` 时无法确认是否已被接纳，
    不发布结算事件。只有 ``ACCEPTED`` 携带任务标识，应用进度经任务观察 API
    以同一 ``task_id`` 查询。
    """

    status: MemoryGenerationAdmissionStatus
    task_id: str | None = None
    task_status: str | None = None
    detail: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status is MemoryGenerationAdmissionStatus.ACCEPTED


__all__ = ["MemoryIntent", "MemoryIntentSubmissionResult"]
