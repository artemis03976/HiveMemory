"""Patchouli 提交边界的公共输入与结果契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# 程序化提交与主动、被动摄入共用来源枚举，不把控制面实现暴露给契约。
InteractionOrigin = Literal["active_chat", "passive_memory", "workspace_port"]


@dataclass(frozen=True)
class InteractionSubmitResult:
    """交互提交收据投影：请求已被队列接纳，应用进度按 interaction_id 跟踪。"""

    interaction_id: str
    work_id: str
    state: str


@dataclass(frozen=True)
class InteractionSubmissionResult:
    """handler 成功后的业务结果；``topic_id`` 是 NEW_TOPIC 的真实落点。"""

    interaction_id: str
    topic_id: str

    @property
    def result_ref(self) -> str:
        return self.topic_id


__all__ = ["InteractionOrigin", "InteractionSubmitResult", "InteractionSubmissionResult"]
