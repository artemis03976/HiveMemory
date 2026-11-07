"""Patchouli 提交边界的公共输入与结果契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# 程序化提交与主动链路、Import Bus 共用来源枚举，不把控制面实现暴露给契约。
# 来源只是观测标签，不驱动任何分支。
InteractionOrigin = Literal["active_chat", "import_bus", "workspace_port"]


InteractionProgress = Literal["pending", "succeeded", "failed", "unknown"]


@dataclass(frozen=True)
class InteractionSubmitResult:
    """一次提交并等待应用的结局。

    ``state`` 是等待结束时的进度：``succeeded`` 带实际所在的话题，``failed``
    带应用失败的异常类名，``pending`` 表示等待到时限仍未应用完成，``unknown``
    只出现在此前接纳、结局已超出保留窗口的交互上。``previously_accepted``
    表示这一交互在本次调用之前已被接纳，本次没有再提交；``new_topic`` 表示
    本次提交以 ``NEW_TOPIC`` 请求建立新话题。
    """

    interaction_id: str
    state: InteractionProgress
    topic_id: str | None = None
    error_class: str | None = None
    previously_accepted: bool = False
    new_topic: bool = False


@dataclass(frozen=True)
class InteractionStatus:
    """已被提交队列接纳的交互的进度。

    ``pending`` 尚未应用完成，``succeeded`` 已应用，``failed`` 应用失败且不再
    重试，``unknown`` 表示接纳过但结局已超出队列的保留窗口。``topic_id`` 只在
    ``succeeded`` 时给出，是该交互实际所在的话题（``NEW_TOPIC`` 已解析）。
    """

    interaction_id: str
    state: InteractionProgress
    topic_id: str | None = None


@dataclass(frozen=True)
class InteractionSubmissionResult:
    """handler 成功后的业务结果；``topic_id`` 是 NEW_TOPIC 解析后实际所在的话题。"""

    interaction_id: str
    topic_id: str

    @property
    def result_ref(self) -> str:
        return self.topic_id


__all__ = [
    "InteractionOrigin",
    "InteractionProgress",
    "InteractionStatus",
    "InteractionSubmitResult",
    "InteractionSubmissionResult",
]
