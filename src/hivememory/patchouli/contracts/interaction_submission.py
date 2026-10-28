"""Patchouli 提交边界的公共输入与结果契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# 程序化提交与主动链路、Import Bus 共用来源枚举，不把控制面实现暴露给契约。
# 来源只是观测标签，不驱动任何分支。
InteractionOrigin = Literal["active_chat", "import_bus", "workspace_port"]


@dataclass(frozen=True)
class InteractionSubmitResult:
    """交互提交收据投影：请求已被队列接纳，应用进度按 interaction_id 跟踪。"""

    interaction_id: str
    work_id: str
    state: str


InteractionProgress = Literal["pending", "succeeded", "failed", "unknown"]


@dataclass(frozen=True)
class InteractionStatus:
    """已被提交队列接纳的交互的进度。

    ``pending`` 尚未应用完成，``succeeded`` 已应用，``failed`` 应用失败且不再
    重试，``unknown`` 表示接纳过但结局已超出队列的保留窗口。``topic_id`` 只在
    ``succeeded`` 时给出，是该交互的真实落点（``NEW_TOPIC`` 已解析）。
    """

    interaction_id: str
    state: InteractionProgress
    topic_id: str | None = None


@dataclass(frozen=True)
class InteractionSubmissionResult:
    """handler 成功后的业务结果；``topic_id`` 是 NEW_TOPIC 的真实落点。"""

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
