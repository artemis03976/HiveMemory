"""本地工作队列的 lane 与失败处理策略契约。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class QueuePolicy:
    """单条 lane 独立持有的容量与执行策略。"""

    capacity: int
    max_concurrency: int
    ordered_by_key: bool = False
    priority_enabled: bool = False
    cancellable: bool = True
    timeout_seconds: float | None = None
    max_attempts: int = 1
    terminal_retention: int = 100
    shutdown_wait_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        # 0 表示不由固定次数截断，后续尝试仍须由 handler decision 明确驱动。
        if self.max_attempts < 0:
            raise ValueError("max_attempts must not be negative")
        if self.terminal_retention < 0:
            raise ValueError("terminal_retention must not be negative")
        if self.shutdown_wait_seconds is not None and self.shutdown_wait_seconds < 0:
            raise ValueError("shutdown_wait_seconds must not be negative")


class FailureAction(str, Enum):
    """handler 对单次执行失败给出的通用处理动作。"""

    RETRY = "retry"
    FAIL = "fail"
    DEAD_LETTER = "dead_letter"
    TREAT_AS_SUCCESS = "treat_as_success"


@dataclass(frozen=True)
class FailureDecision:
    """handler 生成、runtime 执行的失败分类结果。"""

    action: FailureAction
    retry_after_seconds: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.retry_after_seconds is not None and self.retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must not be negative")
        if self.action != FailureAction.RETRY and self.retry_after_seconds is not None:
            raise ValueError("retry_after_seconds is only valid for retry decisions")


__all__ = [
    "FailureAction",
    "FailureDecision",
    "QueuePolicy",
]
