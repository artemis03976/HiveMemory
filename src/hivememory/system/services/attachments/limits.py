"""解析协作式预算（计划 7.7 节）。

预算属于进程内工作量限制，不是进程内存硬配额；协作式检查不能强制中止
卡住的库调用。秒数来自 System 配置（``AttachmentParserConfig``），实现时
用边界样例校准。
"""

from __future__ import annotations

import time
from collections.abc import Callable

from hivememory.system.services.attachments.errors import RESOURCE_LIMIT, AttachmentParseError


class ParseBudget:
    """协作式解析预算。

    在读取和遍历检查点调用 :meth:`check`；超时属于运行结果，不承诺
    跨机器每次都发生相同失败。时钟可注入以便测试使用小额预算。
    """

    def __init__(
        self,
        budget_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._seconds = budget_seconds
        self._clock = clock
        self._started_at = clock()

    def check(self) -> None:
        """在检查点核对剩余预算；超限抛出 ``resource_limit`` 受控失败。"""
        if self._clock() - self._started_at > self._seconds:
            raise AttachmentParseError(
                RESOURCE_LIMIT,
                "附件解析耗时超过预算，请缩小文件后重新上传",
                params={"reason": "parse_budget_exceeded"},
            )


__all__ = ["ParseBudget"]
