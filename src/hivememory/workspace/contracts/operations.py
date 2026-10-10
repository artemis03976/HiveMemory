"""执行者使用的进程操作端口：身份与目标由任务进程绑定。

端口只暴露操作参数，不携带访问 context，也不允许执行者更换目标
Workspace；子线程沿用主线程的同一个端口。进程关闭后明确拒绝调用。
"""

from __future__ import annotations

from typing import Protocol

from hivememory.core.models.pending import PendingAtom, WriteFocus
from hivememory.core.models.reference import ReferenceResolution


class ProcessOperationsClosedError(RuntimeError):
    """操作端口所属进程已经关闭。"""


class ProcessOperations(Protocol):
    """CPU 与执行线程共同消费的操作契约，不暴露进程的访问凭据。"""

    async def submit_write_intent(self, focus: WriteFocus) -> PendingAtom:
        """提交 WRITE，返回已登记的写入意图作为 ACK。"""
        ...

    async def submit_update_intent(
        self, base_alias: str, instruction: str, content: str | None = None
    ) -> PendingAtom:
        """提交 UPDATE，基础引用由能力层解析并验证。"""
        ...

    async def cancel_intents(self, aliases: list[str]) -> list[str]:
        """撤回本进程仍为 PENDING 的指定意图（如未成功结束的子线程提交的意图）。"""
        ...

    async def resolve_references(self, aliases: list[str]) -> list[ReferenceResolution]:
        """按请求顺序解析全部引用，包括不存在的引用。"""
        ...


__all__ = ["ProcessOperations", "ProcessOperationsClosedError"]
