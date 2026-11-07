"""进程工作集 — 单个任务进程这一轮的话题与关闭时仍需释放或等待的资源的登记。

工作集是任务进程状态的一部分（进程状态 = 进程记录 + 工作集），只登记、
不释放：Gateway 为这一轮给出的决定、这一轮的派发话题与正在进行的话题
确定、本轮取得的附件租借与附件编译冻结的实际使用引用。资源由取得它的一方
处理——附件租借由 ``CPUAllocator`` 释放，话题确定由线程表启动、由提交与
关闭流程等待；工作集不持有任何跨进程共享的依赖。执行线程登记在进程的
线程表中，不在工作集。CPU 返回的执行结果与提交任务只存在于阶段内的局部
状态，不进入工作集。

每项待释放的资源都只能取出一次（``take_*``）：关闭流程即使重复执行，
已经释放的资源也不会被再次释放。“租借随进程关闭释放”是唯一的释放事实，
交互提交不负责释放。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from hivememory.core.models.workspace_asset import (
    RepresentationLease,
    WorkspaceAssetRef,
)
from hivememory.core.protocol.gateway import GatewayDecision


@dataclass
class ProcessWorkingSet:
    """一次任务进程的进程内工作集。

    ``gateway_decision`` 在 Gateway 返回后写入，确定话题与提交新话题都按它
    进行。``dispatch_topic_id`` 是这一轮唯一的话题，确定之后冻结，主线程与
    CALL 子线程共享，actor 请求不能指定；Gateway 指向新话题、本轮还没有写入
    时为空。``topic_determination`` 是正在进行或最近一次的话题确定。
    ``used_attachments`` 在附件编译后写入，供封口交互记录使用。
    """

    gateway_decision: GatewayDecision | None = None
    dispatch_topic_id: str | None = None
    topic_determination: asyncio.Task[str] | None = field(default=None, repr=False)
    attachment_leases: list[RepresentationLease] = field(default_factory=list)
    used_attachments: tuple[WorkspaceAssetRef, ...] = ()

    def register_lease(self, lease: RepresentationLease) -> None:
        """登记一项已取得的附件租借，随进程关闭统一释放。"""
        self.attachment_leases.append(lease)

    def remove_lease(self, lease: RepresentationLease) -> None:
        """把单个租借移出工作集（由取得方立即释放时使用）；未登记时忽略。"""
        if lease in self.attachment_leases:
            self.attachment_leases.remove(lease)

    def take_leases(self) -> list[RepresentationLease]:
        """取出全部待释放的租借；重复调用返回空列表。"""
        leases, self.attachment_leases = self.attachment_leases, []
        return leases

    async def settle_topic_determination(self) -> None:
        """等待正在进行的话题确定结束，不论成败；没有正在进行的时立即返回。

        提交前与关闭时调用：确定的结果（或失败）已经交给等待它的写入，这里
        只保证它结束之后才读取派发话题或结束进程。
        """
        determination = self.topic_determination
        if determination is not None and not determination.done():
            await asyncio.wait({determination})


__all__ = [
    "ProcessWorkingSet",
]
