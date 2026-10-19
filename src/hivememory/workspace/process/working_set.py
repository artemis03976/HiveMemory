"""进程工作集 — 单个任务进程关闭时仍需释放或补偿的资源的登记。

工作集是任务进程状态的一部分（进程状态 = 进程记录 + 工作集），
只登记、不释放：持有 Patchouli prepare 的结果（``PreparedAgentRun``，
Patchouli 公开契约）、本轮取得的附件租借、CPU 分配写入的派发话题、附件
编译冻结的实际使用引用与 Actor 阶段签发的线程 context。资源由取得它的一方
释放——附件租借由 ``CPUAllocator`` 释放，线程 context 撤销与 prepare 结果的
cleanup 由编排骨架（``TaskProcessRunner``）处理；工作集不持有任何跨进程
共享的依赖。CPU 返回的执行结果与 finalize 提交任务只存在于阶段内的局部
状态，不进入工作集。

每项待释放的资源都只能取出一次（``take_*``）：关闭流程即使重复执行，
已经释放的资源也不会被再次释放。“租借随进程关闭释放”是唯一的释放事实，
finalize/cleanup 不负责释放。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hivememory.core.access import WorkspaceAccessContext
from hivememory.core.models.workspace_asset import (
    RepresentationLease,
    WorkspaceAssetRef,
)
from hivememory.patchouli.contracts.prepare import PreparedAgentRun


@dataclass
class ProcessWorkingSet:
    """一次任务进程的进程内工作集。

    ``prepared`` 在 prepare 返回后立即写入（身份 scope 校验失败时关闭流程
    仍要把它交回 cleanup，以补偿 prepare 预建的 Topic）；CPU 分配在校验
    通过后读取其中的 Topic 与检索结果。finalize 成功后 Patchouli 接管本轮
    交互，工作集经 :meth:`hand_off_prepared` 记下不再负有补偿义务。
    ``used_attachments`` 在附件编译后写入，供封口交互记录使用。
    ``thread_contexts`` 登记全部执行线程，Actor 阶段结束时统一撤销；关闭流程
    先于任何 await 兜底撤销。
    """

    # CPU 分配冻结真实话题，主线程与 CALL 子线程共享，actor 请求不能指定。
    dispatch_topic_id: str | None = None
    prepared: PreparedAgentRun | None = None
    attachment_leases: list[RepresentationLease] = field(default_factory=list)
    used_attachments: tuple[WorkspaceAssetRef, ...] = ()
    # 工作集登记线程 context 供撤销；进程 context 仍由进程记录持有。
    thread_contexts: dict[str, WorkspaceAccessContext] = field(default_factory=dict)
    # prepare 结果是否已交回 Patchouli（finalize 接管，或关闭流程已取出
    # 请求 cleanup）；交回之后工作集不再负有补偿义务。
    _prepared_handed_off: bool = field(default=False, repr=False)

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

    def take_thread_contexts(self) -> tuple[WorkspaceAccessContext, ...]:
        """取出全部待撤销线程并同步清空登记，使兜底撤销不重复撤销。"""
        contexts = tuple(self.thread_contexts.values())
        self.thread_contexts.clear()
        return contexts

    def hand_off_prepared(self) -> None:
        """finalize 成功：Patchouli 已接管本轮交互，prepare 结果不再需要 cleanup。"""
        self._prepared_handed_off = True

    def take_prepared_for_cleanup(self) -> PreparedAgentRun | None:
        """取出仍需交回 cleanup 的 prepare 结果；只取出一次，交回后返回 ``None``。"""
        if self.prepared is None or self._prepared_handed_off:
            return None
        self._prepared_handed_off = True
        return self.prepared


__all__ = [
    "ProcessWorkingSet",
]
