"""进程工作集 — 单个任务进程的阶段产出与待释放资源的登记。

工作集是任务进程状态的一部分（任务进程 Idea 1.2：进程记录 + 工作集），
只登记、不释放：持有 Patchouli prepare 的结果（``PreparedAgentRun``，
Patchouli 公开契约）、本轮取得的附件租借、附件编译冻结的实际使用引用与
CPU 输出流与 Actor 阶段签发的执行凭据。资源由取得它的一方释放——附件
租借由 ``CPUAllocator`` 释放，执行凭据吊销、CPU 输出流与 prepare 结果的
cleanup 由编排骨架（``TaskProcessRunner``）
处理；工作集不持有任何跨进程共享的依赖。

每项待释放的资源都只能取出一次（``take_*``）：关闭流程因此可以重复执行，
已经释放的资源不会被再次释放。“租借随进程关闭释放”是唯一的释放事实
（Q-1），finalize/cleanup 不再负责释放。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from hivememory.core.models.workspace_asset import (
    RepresentationLease,
    WorkspaceAssetRef,
)
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.contracts import CPUOutput, ExecutionCredential


@dataclass
class ProcessWorkingSet:
    """一次任务进程的进程内工作集。

    ``prepared`` 在 prepare 返回后立即写入（身份 scope 校验失败时关闭流程
    仍要把它交回 cleanup，以补偿 prepare 预建的 Topic）；CPU 分配在校验
    通过后读取其中的 Topic 与检索结果。finalize 成功后 Patchouli 接管本轮
    交互，工作集经 :meth:`hand_off_prepared` 记下不再负有补偿义务。
    ``used_attachments`` 在附件编译后写入，供封口交互记录使用。
    ``cpu_output`` 是 Actor 执行期间打开的 CPU 输出流，进程关闭时必须关闭。
    ``credential`` 是主线程不透明凭据，关闭时在任何 await 之前同步吊销。
    """

    prepared: PreparedAgentRun | None = None
    attachment_leases: list[RepresentationLease] = field(default_factory=list)
    used_attachments: tuple[WorkspaceAssetRef, ...] = ()
    cpu_output: AsyncGenerator[CPUOutput, None] | None = None
    # 工作集只持有不透明的执行凭据；访问 context 与注册目标留在凭据表中。
    credential: ExecutionCredential | None = None
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

    def take_cpu_output(self) -> AsyncGenerator[CPUOutput, None] | None:
        """取出待关闭的 CPU 输出流；已经取出时返回 ``None``。"""
        cpu_output, self.cpu_output = self.cpu_output, None
        return cpu_output

    def take_credential(self) -> ExecutionCredential | None:
        """取出待吊销的主线程凭据；同步清字段，使重复关闭不会再次吊销。"""
        credential, self.credential = self.credential, None
        return credential

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
