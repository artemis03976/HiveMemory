"""进程工作集 — 单个任务进程的 prepare 结果与附件租借持有。

工作集是编排服务的 per-run 状态容器：持有 Patchouli prepare 的结果
（``PreparedAgentRun``，Patchouli 公开契约）、本轮取得的附件租借与附件编译
冻结的实际使用引用。编排骨架从这里读取 prepare 结果完成 CPU 分配，并把它
交回 finalize/cleanup 路由。进程的关闭流程
（``TaskProcess.close``）先调用 :meth:`ProcessWorkingSet.release`，覆盖完成、
取消、失败、断流与 CPU 分配失败各条路径；“租借随进程关闭释放”是唯一的释放事实（Q-1），finalize/cleanup
不再负责释放。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hivememory.core.errors import WorkspaceDomainError
from hivememory.core.models.workspace_asset import (
    RepresentationLease,
    WorkspaceAssetRef,
)
from hivememory.core.ports.workspace_assets import WorkspaceAssetReaderPort
from hivememory.patchouli.contracts.prepare import PreparedAgentRun

logger = logging.getLogger(__name__)


@dataclass
class ProcessWorkingSet:
    """一次任务进程的进程内工作集。

    ``prepared`` 在 prepare 返回后立即写入（身份 scope 校验失败时结束路径仍要
    把它交回 cleanup，以补偿 prepare 预建的 Topic）；CPU 分配在校验通过后读取
    其中的 Topic 与检索结果，结束路径把它交回 Patchouli 的 finalize/cleanup 路由。
    ``used_attachments`` 在附件编译后写入，供封口交互记录使用。
    """

    asset_reader: WorkspaceAssetReaderPort | None
    prepared: PreparedAgentRun | None = None
    attachment_leases: list[RepresentationLease] = field(default_factory=list)
    used_attachments: tuple[WorkspaceAssetRef, ...] = ()

    def register_lease(self, lease: RepresentationLease) -> None:
        """登记一项已取得的附件租借，随进程关闭统一释放。"""
        self.attachment_leases.append(lease)

    def discard_lease(self, lease: RepresentationLease) -> None:
        """释放单个租借并将其移出工作集（版本核对失败时使用）。

        主要调用点总是先 :meth:`register_lease` 再核对；若传入未登记的
        租借，则跳过移除、直接按 Store 幂等语义尝试释放。
        """
        if lease in self.attachment_leases:
            self.attachment_leases.remove(lease)
        self._release_one(lease)

    def release(self) -> None:
        """幂等释放当前登记的全部附件租借。

        逐项释放；Store 关闭等 ``WorkspaceDomainError`` 只记录警告，不
        改变进程已经确定的终态。重复调用是空操作。
        """
        leases, self.attachment_leases = self.attachment_leases, []
        for lease in leases:
            self._release_one(lease)

    def _release_one(self, lease: RepresentationLease) -> None:
        if self.asset_reader is None:
            return
        try:
            self.asset_reader.release_representation_lease(lease.lease_id)
        except WorkspaceDomainError as exc:
            # Store 已关闭等清理路径：记录摘要，不改变进程终态。
            logger.warning(
                "释放附件 lease 失败: lease_id=%s, code=%s",
                lease.lease_id,
                exc.code,
            )


__all__ = [
    "ProcessWorkingSet",
]
