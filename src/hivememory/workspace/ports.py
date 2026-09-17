"""Workspace 基础设施端口：访问准入与派生失效协作。

按父计划 4.2/5.7 节修订：端口只声明 Workspace 基础设施能力（admission、
invalidation，后续阶段加入 cache/Asset lease），**不再是 Actor 的业务
API**——资源读取、领域提交和结果查询统一由既有 application service 与
GlobalSystemBus 公开路由承接。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.workspace.access import (
    CallerPrincipal,
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.workspace.projections import CanonicalResourceChange


@runtime_checkable
class WorkspaceAdmissionPort(Protocol):
    """统一访问边界的类型声明：principal + 请求坐标 → 访问上下文。"""

    async def admit(
        self,
        principal: CallerPrincipal,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
        operation: WorkspaceOperation,
    ) -> WorkspaceAccessContext:
        """校验 principal/用户/Agent/Workspace 映射与 operation 授权后签发上下文。"""
        ...


@runtime_checkable
class ResourceInvalidationPort(Protocol):
    """派生 cache 失效端口：canonical mutation 提交路径同步调用。

    正确性边界（父计划 6.2 节）：失效必须在 mutation 提交的同一同步边界
    完成；进程内事件只作观测，投递失败不得影响 cache 正确性。canonical
    mutation 的内部失效通知不做 Actor admission。实现于 WRX-3。
    """

    async def invalidate(self, change: CanonicalResourceChange) -> None:
        """使指定 canonical 变更涉及的派生视图失效。"""
        ...


__all__ = [
    "WorkspaceAdmissionPort",
    "ResourceInvalidationPort",
]
