"""Workspace 基础设施端口：派生失效协作。

按 A1 计划（docs/plans/v0.7.0-a1-workspace-access-boundary.md 第 6 节迁移
表）修订：认证与准入逻辑收敛到 System 统一认证入口
（``system.access.ActorAuthenticationGateway``），Workspace 侧的准入判定
与共享行为检查由 ``workspace.registry`` / ``workspace.access`` 以具体类
提供，不再保留调用方需分别访问的 ``WorkspaceAdmissionPort`` 认证端口。

端口只声明 Workspace 基础设施能力（当前为 invalidation，后续阶段加入
cache/Asset lease），**不是 Actor 的业务 API**——资源读取、领域提交和
结果查询统一由既有 application service 与 GlobalSystemBus 公开路由承接。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hivememory.workspace.projections import CanonicalResourceChange


@runtime_checkable
class ResourceInvalidationPort(Protocol):
    """派生 cache 失效端口：canonical mutation 提交路径同步调用。

    正确性边界（A2 计划承接父计划 6.2 节）：失效必须在 mutation 提交的
    同一同步边界完成；进程内事件只作观测，投递失败不得影响 cache 正确性。
    canonical mutation 的内部失效通知不做 Actor 认证（A1 第 3.3 节）。
    """

    async def invalidate(self, change: CanonicalResourceChange) -> None:
        """使指定 canonical 变更涉及的派生视图失效。"""
        ...


__all__ = [
    "ResourceInvalidationPort",
]
