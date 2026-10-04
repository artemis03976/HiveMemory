"""Workspace：actor 能力面及其网络共享设施。

- ``authentication``：认证入口——Principal authentication 经
  ``core.access.PrincipalAuthenticator`` 端口委托 System，Workspace 准入由
  本包的 guard 完成；
- ``access`` / ``registry``：Workspace Actor 访问注册表与共享操作授权
  （``WorkspaceAccessGuard``，签发并兑现不透明访问 context）；
- ``cache`` / ``resolution`` / ``runtime``：workspace memory read 能力——
  完整原子缓存、Profile 解析缓存、失效代次与 alias/Profile resolver；
- ``assets``：WorkspaceAsset working set（AssetStore）、解析交接与上传接收；
- ``capability``：actor 可见的能力层，operation 授权在 backing 调用前执行。

依赖方向：只依赖 core、components、engines/infrastructure 与其他子系统公开的
``contracts`` 子包；不导入 system、Alice/AgentRuntime、Gateway 或 Patchouli
内部实现。本包初始化不导入 ``capability``。
"""

from hivememory.workspace.access import WorkspaceAccessGuard
from hivememory.workspace.registry import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)
from hivememory.workspace.runtime import WorkspaceRuntime

__all__ = [
    # 共享行为检查
    "WorkspaceAccessGuard",
    # Workspace Actor 访问注册表
    "WorkspaceActorAccessRecord",
    "WorkspaceActorAccessRegistry",
    # 读取能力与派生缓存的运行时聚合
    "WorkspaceRuntime",
]
