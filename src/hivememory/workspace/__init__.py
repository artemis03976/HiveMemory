"""Workspace：actor 能力面及其网络共享设施。

- ``authentication``：认证入口——统一认证网关（唯一对外认证入口）与
  ``WorkspaceAuthenticator``（第 2 阶段准入、密封 context 的签发与撤销）；
  Principal authentication 经 ``core.access.PrincipalAuthenticator`` 端口
  委托 System；
- ``authorization``：``WorkspaceOperationAuthorizer``——第 3 阶段操作授权、
  进程控制授权与 CPU 执行身份的过渡组装；只依赖访问注册表，读取 context
  密封的授予内容，与认证一侧互不依赖；
- ``registry``：Workspace Actor 访问注册表（准入状态与行为白名单）；
- ``cache`` / ``resolution`` / ``runtime``：workspace memory read 能力——
  完整原子缓存、Profile 解析缓存、失效代次与 alias/Profile resolver；
- ``assets``：WorkspaceAsset working set（AssetStore）、解析交接与上传接收；
- ``capability``：actor 可见的能力层，operation 授权在 backing 调用前执行。

依赖方向：只依赖 core、components、engines/infrastructure 与其他子系统公开的
``contracts`` 子包；不导入 system、Alice/AgentRuntime、Gateway 或 Patchouli
内部实现。本包初始化不导入 ``capability``。
"""

from hivememory.workspace.registry import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)
from hivememory.workspace.runtime import WorkspaceRuntime

# 认证网关、WorkspaceAuthenticator 与 WorkspaceOperationAuthorizer 不在包根
# re-export：它们只能从各自模块导入（workspace.authentication /
# workspace.authorization），避免认证一侧经包根被其他子系统顺手拿走
# （访问边界由 tests/unit/architecture/test_access_boundaries.py 固化）。

__all__ = [
    # Workspace Actor 访问注册表
    "WorkspaceActorAccessRecord",
    "WorkspaceActorAccessRegistry",
    # 读取能力与派生缓存的运行时聚合
    "WorkspaceRuntime",
]
