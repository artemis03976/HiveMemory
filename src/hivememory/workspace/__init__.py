"""Workspace：资源访问边界、能力层、读取视图与任务进程编排。

- ``authentication``：认证入口——统一认证网关（唯一对外认证入口）与
  ``WorkspaceAuthenticator``（第 2 阶段准入、密封 context 的签发与撤销）；
  Principal authentication 经 ``core.access.PrincipalAuthenticator`` 端口
  委托 System；
- ``authorization``：``WorkspaceOperationAuthorizer``——第 3 阶段的 actor
  操作授权、root 授权、任务进程阶段授权与操作快照读取，以及进程控制授权；
  只依赖访问注册表，读取 context 密封的授予内容，与认证一侧互不依赖；
- ``registry``：Workspace Actor 访问注册表（准入状态与 actor 操作白名单）；
- ``cache`` / ``resolution`` / ``runtime``：读取视图——完整原子缓存、
  Profile 解析缓存、失效代次与 alias/Profile resolver；
- ``intents``：写入意图登记；``credentials``：执行凭据表；
- ``capability``：能力层——只供 ``system`` 的 root API（``root``）、只经
  操作入口供 actor 使用的 actor API（``actor``）、操作入口与读取视图的
  总线 backing；
- ``process``：任务进程表、注册入口与四阶段编排（含 CPU 分配）；
- ``assets``：WorkspaceAsset working set（AssetStore）、解析交接与上传接收；
- ``contracts``：其他子系统可导入的唯一公共契约（CPU 端口、操作请求与凭据）。

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
