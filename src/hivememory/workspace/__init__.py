"""Workspace 资源平面：访问边界、读取能力、派生缓存与能力层。

- ``access`` / ``registry``：Workspace Actor 访问注册表、操作定义目录
  （``WorkspaceOperation``）与共享行为检查（``WorkspaceAccessGuard``，A1）；
  调用来源 principal 与唯一对外认证网关归属 System；
- ``cache`` / ``resolution`` / ``runtime``：workspace memory read 能力——
  完整原子缓存、Profile 解析缓存、按 Workspace 维护的失效代次与 alias/Profile
  resolver，交付边界逐次授权，L2 经 backing 协议冷读（A2 §2/§3）；
- ``capability``：in-process 的 workspace server API，由 ``system/application``
  的资源能力部分迁入，operation 授权在 backing 调用前执行（A2 §1.2）。

依赖方向：``access`` / ``registry`` / ``cache`` / ``resolution`` / ``runtime``
只依赖 core 与 workspace 自身；``capability`` 按过渡期分层导入白名单额外依赖
``system.*`` 与 ``patchouli.contracts``（A2 §8 D-2，TODO(A5/A6) 重新整理）。
任何子包都不导入 Alice/AgentRuntime 或 Patchouli 内部 store/familiar/
controller/local route。本包初始化不导入 ``capability``，避免与 System
组合根形成循环导入。
"""

from hivememory.workspace.access import (
    WorkspaceAccessContext,
    WorkspaceAccessGuard,
    WorkspaceOperation,
)
from hivememory.workspace.registry import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)
from hivememory.workspace.runtime import WorkspaceRuntime

__all__ = [
    # 操作目录与共享行为检查
    "WorkspaceAccessContext",
    "WorkspaceAccessGuard",
    "WorkspaceOperation",
    # Workspace Actor 访问注册表
    "WorkspaceActorAccessRecord",
    "WorkspaceActorAccessRegistry",
    # 读取能力与派生缓存的运行时聚合
    "WorkspaceRuntime",
]
