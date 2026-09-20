"""Workspace 基础设施：操作目录、访问注册表、共享行为检查与失效协作。

A1 计划（docs/plans/v0.7.0-a1-workspace-access-boundary.md）确立的分工：
本包持有 Workspace Actor 访问注册表、操作定义目录（``WorkspaceOperation``）
和公共 application 使用的共享行为检查（``WorkspaceAccessGuard``），并承接
派生 cache、失效和快照等基础设施职责。认证凭据簇（``CallerPrincipal``/
``WorkspaceAccessContext``/grant/有效期锚点/受控工厂）随签发者归属
System 统一 Actor Authentication 网关（``system.access``）；guard 通过
中立结构契约消费凭据，本包不导入 System 任何模块。

资源读取、领域提交和结果查询统一由既有 Patchouli application service 与
``GlobalSystemBus`` 公开路由承接，本包不提供第二套业务 API，也不导入
Patchouli 内部 store/familiar/controller/local route 或 Alice/AgentRuntime
的任何实现。

依赖方向：``workspace`` 只依赖 core；System 接入层与 Patchouli
application 消费本包的访问基础设施（WRX-2/3 起扩展 runtime 与 cache）。
"""

from hivememory.workspace.access import (
    IssuedWorkspaceAccess,
    WorkspaceAccessGuard,
    WorkspaceOperation,
)
from hivememory.workspace.ports import (
    ResourceInvalidationPort,
)
from hivememory.workspace.projections import CanonicalResourceChange
from hivememory.workspace.registry import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)

__all__ = [
    # 操作目录与共享行为检查
    "IssuedWorkspaceAccess",
    "WorkspaceAccessGuard",
    "WorkspaceOperation",
    # Workspace Actor 访问注册表
    "WorkspaceActorAccessRecord",
    "WorkspaceActorAccessRegistry",
    # 基础设施端口
    "ResourceInvalidationPort",
    # 基础设施 DTO
    "CanonicalResourceChange",
]
