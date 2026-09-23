"""Workspace 基础设施：操作目录、访问注册表与共享行为检查。

A1 计划（docs/plans/v0.7.0-a1-workspace-access-boundary.md）确立的分工：
本包持有 Workspace Actor 访问注册表、操作定义目录（``WorkspaceOperation``）
和公共 application 使用的共享行为检查（``WorkspaceAccessGuard``）。资源读取、
缓存与失效协作由后续计划定义。``WorkspaceAccessContext`` 是本包
持有的最小准入结果，其签发和有效性由同一 guard 管理。调用来源 principal
与唯一对外认证网关归属 System；本包不导入 System 任何模块。

资源读取、领域提交和结果查询统一由既有 Patchouli application service 与
``GlobalSystemBus`` 公开路由承接，本包不提供第二套业务 API，也不导入
Patchouli 内部 store/familiar/controller/local route 或 Alice/AgentRuntime
的任何实现。

依赖方向：``workspace`` 只依赖 core；System 接入层与 Patchouli
application 消费本包的访问基础设施（WRX-2/3 起扩展 runtime 与 cache）。
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

__all__ = [
    # 操作目录与共享行为检查
    "WorkspaceAccessContext",
    "WorkspaceAccessGuard",
    "WorkspaceOperation",
    # Workspace Actor 访问注册表
    "WorkspaceActorAccessRecord",
    "WorkspaceActorAccessRegistry",
]
