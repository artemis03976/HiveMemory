"""Workspace 基础设施：统一访问边界、授权上下文与派生失效协作。

父计划（docs/plans/v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md，
2026-09-16 application 边界修订）确立的分工：本包只承担
``WorkspaceAccessBoundary``、access context、派生 cache、失效和快照等
基础设施职责；资源读取、领域提交和结果查询统一由既有 Patchouli
application service 与 ``GlobalSystemBus`` 公开路由承接，本包不提供
第二套业务 API，也不导入 Patchouli 内部 store/familiar/controller/local
route 或 Alice/AgentRuntime 的任何实现。

依赖方向：``workspace`` 只依赖 core；Patchouli application 与未来接入的
组合根消费本包的访问与缓存基础设施（WRX-2/3 起扩展 runtime 与 cache）。
"""

from hivememory.workspace.access import (
    CallerPrincipal,
    LocalTrustedAdmissionService,
    WorkspaceAccessContext,
    WorkspaceOperation,
    issue_access_context,
    require_access_context,
)
from hivememory.workspace.ports import (
    ResourceInvalidationPort,
    WorkspaceAdmissionPort,
)
from hivememory.workspace.projections import CanonicalResourceChange

__all__ = [
    # 访问边界
    "CallerPrincipal",
    "LocalTrustedAdmissionService",
    "WorkspaceAccessContext",
    "WorkspaceOperation",
    "issue_access_context",
    "require_access_context",
    # 基础设施端口
    "ResourceInvalidationPort",
    "WorkspaceAdmissionPort",
    # 基础设施 DTO
    "CanonicalResourceChange",
]
