"""Workspace 资源平面：进程级资源访问、缓存、授权与失效的聚合入口。

父计划（docs/plans/v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md）
按阶段落地本包；WRX-1 交付访问边界（access）、端口（ports）、不可变快照
（projections）与 Patchouli 低层 provider 适配（services）。Runtime 聚合
入口于 WRX-2 加入，派生 cache 与失效实现于 WRX-3 加入。

依赖方向：本包不得导入 ``hivememory.alice`` 或 ``hivememory.agent_runtime``。
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
    DomainMutationPort,
    DomainResultPort,
    ResourceInvalidationPort,
    WorkspaceAdmissionPort,
    WorkspaceResourcePort,
)
from hivememory.workspace.projections import (
    CanonicalResourceChange,
    DomainHandle,
    DomainResult,
    DomainSubmission,
    InteractionApplyRequest,
    InteractionApplyResult,
    MemoryIntentRequest,
    MemorySnapshot,
    ProfileSnapshot,
)
from hivememory.workspace.services import (
    PatchouliDomainGateway,
    ProfileResourceService,
    WorkspaceMemoryService,
)

__all__ = [
    # 访问边界
    "CallerPrincipal",
    "LocalTrustedAdmissionService",
    "WorkspaceAccessContext",
    "WorkspaceAdmissionPort",
    "WorkspaceOperation",
    "issue_access_context",
    "require_access_context",
    # 端口
    "DomainMutationPort",
    "DomainResultPort",
    "ResourceInvalidationPort",
    "WorkspaceResourcePort",
    # 快照与 DTO
    "CanonicalResourceChange",
    "DomainHandle",
    "DomainResult",
    "DomainSubmission",
    "InteractionApplyRequest",
    "InteractionApplyResult",
    "MemoryIntentRequest",
    "MemorySnapshot",
    "ProfileSnapshot",
    # 服务
    "PatchouliDomainGateway",
    "ProfileResourceService",
    "WorkspaceMemoryService",
]
