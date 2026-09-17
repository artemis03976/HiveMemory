"""
Workspace 访问边界：CallerPrincipal、operation capability 与受控准入。

父计划（docs/plans/v0.7.0-workspace-resource-system-and-agent-execution-boundaries.md
第 5.6 节）冻结的四层检查顺序在本模块落地为：

    受信入口建立 CallerPrincipal
      -> Workspace admission：principal 能否代表该 Actor 进入该 Workspace
      -> Operation authorization：该 operation 是否在 grant 能力内
      -> 返回只能由本模块工厂构造的 WorkspaceAccessContext

资源 ownership/visibility 与领域 policy 不在本模块判断，仍由资源服务与
Patchouli 在准入之后分层执行。授权上下文（grant）是一次准入的结果，
不写入 Profile/Atom cache，也不因缓存命中而跳过重新校验。

v0.7.0 首版只支持显式的本地/受信配置映射（``LocalTrustedAdmissionService``），
不建设完整成员目录、远程 token/IAM 或跨 Workspace 委托；外部 connector
的 principal 建立与映射由计划 B 负责，A 不把外部认证材料写入资源模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from hivememory.core.errors import (
    AdmissionDeniedError,
    OperationDeniedError,
    OwnerMismatchError,
    ScopeRequiredError,
)
from hivememory.core.models import ActorIdentity, IdentityScope, WorkspaceIdentity


class WorkspaceOperation(str, Enum):
    """Actor→Workspace 的窄能力 operation 枚举（父计划 5.6.2 能力表）。

    每个 capability 只授予其语义声明的能力；"不能推导出的权限"列见父计划。
    ``MANAGEMENT_MEMORY`` 是显式管理 operation：供 owner-management 管理入口
    （如 ``MemoryManagementService``）迁移后使用，绝不进入 Agent resource port。
    """

    RESOURCE_READ = "resource.read"
    RESOURCE_SEARCH = "resource.search"
    PROFILE_READ = "profile.read"
    ASSET_ACQUIRE = "asset.acquire"
    INTERACTION_SUBMIT = "interaction.submit"
    MEMORY_INTENT_SUBMIT = "memory_intent.submit"
    TASK_OBSERVE = "task.observe"
    MANAGEMENT_MEMORY = "management.memory"


@dataclass(frozen=True)
class CallerPrincipal:
    """已被受信入口建立的调用方身份。

    回答"哪个被信任的进程、connector 或本地调用方在发起请求"；请求体中的
    ``user_id``/``agent_id``/``role`` 字符串只是待验证的 claim，不能自封
    principal。``principal_id`` 使用稳定的带命名空间标识（如
    ``local-process:alice-runtime``）。
    """

    principal_id: str
    kind: str = "local-process"

    def __post_init__(self) -> None:
        if not isinstance(self.principal_id, str) or not self.principal_id.strip():
            raise ValueError("principal_id 不能为空")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("kind 不能为空")


class _AdmissionGrant:
    """准入签发凭证（模块私有）。

    ``WorkspaceAccessContext`` 只能携带本类实例构造，而本类仅在
    ``issue_access_context`` 工厂中创建——以此把"谁有权签发访问上下文"
    收敛到 admission 边界，防止调用方用普通字典伪造 grant。
    """

    __slots__ = ("issued_by",)

    def __init__(self, issued_by: str) -> None:
        self.issued_by = issued_by


@dataclass(frozen=True)
class WorkspaceAccessContext:
    """一次准入得到的不可变访问上下文。

    包含已验证的 ``IdentityScope``（冻结 Actor + Workspace 坐标）、请求的
    operation 以及 admission 签发的 grant。``IdentityScope`` 继续只表达
    坐标，不携带 principal、run/frame、policy cache 或"当前 Workspace"。

    构造受控：只能通过 admission 边界的 ``issue_access_context`` 获得，
    进程内 Alice 也必须走同一工厂，不以"内部调用"绕过。
    """

    principal: CallerPrincipal
    identity_scope: IdentityScope
    operation: WorkspaceOperation
    grant: _AdmissionGrant = field(repr=False, compare=False)


def issue_access_context(
    principal: CallerPrincipal,
    identity_scope: IdentityScope,
    operation: WorkspaceOperation,
    *,
    issued_by: str,
) -> WorkspaceAccessContext:
    """admission 边界的受控工厂：签发一个不可变访问上下文。

    只应由 admission 服务调用；其他模块构造不出携带合法 grant 的上下文。
    """
    if not isinstance(principal, CallerPrincipal):
        raise TypeError("principal 必须是 CallerPrincipal")
    if not isinstance(identity_scope, IdentityScope):
        raise ScopeRequiredError("access context 需要已验证的 IdentityScope")
    if not isinstance(operation, WorkspaceOperation):
        raise TypeError("operation 必须是 WorkspaceOperation")
    return WorkspaceAccessContext(
        principal=principal,
        identity_scope=identity_scope,
        operation=operation,
        grant=_AdmissionGrant(issued_by=issued_by),
    )


def require_access_context(
    access: WorkspaceAccessContext | None,
    *,
    operation: WorkspaceOperation,
) -> WorkspaceAccessContext:
    """资源/领域端口的统一消费点：校验 access context 与 operation 匹配。

    - 缺失或类型不符 → ``ScopeRequiredError``（拒绝未经准入的裸 scope）；
    - operation 与签发的 grant 不一致 → ``OperationDeniedError``。
    """
    if not isinstance(access, WorkspaceAccessContext) or not isinstance(
        access.grant, _AdmissionGrant
    ):
        raise ScopeRequiredError("资源/领域入口需要经 admission 签发的 WorkspaceAccessContext")
    if access.operation is not operation:
        raise OperationDeniedError(
            details={
                "granted_operation": access.operation.value,
                "required_operation": operation.value,
            }
        )
    return access


class LocalTrustedAdmissionService:
    """v0.7.0 本地/受信映射 admission 实现。

    端口契约见 ``workspace.ports.WorkspaceAdmissionPort``（Protocol）；本类
    按该形状结构化实现。通过显式配置声明"哪些 principal 可以请求哪些
    operation"；Actor 与 Workspace 的归属关系仍由 ``IdentityScope`` 的
    owner 约束校验（actor user 必须等于 workspace owner），该约束不得在
    请求体中放宽。

    未注册 principal、未授权 operation、owner 不一致都会 fail closed。
    """

    def __init__(
        self,
        trusted_principals: dict[str, WorkspaceOperation | list[WorkspaceOperation]],
        *,
        issued_by: str = "local-trusted-admission",
    ) -> None:
        normalized: dict[str, frozenset[WorkspaceOperation]] = {}
        for principal_id, operations in trusted_principals.items():
            if isinstance(operations, WorkspaceOperation):
                operations = [operations]
            ops = frozenset(operations)
            if not ops:
                raise ValueError(f"principal {principal_id!r} 至少需要一个 operation")
            normalized[principal_id] = ops
        self._trusted = normalized
        self._issued_by = issued_by

    async def admit(
        self,
        principal: CallerPrincipal,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
        operation: WorkspaceOperation,
    ) -> WorkspaceAccessContext:
        """校验 principal 注册与 operation 授权后签发访问上下文。"""
        if not isinstance(principal, CallerPrincipal):
            raise TypeError("principal 必须是 CallerPrincipal")
        if not isinstance(actor, ActorIdentity):
            raise TypeError("actor 必须是 ActorIdentity")
        if not isinstance(workspace, WorkspaceIdentity):
            raise TypeError("workspace 必须是 WorkspaceIdentity")
        if not isinstance(operation, WorkspaceOperation):
            raise TypeError("operation 必须是 WorkspaceOperation")

        allowed = self._trusted.get(principal.principal_id)
        if allowed is None:
            # 未知 principal 一律拒绝；不区分"未注册"与"已吊销"，避免泄漏配置。
            raise AdmissionDeniedError(
                details={"principal_id": principal.principal_id, "reason": "unknown_principal"}
            )
        if operation not in allowed:
            raise OperationDeniedError(
                details={
                    "principal_id": principal.principal_id,
                    "operation": operation.value,
                    "reason": "operation_not_granted",
                }
            )

        try:
            identity_scope = IdentityScope(
                actor_identity=actor,
                workspace_identity=workspace,
            )
        except OwnerMismatchError as exc:
            # owner 约束是 v0.7.0 的准入基线：actor user ≠ workspace owner
            # 时按 admission 拒绝，而不是把矛盾坐标放行到资源层。
            raise AdmissionDeniedError(
                message="actor 与 workspace owner 不一致，admission 拒绝",
                details={
                    "principal_id": principal.principal_id,
                    "actor_user_id": actor.user_id,
                    "owner_user_id": workspace.owner_user_id,
                    "reason": "actor_not_owner",
                },
            ) from exc

        return issue_access_context(
            principal,
            identity_scope,
            operation,
            issued_by=self._issued_by,
        )


__all__ = [
    "CallerPrincipal",
    "WorkspaceOperation",
    "WorkspaceAccessContext",
    "LocalTrustedAdmissionService",
    "issue_access_context",
    "require_access_context",
]
