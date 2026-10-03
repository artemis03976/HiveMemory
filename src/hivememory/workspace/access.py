"""Workspace 准入、访问上下文生命周期与逐次行为授权。

workspace 认证入口（``workspace.authentication``）完成 Principal
authentication 后调用本模块的内部准入方法。WorkspaceAccessGuard 根据
Workspace Actor 注册表确认准入，签发最小上下文，并在每次 API 动作前验证
其有效性及行为白名单（实现 ``core.access.WorkspaceAccessVerifier``）。
资源自身的可见性仍由资源 owner 判断；本模块不依赖 System 或 Patchouli。

context 生命周期（v0.7.0 A1 访问边界返工第 4.4 节）：不设固定有效期；
context 只在三个时点失效——绑定的任务进程以任何结局关闭、请求级 context
随请求结束、System 停止时 guard 关闭使全部 context 失效。
"""

from __future__ import annotations

from weakref import WeakSet

from hivememory.core.access import WorkspaceAccessContext, WorkspaceOperation
from hivememory.core.errors import (
    AdmissionDeniedError,
    OperationDeniedError,
    OwnerMismatchError,
    ScopeRequiredError,
)
from hivememory.core.models import ActorIdentity, IdentityScope, WorkspaceIdentity
from hivememory.workspace.registry import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)


class WorkspaceAccessGuard:
    """持有准入结果的有效性状态，并供公共 application 逐次检查行为许可。

    一个运行实例共享同一个 guard。内部准入只供统一认证网关调用；业务
    入口调用 authorize_operation 后使用返回的可信 scope。签发记录仅为
    私有进程内状态，弱引用避免上下文在所有者释放后持续积累。
    该机制维护可信进程内调用纪律，不隔离任意恶意 Python 代码。
    """

    def __init__(
        self,
        access_registry: WorkspaceActorAccessRegistry,
    ) -> None:
        self._registry = access_registry
        self._closed = False
        self._issued: WeakSet[WorkspaceAccessContext] = WeakSet()

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """结束本运行实例的准入生命周期，既有上下文一并失效。"""
        self._closed = True
        self._issued.clear()

    def invalidate(self, access: WorkspaceAccessContext) -> None:
        """使单个已签发 context 失效（进程关闭或请求结束时调用）。

        失效必须由 context 的绑定所有者触发：任务进程以任何结局关闭时
        失效绑定的 context，请求级 context 在请求结束时失效。对未签发或
        已失效的 context 重复失效是幂等空操作。
        """
        if type(access) is not WorkspaceAccessContext:
            raise TypeError("invalidate 只接受 WorkspaceAccessContext")
        self._issued.discard(access)

    def _admit(
        self, *, actor: ActorIdentity, workspace: WorkspaceIdentity
    ) -> WorkspaceAccessContext:
        """网关内部第二项认证：确认 Workspace 准入并登记签发结果。

        不检查 principal，也不重复校验入参类型与关闭状态——唯一调用方
        System 网关在公开入口已完成来源/adapter 验证、入参类型检查与
        关闭检查（认证流程内无 await，两项检查间不存在状态变化）。
        此方法不作为 adapter 或领域服务的另一认证入口。
        """
        try:
            scope = IdentityScope(actor_identity=actor, workspace_identity=workspace)
        except OwnerMismatchError as exc:
            raise AdmissionDeniedError(
                message="actor 与 workspace owner 不一致，admission 拒绝",
                details={
                    "actor_user_id": actor.user_id,
                    "owner_user_id": workspace.owner_user_id,
                    "reason": "actor_not_owner",
                },
            ) from exc
        record = self._registry.record_for(workspace, actor)
        if record is None or not record.enabled:
            raise AdmissionDeniedError(
                message="该 Actor 在目标 Workspace 没有有效的访问登记",
                details={"reason": "actor_not_admitted"},
            )
        context = WorkspaceAccessContext(identity_scope=scope)
        self._issued.add(context)
        return context

    def verify_context(self, access: WorkspaceAccessContext | None) -> IdentityScope:
        """确认上下文由本实例签发、尚未失效且 Actor 仍有准入，返回可信 scope。

        不检查行为白名单：供 operation 授权已在能力层或任务进程完成的
        backing 读取入口与 Patchouli 公开方法取得可信坐标。资源归属与
        资源 policy 仍由资源 owner 独立校验。
        """
        return self._verified_record(access)[0]

    def authorize_operation(
        self,
        access: WorkspaceAccessContext | None,
        operation: WorkspaceOperation,
    ) -> IdentityScope:
        """检查本实例签发的上下文及当前行为许可，返回可信 scope。

        同一有效凭据可反复调用本检查先后执行不同的获准操作；切换
        Actor/Workspace、凭据失效或网关关闭后必须重新经统一网关认证。
        """
        if not isinstance(operation, WorkspaceOperation):
            raise TypeError("operation 必须是 WorkspaceOperation")
        scope, record = self._verified_record(access)
        # 行为白名单：缺少行为许可是授权失败，不是身份认证失败。
        if operation not in record.allowed_operations:
            raise OperationDeniedError(
                details={
                    "operation": operation.value,
                    "reason": "operation_not_allowed",
                }
            )
        return scope

    def _verified_record(
        self,
        access: WorkspaceAccessContext | None,
    ) -> tuple[IdentityScope, WorkspaceActorAccessRecord]:
        """校验签发、关闭状态与准入记录，返回可信 scope 与当前访问记录。"""
        if type(access) is not WorkspaceAccessContext:
            raise ScopeRequiredError("公共入口需要经统一认证网关签发的 WorkspaceAccessContext")
        if self._closed:
            raise ScopeRequiredError(
                "认证网关已关闭，access context 失效",
                details={"reason": "authentication_gateway_closed"},
            )
        if access not in self._issued:
            raise ScopeRequiredError(
                "access context 未由本运行实例签发",
                details={"reason": "context_not_issued"},
            )
        # 每次动作按完整坐标查询权限，不把配置对象地址或白名单绑定进凭据。
        scope = access.identity_scope
        record = self._registry.record_for(scope.workspace_identity, scope.actor_identity)
        if record is None or not record.enabled:
            raise ScopeRequiredError(
                "该 Actor 已无有效的 Workspace 访问登记",
                details={"reason": "actor_not_admitted"},
            )
        return scope, record


__all__ = [
    "WorkspaceAccessGuard",
]
