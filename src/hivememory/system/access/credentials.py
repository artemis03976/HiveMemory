"""统一认证网关签发的访问凭据：WorkspaceAccessContext 与受控工厂。

凭据随签发者归属 System 认证平面（A1 计划第 2.4/3.1 节）：
``WorkspaceAccessContext`` 只能经本模块的 :func:`issue_access_context`
工厂、由 ``system.access.ActorAuthenticationGateway`` 在两项认证均通过后
签发；grant 绑定签发时的 principal、身份坐标、Workspace Actor 访问记录
与有效期，凭据通过 :meth:`WorkspaceAccessContext.ensure_usable` 自检
完整性——消费端（``workspace.access.WorkspaceAccessGuard``）据此把
"凭据是否仍然可用"与"行为白名单是否允许"分开核对。

历史版本中"一个 context 只能匹配一个 operation"的签名已按 A1 第 6 节
迁移表退出：context 与单次 operation 解耦，实际需要的 operation 由
application 方法绑定、在共享行为检查时传入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from hivememory.core.errors import ScopeRequiredError
from hivememory.core.models import IdentityScope
from hivememory.workspace.registry import WorkspaceActorAccessRecord

from hivememory.system.access.principal import CallerPrincipal

__all__ = [
    "AccessContextValidity",
    "WorkspaceAccessContext",
    "issue_access_context",
]


class AccessContextValidity:
    """认证有效期锚点：由签发网关持有，``close()`` 后其签发的全部 context 失效。

    网关关闭（或所在运行实例结束）后，旧 context 拒绝继续使用；调用侧
    必须重新经过统一网关认证。该对象随 grant 传递给凭据自检，保证
    "网关关闭"这一事实对所有持有者同时生效。
    """

    __slots__ = ("_closed",)

    def __init__(self) -> None:
        self._closed = False

    def close(self) -> None:
        """关闭锚点：之后所有携带本锚点的 context 一并失效。"""
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


class _AccessGrant:
    """统一网关受控签发的内部凭据（模块私有）。

    只能由 :func:`issue_access_context` 工厂创建；绑定签发时的
    principal、身份坐标、Workspace Actor 访问记录与有效期。消费端按
    绑定逐项核对——复制合法凭据后替换身份或权限配置关联会被拒绝，而
    不仅是检查凭据类型。该约束维护可信进程内调用纪律，不宣称隔离任意
    进程内恶意 Python 代码。
    """

    __slots__ = (
        "issued_by",
        "principal_id",
        "identity_scope",
        "access_record",
        "expires_at",
        "validity",
    )

    def __init__(
        self,
        *,
        issued_by: str,
        principal_id: str,
        identity_scope: IdentityScope,
        access_record: WorkspaceActorAccessRecord,
        validity: AccessContextValidity,
        expires_at: float | None,
    ) -> None:
        self.issued_by = issued_by
        self.principal_id = principal_id
        self.identity_scope = identity_scope
        self.access_record = access_record
        self.validity = validity
        self.expires_at = expires_at


@dataclass(frozen=True)
class WorkspaceAccessContext:
    """一次认证得到的不可变访问上下文（与单次 operation 解耦）。

    回答"哪个 Actor 已通过统一网关认证进入哪个 Workspace"：包含已验证
    的 ``IdentityScope``（冻结 Actor + Workspace 坐标）、来源 principal
    与网关签发的 grant。**不携带 operation 字段**——实际需要的 operation
    由 application 方法绑定，在调用共享行为检查时传入；同一有效 context
    可先后执行不同的获准操作。

    ``IdentityScope`` 继续只表达坐标，不携带 principal、run/frame、policy
    cache 或"当前 Workspace"。构造受控：只能经统一认证网关获得；进程内
    调用方（Alice、System application）也必须走同一网关，不以"内部调用"
    绕过。context 不写入 Profile/Atom cache，也不因缓存命中跳过重新校验。
    """

    principal: CallerPrincipal
    identity_scope: IdentityScope
    grant: _AccessGrant = field(repr=False, compare=False)

    def ensure_usable(
        self, *, clock: Callable[[], float]
    ) -> WorkspaceActorAccessRecord:
        """凭据自检：验证完整性/有效期，返回绑定的 Workspace Actor 访问记录。

        由共享行为检查（``workspace.access.WorkspaceAccessGuard``）在白名单
        核对之前调用；检查顺序与拒绝语义（A1 第 3.4 节）：

        - 凭据不是本模块工厂签发（grant 类型不符）→ ``ScopeRequiredError``；
        - principal/身份坐标与签发绑定不一致（替换即拒绝）
          → ``ScopeRequiredError``（reason=``context_binding_invalid``）；
        - 网关已关闭 → ``ScopeRequiredError``
          （reason=``authentication_gateway_closed``）；
        - 超过认证有效区间 → ``ScopeRequiredError``（reason=``context_expired``）。
        """
        grant = self.grant
        if not isinstance(grant, _AccessGrant):
            raise ScopeRequiredError(
                "公共入口需要经统一认证网关签发的 WorkspaceAccessContext"
            )
        # 绑定完整性：principal 与身份坐标必须与签发时一致，替换即拒绝。
        if (
            grant.principal_id != self.principal.principal_id
            or grant.identity_scope != self.identity_scope
        ):
            raise ScopeRequiredError(
                "access context 与签发绑定不一致",
                details={"reason": "context_binding_invalid"},
            )
        # 有效区间：网关关闭或超过声明有效期后拒绝，需重新认证。
        if grant.validity.closed:
            raise ScopeRequiredError(
                "认证网关已关闭，access context 失效",
                details={"reason": "authentication_gateway_closed"},
            )
        if grant.expires_at is not None and clock() >= grant.expires_at:
            raise ScopeRequiredError(
                "access context 已过认证有效区间",
                details={"reason": "context_expired"},
            )
        return grant.access_record


def issue_access_context(
    principal: CallerPrincipal,
    identity_scope: IdentityScope,
    *,
    access_record: WorkspaceActorAccessRecord,
    validity: AccessContextValidity,
    issued_by: str,
    expires_at: float | None = None,
) -> WorkspaceAccessContext:
    """统一认证网关的受控工厂：签发一个不可变访问上下文。

    只应由 ``system.access.ActorAuthenticationGateway`` 在两项认证均通过
    后调用；其他模块构造不出携带合法 grant 的上下文。
    """
    if not isinstance(principal, CallerPrincipal):
        raise TypeError("principal 必须是 CallerPrincipal")
    if not isinstance(identity_scope, IdentityScope):
        raise ScopeRequiredError("access context 需要已验证的 IdentityScope")
    if not isinstance(access_record, WorkspaceActorAccessRecord):
        raise TypeError("access_record 必须是 WorkspaceActorAccessRecord")
    if not isinstance(validity, AccessContextValidity):
        raise TypeError("validity 必须是网关持有的 AccessContextValidity 锚点")
    return WorkspaceAccessContext(
        principal=principal,
        identity_scope=identity_scope,
        grant=_AccessGrant(
            issued_by=issued_by,
            principal_id=principal.principal_id,
            identity_scope=identity_scope,
            access_record=access_record,
            validity=validity,
            expires_at=expires_at,
        ),
    )
