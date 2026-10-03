"""Workspace 认证入口：统一 Actor Authentication 网关。

统一认证编排（A1 访问边界设计）：所有调用侧只调用一个网关，输入受信接入
信息、待解析的 Actor 身份和目标 Workspace；网关内部顺序完成两项认证，全部
成功后返回一个可复用的 ``WorkspaceAccessContext``。

    1. Principal authentication —— 经 ``core.access.PrincipalAuthenticator``
       端口委托 System 实现（接入登记与 adapter 匹配），确认
       CallerPrincipal 和 ActorIdentity；
    2. Workspace authentication / admission —— 核对 Workspace Actor
       准入记录（含 W0 owner 约束），由 Workspace guard 签发准入结果。

认证入口不要求提供待执行的 operation，也不向调用方返回"只完成第一项
认证"的可访问上下文。网关经端口委托 System 完成接入登记校验，并委托 Workspace guard
完成内部准入；网关不持有 Workspace 签发状态或授权配置，也不
执行 search/read/submit，不接受任意 action 代执行业务，也不替代全局
总线路由——认证成功后，调用侧沿既有 adapter/service/bridge 发起业务调用。

阶段拒绝语义（均以稳定 reason 区分）：

- 接入未登记/禁用（不区分，避免泄漏配置）→ ``unknown_principal``；
- adapter 不匹配 → ``adapter_mismatch``；
- principal 身份解析规则不允许该用户 → ``actor_not_allowed_for_principal``；
- actor user ≠ workspace owner（W0 兼容基线）→ ``actor_not_owner``；
- Workspace 无有效 Actor 访问记录 → ``actor_not_admitted``。

以上均为 ``AdmissionDeniedError``（第一、二层失败）；缺少行为许可发生在
后续每次动作的共享行为检查（``workspace.access.WorkspaceAccessGuard``），
属 ``OperationDeniedError``，不在本网关判断。
"""

from __future__ import annotations

from hivememory.core.access import CallerPrincipal, PrincipalAuthenticator, WorkspaceAccessContext
from hivememory.core.errors import AdmissionDeniedError
from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.workspace.access import WorkspaceAccessGuard

__all__ = [
    "ActorAuthenticationGateway",
]


class ActorAuthenticationGateway:
    """统一 Actor Authentication 网关（workspace 认证入口）。

    一次 :meth:`authenticate` 调用完成 Principal authentication 与
    Workspace authentication；两项均通过才签发 ``WorkspaceAccessContext``。
    网关不执行业务、不转发路由；adapter 负责协议解析与接入证据，网关
    负责按登记规则统一验证并作出认证结论。

    生命周期（v0.7.0 A1 访问边界返工第 4.4 节）：context 不设固定有效期，
    只随三个时点失效——绑定的任务进程关闭、请求级 context 随请求结束、
    System 停止。:meth:`close` 只关闭网关自身、拒绝新的认证，已签发
    context 的失效由各自所有者完成；System 停止时在任务进程收尾后另行
    关闭 guard。本类不提供配置热更新——首版本地配置在运行实例内不可变，
    修改经重启生效。
    """

    def __init__(
        self,
        *,
        principals: PrincipalAuthenticator,
        workspace_access: WorkspaceAccessGuard,
    ) -> None:
        self._principals = principals
        self._workspace_access = workspace_access
        self._closed = False

    @property
    def is_closed(self) -> bool:
        """网关或共享 guard 是否已关闭；关闭后不再签发 context。"""
        return self._closed or self._workspace_access.is_closed

    def close(self) -> None:
        """关闭网关：拒绝新的认证请求，不影响已签发 context 的剩余生命周期。"""
        self._closed = True

    def invalidate_context(self, access: WorkspaceAccessContext) -> None:
        """使单个已签发 context 失效；委托给持有签发状态的共享 guard。"""
        self._workspace_access.invalidate(access)

    async def authenticate(
        self,
        *,
        adapter: str,
        principal: CallerPrincipal,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
    ) -> WorkspaceAccessContext:
        """完成两项认证并签发访问上下文；任一失败即拒绝，无部分成功。

        ``adapter`` 是调用来源实际使用的接入方式标识；``principal`` 由
        受信 adapter 依据其接入证据构造。认证成功返回的 context 与单次
        operation 解耦，可在此后各次获准动作中复用。
        """
        if self.is_closed:
            raise AdmissionDeniedError(
                message="认证网关已关闭，拒绝新的认证请求",
                details={"reason": "authentication_gateway_closed"},
            )
        if not isinstance(adapter, str) or not adapter.strip():
            raise TypeError("adapter 必须是非空字符串")
        if not isinstance(principal, CallerPrincipal):
            raise TypeError("principal 必须是 CallerPrincipal")
        if not isinstance(actor, ActorIdentity):
            raise TypeError("actor 必须是 ActorIdentity")
        if not isinstance(workspace, WorkspaceIdentity):
            raise TypeError("workspace 必须是 WorkspaceIdentity")

        # ---- 1. Principal authentication：委托 System 实现的接入登记校验 ----
        self._principals.authenticate_principal(
            adapter=adapter,
            principal=principal,
            actor=actor,
        )

        # ---- 2. Workspace authentication / admission ----
        return self._workspace_access._admit(actor=actor, workspace=workspace)
