"""Workspace 认证入口：统一 Actor Authentication 网关与第 2 阶段认证者。

认证与操作授权分属两个类：

- :class:`ActorAuthenticationGateway` 是唯一对外的认证入口：依次调用
  ``PrincipalAuthenticator``（第 1 阶段）与 :class:`WorkspaceAuthenticator`
  （第 2 阶段），转交 context 的撤销与诊断查询，并负责关闭（拒绝新认证）；
- :class:`WorkspaceAuthenticator` 负责第 2 阶段准入，签发写有授予内容的
  密封 context，并负责撤销。

授予内容密封在 context 内，操作授权者
（``workspace.authorization.WorkspaceOperationAuthorizer``）直接读取，不依赖
认证一侧；认证与授权只经 context 这个类型发生联系。

运行持有者（server、注册入口）只经认证网关接触认证一侧，不直接持有
``WorkspaceAuthenticator``。认证入口不要求提供待执行的 operation，也不向
调用方返回"只完成第一项认证"的可访问上下文；网关不执行业务、不转发路由。

阶段拒绝语义（均以稳定 reason 区分）：

- 接入未登记/禁用（不区分，避免泄漏配置）→ ``unknown_principal``；
- adapter 不匹配 → ``adapter_mismatch``；
- 登记了凭据而未提供一致的凭据 → ``invalid_credential``；
- principal 身份解析规则不允许该用户 → ``actor_not_allowed_for_principal``；
- 不允许声明 ``system`` 的来源声明了它 → ``system_actor_not_allowed``；
- actor user ≠ 要进入的 workspace 的 owner（准入约束，第 2 阶段检查）
  → ``actor_not_owner``；
- Workspace 无有效 Actor 访问记录 → ``actor_not_admitted``。

以上均为 ``AdmissionDeniedError``（第一、二阶段失败）。缺少行为许可发生
在后续每次操作的操作授权（``workspace.authorization``），属
``OperationDeniedError``，不在本模块判断。
"""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakSet

from hivememory.core.access import (
    AccessGrant,
    AccessRunType,
    CallerPrincipal,
    PrincipalAuthenticator,
    RunBinding,
    WorkspaceAccessContext,
)
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import AdmissionDeniedError, ScopeRequiredError
from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.workspace.registry import WorkspaceActorAccessRegistry

__all__ = [
    "AccessGrantSummary",
    "ActorAuthenticationGateway",
    "WorkspaceAuthenticator",
]


@dataclass(frozen=True, slots=True)
class AccessGrantSummary:
    """授予内容的只读摘要，仅供日志与观测标签使用。

    本摘要不能作为授权依据：授权仍必须把 context 交给操作授权者；摘要中
    的坐标不构成准入或授权结论。
    """

    actor_user_id: str
    agent_id: str
    workspace_id: str
    principal_id: str
    run_type: str
    run_id: str
    thread_id: str | None = None


class WorkspaceAuthenticator:
    """第 2 阶段 Workspace 认证：准入、签发与撤销（认证一侧）。

    准入 :meth:`admit` 由认证网关与本类的子线程派生调用；撤销（:meth:`revoke`、
    :meth:`revoke_all`）与诊断查询（:meth:`describe`）只经认证网关转交。
    签发的 context 自带密封的授予内容，本类只记住"签发过哪些仍可能存活
    的 context"，用于 System 停止时撤销全部；没有自身的关闭状态——关闭
    只在认证网关，已签发的 context 照常可用，直到被撤销。该机制维护可信
    进程内调用纪律，不隔离任意恶意 Python 代码。
    """

    def __init__(self, access_registry: WorkspaceActorAccessRegistry) -> None:
        self._registry = access_registry
        self._issued: WeakSet[WorkspaceAccessContext] = WeakSet()

    def admit(
        self,
        *,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
        principal: CallerPrincipal,
        binding: RunBinding,
    ) -> WorkspaceAccessContext:
        """第 2 阶段认证：确认 Workspace 准入，签发写有授予内容的 context。

        owner 约束在本阶段检查：actor 用户必须等于要进入的
        workspace 的 owner。不检查 principal，也不重复校验入参类型与关闭
        状态——认证网关已验证来源与声明；子线程派生则复用有效进程的来源
        与驻留目标，只替换同一用户的 agent。两条调用链均无 await，不存在
        校验与签发之间的状态变化。此方法不作为 adapter 或领域服务的另一认证入口。
        """
        if actor.user_id != workspace.owner_user_id:
            raise AdmissionDeniedError(
                message="actor 与 workspace owner 不一致，admission 拒绝",
                details={
                    "actor_user_id": actor.user_id,
                    "owner_user_id": workspace.owner_user_id,
                    "reason": "actor_not_owner",
                },
            )
        record = self._registry.record_for(workspace, actor)
        if record is None or not record.enabled:
            raise AdmissionDeniedError(
                message="该 Actor 在目标 Workspace 没有有效的访问登记",
                details={"reason": "actor_not_admitted"},
            )
        # 行为白名单不进入授予内容：每次授权都按访问登记即时查询，不缓存
        # 授权结论；有效期也不进入——失效由绑定的运行结束决定。
        context = WorkspaceAccessContext._seal(
            AccessGrant(actor=actor, workspace=workspace, principal=principal, binding=binding)
        )
        self._issued.add(context)
        return context

    def derive_thread_context(self, access: WorkspaceAccessContext) -> WorkspaceAccessContext:
        """沿用已认证的授予内容，为主线程派生一份可独立撤销的 context。

        派生不是重新认证，不访问接入登记或网关关闭状态；准入与操作许可
        仍由授权点即时检查。请求级、线程级以及已撤销的 context 均不能派生。
        """
        if type(access) is not WorkspaceAccessContext:
            raise ScopeRequiredError("派生需要有效的进程 context")
        grant = access._unseal()
        if grant is None or grant.binding.run_type is not AccessRunType.TASK_PROCESS:
            raise ScopeRequiredError("派生需要有效的进程 context")
        context = WorkspaceAccessContext._seal(
            AccessGrant(
                actor=grant.actor,
                workspace=grant.workspace,
                principal=grant.principal,
                binding=RunBinding.for_execution_thread(grant.binding.run_id, "main"),
            )
        )
        self._issued.add(context)
        return context

    def derive_child_thread_context(
        self, access: WorkspaceAccessContext, *, agent_id: str, thread_id: str
    ) -> WorkspaceAccessContext:
        """被调用方沿用进程来源与驻留目标，经第二阶段准入后签发子线程。

        不重新认证 principal；网关关闭不阻止已有进程派生。先完成全部
        校验再签发，失败时不会留下可用凭据。
        """
        if type(access) is not WorkspaceAccessContext:
            raise ScopeRequiredError("派生需要有效的进程 context")
        grant = access._unseal()
        if grant is None or grant.binding.run_type is not AccessRunType.TASK_PROCESS:
            raise ScopeRequiredError("派生需要有效的进程 context")
        if agent_id == SYSTEM_AGENT_ID:
            raise AdmissionDeniedError(
                "子线程不能使用保留 system actor",
                details={"reason": "system_actor_not_allowed"},
            )
        if thread_id == "main":
            raise ScopeRequiredError("子线程不能使用主线程标识")
        return self.admit(
            actor=ActorIdentity(user_id=grant.actor.user_id, agent_id=agent_id),
            workspace=grant.workspace,
            principal=grant.principal,
            binding=RunBinding.for_execution_thread(grant.binding.run_id, thread_id),
        )

    def revoke_thread_context(self, access: WorkspaceAccessContext) -> None:
        """只撤销线程 context，避免窄化端口误撤销仍需收尾的进程 context。"""
        if type(access) is not WorkspaceAccessContext:
            raise ScopeRequiredError("撤销线程需要线程 context")
        grant = access._unseal()
        if grant is not None and grant.binding.run_type is not AccessRunType.EXECUTION_THREAD:
            raise ScopeRequiredError("撤销线程需要线程 context")
        self.revoke(access)

    def revoke(self, access: WorkspaceAccessContext) -> None:
        """撤销单个 context（进程关闭或请求结束时调用）。

        撤销必须由 context 的绑定所有者触发：任务进程以任何结局关闭时
        撤销绑定的 context，请求级 context 在请求结束时撤销。重复撤销是
        幂等空操作。
        """
        if type(access) is not WorkspaceAccessContext:
            raise TypeError("revoke 只接受 WorkspaceAccessContext")
        access._revoke()
        self._issued.discard(access)

    def revoke_all(self) -> None:
        """System 停止时撤销全部已签发 context，不留下仍可使用的凭据。

        撤销后 context 不再能通过授权（``context_not_issued``）；操作授权者
        无状态，不需要随之关闭。
        """
        for context in list(self._issued):
            context._revoke()
        self._issued.clear()

    def describe(self, access: WorkspaceAccessContext) -> AccessGrantSummary | None:
        """返回授予内容的只读摘要；已撤销返回 ``None``。

        只用于日志与观测标签（如取消入口为观测事件取请求方的驻留
        workspace 标签），不能作为授权依据。
        """
        if type(access) is not WorkspaceAccessContext:
            raise TypeError("describe 只接受 WorkspaceAccessContext")
        grant = access._unseal()
        if grant is None:
            return None
        return AccessGrantSummary(
            actor_user_id=grant.actor.user_id,
            agent_id=grant.actor.agent_id,
            workspace_id=grant.workspace.workspace_id,
            principal_id=grant.principal.principal_id,
            run_type=grant.binding.run_type.value,
            run_id=grant.binding.run_id,
            thread_id=grant.binding.thread_id,
        )


class ActorAuthenticationGateway:
    """统一 Actor Authentication 网关（workspace 认证入口）。

    一次 :meth:`authenticate` 调用编排两项认证——Principal
    authentication（第 1 阶段，经端口委托 System）与 Workspace
    authentication（第 2 阶段，委托 :class:`WorkspaceAuthenticator`）；
    两项均通过才签发 ``WorkspaceAccessContext``，签发时写入运行绑定：
    任务进程 context 绑定本进程的 ``process_id``，请求级
    context 绑定入口为本次请求生成的标识。网关是唯一对外的认证入口：
    context 的撤销与诊断查询都经这里转交认证一侧，运行持有者不
    直接接触 ``WorkspaceAuthenticator``。

    生命周期：context 不设固定有效期，
    随绑定的任务进程关闭、CPU 执行结束、请求结束或 System 停止而失效。
    :meth:`close` 关闭网关并拒绝新的认证，已签发 context 照常
    可用；任务进程收尾后经 :meth:`revoke_all_contexts` 撤销全部已签发
    context。本类不提供配置热更新——
    本地配置在运行实例内不可变，修改经重启生效。
    """

    def __init__(
        self,
        *,
        principals: PrincipalAuthenticator,
        authenticator: WorkspaceAuthenticator,
    ) -> None:
        self._principals = principals
        self._authenticator = authenticator
        self._closed = False

    @property
    def is_closed(self) -> bool:
        """网关是否已关闭；关闭后拒绝新认证，已有进程仍可派生线程。"""
        return self._closed

    def close(self) -> None:
        """关闭网关：拒绝新的认证请求，不影响已签发 context 的剩余生命周期。"""
        self._closed = True

    def revoke_context(self, access: WorkspaceAccessContext) -> None:
        """撤销单个已签发 context；转交认证一侧（进程关闭或请求结束）。"""
        self._authenticator.revoke(access)

    def derive_thread_context(self, access: WorkspaceAccessContext) -> WorkspaceAccessContext:
        """实现执行器的派生端口；关闭网关只拒绝新认证，不禁止已有进程派生。"""
        return self._authenticator.derive_thread_context(access)

    def derive_child_thread_context(
        self, access: WorkspaceAccessContext, *, agent_id: str, thread_id: str
    ) -> WorkspaceAccessContext:
        """实现进程侧的子线程派生端口；准入仍由认证一侧检查。"""
        return self._authenticator.derive_child_thread_context(
            access, agent_id=agent_id, thread_id=thread_id
        )

    def revoke_thread_context(self, access: WorkspaceAccessContext) -> None:
        """实现执行器的同步撤销端口，不暴露进程或请求 context 的撤销能力。"""
        self._authenticator.revoke_thread_context(access)

    def revoke_all_contexts(self) -> None:
        """System 停止时撤销全部已签发 context；此后它们不再能通过授权。"""
        self._authenticator.revoke_all()

    def describe_context(self, access: WorkspaceAccessContext) -> AccessGrantSummary | None:
        """返回授予内容的只读摘要（日志与观测标签专用）；转交认证一侧。"""
        return self._authenticator.describe(access)

    async def authenticate(
        self,
        *,
        adapter: str,
        principal: CallerPrincipal,
        credential: str | None,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
        binding: RunBinding,
    ) -> WorkspaceAccessContext:
        """完成两项认证并签发访问上下文；任一失败即拒绝，无部分成功。

        ``adapter`` 是调用来源实际使用的接入方式标识，由入口代码固定传入；
        ``principal`` 是入口传入的调用来源，``credential`` 是它的接入证据
        ——进程内入口传 ``None``，外部入口原样传入请求中的凭据，两者都交给
        第 1 阶段核对，入口不自行判定身份；``binding`` 是本次运行绑定——
        任务进程注册传 :meth:`RunBinding.for_task_process`，请求级访问传
        :meth:`RunBinding.for_request`。认证成功返回的 context 与单次
        operation 解耦，可在此后各次获准操作中经操作授权者逐次授权。
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
        if credential is not None and not isinstance(credential, str):
            raise TypeError("credential 必须是字符串或 None")
        if not isinstance(actor, ActorIdentity):
            raise TypeError("actor 必须是 ActorIdentity")
        if not isinstance(workspace, WorkspaceIdentity):
            raise TypeError("workspace 必须是 WorkspaceIdentity")
        if not isinstance(binding, RunBinding):
            raise TypeError("binding 必须是 RunBinding")
        if binding.run_type is AccessRunType.EXECUTION_THREAD:
            raise ScopeRequiredError("线程 context 必须从进程 context 派生")

        # ---- 1. Principal authentication：委托 System 实现的接入登记校验 ----
        self._principals.authenticate_principal(
            adapter=adapter,
            principal=principal,
            credential=credential,
            actor=actor,
        )

        # ---- 2. Workspace authentication / admission（签发即绑定运行） ----
        return self._authenticator.admit(
            actor=actor,
            workspace=workspace,
            principal=principal,
            binding=binding,
        )
