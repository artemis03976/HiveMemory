"""Workspace 认证入口：统一 Actor Authentication 网关与第 2 阶段认证者。

认证与操作授权分属两个类：

- :class:`ActorAuthenticationGateway` 是唯一对外的认证入口：依次调用
  ``PrincipalAuthenticator``（第 1 阶段）与 :class:`WorkspaceAuthenticator`
  （第 2 阶段），转交子工作 actor 的准入，并负责关闭（拒绝新认证）；
- :class:`WorkspaceAuthenticator` 负责第 2 阶段准入，准入通过即构造工作 actor。

认证的结果是工作 actor（``core.access.WorkingActor``），操作授权者
（``workspace.authorization.WorkspaceOperationAuthorizer``）只读取它，不依赖
认证一侧；认证与授权只经这个类型发生联系。

运行持有者（server、注册入口、Import Bus 服务）只经认证网关接触认证一侧，
不直接持有 ``WorkspaceAuthenticator``。认证入口不要求提供待执行的 operation；
网关不执行业务、不转发路由。

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

from hivememory.core.access import CallerPrincipal, PrincipalAuthenticator, WorkingActor
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import AdmissionDeniedError
from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.workspace.registry import WorkspaceActorAccessRegistry

__all__ = [
    "ActorAuthenticationGateway",
    "WorkspaceAuthenticator",
]


class WorkspaceAuthenticator:
    """第 2 阶段 Workspace 认证：准入并构造工作 actor（认证一侧）。

    准入 :meth:`admit` 由认证网关与本类的子工作 actor 准入调用。本类无状态：
    工作 actor 是普通值，不登记、不撤销；准入与白名单在每次授权时由操作
    授权者即时查询。
    """

    def __init__(self, access_registry: WorkspaceActorAccessRegistry) -> None:
        self._registry = access_registry

    def admit(
        self,
        *,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
        principal: CallerPrincipal,
    ) -> WorkingActor:
        """第 2 阶段认证：确认 Workspace 准入，返回驻留该 Workspace 的工作 actor。

        owner 约束在本阶段检查：actor 用户必须等于要进入的 workspace 的
        owner。不检查 principal，也不重复校验入参类型与关闭状态——认证网关
        已验证来源与声明；子工作 actor 的准入则复用主工作 actor 的来源与
        驻留 Workspace，只替换同一用户的 agent。此方法不作为 adapter 或领域
        服务的另一认证入口。
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
        # 行为白名单不进入工作 actor：每次授权都按访问登记即时查询，不缓存
        # 授权结论。
        return WorkingActor(principal=principal, actor=actor, workspace=workspace)

    def admit_sub_actor(self, main: WorkingActor, agent_id: str) -> WorkingActor:
        """被调用方沿用主工作 actor 的来源与驻留 Workspace，经第 2 阶段准入。

        不重新认证 principal；保留的 ``system`` 不代表任何被调用的 Agent，
        在准入前拒绝。
        """
        if agent_id == SYSTEM_AGENT_ID:
            raise AdmissionDeniedError(
                "子线程不能使用保留 system actor",
                details={"reason": "system_actor_not_allowed"},
            )
        return self.admit(
            actor=ActorIdentity(user_id=main.actor.user_id, agent_id=agent_id),
            workspace=main.workspace,
            principal=main.principal,
        )


class ActorAuthenticationGateway:
    """统一 Actor Authentication 网关（workspace 认证入口）。

    一次 :meth:`authenticate` 调用编排两项认证——Principal
    authentication（第 1 阶段，经端口委托 System）与 Workspace
    authentication（第 2 阶段，委托 :class:`WorkspaceAuthenticator`）；
    两项均通过才返回工作 actor。网关也实现进程侧的子工作 actor 准入端口
    （``core.access.SubActorAdmission``），运行持有者不直接接触
    ``WorkspaceAuthenticator``。

    :meth:`close` 关闭网关并拒绝新的认证；已有进程仍可为子线程做准入。本类
    不提供配置热更新——本地配置在运行实例内不可变，修改经重启生效。
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
        """网关是否已关闭；关闭后拒绝新认证，已有进程仍可为子线程做准入。"""
        return self._closed

    def close(self) -> None:
        """关闭网关：拒绝新的认证请求，不影响已有进程派生子线程。"""
        self._closed = True

    def admit_sub_actor(self, main: WorkingActor, agent_id: str) -> WorkingActor:
        """实现进程侧的子工作 actor 准入端口；关闭网关不禁止已有进程派生。"""
        return self._authenticator.admit_sub_actor(main, agent_id)

    async def authenticate(
        self,
        *,
        adapter: str,
        principal: CallerPrincipal,
        credential: str | None,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
    ) -> WorkingActor:
        """完成两项认证并返回工作 actor；任一失败即拒绝，无部分成功。

        ``adapter`` 是调用来源实际使用的接入方式标识，由入口代码固定传入；
        ``principal`` 是入口传入的调用来源，``credential`` 是它的接入证据
        ——进程内入口传 ``None``，外部入口原样传入请求中的凭据，两者都交给
        第 1 阶段核对，入口不自行判定身份。返回的工作 actor 与单次 operation
        解耦，此后每次操作都由操作授权者逐次授权。
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

        # ---- 1. Principal authentication：委托 System 实现的接入登记校验 ----
        self._principals.authenticate_principal(
            adapter=adapter,
            principal=principal,
            credential=credential,
            actor=actor,
        )

        # ---- 2. Workspace authentication / admission ----
        return self._authenticator.admit(
            actor=actor,
            workspace=workspace,
            principal=principal,
        )
