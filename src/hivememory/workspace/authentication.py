"""Workspace 认证入口：统一 Actor Authentication 网关与第 2 阶段认证者。

认证与操作授权分属两个类（A1 访问边界返工第 4.2 节，身份与访问体系
Idea I-10）：

- :class:`ActorAuthenticationGateway` 是唯一对外的认证入口：依次调用
  ``PrincipalAuthenticator``（第 1 阶段）与 :class:`WorkspaceAuthenticator`
  （第 2 阶段），转交 context 的失效、清空与诊断查询，并负责关闭；
- :class:`WorkspaceAuthenticator` 负责第 2 阶段准入与 context 签发，
  持有"context → 授予记录"的弱引用字典（认证一侧），并向操作授权者
  （``workspace.authorization.WorkspaceOperationAuthorizer``）提供只读的
  兑现接口。

运行持有者（server、注册入口）只经认证网关接触认证一侧，不直接持有
``WorkspaceAuthenticator``。认证入口不要求提供待执行的 operation，也不向
调用方返回"只完成第一项认证"的可访问上下文；网关不执行业务、不转发路由。

阶段拒绝语义（均以稳定 reason 区分）：

- 接入未登记/禁用（不区分，避免泄漏配置）→ ``unknown_principal``；
- adapter 不匹配 → ``adapter_mismatch``；
- principal 身份解析规则不允许该用户 → ``actor_not_allowed_for_principal``；
- actor user ≠ 要进入的 workspace 的 owner（W0 兼容基线，第 2 阶段检查）
  → ``actor_not_owner``；
- Workspace 无有效 Actor 访问记录 → ``actor_not_admitted``。

以上均为 ``AdmissionDeniedError``（第一、二阶段失败）。缺少行为许可发生
在后续每次操作的操作授权（``workspace.authorization``），属
``OperationDeniedError``，不在本模块判断。
"""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary

from hivememory.core.access import (
    CallerPrincipal,
    PrincipalAuthenticator,
    RunBinding,
    WorkspaceAccessContext,
)
from hivememory.core.errors import AdmissionDeniedError, ScopeRequiredError
from hivememory.core.models import ActorIdentity, WorkspaceIdentity
from hivememory.workspace.registry import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)

__all__ = [
    "AccessGrantSummary",
    "ActorAuthenticationGateway",
    "RedeemedAccess",
    "WorkspaceAuthenticator",
]


@dataclass(frozen=True, slots=True)
class AccessGrantSummary:
    """授予记录的只读摘要，仅供日志与观测标签使用。

    本摘要不能作为身份兑现入口：授权仍必须把 context 交给操作授权者；
    摘要中的坐标不构成准入或授权结论。
    """

    actor_user_id: str
    agent_id: str
    workspace_id: str
    principal_id: str
    run_type: str
    run_id: str


@dataclass(frozen=True, slots=True)
class RedeemedAccess:
    """兑现结果：授予记录的只读投影，供操作授权者做第 3 阶段判断。

    ``access_record`` 是当前访问登记记录（含行为白名单）；授权判断所需的
    actor 与驻留 workspace 都来自认证一侧的授予记录，不是调用方传入的
    声明。
    """

    actor: ActorIdentity
    workspace: WorkspaceIdentity
    access_record: WorkspaceActorAccessRecord


@dataclass(frozen=True, slots=True)
class _GrantRecord:
    """认证一侧内部保存的一次授予记录（身份与访问体系 Idea 第 5 节）。

    行为白名单不进入授予记录：每次授权都经兑现接口按访问登记即时查询，
    不缓存授权结论；有效期也不进入——失效由绑定的运行结束决定。
    """

    actor: ActorIdentity
    workspace: WorkspaceIdentity
    principal: CallerPrincipal
    binding: RunBinding


class WorkspaceAuthenticator:
    """第 2 阶段 Workspace 认证与访问 context 签发（认证一侧）。

    准入 :meth:`admit` 只由认证网关调用；:meth:`redeem` / :meth:`peek` 是
    向操作授权者提供的只读兑现接口——授予记录不外泄可变引用，兑现不改变
    签发状态。失效（:meth:`invalidate`）、清空（:meth:`clear`）与诊断查询
    （:meth:`describe`）只经认证网关转交。该机制维护可信进程内调用纪律，
    不隔离任意恶意 Python 代码。
    """

    def __init__(self, access_registry: WorkspaceActorAccessRegistry) -> None:
        self._registry = access_registry
        self._closed = False
        self._issued: WeakKeyDictionary[WorkspaceAccessContext, _GrantRecord] = WeakKeyDictionary()

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """结束认证一侧的生命周期：拒绝新的准入，既有授予记录一并失效。"""
        self._closed = True
        self._issued.clear()

    def clear(self) -> None:
        """System 停止时清空全部授予记录（P-6 的收尾步骤）。

        清空后已签发 context 的兑现自然失败（``context_not_issued``）；
        操作授权者无状态，不需要随之关闭。
        """
        self._issued.clear()

    def invalidate(self, access: WorkspaceAccessContext) -> None:
        """使单个已签发 context 失效（进程关闭或请求结束时调用）。

        失效必须由 context 的绑定所有者触发：任务进程以任何结局关闭时
        失效绑定的 context，请求级 context 在请求结束时失效。对未签发或
        已失效的 context 重复失效是幂等空操作。
        """
        if type(access) is not WorkspaceAccessContext:
            raise TypeError("invalidate 只接受 WorkspaceAccessContext")
        self._issued.pop(access, None)

    def describe(self, access: WorkspaceAccessContext) -> AccessGrantSummary | None:
        """返回授予记录的只读摘要；未签发或已失效返回 ``None``。

        只用于日志与观测标签（如取消入口为观测事件取请求方的驻留
        workspace 标签），不能作为身份兑现入口。
        """
        if type(access) is not WorkspaceAccessContext:
            raise TypeError("describe 只接受 WorkspaceAccessContext")
        grant = self._issued.get(access)
        if grant is None:
            return None
        return AccessGrantSummary(
            actor_user_id=grant.actor.user_id,
            agent_id=grant.actor.agent_id,
            workspace_id=grant.workspace.workspace_id,
            principal_id=grant.principal.principal_id,
            run_type=grant.binding.run_type.value,
            run_id=grant.binding.run_id,
        )

    def admit(
        self,
        *,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
        principal: CallerPrincipal,
        binding: RunBinding,
    ) -> WorkspaceAccessContext:
        """第 2 阶段认证：确认 Workspace 准入并写入授予记录、签发 context。

        owner 约束（W0 基线）在本阶段检查：actor 用户必须等于要进入的
        workspace 的 owner。不检查 principal，也不重复校验入参类型与关闭
        状态——唯一调用方认证网关在公开入口已完成来源/adapter 验证、入参
        类型检查与关闭检查（认证流程内无 await，两项检查间不存在状态
        变化）。此方法不作为 adapter 或领域服务的另一认证入口。
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
        context = WorkspaceAccessContext()
        self._issued[context] = _GrantRecord(
            actor=actor,
            workspace=workspace,
            principal=principal,
            binding=binding,
        )
        return context

    def redeem(self, access: WorkspaceAccessContext | None) -> RedeemedAccess:
        """只读兑现：确认签发与准入仍然有效，返回授予记录的只读投影。

        供操作授权者在每次操作授权前调用：context 未签发或已失效、认证
        一侧已关闭、或准入记录已不再有效时分别以稳定 reason 拒绝。每次
        兑现都按完整坐标查询访问登记，不缓存授权结论。
        """
        if type(access) is not WorkspaceAccessContext:
            raise ScopeRequiredError("公共入口需要经统一认证网关签发的 WorkspaceAccessContext")
        if self._closed:
            raise ScopeRequiredError(
                "认证一侧已关闭，access context 失效",
                details={"reason": "authentication_gateway_closed"},
            )
        grant = self._issued.get(access)
        if grant is None:
            raise ScopeRequiredError(
                "access context 未由本运行实例签发",
                details={"reason": "context_not_issued"},
            )
        # 每次兑现按完整坐标查询准入：授予记录不绑定白名单快照。
        record = self._registry.record_for(grant.workspace, grant.actor)
        if record is None or not record.enabled:
            raise ScopeRequiredError(
                "该 Actor 已无有效的 Workspace 访问登记",
                details={"reason": "actor_not_admitted"},
            )
        return RedeemedAccess(actor=grant.actor, workspace=grant.workspace, access_record=record)

    def peek(self, access: WorkspaceAccessContext) -> RedeemedAccess | None:
        """只查签发记录取回兑现投影；未签发或已失效返回 ``None``。

        供操作授权者的进程控制授权在记录侧使用：进程记录中的 context 在
        收尾窗口内可能已失效，此时按"不可控"处理而不是接线缺陷。不做
        关闭检查——关闭只拒绝新的准入与兑现，已登记进程的收尾窗口按
        不可控呈现；准入复查失败同样折叠为 ``None``（当前注册表进程内
        不可变，此路径不可观测）。
        """
        if type(access) is not WorkspaceAccessContext:
            return None
        grant = self._issued.get(access)
        if grant is None:
            return None
        record = self._registry.record_for(grant.workspace, grant.actor)
        if record is None or not record.enabled:
            return None
        return RedeemedAccess(actor=grant.actor, workspace=grant.workspace, access_record=record)


class ActorAuthenticationGateway:
    """统一 Actor Authentication 网关（workspace 认证入口）。

    一次 :meth:`authenticate` 调用编排两项认证——Principal
    authentication（第 1 阶段，经端口委托 System）与 Workspace
    authentication（第 2 阶段，委托 :class:`WorkspaceAuthenticator`）；
    两项均通过才签发 ``WorkspaceAccessContext``，签发时写入运行绑定
    （I-3）：任务进程 context 绑定本进程的 ``process_id``，请求级
    context 绑定入口为本次请求生成的标识。网关是唯一对外的认证入口：
    context 的失效、清空与诊断查询都经这里转交认证一侧，运行持有者不
    直接接触 ``WorkspaceAuthenticator``。

    生命周期（v0.7.0 A1 访问边界返工第 4.8 节）：context 不设固定有效期，
    只随三个时点失效——绑定的任务进程关闭、请求级 context 随请求结束、
    System 停止。:meth:`close` 关闭网关并拒绝新的认证；任务进程收尾后经
    :meth:`clear_contexts` 清空全部授予记录。本类不提供配置热更新——
    首版本地配置在运行实例内不可变，修改经重启生效。
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
        """网关或认证一侧是否已关闭；关闭后不再签发 context。"""
        return self._closed or self._authenticator.is_closed

    def close(self) -> None:
        """关闭网关：拒绝新的认证请求，不影响已签发 context 的剩余生命周期。"""
        self._closed = True

    def invalidate_context(self, access: WorkspaceAccessContext) -> None:
        """使单个已签发 context 失效；转交认证一侧（进程关闭或请求结束）。"""
        self._authenticator.invalidate(access)

    def clear_contexts(self) -> None:
        """System 停止时清空全部授予记录；此后已签发 context 的兑现自然失败。"""
        self._authenticator.clear()

    def describe_context(self, access: WorkspaceAccessContext) -> AccessGrantSummary | None:
        """返回授予记录的只读摘要（日志与观测标签专用）；转交认证一侧。"""
        return self._authenticator.describe(access)

    async def authenticate(
        self,
        *,
        adapter: str,
        principal: CallerPrincipal,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
        binding: RunBinding,
    ) -> WorkspaceAccessContext:
        """完成两项认证并签发访问上下文；任一失败即拒绝，无部分成功。

        ``adapter`` 是调用来源实际使用的接入方式标识；``principal`` 由
        受信 adapter 依据其接入证据构造；``binding`` 是本次运行绑定——
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
        if not isinstance(actor, ActorIdentity):
            raise TypeError("actor 必须是 ActorIdentity")
        if not isinstance(workspace, WorkspaceIdentity):
            raise TypeError("workspace 必须是 WorkspaceIdentity")
        if not isinstance(binding, RunBinding):
            raise TypeError("binding 必须是 RunBinding")

        # ---- 1. Principal authentication：委托 System 实现的接入登记校验 ----
        self._principals.authenticate_principal(
            adapter=adapter,
            principal=principal,
            actor=actor,
        )

        # ---- 2. Workspace authentication / admission（签发即绑定运行） ----
        return self._authenticator.admit(
            actor=actor,
            workspace=workspace,
            principal=principal,
            binding=binding,
        )
