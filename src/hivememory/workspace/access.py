"""Workspace 准入、访问 context 生命周期与逐次操作授权。

workspace 认证入口（``workspace.authentication``）完成 Principal
authentication 后调用本模块的内部准入方法。WorkspaceAccessGuard 在签发
context 时把授予记录（已验证 actor、驻留 workspace、来源 principal、运行
绑定）保存在内部的弱引用字典中；授权点把 context 交回 guard 兑现身份，
context 对外不透明（身份与访问体系 Idea I-1）。

guard 的公开检查接口（v0.7.0 A1 访问边界返工第 4.2 节）：

- :meth:`WorkspaceAccessGuard.authorize_operation`：第 3 阶段操作授权——
  显式接收目标 workspace，返回组装好的 ``IdentityScope``；只由授权点
  （注册入口、能力层、任务进程的阶段检查）调用；
- :meth:`WorkspaceAccessGuard.authorize_process_control`：进程控制授权——
  比对请求方 context 与进程记录中的 context（P-7），供取消与状态查询
  入口判断；
- :meth:`WorkspaceAccessGuard.cpu_execution_identity`：CPU 执行身份的
  过渡组装方法（I-9）——只做目标与 owner 检查、不检查 operation，只由
  任务进程在 CPU 分配时调用，Alice 的能力层调用迁移完成后删除。

资源自身的可见性仍由资源 owner 判断；本模块不依赖 System 或 Patchouli，
也不向 Gateway、Patchouli、引擎或存储注入身份。

context 生命周期（v0.7.0 A1 访问边界返工第 4.8 节）：不设固定有效期；
context 只在三个时点失效——绑定的任务进程以任何结局关闭、请求级 context
随请求结束、System 停止时 guard 关闭使全部 context 失效。
"""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakKeyDictionary

from hivememory.core.access import (
    CallerPrincipal,
    RunBinding,
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.core.errors import (
    AdmissionDeniedError,
    OperationDeniedError,
    ScopeRequiredError,
)
from hivememory.core.models import ActorIdentity, IdentityScope, WorkspaceIdentity
from hivememory.workspace.registry import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)


@dataclass(frozen=True, slots=True)
class AccessGrantSummary:
    """授予记录的只读摘要，仅供日志与诊断使用。

    本摘要不能作为身份兑现入口：授权仍必须把 context 交回 guard 的检查
    方法；摘要中的坐标不构成准入或授权结论。
    """

    actor_user_id: str
    agent_id: str
    workspace_id: str
    principal_id: str
    run_type: str
    run_id: str


@dataclass(frozen=True, slots=True)
class _GrantRecord:
    """guard 内部保存的一次授予记录（身份与访问体系 Idea 第 5 节）。

    行为白名单不进入授予记录：每次授权都按访问登记即时查询，不缓存授权
    结论；有效期也不进入——失效由绑定的运行结束决定。
    """

    actor: ActorIdentity
    workspace: WorkspaceIdentity
    principal: CallerPrincipal
    binding: RunBinding


class WorkspaceAccessGuard:
    """签发不透明访问 context，并供授权点逐次兑现身份与检查操作许可。

    一个运行实例共享同一个 guard。内部准入只供统一认证网关调用；业务
    授权点调用 :meth:`authorize_operation` 后使用返回的 ``IdentityScope``。
    授予记录是私有的进程内状态，以弱引用字典保存——context 在其绑定
    所有者释放后随之失效，不会持续积累。该机制维护可信进程内调用纪律，
    不隔离任意恶意 Python 代码。
    """

    def __init__(
        self,
        access_registry: WorkspaceActorAccessRegistry,
    ) -> None:
        self._registry = access_registry
        self._closed = False
        self._issued: WeakKeyDictionary[WorkspaceAccessContext, _GrantRecord] = WeakKeyDictionary()

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """结束本运行实例的准入生命周期，既有 context 一并失效。"""
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
        self._issued.pop(access, None)

    def describe(self, access: WorkspaceAccessContext) -> AccessGrantSummary | None:
        """返回授予记录的只读摘要；未签发或已失效返回 ``None``。

        只用于日志与诊断（如取消入口为观测事件取请求方的驻留 workspace
        标签），不能作为身份兑现入口。
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

    def _admit(
        self,
        *,
        actor: ActorIdentity,
        workspace: WorkspaceIdentity,
        principal: CallerPrincipal,
        binding: RunBinding,
    ) -> WorkspaceAccessContext:
        """网关内部第二项认证：确认 Workspace 准入并写入授予记录。

        owner 约束（W0 基线）在本阶段检查：actor 用户必须等于要进入的
        workspace 的 owner。不检查 principal，也不重复校验入参类型与关闭
        状态——唯一调用方 System 网关在公开入口已完成来源/adapter 验证、
        入参类型检查与关闭检查（认证流程内无 await，两项检查间不存在
        状态变化）。此方法不作为 adapter 或领域服务的另一认证入口。
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

    def authorize_operation(
        self,
        access: WorkspaceAccessContext,
        operation: WorkspaceOperation,
        target_workspace: WorkspaceIdentity,
    ) -> IdentityScope:
        """第 3 阶段操作授权：确认 context 有效，返回目标 workspace 的可信 scope。

        授权点显式传入这次操作的目标 workspace（I-4）：目标不等于驻留
        workspace、actor 用户不是目标 workspace 的 owner、或访问登记的
        白名单不含该 operation 时分别以稳定 reason 拒绝。同一有效凭据可
        反复调用本检查先后执行不同的获准操作；切换 Actor/Workspace、凭据
        失效或网关关闭后必须重新经统一网关认证。
        """
        if not isinstance(operation, WorkspaceOperation):
            raise TypeError("operation 必须是 WorkspaceOperation")
        if not isinstance(target_workspace, WorkspaceIdentity):
            raise TypeError("target_workspace 必须是 WorkspaceIdentity")
        grant, record = self._verified_grant(access)
        # 目标 workspace 检查（I-4）：当前只接受等于驻留 workspace 的目标，
        # 其余一律拒绝；受限穿透访问落地后在此放宽。
        if target_workspace != grant.workspace:
            raise OperationDeniedError(
                details={
                    "operation": operation.value,
                    "reason": "target_workspace_not_resident",
                    "resident_workspace_id": grant.workspace.workspace_id,
                    "target_workspace_id": target_workspace.workspace_id,
                }
            )
        # owner 约束（W0 基线）移到第 3 阶段：身份类型不再承担授权规则。
        if grant.actor.user_id != target_workspace.owner_user_id:
            raise OperationDeniedError(
                details={
                    "operation": operation.value,
                    "reason": "target_owner_mismatch",
                    "actor_user_id": grant.actor.user_id,
                    "owner_user_id": target_workspace.owner_user_id,
                }
            )
        # 行为白名单：缺少行为许可是授权失败，不是身份认证失败。
        if operation not in record.allowed_operations:
            raise OperationDeniedError(
                details={
                    "operation": operation.value,
                    "reason": "operation_not_allowed",
                }
            )
        return IdentityScope(
            actor_identity=grant.actor,
            workspace_identity=target_workspace,
        )

    def authorize_process_control(
        self,
        requestor: WorkspaceAccessContext,
        record_access: WorkspaceAccessContext,
    ) -> bool:
        """进程控制授权：比对请求方与进程记录的驻留坐标（P-7）。

        两份 context 驻留在同一 owner 与 workspace 时返回 ``True``（沿用
        既有进程表比对规则）；请求方 context 无效以 ``ScopeRequiredError``
        拒绝（生产入口经网关签发后不应出现）。进程记录侧只查签发记录、
        不要求准入复查：进程收尾与注销之间的窗口内到达的控制请求按
        ``False`` 呈现为 ``not_found``，不泄露目标进程是否存在。
        """
        requestor_grant = self._verified_grant(requestor)[0]
        record_grant = self._peek_grant(record_access)
        if record_grant is None:
            return False
        return requestor_grant.workspace == record_grant.workspace

    def cpu_execution_identity(
        self,
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
    ) -> IdentityScope:
        """CPU 执行身份的过渡组装（I-9）：只做目标与 owner 检查，不检查 operation。

        只供任务进程在 CPU 分配时调用——CPU 执行本身没有对应的 operation，
        Alice 直接调用 Patchouli 的缺口由 Alice 的能力层调用迁移解决；
        该迁移完成后本方法删除。
        """
        if not isinstance(target_workspace, WorkspaceIdentity):
            raise TypeError("target_workspace 必须是 WorkspaceIdentity")
        grant = self._verified_grant(access)[0]
        if target_workspace != grant.workspace:
            raise OperationDeniedError(
                details={
                    "reason": "target_workspace_not_resident",
                    "resident_workspace_id": grant.workspace.workspace_id,
                    "target_workspace_id": target_workspace.workspace_id,
                }
            )
        if grant.actor.user_id != target_workspace.owner_user_id:
            raise OperationDeniedError(
                details={
                    "reason": "target_owner_mismatch",
                    "actor_user_id": grant.actor.user_id,
                    "owner_user_id": target_workspace.owner_user_id,
                }
            )
        return IdentityScope(
            actor_identity=grant.actor,
            workspace_identity=target_workspace,
        )

    def _peek_grant(
        self,
        access: WorkspaceAccessContext,
    ) -> _GrantRecord | None:
        """只查签发记录取回授予记录；未签发或已失效返回 ``None``。

        供进程控制授权的记录侧使用：进程记录中的 context 在收尾窗口内
        可能已失效，此时按"不可控"处理而不是接线缺陷。
        """
        if type(access) is not WorkspaceAccessContext:
            return None
        return self._issued.get(access)

    def _verified_grant(
        self,
        access: WorkspaceAccessContext | None,
    ) -> tuple[_GrantRecord, WorkspaceActorAccessRecord]:
        """校验签发、关闭状态与准入记录，返回授予记录与当前访问记录。"""
        if type(access) is not WorkspaceAccessContext:
            raise ScopeRequiredError("公共入口需要经统一认证网关签发的 WorkspaceAccessContext")
        if self._closed:
            raise ScopeRequiredError(
                "认证网关已关闭，access context 失效",
                details={"reason": "authentication_gateway_closed"},
            )
        grant = self._issued.get(access) if access is not None else None
        if grant is None:
            raise ScopeRequiredError(
                "access context 未由本运行实例签发",
                details={"reason": "context_not_issued"},
            )
        # 每次动作按完整坐标查询权限：授予记录不保存行为白名单快照，
        # 配置对象地址或白名单不绑定进凭据。
        record = self._registry.record_for(grant.workspace, grant.actor)
        if record is None or not record.enabled:
            raise ScopeRequiredError(
                "该 Actor 已无有效的 Workspace 访问登记",
                details={"reason": "actor_not_admitted"},
            )
        return grant, record


__all__ = [
    "AccessGrantSummary",
    "WorkspaceAccessGuard",
]
