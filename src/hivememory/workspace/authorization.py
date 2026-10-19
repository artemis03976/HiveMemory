"""Workspace 授权：actor 操作、root 管理、系统阶段与进程控制。

:class:`WorkspaceOperationAuthorizer` 是授权点（能力层、任务进程的阶段
检查、注册入口的取消）使用的操作授权者：它无状态，只
依赖 Workspace Actor 访问登记——每次授权都读取 context 密封的授予内容，
再校验目标 workspace 与 owner，按调用面检查 actor 操作白名单或保留的
system 身份；任务进程阶段不检查操作白名单。授权成功后组装并返回
``IdentityScope``。

认证与授权互不依赖：本模块不导入认证一侧（``workspace.authentication``），
只经 context 这个类型与它发生联系；也不导入任务进程与能力层。授权点
显式接收目标 workspace：它是这次操作的参数，不是身份；当前只
接受等于驻留 workspace 的目标，受限穿透访问落地后在此放宽。
"""

from __future__ import annotations

from hivememory.core.access import (
    AccessGrant,
    AccessRunType,
    WorkspaceAccessContext,
    WorkspaceOperation,
)
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import (
    OperationDeniedError,
    ScopeRequiredError,
    ThreadContextRevokedError,
)
from hivememory.core.models import IdentityScope, WorkspaceIdentity
from hivememory.workspace.registry import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)

__all__ = [
    "WorkspaceOperationAuthorizer",
]


class WorkspaceOperationAuthorizer:
    """逐次授权：共享准入与目标校验，按 API 调用面检查权限并组装可信 scope。

    同一有效凭据可反复授权执行不同的获准操作；切换 Actor/Workspace 或
    凭据被撤销后必须重新经认证网关认证。授权规则来自同一份 Workspace
    访问登记的白名单，每次授权即时查询、不缓存结论。
    """

    def __init__(self, access_registry: WorkspaceActorAccessRegistry) -> None:
        self._registry = access_registry

    def authorize_operation(
        self,
        access: WorkspaceAccessContext,
        operation: WorkspaceOperation,
        target_workspace: WorkspaceIdentity,
    ) -> IdentityScope:
        """第 3 阶段操作授权：确认 context 有效，返回目标 workspace 的可信 scope。

        目标不等于驻留 workspace、actor 用户不是目标 workspace 的 owner、
        或访问登记的白名单不含该 operation 时分别以稳定 reason 拒绝。
        """
        if not isinstance(operation, WorkspaceOperation):
            raise TypeError("operation 必须是 WorkspaceOperation")
        grant, record = self._redeem(access)
        self._check_target(grant, target_workspace, operation=operation)
        # 行为白名单：缺少行为许可是授权失败，不是身份认证失败。
        if operation not in record.allowed_operations:
            raise OperationDeniedError(
                details={
                    "operation": operation.value,
                    "reason": "operation_not_allowed",
                }
            )
        return IdentityScope(actor_identity=grant.actor, workspace_identity=target_workspace)

    def validate_thread_context(self, access: WorkspaceAccessContext) -> None:
        """入口分派前检查线程 context 的类型与寿命，不代替逐次操作授权。"""
        self._thread_grant(access)

    def get_thread_process_id(self, access: WorkspaceAccessContext) -> str:
        """只为能力层读取有效线程的进程关联，调用方不能自行指定关联。"""
        return self._thread_grant(access).binding.run_id

    def authorize_thread_operation(
        self, access: WorkspaceAccessContext, operation: WorkspaceOperation
    ) -> IdentityScope:
        """线程只在驻留 Workspace 发起操作，目标从密封授予内容取得。"""
        grant = self._thread_grant(access)
        return self.authorize_operation(access, operation, grant.workspace)

    @staticmethod
    def _thread_grant(access: WorkspaceAccessContext) -> AccessGrant:
        """线程已撤销属于生命周期拒绝，类型或运行种类错误属于接线缺陷。"""
        if type(access) is not WorkspaceAccessContext:
            raise ScopeRequiredError("actor 操作需要线程 context")
        grant = access._unseal()
        if grant is None:
            raise ThreadContextRevokedError("Thread context is revoked")
        if grant.binding.run_type is not AccessRunType.EXECUTION_THREAD:
            raise ScopeRequiredError("actor 操作需要线程 context")
        return grant

    def authorize_process_control(
        self,
        requestor: WorkspaceAccessContext,
        record_access: WorkspaceAccessContext,
    ) -> bool:
        """进程控制授权：比对请求方与进程记录的驻留坐标。

        两份 context 驻留在同一 owner 与 workspace 时返回 ``True``（沿用
        既有进程表比对规则）；请求方 context 无效以 ``ScopeRequiredError``
        拒绝（生产入口经网关签发后不应出现）。进程记录侧的 context 已撤销
        或准入已失效时返回 ``False``：进程收尾与注销之间的窗口内到达的
        控制请求呈现为 ``not_found``，不泄露目标进程是否存在。
        """
        requestor_grant, _ = self._redeem(requestor)
        record_grant = (
            record_access._unseal() if type(record_access) is WorkspaceAccessContext else None
        )
        if record_grant is None or self._admission_record(record_grant) is None:
            return False
        # WorkspaceIdentity 相等性覆盖 owner_user_id 与 workspace 坐标。
        return requestor_grant.workspace == record_grant.workspace

    def authorize_stage(
        self,
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
    ) -> IdentityScope:
        """任务进程阶段授权：校验准入与目标，不检查 actor 操作白名单。

        阶段调用由进程负责，scope 保留进程 actor 的身份以执行资源可见性
        检查；它只交给紧随的阶段调用，不进入 CPU 输入清单或线程 context。
        生产调用面限定在 ``workspace.process``。
        """
        grant, _ = self._redeem(access)
        self._check_target(grant, target_workspace)
        return IdentityScope(actor_identity=grant.actor, workspace_identity=target_workspace)

    def authorize_root(
        self,
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
    ) -> IdentityScope:
        """root API 授权：有效准入与目标校验后，只允许保留的 system actor。

        root 是授权规则，不是身份类型或可登记的 actor 操作；空白名单不影响
        system 的管理能力，其他 actor 即使持有全部操作也不能进入 root API。
        """
        grant, _ = self._redeem(access)
        self._check_target(grant, target_workspace)
        if grant.actor.agent_id != SYSTEM_AGENT_ID:
            raise OperationDeniedError(details={"reason": "root_required"})
        return IdentityScope(actor_identity=grant.actor, workspace_identity=target_workspace)

    def get_stage_operations(
        self,
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
    ) -> frozenset[WorkspaceOperation]:
        """任务进程在 CPU 分配或子线程派生时读取对应 actor 的操作集合。

        与阶段授权相同地兑现凭据、确认准入并校验目标；空白名单照常返回。
        返回值只用于 CPU 提示词与执行策略，不授予权限，也不携带访问身份；
        actor API 仍逐次授权。生产调用面限定在 ``workspace.process``。
        """
        grant, record = self._redeem(access)
        self._check_target(grant, target_workspace)
        return record.allowed_operations

    # ---- 内部辅助 ----

    def _redeem(
        self, access: WorkspaceAccessContext
    ) -> tuple[AccessGrant, WorkspaceActorAccessRecord]:
        """读取 context 密封的授予内容，并按完整坐标确认准入仍然有效。

        不是签发得到的 context 或已撤销的 context 以 ``context_not_issued``
        拒绝；准入记录已不再有效时以 ``actor_not_admitted`` 拒绝。授予内容
        不绑定白名单快照，每次都即时查询访问登记。
        """
        if type(access) is not WorkspaceAccessContext:
            raise ScopeRequiredError("公共入口需要经统一认证网关签发的 WorkspaceAccessContext")
        grant = access._unseal()
        if grant is None:
            raise ScopeRequiredError(
                "access context 未签发或已撤销",
                details={"reason": "context_not_issued"},
            )
        record = self._admission_record(grant)
        if record is None:
            raise ScopeRequiredError(
                "该 Actor 已无有效的 Workspace 访问登记",
                details={"reason": "actor_not_admitted"},
            )
        return grant, record

    def _admission_record(self, grant: AccessGrant) -> WorkspaceActorAccessRecord | None:
        record = self._registry.record_for(grant.workspace, grant.actor)
        if record is None or not record.enabled:
            return None
        return record

    @staticmethod
    def _check_target(
        grant: AccessGrant,
        target_workspace: WorkspaceIdentity,
        *,
        operation: WorkspaceOperation | None = None,
    ) -> None:
        """操作授权的目标 workspace 与 owner 检查。

        目标检查当前只接受等于驻留 workspace 的目标，受限穿透访问
        落地后在此放宽；owner 约束（成员模型落地前的准入约束）在第 3 阶段检查，身份类型
        不承担授权规则。
        """
        if not isinstance(target_workspace, WorkspaceIdentity):
            raise TypeError("target_workspace 必须是 WorkspaceIdentity")
        operation_detail = {"operation": operation.value} if operation is not None else {}
        if target_workspace != grant.workspace:
            raise OperationDeniedError(
                details={
                    **operation_detail,
                    "reason": "target_workspace_not_resident",
                    "resident_workspace_id": grant.workspace.workspace_id,
                    "target_workspace_id": target_workspace.workspace_id,
                }
            )
        if grant.actor.user_id != target_workspace.owner_user_id:
            raise OperationDeniedError(
                details={
                    **operation_detail,
                    "reason": "target_owner_mismatch",
                    "actor_user_id": grant.actor.user_id,
                    "owner_user_id": target_workspace.owner_user_id,
                }
            )
