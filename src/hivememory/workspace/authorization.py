"""Workspace 操作授权：第 3 阶段授权、进程控制授权与 CPU 执行身份。

:class:`WorkspaceOperationAuthorizer` 是授权点（能力层、任务进程的阶段
检查、注册入口的取消与状态查询）使用的操作授权者（A1 访问边界返工第
4.2 节，身份与访问体系 Idea I-10 及其 2026-10-04 补充）：它无状态，只
依赖 Workspace Actor 访问登记——每次授权都读取 context 密封的授予内容，
再对目标 workspace、owner 与访问登记白名单做检查，组装并返回
``IdentityScope``。

认证与授权互不依赖：本模块不导入认证一侧（``workspace.authentication``），
只经 context 这个类型与它发生联系；也不导入任务进程与能力层。授权点
显式接收目标 workspace（I-4）：它是这次操作的参数，不是身份；当前只
接受等于驻留 workspace 的目标，受限穿透访问落地后在此放宽。
"""

from __future__ import annotations

from hivememory.core.access import AccessGrant, WorkspaceAccessContext, WorkspaceOperation
from hivememory.core.errors import OperationDeniedError, ScopeRequiredError
from hivememory.core.models import IdentityScope, WorkspaceIdentity
from hivememory.workspace.registry import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)

__all__ = [
    "WorkspaceOperationAuthorizer",
]


class WorkspaceOperationAuthorizer:
    """第 3 阶段操作授权：目标、owner 与白名单检查，组装可信 scope。

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
        或访问登记的白名单不含该 operation 时分别以稳定 reason 拒绝
        （A1 访问边界返工第 4.2 节）。
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

    def authorize_process_control(
        self,
        requestor: WorkspaceAccessContext,
        record_access: WorkspaceAccessContext,
    ) -> bool:
        """进程控制授权：比对请求方与进程记录的驻留坐标（P-7）。

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

    def cpu_execution_identity(
        self,
        access: WorkspaceAccessContext,
        target_workspace: WorkspaceIdentity,
    ) -> IdentityScope:
        """CPU 执行身份的过渡组装（I-9）：只做目标与 owner 检查，不检查 operation。

        只供任务进程在 CPU 分配时调用——CPU 执行本身没有对应的 operation，
        Alice 直接调用 Patchouli 的缺口由 Alice 的能力层调用迁移解决；
        该迁移完成后本方法删除。调用点由分层测试约束在
        ``workspace/process`` 之内。
        """
        grant, _ = self._redeem(access)
        self._check_target(grant, target_workspace, operation=None)
        return IdentityScope(actor_identity=grant.actor, workspace_identity=target_workspace)

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
        operation: WorkspaceOperation | None,
    ) -> None:
        """目标 workspace 与 owner 检查：操作授权与 CPU 执行身份共用的唯一实现。

        目标检查（I-4）当前只接受等于驻留 workspace 的目标，受限穿透访问
        落地后在此放宽；owner 约束（W0 基线）在第 3 阶段检查，身份类型
        不承担授权规则（I-5）。
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
