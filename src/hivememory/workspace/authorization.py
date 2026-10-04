"""Workspace 操作授权：第 3 阶段授权、进程控制授权与 CPU 执行身份。

:class:`WorkspaceOperationAuthorizer` 是授权点（能力层、任务进程的阶段
检查、注册入口的取消与状态查询）使用的操作授权者（A1 访问边界返工第
4.2 节，身份与访问体系 Idea I-10）：它自身无 per-context 状态，只持有
认证一侧（``workspace.authentication.WorkspaceAuthenticator``）的只读
兑现接口——每次授权都把 context 兑现为授予记录的只读投影，再对目标
workspace、owner 与访问登记白名单做检查，组装并返回 ``IdentityScope``。

授权点显式接收目标 workspace（I-4）：它是这次操作的参数，不是身份；
当前只接受等于驻留 workspace 的目标，受限穿透访问落地后在此放宽。授权
点以下只流动本类组装的 ``IdentityScope``；本模块不导入任务进程与能力
层，也不导入认证网关。
"""

from __future__ import annotations

from hivememory.core.access import WorkspaceAccessContext, WorkspaceOperation
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import IdentityScope, WorkspaceIdentity
from hivememory.workspace.authentication import WorkspaceAuthenticator

__all__ = [
    "WorkspaceOperationAuthorizer",
]


class WorkspaceOperationAuthorizer:
    """第 3 阶段操作授权：目标、owner 与白名单检查，组装可信 scope。

    ``authenticator`` 是认证一侧：本类只经其只读兑现接口确认 context
    有效，不接触签发与失效。同一有效凭据可反复授权执行不同的获准操作；
    切换 Actor/Workspace、凭据失效或认证一侧清空后必须重新经认证网关
    认证。授权规则来自同一份 Workspace 访问登记的白名单，每次授权即时
    查询、不缓存结论。
    """

    def __init__(self, authenticator: WorkspaceAuthenticator) -> None:
        self._authenticator = authenticator

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
        if not isinstance(target_workspace, WorkspaceIdentity):
            raise TypeError("target_workspace 必须是 WorkspaceIdentity")
        redeemed = self._authenticator.redeem(access)
        # 目标 workspace 检查（I-4）：当前只接受等于驻留 workspace 的目标，
        # 其余一律拒绝；受限穿透访问落地后在此放宽。
        if target_workspace != redeemed.workspace:
            raise OperationDeniedError(
                details={
                    "operation": operation.value,
                    "reason": "target_workspace_not_resident",
                    "resident_workspace_id": redeemed.workspace.workspace_id,
                    "target_workspace_id": target_workspace.workspace_id,
                }
            )
        # owner 约束（W0 基线）在第 3 阶段检查：身份类型不承担授权规则。
        if redeemed.actor.user_id != target_workspace.owner_user_id:
            raise OperationDeniedError(
                details={
                    "operation": operation.value,
                    "reason": "target_owner_mismatch",
                    "actor_user_id": redeemed.actor.user_id,
                    "owner_user_id": target_workspace.owner_user_id,
                }
            )
        # 行为白名单：缺少行为许可是授权失败，不是身份认证失败。
        if operation not in redeemed.access_record.allowed_operations:
            raise OperationDeniedError(
                details={
                    "operation": operation.value,
                    "reason": "operation_not_allowed",
                }
            )
        return IdentityScope(
            actor_identity=redeemed.actor,
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
        requestor_redeemed = self._authenticator.redeem(requestor)
        record_redeemed = self._authenticator.peek(record_access)
        if record_redeemed is None:
            return False
        # WorkspaceIdentity 相等性覆盖 owner_user_id 与 workspace 坐标。
        return requestor_redeemed.workspace == record_redeemed.workspace

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
        if not isinstance(target_workspace, WorkspaceIdentity):
            raise TypeError("target_workspace 必须是 WorkspaceIdentity")
        redeemed = self._authenticator.redeem(access)
        if target_workspace != redeemed.workspace:
            raise OperationDeniedError(
                details={
                    "reason": "target_workspace_not_resident",
                    "resident_workspace_id": redeemed.workspace.workspace_id,
                    "target_workspace_id": target_workspace.workspace_id,
                }
            )
        if redeemed.actor.user_id != target_workspace.owner_user_id:
            raise OperationDeniedError(
                details={
                    "reason": "target_owner_mismatch",
                    "actor_user_id": redeemed.actor.user_id,
                    "owner_user_id": target_workspace.owner_user_id,
                }
            )
        return IdentityScope(
            actor_identity=redeemed.actor,
            workspace_identity=target_workspace,
        )
