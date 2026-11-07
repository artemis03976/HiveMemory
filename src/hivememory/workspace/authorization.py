"""Workspace 授权：actor 执行期间的操作授权与系统级操作的 root 授权。

:class:`WorkspaceOperationAuthorizer` 是授权点使用的操作授权者，只有两个
授权方法，按“谁在执行这项操作”区分：

- :meth:`~WorkspaceOperationAuthorizer.authorize_agent`：Agent 在 actor 执行
  期间发起的操作（actor API，含任务进程以主线程提交的预检索），按访问登记的
  白名单授权；
- :meth:`~WorkspaceOperationAuthorizer.authorize_root`：系统以保留的
  ``system`` 执行的操作（root API、任务进程的系统阶段与取消、Import Bus
  的确定话题与交互提交），不检查白名单。

两者共用三项前提，按顺序检查：准入记录有效、目标等于驻留 Workspace、actor
的用户是目标 Workspace 的 owner。授权成功后组装并返回 ``IdentityScope``。

授权者无状态，只依赖 Workspace Actor 访问登记，每次授权即时查询、不缓存
结论。它不导入认证一侧（``workspace.authentication``），只经工作 actor 这个
类型与它发生联系；也不导入任务进程与能力层。目标 Workspace 作为参数单独
传入：它是这次操作的参数，不是身份；目标只能等于驻留 Workspace，放宽这条
规则时授权点的签名不变。
"""

from __future__ import annotations

from hivememory.core.access import WorkingActor, WorkspaceOperation
from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import OperationDeniedError, ScopeRequiredError
from hivememory.core.models import IdentityScope, WorkspaceIdentity
from hivememory.workspace.registry import (
    WorkspaceActorAccessRecord,
    WorkspaceActorAccessRegistry,
)

__all__ = ["WorkspaceOperationAuthorizer"]


class WorkspaceOperationAuthorizer:
    """逐次授权：共享准入与目标校验，按执行者检查白名单或 root，并组装可信 scope。

    同一工作 actor 可反复授权执行不同的获准操作；授权规则来自同一份
    Workspace 访问登记，每次授权即时查询、不缓存结论。
    """

    def __init__(self, access_registry: WorkspaceActorAccessRegistry) -> None:
        self._registry = access_registry

    def authorize_agent(
        self,
        actor: WorkingActor,
        operation: WorkspaceOperation,
        target_workspace: WorkspaceIdentity,
    ) -> IdentityScope:
        """actor 操作授权：共同前提通过后，要求访问登记的白名单含该 operation。

        缺少行为许可是授权失败，不是身份认证失败，以 ``operation_not_allowed``
        拒绝。生产调用面限定在 actor API。
        """
        if not isinstance(operation, WorkspaceOperation):
            raise TypeError("operation 必须是 WorkspaceOperation")
        record = self._check_premises(actor, target_workspace, operation=operation)
        if operation not in record.allowed_operations:
            raise OperationDeniedError(
                details={
                    "operation": operation.value,
                    "reason": "operation_not_allowed",
                }
            )
        return IdentityScope(actor_identity=actor.actor, workspace_identity=target_workspace)

    def authorize_root(
        self,
        actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
    ) -> IdentityScope:
        """root 授权：共同前提通过后，只允许保留的 system actor。

        root 是授权规则，不是身份类型或可登记的 actor 操作；空白名单不影响
        system 的系统级操作，其他 actor 即使持有全部操作也不能通过，以
        ``root_required`` 拒绝。生产调用面限定在 root API、任务进程与 Import
        Bus 服务。
        """
        self._check_premises(actor, target_workspace)
        if actor.actor.agent_id != SYSTEM_AGENT_ID:
            raise OperationDeniedError(details={"reason": "root_required"})
        return IdentityScope(actor_identity=actor.actor, workspace_identity=target_workspace)

    def allowed_operations(self, actor: WorkingActor) -> frozenset[WorkspaceOperation]:
        """读取工作 actor 当前获准的 actor 操作集合；只读查询，不授予权限。

        供线程表生成线程描述中的操作快照（提示词与执行策略）；actor API 仍
        逐次授权。准入记录已不再有效时与授权一样以
        ``actor_not_admitted`` 拒绝；空白名单照常返回。
        """
        return self._admission_record(actor).allowed_operations

    # ---- 内部辅助 ----

    def _check_premises(
        self,
        actor: WorkingActor,
        target_workspace: WorkspaceIdentity,
        *,
        operation: WorkspaceOperation | None = None,
    ) -> WorkspaceActorAccessRecord:
        """两个授权方法的共同前提：准入有效、目标是驻留 Workspace、actor 用户是 owner。

        owner 约束在认证时已经检查，这里对目标 Workspace 再检查一次：身份类型
        不承担授权规则。
        """
        if not isinstance(target_workspace, WorkspaceIdentity):
            raise TypeError("target_workspace 必须是 WorkspaceIdentity")
        record = self._admission_record(actor)
        operation_detail = {"operation": operation.value} if operation is not None else {}
        if target_workspace != actor.workspace:
            raise OperationDeniedError(
                details={
                    **operation_detail,
                    "reason": "target_workspace_not_resident",
                    "resident_workspace_id": actor.workspace.workspace_id,
                    "target_workspace_id": target_workspace.workspace_id,
                }
            )
        if actor.actor.user_id != target_workspace.owner_user_id:
            raise OperationDeniedError(
                details={
                    **operation_detail,
                    "reason": "target_owner_mismatch",
                    "actor_user_id": actor.actor.user_id,
                    "owner_user_id": target_workspace.owner_user_id,
                }
            )
        return record

    def _admission_record(self, actor: WorkingActor) -> WorkspaceActorAccessRecord:
        """按驻留坐标确认准入仍然有效；无效时以 ``actor_not_admitted`` 拒绝。"""
        record = self._registry.record_for(actor.workspace, actor.actor)
        if record is None or not record.enabled:
            raise ScopeRequiredError(
                "该 Actor 已无有效的 Workspace 访问登记",
                details={"reason": "actor_not_admitted"},
            )
        return record
