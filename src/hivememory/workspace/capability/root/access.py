"""root 访问记录 API：授权后读取与编辑目标 Workspace 的访问记录。

访问记录（准入与 actor 操作白名单）属于 workspace，只有保留的 system 能管理，
不新增可登记给 actor 的管理 operation。每次调用都在入口 root 授权一次，只能
读写目标 Workspace 坐标下的记录。

保留的 ``system`` 记录不能经这里修改或删除：root 授权以它准入 ``system``，
禁用或删除它会让管理接口自身立即失效，而恢复它恰恰需要管理接口；它只能经
登记文件修改。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.errors import AccessRegistrationConflictError
from hivememory.workspace.registry import WorkspaceActorAccessRecord

if TYPE_CHECKING:
    from hivememory.core.access import WorkingActor, WorkspaceOperation
    from hivememory.core.models import WorkspaceIdentity
    from hivememory.workspace.authorization import WorkspaceOperationAuthorizer
    from hivememory.workspace.registry import WorkspaceActorAccessRegistry


class RootAccessService:
    """访问记录的管理入口：列出、新增或修改、删除。

    ``agent_id`` 为 ``None`` 表示用户级记录（覆盖其余所有具体 Agent）。记录的
    owner 与用户都取目标 Workspace 的 owner：准入要求二者相等。编辑写回生效
    的登记文件后立即生效，授权者在下一次授权时按新内容判定。
    """

    def __init__(
        self,
        registry: WorkspaceActorAccessRegistry,
        *,
        operation_authorizer: WorkspaceOperationAuthorizer,
    ) -> None:
        self._registry = registry
        self._authorizer = operation_authorizer

    def list_records(
        self, *, target_workspace: WorkspaceIdentity, actor: WorkingActor
    ) -> list[WorkspaceActorAccessRecord]:
        """目标 Workspace 的访问记录，不含保留的 ``system`` 记录。"""
        scope = self._authorizer.authorize_root(actor, target_workspace)
        return [
            record
            for record in self._registry.records_in(scope.workspace_identity)
            if record.agent_id != SYSTEM_AGENT_ID
        ]

    def put_record(
        self,
        agent_id: str | None,
        *,
        enabled: bool,
        allowed_operations: frozenset[WorkspaceOperation],
        target_workspace: WorkspaceIdentity,
        actor: WorkingActor,
    ) -> WorkspaceActorAccessRecord:
        """新增或替换一条访问记录；禁用的记录不能带白名单（与装载规则相同）。"""
        scope = self._authorizer.authorize_root(actor, target_workspace)
        _reject_system(agent_id)
        workspace = scope.workspace_identity
        record = WorkspaceActorAccessRecord(
            owner_user_id=workspace.owner_user_id,
            workspace_id=workspace.workspace_id,
            user_id=workspace.owner_user_id,
            agent_id=agent_id,
            enabled=enabled,
            allowed_operations=allowed_operations,
        )
        self._registry.put(record)
        return record

    def delete_record(
        self,
        agent_id: str | None,
        *,
        target_workspace: WorkspaceIdentity,
        actor: WorkingActor,
    ) -> None:
        """删除一条访问记录；删除用户级记录后，没有精确记录的 Agent 不能进入。"""
        scope = self._authorizer.authorize_root(actor, target_workspace)
        _reject_system(agent_id)
        workspace = scope.workspace_identity
        self._registry.delete(
            (workspace.owner_user_id, workspace.workspace_id, workspace.owner_user_id, agent_id)
        )


def _reject_system(agent_id: str | None) -> None:
    if agent_id == SYSTEM_AGENT_ID:
        raise AccessRegistrationConflictError(
            details={"reason": "protected", "agent_id": SYSTEM_AGENT_ID}
        )
