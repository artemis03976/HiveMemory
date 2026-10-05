"""Workspace Actor 访问注册表：准入状态与行为白名单的权威配置。

两类登记之一（v0.7.0 A1 访问边界返工第 4.7 节）：本注册表由 Workspace
访问基础设施持有和查询，回答"这个 Actor 在这个 Workspace 是否被允许进入、
进入后允许执行哪些 operation"。一条访问记录同时承载准入状态与行为白名单，
不强制拆成两个 store。

职责边界：

- 只保存授权配置，不保存 Memory 可见性，不执行 Patchouli 业务；
- 键使用完整 Workspace 归属坐标（``owner_user_id + workspace_id``）与
  Actor 的 ``user_id + agent_id``；外部会话、run/frame 和调用协议
  不进入权限键；
- ``team_id`` 由可信身份关系提供给资源 policy，不进入本注册表；
- 首版为进程内不可变本地配置（``configs/workspace_actors.yaml``），
  配置修改经重启生效，不提供热更新、持久化或管理 API；System composition
  负责装载配置并注入给统一认证网关与共享行为检查。

用户级记录（v0.7.0 简化）：``agent_id`` 为 ``None`` 的记录覆盖该用户的
所有具体 Agent，但不覆盖保留的 ``system``——system 必须单独显式登记。
匹配时精确记录优先；精确记录禁用即拒绝，不回落到用户级记录；每个
(owner, workspace, user) 至多一条用户级记录，重复在装载期失败。

W0 兼容基线：``actor.user_id == workspace.owner_user_id`` 仍是准入前提，
因此本注册表在装载期拒绝跨 owner 的访问记录（跨 owner 成员模型不在
当前范围内）；相同 owner 也不表示自动获准——缺失记录即准入失败。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hivememory.core.constants import SYSTEM_AGENT_ID

if TYPE_CHECKING:
    # 注解引用保持 workspace 包内单向依赖：registry 不在运行期导入 access。
    from hivememory.core.access import WorkspaceOperation
    from hivememory.core.models import ActorIdentity, WorkspaceIdentity

__all__ = [
    "WorkspaceActorAccessRecord",
    "WorkspaceActorAccessRegistry",
]


@dataclass(frozen=True)
class WorkspaceActorAccessRecord:
    """一条 Workspace Actor 访问记录：准入 + 行为白名单。

    ``enabled=False`` 表示该 Actor 当前不被允许进入该 Workspace；
    ``allowed_operations`` 是进入后的行为上限，**允许为空**——空集合代表
    可进入但未获准执行任何资源操作，不是身份认证失败。

    ``agent_id`` 为 ``None`` 表示用户级记录：覆盖该用户的所有具体 Agent，
    不覆盖保留的 ``system``。
    """

    owner_user_id: str
    workspace_id: str
    user_id: str
    agent_id: str | None
    enabled: bool = True
    allowed_operations: frozenset[WorkspaceOperation] = frozenset()

    @property
    def key(self) -> tuple[str, str, str, str | None]:
        """注册表内部唯一键：完整 Workspace 坐标 + Actor 坐标。"""
        return (self.owner_user_id, self.workspace_id, self.user_id, self.agent_id)


class WorkspaceActorAccessRegistry:
    """进程内不可变的 Workspace Actor 访问注册表（v0.7.0 本地配置）。

    供认证一侧（``workspace.authentication.WorkspaceAuthenticator``）的
    内部准入与操作授权者（``workspace.authorization``）的逐次授权查询；
    认证网关经认证一侧完成 Workspace 准入，不直接读取本注册表。
    未登记 Workspace、未知 Actor 或缺失访问记录一律查无结果，由调用方
    fail closed；本类不区分"未登记"与"已吊销"，避免泄漏配置细节。
    """

    def __init__(self, records: list[WorkspaceActorAccessRecord]) -> None:
        """装载访问记录；重复键、跨 owner 记录与配置矛盾在装载期显式失败。"""
        by_key: dict[tuple[str, str, str, str | None], WorkspaceActorAccessRecord] = {}
        for record in records:
            if not isinstance(record, WorkspaceActorAccessRecord):
                raise TypeError("访问记录必须是 WorkspaceActorAccessRecord")
            if not record.enabled and record.allowed_operations:
                # 禁用记录上的白名单没有意义，属于配置矛盾，装载期拒绝。
                raise ValueError(f"Workspace Actor 访问记录已禁用却配置了行为白名单: {record.key}")
            if record.user_id != record.owner_user_id:
                # W0 兼容基线：准入要求 actor user 等于 workspace owner；
                # 跨 owner 成员记录在成员模型落地前不可表达。
                raise ValueError(
                    "Workspace Actor 访问记录的 user 必须等于 workspace owner"
                    f"（W0 兼容基线）: {record.key}"
                )
            if record.key in by_key:
                # 用户级记录键以 ``agent_id=None`` 收敛：同一 (owner,
                # workspace, user) 的第二条用户级记录在此被拒绝。
                raise ValueError(f"Workspace Actor 访问记录键重复: {record.key}")
            by_key[record.key] = record
        self._records = by_key

    def record_for(
        self,
        workspace_identity: WorkspaceIdentity,
        actor_identity: ActorIdentity,
    ) -> WorkspaceActorAccessRecord | None:
        """按完整坐标查询访问记录；查无结果返回 ``None``（调用方拒绝）。

        精确记录优先：命中（含禁用）即返回，由调用方按 ``enabled`` 拒绝，
        不回落到用户级记录。未命中精确记录且 Actor 不是保留 ``system``
        时，回落到该用户的用户级记录（``agent_id=None``）；``system``
        不被用户级记录覆盖，必须单独登记。
        """
        coordinates = (
            workspace_identity.owner_user_id,
            workspace_identity.workspace_id,
            actor_identity.user_id,
        )
        exact = self._records.get((*coordinates, actor_identity.agent_id))
        if exact is not None:
            return exact
        if actor_identity.agent_id == SYSTEM_AGENT_ID:
            return None
        return self._records.get((*coordinates, None))
