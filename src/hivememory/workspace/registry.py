"""Workspace Actor 访问注册表：准入状态与行为白名单的权威配置。

A1 计划（docs/plans/v0.7.0-a1-workspace-access-boundary.md 第 2.2/3.2 节）
确立的两类登记之一：本注册表由 Workspace 访问基础设施持有和查询，回答
"这个 Actor 在这个 Workspace 是否被允许进入、进入后允许执行哪些 operation"。
一条访问记录同时承载准入状态与行为白名单，不强制拆成两个 store。

职责边界：

- 只保存授权配置，不保存 Memory 可见性，不执行 Patchouli 业务；
- 键使用完整 Workspace 归属坐标（``owner_user_id + workspace_id``）与
  Actor 的 ``user_id + agent_id``；``session_id``、run/frame 和调用协议
  不进入权限键；
- ``team_id`` 由可信身份关系提供给资源 policy，不进入本注册表；
- 首版为进程内不可变本地配置，配置修改经重启生效，不提供热更新、
  持久化或管理 API；System composition 负责装载配置并注入给统一认证
  网关与共享行为检查。

W0 兼容基线：``actor.user_id == workspace.owner_user_id`` 仍是准入前提，
因此本注册表在装载期拒绝跨 owner 的访问记录（跨 owner 成员模型不在
A1 范围内）；相同 owner 也不表示自动获准——缺失记录即准入失败。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hivememory.core.models import ActorIdentity, WorkspaceIdentity
    # 注解引用保持 workspace 包内单向依赖：registry 不在运行期导入 access。
    from hivememory.workspace.access import WorkspaceOperation

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
    """

    owner_user_id: str
    workspace_id: str
    user_id: str
    agent_id: str
    enabled: bool = True
    allowed_operations: frozenset[WorkspaceOperation] = field(
        default=frozenset(), compare=True
    )

    @property
    def key(self) -> tuple[str, str, str, str]:
        """注册表内部唯一键：完整 Workspace 坐标 + Actor 坐标。"""
        return (self.owner_user_id, self.workspace_id, self.user_id, self.agent_id)


class WorkspaceActorAccessRegistry:
    """进程内不可变的 Workspace Actor 访问注册表（v0.7.0 首版本地配置）。

    供 System 统一认证网关（准入判定）与共享行为检查（白名单比对）查询。
    未登记 Workspace、未知 Actor 或缺失访问记录一律查无结果，由调用方
    fail closed；本类不区分"未登记"与"已吊销"，避免泄漏配置细节。
    """

    def __init__(self, records: list[WorkspaceActorAccessRecord]) -> None:
        """装载访问记录；重复键与跨 owner 记录在装载期显式失败。"""
        by_key: dict[tuple[str, str, str, str], WorkspaceActorAccessRecord] = {}
        for record in records:
            if not isinstance(record, WorkspaceActorAccessRecord):
                raise TypeError("访问记录必须是 WorkspaceActorAccessRecord")
            if not record.enabled and record.allowed_operations:
                # 禁用记录上的白名单没有意义，属于配置矛盾，装载期拒绝。
                raise ValueError(
                    f"Workspace Actor 访问记录已禁用却配置了行为白名单: {record.key}"
                )
            if record.user_id != record.owner_user_id:
                # W0 兼容基线：准入要求 actor user 等于 workspace owner；
                # 跨 owner 成员记录在成员模型落地前不可表达。
                raise ValueError(
                    "Workspace Actor 访问记录的 user 必须等于 workspace owner"
                    f"（W0 兼容基线）: {record.key}"
                )
            if record.key in by_key:
                raise ValueError(f"Workspace Actor 访问记录键重复: {record.key}")
            by_key[record.key] = record
        self._records = by_key

    def record_for(
        self,
        workspace_identity: WorkspaceIdentity,
        actor_identity: ActorIdentity,
    ) -> WorkspaceActorAccessRecord | None:
        """按完整坐标查询访问记录；查无结果返回 ``None``（调用方拒绝）。

        键包含 ``owner_user_id``：两个 owner 使用相同 ``workspace_id`` 时
        记录不串扰。
        """
        return self._records.get(
            (
                workspace_identity.owner_user_id,
                workspace_identity.workspace_id,
                actor_identity.user_id,
                actor_identity.agent_id,
            )
        )
