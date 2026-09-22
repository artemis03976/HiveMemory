"""Memory ownership 硬边界与 Workspace 内 actor 读取策略。"""

from __future__ import annotations

from hivememory.core.models import (
    ActorIdentity,
    MemoryAtom,
    MemoryVisibility,
    WorkspaceIdentity,
)


def memory_belongs_to_workspace(
    memory: MemoryAtom,
    workspace_identity: WorkspaceIdentity,
) -> bool:
    """先验证唯一 ownership，任何 actor policy 都不能绕过此结果。"""
    return memory.workspace_identity == workspace_identity


def memory_visible_to_actor(memory: MemoryAtom, actor_identity: ActorIdentity) -> bool:
    """在 ownership 已通过后执行 Workspace 内 actor read policy。"""
    policy = memory.meta.access_policy
    if policy.visibility == MemoryVisibility.PUBLIC:
        return True
    if policy.visibility == MemoryVisibility.PRIVATE:
        return policy.target_agent_id == actor_identity.agent_id
    if policy.visibility == MemoryVisibility.TEAM:
        return bool(actor_identity.team_id and policy.target_team_id == actor_identity.team_id)
    return False


def memory_is_readable(
    memory: MemoryAtom,
    *,
    workspace_identity: WorkspaceIdentity,
    actor_identity: ActorIdentity,
    enforce_actor_visibility: bool = True,
) -> bool:
    """按固定顺序组合 ownership hard filter 与 actor read policy。

    ``enforce_actor_visibility=False`` 仅用于 owner-management 读取（D4）：
    ownership hard boundary 仍然生效，Workspace 内 actor 可见性策略被跳过，
    因此管理入口可以读取该 Workspace 的 PRIVATE/TEAM/PUBLIC 全部 Memory；
    Agent retrieval 与运行上下文必须保持默认 True。
    """
    if not memory_belongs_to_workspace(memory, workspace_identity):
        return False
    if not enforce_actor_visibility:
        return True
    return memory_visible_to_actor(memory, actor_identity)


__all__ = [
    "memory_belongs_to_workspace",
    "memory_visible_to_actor",
    "memory_is_readable",
]
