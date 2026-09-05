"""Memory ownership hard boundary 与 Workspace 内 actor read policy。"""

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
        return bool(
            actor_identity.team_id
            and policy.target_team_id == actor_identity.team_id
        )
    return False


def memory_is_readable(
    memory: MemoryAtom,
    *,
    workspace_identity: WorkspaceIdentity,
    actor_identity: ActorIdentity,
) -> bool:
    """按固定顺序组合 ownership hard filter 与 actor read policy。"""
    return memory_belongs_to_workspace(memory, workspace_identity) and memory_visible_to_actor(
        memory,
        actor_identity,
    )


__all__ = [
    "memory_belongs_to_workspace",
    "memory_visible_to_actor",
    "memory_is_readable",
]
