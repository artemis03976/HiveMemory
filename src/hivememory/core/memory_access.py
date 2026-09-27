"""Memory ownership 硬边界与 Workspace 内 actor 读取策略（授权谓词唯一实现）。

A2 §5.1 / §8.2：授权谓词上收 core，Patchouli 存储边界与 workspace 读取
resolver 共用同一份实现，杜绝两侧对 ``MemoryAccessPolicy`` 的解释分叉。
谓词只依赖原子自身的 policy 与可信身份坐标，结论逐次评估、不缓存。
"""

from __future__ import annotations

from hivememory.core.models import (
    ActorIdentity,
    MemoryAccessPolicy,
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


def access_policy_permits(policy: MemoryAccessPolicy, actor_identity: ActorIdentity) -> bool:
    """在 ownership 已通过后，按资源 policy 判定当前 Actor 能否读取。

    只接收 policy 本身：Profile 解析缓存条目只随存源原子的 policy 依据，
    不持有完整原子，命中时以本函数逐次授权。
    """
    if policy.visibility == MemoryVisibility.PUBLIC:
        return True
    if policy.visibility == MemoryVisibility.PRIVATE:
        return policy.target_agent_id == actor_identity.agent_id
    if policy.visibility == MemoryVisibility.TEAM:
        return bool(actor_identity.team_id and policy.target_team_id == actor_identity.team_id)
    return False


def memory_visible_to_actor(memory: MemoryAtom, actor_identity: ActorIdentity) -> bool:
    """在 ownership 已通过后执行 Workspace 内 actor read policy。"""
    return access_policy_permits(memory.meta.access_policy, actor_identity)


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
    Agent retrieval、运行上下文与 workspace 读取 resolver 必须保持默认 True。
    """
    if not memory_belongs_to_workspace(memory, workspace_identity):
        return False
    if not enforce_actor_visibility:
        return True
    return memory_visible_to_actor(memory, actor_identity)


__all__ = [
    "access_policy_permits",
    "memory_belongs_to_workspace",
    "memory_is_readable",
    "memory_visible_to_actor",
]
