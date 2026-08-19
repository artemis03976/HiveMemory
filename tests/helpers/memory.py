"""非 Workspace 专项测试使用的 schema v2 Memory 构造辅助。"""

from typing import Any

from hivememory.core.models import (
    Identity,
    MemoryAccessPolicy,
    MemoryCreationContext,
    MemoryVisibility,
    MetaData,
    WorkspaceIdentity,
)


def make_memory_metadata(
    *,
    source_agent_id: str,
    user_id: str,
    team_id: str | None = None,
    visibility: MemoryVisibility | str = MemoryVisibility.PUBLIC,
    workspace_id: str = "main_workspace",
    access_policy: MemoryAccessPolicy | None = None,
    **values: Any,
) -> MetaData:
    """把旧测试夹具的显式语义转换为 canonical v2 metadata。"""
    normalized = visibility.value if hasattr(visibility, "value") else str(visibility)
    if access_policy is None:
        if normalized == "PUBLIC":
            access_policy = MemoryAccessPolicy.public()
        elif normalized == "PRIVATE":
            access_policy = MemoryAccessPolicy(
                visibility=MemoryVisibility.PRIVATE,
                target_agent_id=source_agent_id,
            )
        elif normalized in {"TEAM", "WORKSPACE"}:
            if not team_id:
                raise ValueError("TEAM 测试 metadata 必须提供 team_id")
            access_policy = MemoryAccessPolicy(
                visibility=MemoryVisibility.TEAM,
                target_team_id=team_id,
            )
        else:
            raise ValueError(f"不支持的测试 visibility: {normalized}")
    return MetaData(
        workspace_identity=WorkspaceIdentity(
            owner_user_id=user_id,
            workspace_key=workspace_id,
            workspace_id=workspace_id,
        ),
        source_agent_id=source_agent_id,
        source_team_id=team_id,
        access_policy=access_policy,
        **values,
    )


def make_memory_creation_context(
    *,
    user_id: str = "u1",
    agent_id: str = "a1",
    team_id: str | None = None,
    workspace_id: str = "main_workspace",
) -> MemoryCreationContext:
    """构造不读取进程状态的显式生成 scope。"""
    return MemoryCreationContext(
        actor_identity=Identity(
            user_id=user_id,
            agent_id=agent_id,
            team_id=team_id,
        ),
        workspace_identity=WorkspaceIdentity(
            owner_user_id=user_id,
            workspace_key=workspace_id,
            workspace_id=workspace_id,
        ),
    )


__all__ = ["make_memory_creation_context", "make_memory_metadata"]
