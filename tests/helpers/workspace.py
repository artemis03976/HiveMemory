"""测试专用 WorkspaceAccessContext 与 RuntimeScope 构造器。"""

from hivememory.core.models import (
    Identity,
    RuntimeScope,
    WorkspaceAccessContext,
    build_internal_workspace_access,
)


def make_access_context(
    *,
    actor_identity: Identity | None = None,
    user_id: str = "test_user",
    agent_id: str = "test_agent",
    workspace_id: str = "main_workspace",
    interaction_id: str = "test_interaction",
) -> WorkspaceAccessContext:
    """显式构造测试 scope，绝不读取进程当前 Workspace。"""
    return build_internal_workspace_access(
        actor_identity or Identity(user_id=user_id, agent_id=agent_id),
        workspace_id,
        interaction_id,
    )


def make_runtime_scope(
    *,
    actor_identity: Identity | None = None,
    user_id: str = "test_user",
    agent_id: str = "test_agent",
    run_id: str = "test_run",
    frame_id: str = "test_frame",
    workspace_id: str = "main_workspace",
    interaction_id: str = "test_interaction",
) -> RuntimeScope:
    """构造携带完整 Workspace hard boundary 的 Alice 执行坐标。"""
    return RuntimeScope(
        access_context=make_access_context(
            actor_identity=actor_identity,
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            interaction_id=interaction_id,
        ),
        run_id=run_id,
        frame_id=frame_id,
    )
