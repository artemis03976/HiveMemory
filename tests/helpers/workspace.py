"""测试专用 IdentityScope 与 RuntimeScope 构造器。"""

from hivememory.core.models import (
    ActorIdentity,
    RuntimeScope,
    IdentityScope,
    build_internal_identity_scope,
)


def make_identity_scope(
    *,
    actor_identity: ActorIdentity | None = None,
    user_id: str = "test_user",
    agent_id: str = "test_agent",
    workspace_id: str = "main_workspace",
    interaction_id: str | None = None,
) -> IdentityScope:
    """显式构造测试 scope，绝不读取进程当前 Workspace。

    ``interaction_id`` 仅为兼容现有测试调用签名而接收；它不会写入
    ``IdentityScope``。需要保存 interaction ID 的测试载体应独立持有该字段。
    """
    return build_internal_identity_scope(
        actor_identity or ActorIdentity(user_id=user_id, agent_id=agent_id),
        workspace_id,
    )


def make_runtime_scope(
    *,
    actor_identity: ActorIdentity | None = None,
    user_id: str = "test_user",
    agent_id: str = "test_agent",
    run_id: str = "test_run",
    frame_id: str = "test_frame",
    workspace_id: str = "main_workspace",
    interaction_id: str | None = None,
) -> RuntimeScope:
    """构造携带完整 Workspace hard boundary 的 Alice 执行坐标。"""
    return RuntimeScope(
        identity_scope=make_identity_scope(
            actor_identity=actor_identity,
            user_id=user_id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            interaction_id=interaction_id,
        ),
        run_id=run_id,
        frame_id=frame_id,
    )
