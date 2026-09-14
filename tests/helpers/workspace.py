"""测试专用 IdentityScope 与 RuntimeScope 构造器。"""

from hivememory.core.constants import SYSTEM_AGENT_ID
from hivememory.core.models import (
    ActorIdentity,
    RuntimeScope,
    IdentityScope,
    WorkspaceIdentity,
    build_internal_identity_scope,
)


def make_workspace_identity(
    *,
    owner_user_id: str = "test_user",
    workspace_id: str = "main_workspace",
) -> WorkspaceIdentity:
    """构造与 ``make_identity_scope`` 同构的 Workspace 归属坐标。

    供 Workspace 分区 cache 的读写测试显式传入 key 坐标，默认值与
    ``make_identity_scope``/``make_runtime_scope`` 的默认 scope 一致。
    """
    return WorkspaceIdentity(
        owner_user_id=owner_user_id,
        workspace_key=workspace_id,
        workspace_id=workspace_id,
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


def make_management_identity_scope(
    *,
    user_id: str = "test_user",
    workspace_id: str = "main_workspace",
) -> IdentityScope:
    """构造 server 非 Agent action 语义的管理 scope（保留 ``system`` actor）。

    对齐 ``server.deps.resolve_request_identity_scope`` 的公共入口行为：
    管理读取/管理写入等没有具体 Agent 作为操作来源主体的操作，由 server
    注入 ``SYSTEM_AGENT_ID`` 后冻结。
    """
    return make_identity_scope(
        user_id=user_id,
        agent_id=SYSTEM_AGENT_ID,
        workspace_id=workspace_id,
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
