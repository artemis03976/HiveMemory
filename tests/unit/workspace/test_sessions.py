"""会话服务的容量、归属隔离、认领与快照保护。"""

import pytest

from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.errors import (
    SessionBusyError,
    SessionCapacityError,
    SessionClosedError,
    SessionNotFoundError,
)
from hivememory.core.protocol.models import InteractionPayload
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.workspace import make_workspace_identity


def test_capacity_rejects_without_evicting_existing_history():
    """容量拒绝不能破坏已有轮次或让其他 Workspace 消耗同一配额。"""
    service = ConversationSessionService(
        WorkspaceSessionConfig(max_sessions_per_workspace=1, max_turns_per_session=1)
    )
    workspace = make_workspace_identity(owner_user_id="u1")
    session = service.create(workspace)
    with pytest.raises(SessionCapacityError):
        service.create(workspace)
    service.create(make_workspace_identity(owner_user_id="u2"))
    service.claim(workspace, session.session_id, "p1")
    service.append(
        workspace,
        session.session_id,
        process_id="p1",
        agent_id="alice",
        payload=InteractionPayload(user_message="首轮", assistant_final_text="回复"),
        topic_id="t1",
    )
    service.release(workspace, session.session_id, "p1")
    with pytest.raises(SessionCapacityError):
        service.claim(workspace, session.session_id, "p2")
    saved = service.get(workspace, session.session_id)
    assert saved.title == "首轮"
    assert [(t.interaction_id, t.payload.assistant_final_text) for t in saved.turns] == [
        ("p1", "回复")
    ]


@pytest.mark.parametrize("operation", ["get", "rename", "delete", "claim", "release"])
def test_foreign_workspace_cannot_access_session(operation):
    """所有按 ID 访问的入口都先校验资源归属。"""
    service = ConversationSessionService(WorkspaceSessionConfig())
    owner = make_workspace_identity(owner_user_id="u1")
    other = make_workspace_identity(owner_user_id="u2")
    session = service.create(owner)
    arguments = {"rename": ("新标题",), "claim": ("p1",), "release": ("p1",)}.get(operation, ())
    with pytest.raises(SessionNotFoundError):
        getattr(service, operation)(other, session.session_id, *arguments)
    assert service.list_sessions(other) == ()
    assert [s.session_id for s in service.list_sessions(owner)] == [session.session_id]


def test_claim_excludes_concurrent_writer_and_stale_release():
    """忙会话不能删除或再次认领，旧进程释放不能影响后继进程。"""
    service = ConversationSessionService(WorkspaceSessionConfig())
    workspace = make_workspace_identity()
    session = service.create(workspace)
    service.claim(workspace, session.session_id, "p1")
    with pytest.raises(SessionBusyError):
        service.claim(workspace, session.session_id, "p2")
    with pytest.raises(SessionBusyError):
        service.delete(workspace, session.session_id)
    service.release(workspace, session.session_id, "p1")
    service.claim(workspace, session.session_id, "p2")
    service.release(workspace, session.session_id, "p1")
    assert service.get(workspace, session.session_id).active_process_id == "p2"


def test_payload_snapshots_do_not_mutate_stored_history():
    """入参或历史读取结果被修改时，已封口正文保持原样。"""
    service = ConversationSessionService(WorkspaceSessionConfig())
    workspace = make_workspace_identity()
    session = service.create(workspace)
    service.claim(workspace, session.session_id, "p1")
    payload = InteractionPayload(user_message="问题", assistant_final_text="原文")
    service.append(
        workspace, session.session_id, process_id="p1", agent_id="a", payload=payload, topic_id="t"
    )
    payload.assistant_final_text = "入参篡改"
    service.get(workspace, session.session_id).turns[0].payload.assistant_final_text = "读取篡改"
    assert (
        service.get(workspace, session.session_id).turns[0].payload.assistant_final_text == "原文"
    )


@pytest.mark.parametrize(
    "operation", ["create", "list_sessions", "get", "rename", "delete", "claim", "release"]
)
def test_closed_service_rejects_reads_and_writes(operation):
    """关闭不是空列表或会话未找到，而是显式的设施不可用。"""
    service = ConversationSessionService(WorkspaceSessionConfig())
    workspace = make_workspace_identity()
    session = service.create(workspace)
    service.close_and_clear()
    arguments = {
        "create": (),
        "list_sessions": (),
        "get": (session.session_id,),
        "rename": (session.session_id, "标题"),
        "delete": (session.session_id,),
        "claim": (session.session_id, "p"),
        "release": (session.session_id, "p"),
    }[operation]
    with pytest.raises(SessionClosedError):
        getattr(service, operation)(workspace, *arguments)
