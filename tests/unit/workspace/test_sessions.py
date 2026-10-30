"""会话服务的容量、归属隔离、认领与快照保护。"""

import pytest

from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.workspace.contracts.errors import (
    SessionBusyError,
    SessionCapacityError,
    SessionCheckpointError,
    SessionClosedError,
    SessionNotFoundError,
)
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.sessions import make_compaction_checkpoint, make_interaction_payload
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
        payload=make_interaction_payload("首轮", "回复"),
        topic_id="t1",
    )
    service.release(workspace, session.session_id, "p1")
    with pytest.raises(SessionCapacityError):
        service.claim(workspace, session.session_id, "p2")
    saved = service.get(workspace, session.session_id)
    assert saved.title == "首轮"
    assert [(t.interaction_id, t.payload.reply_text) for t in saved.turns] == [("p1", "回复")]


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
    payload = make_interaction_payload("问题", "原文")
    service.append(
        workspace, session.session_id, process_id="p1", agent_id="a", payload=payload, topic_id="t"
    )
    payload.turn_events.pop()
    service.get(workspace, session.session_id).turns[0].payload.turn_events.clear()
    assert service.get(workspace, session.session_id).turns[0].payload.reply_text == "原文"


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


@pytest.fixture
def checkpoint_session():
    """两轮旧历史与一个持有认领的后继进程。"""
    service = ConversationSessionService(WorkspaceSessionConfig())
    workspace = make_workspace_identity()
    session_id = service.create(workspace).session_id
    for process in ("old-1", "old-2"):
        service.claim(workspace, session_id, process)
        service.append(
            workspace,
            session_id,
            process_id=process,
            agent_id="alice",
            payload=make_interaction_payload(process),
            topic_id="topic",
        )
        service.release(workspace, session_id, process)
    service.claim(workspace, session_id, "current")
    return service, workspace, session_id


@pytest.mark.parametrize(
    "reduced,summarized",
    [
        ("foreign-turn", None),
        ("old-2", "foreign-turn"),
        ("old-1", "old-2"),
        ("old-1", "old-1"),
        ("old-2", None),
    ],
    ids=["unknown-reduction", "unknown-summary", "summary-ahead", "reduction-back", "summary-back"],
)
def test_invalid_checkpoint_preserves_previous_view(checkpoint_session, reduced, summarized):
    """游标存在性、覆盖顺序与两个独立的单调约束都在会话所有者处拒绝。"""
    service, workspace, session_id = checkpoint_session
    previous = make_compaction_checkpoint(reduced_through="old-2", summarized_through="old-1")
    service.accept_checkpoint(workspace, session_id, process_id="current", checkpoint=previous)
    with pytest.raises(SessionCheckpointError):
        service.accept_checkpoint(
            workspace,
            session_id,
            process_id="current",
            checkpoint=make_compaction_checkpoint(
                reduced_through=reduced, summarized_through=summarized
            ),
        )
    assert service.get(workspace, session_id).compaction_checkpoint == previous


def test_checkpoint_of_another_format_rebuilds_from_earlier_cursors(checkpoint_session):
    """单调约束只在同一产生方与格式版本内成立，新格式的重建视图可以替换旧视图。"""
    service, workspace, session_id = checkpoint_session
    service.accept_checkpoint(
        workspace,
        session_id,
        process_id="current",
        checkpoint=make_compaction_checkpoint(
            reduced_through="old-2", summarized_through="old-1", summary="旧格式摘要"
        ),
    )
    rebuilt = make_compaction_checkpoint(format_version=2, reduced_through="old-1")
    service.accept_checkpoint(workspace, session_id, process_id="current", checkpoint=rebuilt)
    assert service.get(workspace, session_id).compaction_checkpoint == rebuilt


def test_checkpoint_requires_claim_and_workspace_ownership(checkpoint_session):
    """不同 Workspace 或非认领进程不能改变检查点。"""
    service, workspace, session_id = checkpoint_session
    checkpoint = make_compaction_checkpoint(reduced_through="old-1")
    with pytest.raises(SessionNotFoundError):
        service.accept_checkpoint(
            make_workspace_identity(owner_user_id="other"),
            session_id,
            process_id="current",
            checkpoint=checkpoint,
        )
    with pytest.raises(SessionBusyError):
        service.accept_checkpoint(workspace, session_id, process_id="stale", checkpoint=checkpoint)
    assert service.get(workspace, session_id).compaction_checkpoint is None


def test_checkpoint_advances_without_mutating_history_and_follows_session_lifetime(
    checkpoint_session,
):
    """检查点可同位更新或前进，列表保持轻量，删除与关闭使其不可再读取。"""
    service, workspace, session_id = checkpoint_session
    original_turns = service.get(workspace, session_id).turns
    first = make_compaction_checkpoint(reduced_through="old-1")
    advanced = make_compaction_checkpoint(
        reduced_through="old-2",
        summarized_through="old-1",
        summary="已完成分析",
        references=("mem_a",),
    )
    for checkpoint in (first, first, advanced):
        service.accept_checkpoint(
            workspace, session_id, process_id="current", checkpoint=checkpoint
        )
    saved = service.get(workspace, session_id)
    assert saved.compaction_checkpoint == advanced
    assert saved.turns == original_turns
    assert "compaction_checkpoint" not in service.list_sessions(workspace)[0].model_dump()
    service.release(workspace, session_id, "current")
    service.delete(workspace, session_id)
    with pytest.raises(SessionNotFoundError):
        service.get(workspace, session_id)
    new_id = service.create(workspace).session_id
    assert service.get(workspace, new_id).compaction_checkpoint is None
    service.close_and_clear()
    with pytest.raises(SessionClosedError):
        service.accept_checkpoint(workspace, new_id, process_id="current", checkpoint=first)
