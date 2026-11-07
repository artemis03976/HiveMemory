"""真实注册入口、会话服务与 root 授权的协作；CPU 和子系统边界使用替身。"""

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.errors import OperationDeniedError
from hivememory.core.models import OMNI_DOLL_PROFILE, ActorIdentity, ResolvedAgentProfile
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.workspace.capability.root.session import RootSessionService
from hivememory.workspace.contracts.errors import SessionBusyError, SessionNotFoundError
from hivememory.workspace.process.table import ProcessTable
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import PatchouliTurnStub, collect_events, make_task_process_service
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_workspace_identity,
)


def composition():
    """同一 Workspace 登记两个 Agent 与管理 actor，便于验证逐轮切换。"""
    return make_access_composition(
        [make_actor_access_record(agent_id=a) for a in ("system", "a", "b")]
    )


@pytest.mark.asyncio
async def test_root_crud_and_actor_rejection():
    access = composition()
    store = ConversationSessionService(WorkspaceSessionConfig())
    root = RootSessionService(store, operation_authorizer=access.authorizer)
    management = dict(
        target_workspace=access.default_workspace,
        actor=await access.authenticate(agent_id="system"),
    )
    session = root.create_session(**management)
    root.rename_session(session.session_id, title="恢复的会话", **management)
    assert [(s.title, s.turn_count) for s in root.list_sessions(**management)] == [
        ("恢复的会话", 0)
    ]
    assert root.get_session(session.session_id, **management).turns == ()
    actor = await access.authenticate(agent_id="a")
    with pytest.raises(OperationDeniedError):
        root.get_session(session.session_id, target_workspace=access.default_workspace, actor=actor)
    root.delete_session(session.session_id, **management)
    assert root.list_sessions(**management) == ()


@pytest.mark.asyncio
async def test_registration_claim_release_and_rejection_leave_no_process():
    access = composition()
    store = ConversationSessionService(WorkspaceSessionConfig())
    table = ProcessTable()
    service = make_task_process_service(
        GlobalSystemBus(),
        cpu=ScriptedCPU(),
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        sessions=store,
        process_table=table,
    )
    args = dict(
        adapter="local",
        principal=access.principal,
        actor=ActorIdentity(user_id=access.default_workspace.owner_user_id, agent_id="a"),
        workspace=access.default_workspace,
        message="问题",
    )
    handle = await service.register_process(**args, process_id="p1")
    with pytest.raises(SessionBusyError):
        await service.register_process(**args, process_id="p2", session_id=handle.session_id)
    assert table.get("p2") is None
    foreign = store.create(make_workspace_identity(owner_user_id="foreign"))
    for session_id in ("missing", foreign.session_id):
        with pytest.raises(SessionNotFoundError):
            await service.register_process(**args, process_id="p2", session_id=session_id)
        assert table.get("p2") is None
    await service.close_process(handle)
    next_handle = await service.register_process(
        **args, process_id="p2", session_id=handle.session_id
    )
    assert next_handle.session_id == handle.session_id
    await service.shutdown()
    assert store.get(access.default_workspace, handle.session_id).active_process_id is None
    assert table.snapshot() == ()


@pytest.mark.asyncio
async def test_submitted_turns_are_saved_with_applied_topic_and_agents_can_switch():
    """同一会话逐轮切换执行 Agent：每轮提交交互后按实际所在的话题写入会话。

    Gateway 指向新话题、这一轮没有写入时，交互按新话题决定提交，会话记录的是
    应用后实际所在的话题，而不是 Gateway 的目标标记。
    """
    access = composition()
    store = ConversationSessionService(WorkspaceSessionConfig())
    bus = GlobalSystemBus()

    async def gateway(**kwargs):
        return GatewayDecisionOutcome(
            decision=make_gateway_decision(target_topic_id="NEW_TOPIC", new_topic_title="新话题")
        )

    async def profile(agent_id, **kwargs):
        return ResolvedAgentProfile(
            profile=OMNI_DOLL_PROFILE,
        )

    patchouli = PatchouliTurnStub().install(bus)
    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, profile)
    service = make_task_process_service(
        bus,
        cpu=ScriptedCPU(result=make_cpu_result()),
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        sessions=store,
    )
    session_id = None
    for i, agent in enumerate(("a", "b")):
        handle = await service.register_process(
            adapter="local",
            principal=access.principal,
            actor=ActorIdentity(user_id=access.default_workspace.owner_user_id, agent_id=agent),
            workspace=access.default_workspace,
            process_id=f"p{i}",
            message=f"问题{i}",
            session_id=session_id,
        )
        session_id = handle.session_id
        events = await collect_events(service, handle)
        assert events[0]["data"]["session_id"] == session_id
    turns = store.get(access.default_workspace, session_id).turns
    assert [s["payload"] for s in patchouli.submissions] == [t.payload for t in turns]
    assert [(t.payload.user_message, t.agent_id, t.topic_id) for t in turns] == [
        ("问题0", "a", "topic-new"),
        ("问题1", "b", "topic-new"),
    ]
