"""真实注册入口、会话服务与 root 授权的协作；CPU 和子系统边界使用替身。"""

import pytest

from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.errors import OperationDeniedError, SessionBusyError, SessionNotFoundError
from hivememory.core.models import OMNI_DOLL_PROFILE, ActorIdentity, ResolvedAgentProfile
from hivememory.core.protocol.gateway import GatewayDecisionOutcome
from hivememory.core.protocol.models import RetrievalResponse
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.capability.root.session import RootSessionService
from hivememory.workspace.process.table import ProcessTable
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.process import make_task_process_service
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
        access=await access.authenticate(agent_id="system"),
    )
    session = root.create_session(**management)
    root.rename_session(session.session_id, title="恢复的会话", **management)
    assert [(s.title, s.turn_count) for s in root.list_sessions(**management)] == [
        ("恢复的会话", 0)
    ]
    assert root.get_session(session.session_id, **management).turns == ()
    actor = await access.authenticate(agent_id="a")
    with pytest.raises(OperationDeniedError):
        root.get_session(
            session.session_id, target_workspace=access.default_workspace, access=actor
        )
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
async def test_completed_turn_is_saved_before_finalize_and_agents_can_switch():
    access = composition()
    store = ConversationSessionService(WorkspaceSessionConfig())
    bus = GlobalSystemBus()
    submitted = []

    async def gateway(**kwargs):
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def profile(agent_id, **kwargs):
        return ResolvedAgentProfile(
            profile=OMNI_DOLL_PROFILE,
        )

    async def prepare(*, identity_scope, interaction_id, **kwargs):
        return PreparedAgentRun(
            belong_to=identity_scope.workspace_identity,
            interaction_id=interaction_id,
            topic_id="shared-topic",
            is_new_topic=False,
            retrieval_result=RetrievalResponse.from_memories([]),
        )

    async def finalize(*, payload, prepared_run, **kwargs):
        history = store.get(access.default_workspace, session_id).turns
        assert history[-1].interaction_id == prepared_run.interaction_id
        assert history[-1].payload == payload
        submitted.append(payload.user_message)

    bus.register(GatewayPublicRoutes.PROCESS, gateway)
    bus.register(PatchouliRoutes.GET_AGENT_PROFILE, profile)
    bus.register(PatchouliRoutes.PREPARE_AGENT_RUN, prepare)
    bus.register(PatchouliRoutes.FINALIZE_AGENT_RUN, finalize)
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
        result = await service.run_process(handle, stream=False)
        assert result.session_id == session_id
    assert submitted == ["问题0", "问题1"]
    assert [
        (t.agent_id, t.topic_id) for t in store.get(access.default_workspace, session_id).turns
    ] == [("a", "shared-topic"), ("b", "shared-topic")]
