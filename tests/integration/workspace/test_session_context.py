"""会话、进程 CPU 分配、Alice 提示词与 Gateway 输入边界的协作。"""

import pytest

from hivememory.alice.context.view import SessionContextView
from hivememory.components.bus.global_bus import GlobalSystemBus
from hivememory.config.alice import CompactionConfig
from hivememory.config.gateway import SystemGatewayConfig
from hivememory.config.workspace import WorkspaceSessionConfig
from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    ResolvedAgentProfile,
    TopicSnapshot,
)
from hivememory.core.protocol.gateway import (
    GatewayDecisionOutcome,
    GatewayIngressMode,
    GatewaySessionTurn,
)
from hivememory.gateway.contracts import GatewayPublicRoutes
from hivememory.gateway.runtime import GatewayRuntime
from hivememory.gateway.service import GatewayService
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.patchouli.contracts.prepare import PreparedAgentRun
from hivememory.workspace.sessions import ConversationSessionService
from tests.helpers.chat_handoff import make_gateway_decision
from tests.helpers.cpu import ScriptedCPU, make_cpu_result
from tests.helpers.llm import DEFAULT_LLM_PARAMS, ScriptedLLMService
from tests.helpers.process import make_task_process_service
from tests.helpers.sessions import make_compaction_checkpoint
from tests.helpers.workspace import (
    make_access_composition,
    make_actor_access_record,
    make_identity_scope,
)


@pytest.mark.asyncio
async def test_interleaved_sessions_share_topic_but_not_cpu_or_gateway_history():
    """同 Topic 的另一会话不能污染历史，Alice 全量保留，Gateway 只取配置的最近轮次。"""
    access = make_access_composition([make_actor_access_record(agent_id="a")])
    bus = GlobalSystemBus()
    gateway_history = []

    async def gateway(*, session_turns, **kwargs):
        gateway_history.append(session_turns)
        return GatewayDecisionOutcome(decision=make_gateway_decision())

    async def profile(agent_id, **kwargs):
        return ResolvedAgentProfile(profile=OMNI_DOLL_PROFILE)

    async def prepare(*, identity_scope, interaction_id, **kwargs):
        return PreparedAgentRun(
            belong_to=identity_scope.workspace_identity,
            interaction_id=interaction_id,
            topic_id="shared-topic",
            is_new_topic=False,
        )

    async def finalize(**kwargs):
        pass

    for route, handler in [
        (GatewayPublicRoutes.PROCESS, gateway),
        (PatchouliRoutes.GET_AGENT_PROFILE, profile),
        (PatchouliRoutes.PREPARE_AGENT_RUN, prepare),
        (PatchouliRoutes.FINALIZE_AGENT_RUN, finalize),
    ]:
        bus.register(route, handler)
    cpu = ScriptedCPU(result=make_cpu_result(final_text="回复"))
    session_store = ConversationSessionService(WorkspaceSessionConfig())
    service = make_task_process_service(
        bus,
        cpu=cpu,
        access_gateway=access.gateway,
        operation_authorizer=access.authorizer,
        gateway_session_turn_limit=2,
        sessions=session_store,
    )
    sessions = {"A": None, "B": None}
    for i, name in enumerate(["A", "A", "B", "A", "A", "A", "A", "A"]):
        handle = await service.register_process(
            adapter="local",
            principal=access.principal,
            actor=ActorIdentity(user_id=access.default_workspace.owner_user_id, agent_id="a"),
            workspace=access.default_workspace,
            process_id=f"p{i}",
            message=f"{name}{i}",
            session_id=sessions[name],
        )
        if i == 7:
            session_store.accept_checkpoint(
                access.default_workspace,
                handle.session_id,
                process_id=handle.process_id,
                checkpoint=make_compaction_checkpoint(reduced_through="p1"),
            )
        sessions[name] = handle.session_id
        await service.run_process(handle, stream=False)
    manifest = cpu.calls[-1].manifest
    assert [t.payload.user_message for t in manifest.session_history] == [
        "A0",
        "A1",
        "A3",
        "A4",
        "A5",
        "A6",
    ]
    assert [t.user_message for t in gateway_history[-1]] == ["A5", "A6"]
    assert gateway_history[2] == ()
    messages = SessionContextView(
        instructions=manifest.instructions,
        turn_context=manifest.turn_context,
        user_message=manifest.user_message,
        language=manifest.language,
        mtp_prompt="",
        session_history=manifest.session_history,
        current_agent_id=manifest.labels.agent_id,
        checkpoint=manifest.compaction_checkpoint,
        input_token_limit=100000,
        config=CompactionConfig(),
    ).messages
    assert [m["content"] for m in messages if m["role"] == "user"] == [
        "A0",
        "A1",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
    ]
    assert manifest.topic_id == "shared-topic"
    assert manifest.compaction_checkpoint.reduced_through == "p1"


@pytest.mark.asyncio
async def test_gateway_routes_and_rewrites_with_session_history_without_exporting_it():
    """真实 workflow 将同一会话历史交给两个 Engine，菜单保留，公共决定不携带历史。"""
    bus = GlobalSystemBus()
    scope = make_identity_scope()
    topic = TopicSnapshot(
        last_update=1_700_000_000.0,
        topic_id="topic-1",
        topic_title="Docker",
        state_summary="Topic 背景",
        workspace_identity=scope.workspace_identity,
    )

    async def topics(**kwargs):
        return (topic,)

    async def get_topic(*args, **kwargs):
        return None

    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, topics)
    bus.register(PatchouliRoutes.TOPIC_GET_DATA, get_topic)
    llm = ScriptedLLMService(
        '{"target_topic":"topic-1","reason":"会话指代 Docker"}',
        '{"intent_type":"RAG","rewritten_query":"Docker 容器如何重启","search_keywords":["Docker"],"memory_write_signal":"WRITE"}',
    )
    service = GatewayService(
        GatewayRuntime(
            config=SystemGatewayConfig(),
            global_bus=bus,
            llm_service=llm,
            llm_params=DEFAULT_LLM_PARAMS,
        )
    )
    result = await service.process(
        "它怎么重启？",
        identity_scope=scope,
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
        session_turns=(
            GatewaySessionTurn(user_message="Docker 容器已启动", assistant_text="可以继续配置"),
        ),
    )
    assert result.decision.target_topic_id == "topic-1"
    assert result.decision.rewritten_query == "Docker 容器如何重启"
    assert len(llm.calls) == 2
    for call in llm.calls:
        assert call.messages[1:] == [
            {"role": "user", "content": "Docker 容器已启动"},
            {"role": "assistant", "content": "可以继续配置"},
            {"role": "user", "content": "它怎么重启？"},
        ]
    assert "Docker" in llm.calls[0].messages[0]["content"]
    assert "session_turns" not in result.decision.model_dump()
