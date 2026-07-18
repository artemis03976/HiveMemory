"""ChatApplicationService / PassiveIngressService 委托测试"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    Identity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
    MetaData,
    PayloadLayer,
)
from hivememory.core.protocol.gateway import (
    GatewayDecision,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.core.protocol.models import (
    AgentRunContext,
    AgentRunResult,
    RetrievalResponse,
)
from hivememory.patchouli.models import (
    PreparedAgentRun,
    StreamPrelude,
)
from hivememory.system.application.agent_service import AgentApplicationService
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus


def _make_prepared_run(**overrides) -> PreparedAgentRun:
    identity = Identity(user_id="u1", agent_id="omni_doll")
    gateway_decision = GatewayDecision(
        target_topic_id="topic_1",
        rewritten_query="resolved",
        search_keywords=("k",),
        memory_write_signal=MemoryWriteSignal.WRITE,
        retrieval_plan=RetrievalPlan(),
        intent_type=IntentType.RAG,
    )
    defaults = dict(
        agent_run_context=AgentRunContext(
            identity=identity,
            topic_id="topic_1",
            user_message="hi",
            topic_context=None,
            retrieval_result=RetrievalResponse(),
            agent_profile=OMNI_DOLL_PROFILE,
            storage_available=True,
        ),
        stream_prelude=StreamPrelude(
            topic_id="topic_1",
            is_new_topic=False,
            pool_topics=[],
            memory_refs=[],
        ),
        gateway_decision=gateway_decision,
        generation_options=None,
    )
    defaults.update(overrides)
    return PreparedAgentRun(**defaults)


def _make_chat_result() -> AgentRunResult:
    return AgentRunResult(
        final_text="hello!",
        mtp_iterations=0,
        total_iterations=1,
        turn_events=[],
    )


@pytest.fixture
def mock_global_bus():
    """模拟 GlobalSystemBus，根据路由返回不同结果。"""
    bus = MagicMock(spec=GlobalSystemBus)

    prepared = _make_prepared_run()
    chat_result = _make_chat_result()

    async def route_dispatch(route, *args, **kwargs):
        if route == GlobalRoutes.PATCHOULI_PREPARE_AGENT_RUN:
            return prepared
        elif route == GlobalRoutes.ALICE_RUN_AGENT:
            return chat_result
        elif route == GlobalRoutes.PATCHOULI_FINALIZE_AGENT_RUN:
            return None
        elif route == GlobalRoutes.PATCHOULI_CLEANUP_PREPARED_AGENT_RUN:
            return True
        elif route == GlobalRoutes.ALICE_RUN_AGENT_STREAM:
            async def _stream():
                yield {"event": "token", "data": {"content": "hi"}}
                yield {"event": "done", "data": chat_result.model_dump()}
            return _stream()
        return None

    bus.request = AsyncMock(side_effect=route_dispatch)
    return bus


@pytest.fixture
def passive_config():
    scheduler_tasks = MagicMock()
    scheduler_tasks.observer_idle_flush_timeout_seconds = 30.0
    scheduler_tasks.observer_idle_flush_interval_seconds = 30.0
    scheduler_tasks.enable_observer_idle_flush = True

    scheduler = MagicMock()
    scheduler.tick_seconds = 0.01
    scheduler.shutdown_wait_seconds = 0.1
    scheduler.enabled = False
    scheduler.tasks = scheduler_tasks

    config = MagicMock()
    config.scheduler = scheduler
    return config


def _make_analysis_result(
    *,
    target_topic: str = "NEW_TOPIC",
    memory: str | None = "<mem>ctx</mem>",
    worth_saving: bool = True,
) -> tuple[GatewayDecision, RetrievalResponse]:
    gateway_decision = GatewayDecision(
        target_topic_id=target_topic,
        rewritten_query="resolved query",
        search_keywords=("resolved",),
        memory_write_signal=(
            MemoryWriteSignal.WRITE
            if worth_saving
            else MemoryWriteSignal.SKIP
        ),
        retrieval_plan=RetrievalPlan(),
        intent_type=IntentType.RAG,
    )
    retrieval_result = RetrievalResponse(
        memories=[],
    )
    _ = memory
    return gateway_decision, retrieval_result


def _make_memory_atom(title: str = "Test", user_id: str = "u1") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=MetaData(source_agent_id="a1", user_id=user_id),
        index=IndexLayer(
            title=title,
            summary="A test memory summary",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="test content"),
    )


class TestAgentApplicationService:
    @pytest.fixture
    def service(self, mock_global_bus, passive_config):
        return AgentApplicationService(
            global_bus=mock_global_bus,
            config=passive_config,
        )

    @pytest.mark.asyncio
    async def test_create_agent_profile_uses_public_route(self, service, mock_global_bus):
        created = _make_memory_atom(title="Worker")
        mock_global_bus.request.side_effect = None
        mock_global_bus.request.return_value = created

        atom = await service.create_agent_profile(
            title="Worker",
            alias="worker",
            summary="",
            content="persona",
            tags=["agent"],
            agent_config={"allowed_mtp_verbs": ["SEARCH"]},
        )

        mock_global_bus.request.assert_awaited_once()
        route, payload = mock_global_bus.request.await_args.args
        assert route == GlobalRoutes.PATCHOULI_AGENT_PROFILE_CREATE
        assert payload.index.memory_type == MemoryType.AGENT_PROFILE
        assert payload.index.summary == "Worker agent profile"
        assert payload.index.alias == "worker"
        assert payload.payload.content == "persona"
        assert payload.payload.artifacts.agent_config == {"allowed_mtp_verbs": ["SEARCH"]}
        assert atom is created

    @pytest.mark.asyncio
    async def test_list_agent_profiles_uses_public_route(self, service, mock_global_bus):
        mock_global_bus.request.side_effect = None
        mock_global_bus.request.return_value = []

        assert await service.list_agent_profiles() == []
        mock_global_bus.request.assert_awaited_once_with(
            GlobalRoutes.PATCHOULI_AGENT_PROFILE_LIST,
            limit=100,
        )


