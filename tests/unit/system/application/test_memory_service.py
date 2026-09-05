"""MemoryApplicationService 委托测试。"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from hivememory.core.models import (
    OMNI_DOLL_PROFILE,
    ActorIdentity,
    IndexLayer,
    MemoryAtom,
    MemoryType,
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
from hivememory.system.application.memory_service import (
    MemoryApplicationService,
    MemoryLifecycleUnavailableError,
    MemoryNotFoundError,
)
from hivememory.system.contracts.routes import GlobalRoutes
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from tests.helpers.workspace import make_identity_scope
from tests.helpers.memory import make_memory_metadata


def _make_prepared_run(**overrides) -> PreparedAgentRun:
    identity = ActorIdentity(user_id="u1", agent_id="omni_doll")
    identity_scope = make_identity_scope(
        actor_identity=identity,
        interaction_id="interaction-test",
    )
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
            identity_scope=identity_scope,
            interaction_id="test-interaction",
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


def _make_memory_atom(title: str = "Test", user_id: str = "u1") -> MemoryAtom:
    return MemoryAtom(
        id=uuid4(),
        meta=make_memory_metadata(source_agent_id="a1", user_id=user_id),
        index=IndexLayer(
            title=title,
            summary="A test memory summary",
            tags=["test"],
            memory_type=MemoryType.FACT,
        ),
        payload=PayloadLayer(content="test content"),
    )


class TestMemoryApplicationService:
    @pytest.fixture
    def service(self, mock_global_bus, passive_config):
        return MemoryApplicationService(
            global_bus=mock_global_bus,
            config=passive_config,
        )

    @pytest.mark.asyncio
    async def test_create_memory_uses_public_route(self, service, mock_global_bus):
        created = _make_memory_atom(title="Created memory")
        mock_global_bus.request.side_effect = None
        mock_global_bus.request.return_value = created

        atom = await service.create_memory(
            title="Created memory",
            summary="A sufficiently long memory summary",
            content="Created memory content",
            memory_type="FACT",
            tags=["created", "ui"],
            alias="created-memory",
            user_id="u1",
        )

        mock_global_bus.request.assert_awaited_once()
        route, identity_scope, payload = mock_global_bus.request.await_args.args
        assert route == GlobalRoutes.PATCHOULI_MEMORY_CREATE
        assert payload.meta.source_agent_id == "ui"
        assert payload.workspace_identity == identity_scope.workspace_identity
        assert payload.workspace_identity.owner_user_id == "u1"
        assert payload.index.memory_type == MemoryType.FACT
        assert payload.index.alias == "created-memory"

    @pytest.mark.asyncio
    async def test_get_memory_not_found_raises_domain_error(self, service, mock_global_bus):
        mock_global_bus.request.side_effect = None
        mock_global_bus.request.return_value = None

        with pytest.raises(MemoryNotFoundError):
            await service.get_memory(uuid4(), user_id="u1")

    @pytest.mark.asyncio
    async def test_record_feedback_without_lifecycle_raises_domain_error(
        self,
        service,
        mock_global_bus,
    ):
        mock_global_bus.request.side_effect = RuntimeError(
            "Memory lifecycle engine is unavailable"
        )

        with pytest.raises(MemoryLifecycleUnavailableError):
            await service.record_feedback(
                uuid4(),
                user_id="u1",
                positive=True,
                source="ui.memory_ref",
            )


