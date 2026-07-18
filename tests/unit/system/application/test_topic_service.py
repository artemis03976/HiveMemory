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
from hivememory.system.application.topic_service import TopicApplicationService
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


class TestTopicApplicationService:
    @pytest.fixture
    def bus(self):
        return GlobalSystemBus()

    @pytest.fixture
    def service(self, bus, passive_config):
        return TopicApplicationService(
            global_bus=bus,
            config=passive_config,
        )

    @pytest.mark.asyncio
    async def test_list_active_topics_uses_public_route(self, service, bus):
        handler = AsyncMock(return_value=["snapshot"])
        bus.register(GlobalRoutes.PATCHOULI_TOPIC_LIST_ACTIVE, handler)

        assert await service.list_active_topics(user_id="u1") == ["snapshot"]
        handler.assert_awaited_once()
        identity = handler.await_args.kwargs["identity"]
        assert identity.user_id == "u1"

    @pytest.mark.asyncio
    async def test_settle_topic_uses_public_route(self, service, bus):
        task = MagicMock(task_id="memtask_1", topic_id="t1")
        handler = AsyncMock(return_value=task)
        bus.register(GlobalRoutes.PATCHOULI_MANUAL_SETTLE_TOPIC, handler)

        result = await service.settle_topic(topic_id="t1")

        assert result == {"success": True, "task_id": "memtask_1", "topic_id": "t1"}
        handler.assert_awaited_once_with(topic_id="t1")

    @pytest.mark.asyncio
    async def test_evict_topic_uses_public_route(self, service, bus):
        handler = AsyncMock(return_value={"success": True, "message": "话题 t1 已删除"})
        bus.register(GlobalRoutes.PATCHOULI_EVICT_TOPIC, handler)

        result = await service.evict_topic(topic_id="t1")

        assert result == {"success": True, "message": "话题 t1 已删除"}
        handler.assert_awaited_once_with(topic_id="t1")


