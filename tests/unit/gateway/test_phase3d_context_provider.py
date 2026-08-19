"""Gateway Phase 3D Context Provider 与 Global Bus 测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from hivememory.core.models import (
    TopicLastTurn,
    TopicSnapshot,
)
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    IntentType,
    MemoryWriteSignal,
    RetrievalPlan,
)
from hivememory.engines.gateway.interceptors import create_interceptor
from hivememory.engines.gateway.models import TopicRoutingResult
from hivememory.gateway.analysis import UserQueryAnalysisResult
from hivememory.gateway.commands import (
    SystemCommandDispatcher,
    create_builtin_command_registry,
)
from hivememory.gateway.context import GlobalBusGatewayContextProvider
from hivememory.gateway.errors import RecoverableGatewayError
from hivememory.gateway.workflow.topology import build_gateway_workflow
from hivememory.patchouli.contracts import PatchouliRoutes
from hivememory.system.config import (
    GatewayContextPreparationConfig,
    RuleInterceptorConfig,
    TopicRouterConfig,
    UserQueryAnalysisConfig,
)
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.bus.global_bus import GlobalSystemBus
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from tests.helpers.workspace import make_access_context


@pytest.mark.asyncio
async def test_provider_prepares_candidate_topics_and_menu() -> None:
    bus = GlobalSystemBus()
    access_context = make_access_context(user_id="u1")
    calls = []
    snapshots = (
        TopicSnapshot(
            topic_id="topic-1",
            topic_title="Gateway",
            state_summary="正在实现 Phase 3D",
            last_turn=TopicLastTurn(user="继续", assistant="处理中"),
            workspace_identity=access_context.workspace_identity,
        ),
    )

    async def list_active_topics(**kwargs):
        calls.append(kwargs)
        # 依据 include_empty 参数返回不同结果，使下方透传断言有约束力
        return snapshots if kwargs["include_empty"] else ()

    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, list_active_topics)
    provider = GlobalBusGatewayContextProvider(
        global_bus=bus,
        include_empty_topics=True,
    )

    result = await provider.prepare_candidate_topics(access_context=access_context)

    assert result.topic_snapshots == snapshots
    assert "topic-1" in result.active_topics_menu
    assert "正在实现 Phase 3D" in result.active_topics_menu
    assert "User: 继续" in result.active_topics_menu


@pytest.mark.asyncio
async def test_provider_new_topic_does_not_issue_bus_request() -> None:
    provider = GlobalBusGatewayContextProvider(global_bus=GlobalSystemBus())

    result = await provider.prepare_routed_topic(
        access_context=make_access_context(user_id="u1"),
        topic_id="NEW_TOPIC",
    )

    assert result is None


@pytest.mark.asyncio
async def test_provider_converts_bus_unavailable_to_recoverable_error() -> None:
    provider = GlobalBusGatewayContextProvider(global_bus=GlobalSystemBus())

    with pytest.raises(RecoverableGatewayError, match="candidate topics"):
        await provider.prepare_candidate_topics(
            access_context=make_access_context(user_id="u1")
        )


@pytest.mark.asyncio
async def test_provider_rejects_mutable_or_noncanonical_route_results() -> None:
    bus = GlobalSystemBus()

    async def list_active_topics(**_kwargs):
        return []

    async def get_topic_data(**_kwargs):
        return {"topic_id": "topic-1"}

    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, list_active_topics)
    bus.register(PatchouliRoutes.TOPIC_GET_DATA, get_topic_data)
    provider = GlobalBusGatewayContextProvider(global_bus=bus)
    access_context = make_access_context(user_id="u1")

    with pytest.raises(TypeError, match="tuple"):
        await provider.prepare_candidate_topics(access_context=access_context)
    with pytest.raises(TypeError, match="TopicData"):
        await provider.prepare_routed_topic(
            access_context=access_context,
            topic_id="topic-1",
        )


def _build_provider_workflow(
    *,
    bus: GlobalSystemBus,
    router,
    resolver,
    context_config: GatewayContextPreparationConfig,
    events: RecordingRuntimeEventSink,
):
    registry = create_builtin_command_registry()
    return build_gateway_workflow(
        interceptor=create_interceptor(RuleInterceptorConfig(), registry),
        command_dispatcher=SystemCommandDispatcher(registry),
        context_provider=GlobalBusGatewayContextProvider(global_bus=bus),
        topic_router=router,
        analysis_resolver=resolver,
        context_config=context_config,
        topic_router_config=TopicRouterConfig(timeout_ms=20),
        analysis_config=UserQueryAnalysisConfig(overall_timeout_ms=20),
        runtime_events=events,
    )


def _resolver():
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(
        return_value=UserQueryAnalysisResult(
            intent_type=IntentType.RAG,
            rewritten_query="问题",
            memory_write_signal=MemoryWriteSignal.WRITE,
            retrieval_plan=RetrievalPlan(),
        )
    )
    return resolver


@pytest.mark.asyncio
async def test_candidate_preparation_has_independent_timeout_fallback() -> None:
    bus = GlobalSystemBus()

    async def slow_list(**_kwargs):
        await asyncio.sleep(0.05)
        return ()

    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, slow_list)
    router = AsyncMock()
    router.route = AsyncMock(
        return_value=TopicRoutingResult(topic_id="NEW_TOPIC")
    )
    resolver = _resolver()
    events = RecordingRuntimeEventSink()
    workflow = _build_provider_workflow(
        bus=bus,
        router=router,
        resolver=resolver,
        context_config=GatewayContextPreparationConfig(
            candidate_topics_timeout_ms=1,
            routed_topic_timeout_ms=20,
        ),
        events=events,
    )

    result = await workflow.run(
        "需要处理的问题",
        access_context=make_access_context(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"
    assert router.route.await_args.kwargs["topic_snapshots"] == ()
    fallback_steps = [
        event.data["step_id"]
        for event in events.events
        if event.event_type == RuntimeEventType.GATEWAY_STEP_COMPLETED.value
        and event.data["is_fallback"]
    ]
    assert fallback_steps == ["candidate_topics_preparation"]


@pytest.mark.asyncio
async def test_routed_topic_preparation_has_independent_timeout_fallback() -> None:
    bus = GlobalSystemBus()

    async def list_topics(**_kwargs):
        return (
            TopicSnapshot(
                topic_id="topic-1",
                topic_title="Gateway",
                workspace_identity=make_access_context(user_id="u1").workspace_identity,
            ),
        )

    async def slow_get(**_kwargs):
        await asyncio.sleep(0.05)
        return None

    bus.register(PatchouliRoutes.TOPIC_LIST_ACTIVE, list_topics)
    bus.register(PatchouliRoutes.TOPIC_GET_DATA, slow_get)
    router = AsyncMock()
    router.route = AsyncMock(
        return_value=TopicRoutingResult(topic_id="topic-1")
    )
    resolver = _resolver()
    events = RecordingRuntimeEventSink()
    workflow = _build_provider_workflow(
        bus=bus,
        router=router,
        resolver=resolver,
        context_config=GatewayContextPreparationConfig(
            candidate_topics_timeout_ms=20,
            routed_topic_timeout_ms=1,
        ),
        events=events,
    )

    result = await workflow.run(
        "需要处理的问题",
        access_context=make_access_context(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"
    context = resolver.resolve.await_args.args[0]
    assert context.routed_topic_data is None
    fallback_steps = [
        event.data["step_id"]
        for event in events.events
        if event.event_type == RuntimeEventType.GATEWAY_STEP_COMPLETED.value
        and event.data["is_fallback"]
    ]
    assert fallback_steps == ["routed_topic_preparation"]
