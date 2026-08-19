"""Gateway Phase 3C 固定 workflow 与通用 Step 测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.core.models import TopicSnapshot
from hivememory.core.protocol.gateway import (
    GatewayIngressMode,
    IntentType,
    MemoryWriteSignal,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.engines.gateway.interceptors import create_interceptor
from hivememory.engines.gateway.models import TopicRoutingResult
from hivememory.gateway.analysis import UserQueryAnalysisResult
from hivememory.gateway.commands import (
    SystemCommandDispatcher,
    create_builtin_command_registry,
)
from hivememory.gateway.context import CandidateTopics
from hivememory.gateway.workflow import (
    GatewayExecutionState,
    GatewayWorkflowStep,
    RecoverableGatewayError,
)
from hivememory.gateway.workflow.topology import build_gateway_workflow
from hivememory.system.config import (
    GatewayContextPreparationConfig,
    RuleInterceptorConfig,
    TopicRouterConfig,
    UserQueryAnalysisConfig,
)
from hivememory.system.contracts.runtime_events import RuntimeEventType
from hivememory.system.runtime.events import RecordingRuntimeEventSink
from tests.helpers.workspace import make_access_context


class _Provider:
    def __init__(self) -> None:
        self.prepare_candidate_topics = AsyncMock(
            return_value=CandidateTopics(
                topic_snapshots=(
                    TopicSnapshot(
                        topic_id="topic-1",
                        topic_title="Gateway",
                        workspace_identity=make_access_context(user_id="u1").workspace_identity,
                    ),
                ),
                active_topics_menu="topic-1: Gateway",
            )
        )
        self.prepare_routed_topic = AsyncMock(return_value=None)


class _Router:
    def __init__(self) -> None:
        self.route = AsyncMock(
            return_value=TopicRoutingResult(topic_id="topic-1", reason="匹配")
        )


class _Resolver:
    def __init__(self) -> None:
        self.resolve = AsyncMock(
            return_value=UserQueryAnalysisResult(
                intent_type=IntentType.RAG,
                rewritten_query="分析后的问题",
                # list 输入：验证 finalize 投影前 pydantic 归一化为 tuple
                search_keywords=["gateway"],
                memory_write_signal=MemoryWriteSignal.WRITE,
                retrieval_plan=RetrievalPlan(
                    mode=RetrievalMode.HYBRID,
                    top_k=7,
                ),
            )
        )


def _make_workflow(
    *,
    provider=None,
    router=None,
    resolver=None,
    events=None,
):
    registry = create_builtin_command_registry()
    return build_gateway_workflow(
        interceptor=create_interceptor(RuleInterceptorConfig(), registry),
        command_dispatcher=SystemCommandDispatcher(registry),
        context_provider=provider,
        topic_router=router,
        analysis_resolver=resolver,
        context_config=GatewayContextPreparationConfig(
            candidate_topics_timeout_ms=20,
            routed_topic_timeout_ms=20,
        ),
        topic_router_config=TopicRouterConfig(timeout_ms=20),
        analysis_config=UserQueryAnalysisConfig(overall_timeout_ms=20),
        runtime_events=events,
    )


@pytest.mark.asyncio
async def test_command_branch_dispatches_and_short_circuits_decision_prefix() -> None:
    provider = _Provider()
    router = _Router()
    resolver = _Resolver()
    workflow = _make_workflow(
        provider=provider,
        router=router,
        resolver=resolver,
    )

    result = await workflow.run(
        "/clear",
        access_context=make_access_context(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "command"
    assert result.command_execution_result.client_action["type"] == "clear_chat"
    provider.prepare_candidate_topics.assert_not_awaited()
    provider.prepare_routed_topic.assert_not_awaited()
    router.route.assert_not_awaited()
    resolver.resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_simple_chat_keeps_topic_prefix_and_skips_resolver() -> None:
    provider = _Provider()
    router = _Router()
    resolver = _Resolver()
    workflow = _make_workflow(
        provider=provider,
        router=router,
        resolver=resolver,
    )

    result = await workflow.run(
        "你好",
        access_context=make_access_context(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"
    assert result.decision.target_topic_id == "topic-1"
    assert result.decision.intent_type == IntentType.CHAT
    assert result.decision.rewritten_query == "你好"
    assert result.decision.search_keywords == ()
    assert result.decision.memory_write_signal == MemoryWriteSignal.SKIP
    assert result.decision.retrieval_plan.mode == RetrievalMode.SKIP
    provider.prepare_candidate_topics.assert_awaited_once()
    provider.prepare_routed_topic.assert_awaited_once()
    router.route.assert_awaited_once()
    resolver.resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_standard_branch_applies_one_complete_analysis_result() -> None:
    provider = _Provider()
    router = _Router()
    resolver = _Resolver()
    workflow = _make_workflow(
        provider=provider,
        router=router,
        resolver=resolver,
    )

    result = await workflow.run(
        "继续实现 Gateway",
        access_context=make_access_context(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"
    assert result.decision.search_keywords == ("gateway",)
    resolver.resolve.assert_awaited_once()
    context = resolver.resolve.await_args.args[0]
    assert context.topic_id == "topic-1"
    assert context.candidate_topics.topic_snapshots[0].topic_id == "topic-1"


@pytest.mark.asyncio
async def test_passive_slash_input_is_not_a_system_command() -> None:
    provider = _Provider()
    router = _Router()
    resolver = _Resolver()
    workflow = _make_workflow(
        provider=provider,
        router=router,
        resolver=resolver,
    )

    result = await workflow.run(
        "/clear",
        access_context=make_access_context(user_id="u1"),
        ingress_mode=GatewayIngressMode.PASSIVE_MEMORY,
    )

    assert result.kind == "decision"
    resolver.resolve.assert_awaited_once()
    provider.prepare_candidate_topics.assert_awaited_once()


@pytest.mark.asyncio
async def test_declared_fallbacks_form_a_complete_conservative_decision() -> None:
    events = RecordingRuntimeEventSink()
    workflow = _make_workflow(events=events)

    result = await workflow.run(
        "需要检索的问题",
        access_context=make_access_context(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"
    assert result.decision.target_topic_id == "NEW_TOPIC"
    assert result.decision.intent_type == IntentType.RAG
    assert result.decision.rewritten_query == "需要检索的问题"
    assert result.decision.search_keywords == ()
    assert result.decision.memory_write_signal == MemoryWriteSignal.UNKNOWN
    assert result.decision.retrieval_plan.mode == RetrievalMode.HYBRID
    assert events.events[0].event_type == (
        RuntimeEventType.GATEWAY_WORKFLOW_STARTED.value
    )
    assert events.events[-1].event_type == (
        RuntimeEventType.GATEWAY_WORKFLOW_COMPLETED.value
    )
    completed = [
        event
        for event in events.events
        if event.event_type == RuntimeEventType.GATEWAY_STEP_COMPLETED.value
    ]
    assert [event.data["is_fallback"] for event in completed] == [
        False,
        True,
        True,
        True,
        True,
    ]


@pytest.mark.asyncio
async def test_step_fallback_only_covers_invoke_failures() -> None:
    workflow = _make_workflow()
    state = GatewayExecutionState(
        raw_message="问题",
        access_context=make_access_context(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )
    fallback = Mock(return_value={"topic_id": "NEW_TOPIC"})

    selector_error = GatewayWorkflowStep(
        step_id="selector_error",
        select_input=Mock(side_effect=ValueError("selector")),
        invoke=AsyncMock(),
        project=Mock(),
        fallback=fallback,
    )
    with pytest.raises(ValueError, match="selector"):
        await workflow._run_step(state, selector_error, 0)
    fallback.assert_not_called()

    projector_error = GatewayWorkflowStep(
        step_id="projector_error",
        select_input=lambda _snapshot: "input",
        invoke=AsyncMock(return_value="output"),
        project=Mock(side_effect=ValueError("projector")),
        fallback=fallback,
    )
    with pytest.raises(ValueError, match="projector"):
        await workflow._run_step(state, projector_error, 0)
    fallback.assert_not_called()

    unknown_update = GatewayWorkflowStep(
        step_id="unknown_update",
        select_input=lambda _snapshot: "input",
        invoke=AsyncMock(return_value="output"),
        project=lambda _output: {"unknown": True},
        fallback=fallback,
    )
    with pytest.raises(ValueError, match="未知字段"):
        await workflow._run_step(state, unknown_update, 0)
    fallback.assert_not_called()

    recoverable = GatewayWorkflowStep(
        step_id="recoverable",
        select_input=lambda _snapshot: "input",
        invoke=AsyncMock(side_effect=RecoverableGatewayError("recoverable")),
        project=Mock(),
        fallback=fallback,
    )
    await workflow._run_step(state, recoverable, 0)
    assert state.topic_id == "NEW_TOPIC"
    fallback.assert_called_once()


@pytest.mark.asyncio
async def test_step_timeout_uses_local_fallback() -> None:
    workflow = _make_workflow()
    state = GatewayExecutionState(
        raw_message="问题",
        access_context=make_access_context(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    async def slow_invoke(_selected: str) -> str:
        await asyncio.sleep(0.05)
        return "late"

    step = GatewayWorkflowStep(
        step_id="timeout",
        select_input=lambda _snapshot: "input",
        invoke=slow_invoke,
        project=Mock(),
        timeout_ms=1,
        fallback=lambda _selected, _error: {"topic_id": "NEW_TOPIC"},
    )

    await workflow._run_step(state, step, 0)

    assert state.topic_id == "NEW_TOPIC"
    step.project.assert_not_called()


@pytest.mark.asyncio
async def test_event_sink_failure_does_not_change_business_result() -> None:
    class BrokenEventSink:
        def emit(self, _event) -> None:
            raise RuntimeError("sink failed")

    workflow = _make_workflow(events=BrokenEventSink())

    result = await workflow.run(
        "问题",
        access_context=make_access_context(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"


@pytest.mark.asyncio
async def test_invariant_failure_emits_workflow_failed_event() -> None:
    events = RecordingRuntimeEventSink()
    workflow = _make_workflow(events=events)
    workflow._entry_step = GatewayWorkflowStep(
        step_id="broken_entry",
        select_input=lambda _snapshot: "input",
        invoke=AsyncMock(return_value="output"),
        project=lambda _output: {"unknown": True},
        fallback=lambda _selected, _error: {},
    )

    with pytest.raises(ValueError, match="未知字段"):
        await workflow.run(
            "问题",
            access_context=make_access_context(user_id="u1"),
            ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
        )

    assert events.events[-1].event_type == (
        RuntimeEventType.GATEWAY_WORKFLOW_FAILED.value
    )
