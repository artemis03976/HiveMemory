"""Gateway 固定 workflow 与通用 Step 测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from hivememory.components.events.bus import RecordingRuntimeEventSink
from hivememory.config.gateway import (
    GatewayContextPreparationConfig,
    RuleInterceptorConfig,
    TopicRouterConfig,
    UserQueryAnalysisConfig,
)
from hivememory.core.contracts.runtime_events import RuntimeEventType
from hivememory.core.models import TopicSnapshot
from hivememory.core.protocol.gateway import (
    CommandParseResult,
    CommandParseStatus,
    GatewayIngressMode,
    GatewaySessionTurn,
    IntentType,
    RetrievalMode,
    RetrievalPlan,
)
from hivememory.engines.gateway.interceptors import create_interceptor
from hivememory.engines.gateway.models import TopicRoutingResult
from hivememory.engines.gateway.topic_router import TopicRouterError
from hivememory.gateway.analysis import UserQueryAnalysisResult
from hivememory.gateway.commands import create_builtin_command_registry
from hivememory.gateway.context import CandidateTopics
from hivememory.gateway.workflow import (
    GatewayExecutionState,
    GatewayWorkflowStep,
    RecoverableGatewayError,
)
from hivememory.gateway.workflow.topology import build_gateway_workflow
from tests.helpers.workspace import make_identity_scope


class _Provider:
    def __init__(self) -> None:
        self.prepare_candidate_topics = AsyncMock(
            return_value=CandidateTopics(
                topic_snapshots=(
                    TopicSnapshot(
                        last_update=1_700_000_000.0,
                        topic_id="topic-1",
                        topic_title="Gateway",
                        workspace_identity=make_identity_scope(user_id="u1").workspace_identity,
                    ),
                ),
                active_topics_menu="topic-1: Gateway",
            )
        )
        self.prepare_routed_topic = AsyncMock(return_value=None)


class _Router:
    def __init__(self) -> None:
        self.route = AsyncMock(return_value=TopicRoutingResult(topic_id="topic-1", reason="匹配"))


class _Resolver:
    def __init__(self) -> None:
        self.resolve = AsyncMock(
            return_value=UserQueryAnalysisResult(
                intent_type=IntentType.RAG,
                rewritten_query="分析后的问题",
                # list 输入：验证 finalize 投影前 pydantic 归一化为 tuple
                search_keywords=["gateway"],
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
async def test_command_branch_returns_parse_result_and_skips_decision_prefix() -> None:
    """主动模式命令只解析：命令结果携带 matched 解析产物，且不运行任何决策步骤。"""
    provider = _Provider()
    router = _Router()
    resolver = _Resolver()
    workflow = _make_workflow(
        provider=provider,
        router=router,
        resolver=resolver,
    )

    result = await workflow.run(
        "/help",
        identity_scope=make_identity_scope(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "command"
    assert result.command_parse_result.parse_status == CommandParseStatus.MATCHED
    assert result.command_parse_result.command_id == "system.help"
    provider.prepare_candidate_topics.assert_not_awaited()
    provider.prepare_routed_topic.assert_not_awaited()
    router.route.assert_not_awaited()
    resolver.resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_slash_input_returns_command_outcome_with_unknown_parse() -> None:
    """未知 slash 输入仍以命令结果收口，解析状态为 unknown；完成事件的
    command_id 在解析失败时回退为解析出的命令名。"""
    events = RecordingRuntimeEventSink()
    workflow = _make_workflow(events=events)

    result = await workflow.run(
        "/no_such_command",
        identity_scope=make_identity_scope(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "command"
    assert result.command_parse_result.parse_status == CommandParseStatus.UNKNOWN
    assert result.command_parse_result.command_id is None
    assert result.command_parse_result.name == "/no_such_command"
    completed = events.events[-1]
    assert completed.event_type == RuntimeEventType.GATEWAY_WORKFLOW_COMPLETED.value
    assert completed.data["outcome_kind"] == "command"
    assert completed.data["command_id"] == "/no_such_command"


def test_passive_state_with_command_flow_end_cannot_finalize() -> None:
    """被动模式不得产生命令结果：命令流程删除执行步骤后，finalize 是唯一守卫。"""
    state = GatewayExecutionState(
        raw_message="/help",
        identity_scope=make_identity_scope(user_id="u1"),
        ingress_mode=GatewayIngressMode.PASSIVE_MEMORY,
        command_parse_result=CommandParseResult(
            command_id="system.help",
            raw_input="/help",
            name="/help",
            parse_status=CommandParseStatus.MATCHED,
        ),
        flow_end_reason="system_command",
    )

    with pytest.raises(RuntimeError, match="PASSIVE_MEMORY 不得产生 command outcome"):
        state.finalize()


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
        identity_scope=make_identity_scope(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"
    assert result.decision.target_topic_id == "topic-1"
    assert result.decision.intent_type == IntentType.CHAT
    assert result.decision.rewritten_query == "你好"
    assert result.decision.search_keywords == ()
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
        identity_scope=make_identity_scope(user_id="u1"),
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
        identity_scope=make_identity_scope(user_id="u1"),
        ingress_mode=GatewayIngressMode.PASSIVE_MEMORY,
    )

    assert result.kind == "decision"
    provider.prepare_candidate_topics.assert_awaited_once()


@pytest.mark.asyncio
async def test_passive_memory_routes_topic_without_query_analysis() -> None:
    """被动入口照常路由话题，但不运行查询分析：决定以不检索的默认值收口。"""
    router = _Router()
    resolver = _Resolver()
    workflow = _make_workflow(provider=_Provider(), router=router, resolver=resolver)

    result = await workflow.run(
        "继续实现 Gateway",
        identity_scope=make_identity_scope(user_id="u1"),
        ingress_mode=GatewayIngressMode.PASSIVE_MEMORY,
    )

    assert result.kind == "decision"
    assert result.decision.target_topic_id == "topic-1"
    assert result.decision.rewritten_query == "继续实现 Gateway"
    assert result.decision.retrieval_plan.mode == RetrievalMode.SKIP
    router.route.assert_awaited_once()
    resolver.resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_declared_fallbacks_form_a_complete_conservative_decision() -> None:
    events = RecordingRuntimeEventSink()
    workflow = _make_workflow(events=events)

    result = await workflow.run(
        "需要检索的问题",
        identity_scope=make_identity_scope(user_id="u1"),
        ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
    )

    assert result.kind == "decision"
    assert result.decision.target_topic_id == "NEW_TOPIC"
    assert result.decision.intent_type == IntentType.RAG
    assert result.decision.rewritten_query == "需要检索的问题"
    assert result.decision.search_keywords == ()
    assert result.decision.retrieval_plan.mode == RetrievalMode.HYBRID
    assert events.events[0].event_type == (RuntimeEventType.GATEWAY_WORKFLOW_STARTED.value)
    assert events.events[-1].event_type == (RuntimeEventType.GATEWAY_WORKFLOW_COMPLETED.value)
    assert {event.workspace_id for event in events.events} == {"main_workspace"}
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


@pytest.mark.parametrize(
    ("previous_topic_id", "expected_target"),
    [("topic-1", "topic-1"), ("topic-evicted", "NEW_TOPIC")],
    ids=["previous-still-a-candidate", "previous-no-longer-a-candidate"],
)
@pytest.mark.asyncio
async def test_topic_routing_failure_continues_previous_topic_while_it_is_a_candidate(
    previous_topic_id: str, expected_target: str
) -> None:
    """话题路由失败时续接上一轮落位的话题；该话题已不在候选中时才新建。"""
    router = _Router()
    router.route = AsyncMock(side_effect=TopicRouterError("路由模型不可用"))
    workflow = _make_workflow(provider=_Provider(), router=router, resolver=_Resolver())

    result = await workflow.run(
        "继续改",
        identity_scope=make_identity_scope(user_id="u1"),
        ingress_mode=GatewayIngressMode.PASSIVE_MEMORY,
        session_turns=(
            GatewaySessionTurn(user_message="改登录页", topic_id="topic-0"),
            GatewaySessionTurn(user_message="修表单校验", topic_id=previous_topic_id),
        ),
    )

    assert result.kind == "decision"
    assert result.decision.target_topic_id == expected_target
    assert result.decision.new_topic_title is None


@pytest.mark.asyncio
async def test_step_fallback_only_covers_invoke_failures() -> None:
    workflow = _make_workflow()
    state = GatewayExecutionState(
        raw_message="问题",
        identity_scope=make_identity_scope(user_id="u1"),
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
        identity_scope=make_identity_scope(user_id="u1"),
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
        identity_scope=make_identity_scope(user_id="u1"),
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
            identity_scope=make_identity_scope(user_id="u1"),
            ingress_mode=GatewayIngressMode.ACTIVE_CHAT,
        )

    failure = events.events[-1]
    assert failure.event_type == RuntimeEventType.GATEWAY_WORKFLOW_FAILED.value
    assert failure.reason == "workflow_failed"
    assert failure.message == "Gateway 处理失败"
    assert failure.data["error_class"] == "ValueError"
    assert "未知字段" not in failure.model_dump_json()
